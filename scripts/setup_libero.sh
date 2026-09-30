#!/usr/bin/env bash
# Build <repo>/.venv (Python 3.10) for a LIBERO-family simulator repo -- LIBERO, LIBERO-plus
# or LIBERO-PRO. All three install as the same `libero` package, so each one gets its own
# venv. Show-Harness code runs INSIDE these venvs (through PYTHONPATH), never the other way
# round, so the harness requirements go in too.
#
#   bash scripts/setup_libero.sh <repo_dir> [extra `uv pip install` args ...]
#   bash scripts/setup.sh libero <repo_dir>          # same thing through the usual entry point
#
# LIBERO-plus additionally needs ImageMagick (wand, imported at the top of envs/env_wrapper.py).
# Without root, `apt install libmagickwand-dev` is out; unpack the same Ubuntu debs into a
# prefix and point MAGICK_HOME at it:
#
#   apt-get download libmagickwand-6.q16-6 libmagickcore-6.q16-6 libmagickcore-6.q16-6-extra \
#                    libmagickwand-6.q16-dev libmagickcore-6.q16-dev libfftw3-double3 liblqr-1-0
#   for d in *.deb; do dpkg -x "$d" _root; done
#   mkdir -p <prefix>/lib && cp -a _root/usr/lib/x86_64-linux-gnu/* <prefix>/lib/
#   MAGICK_HOME=<prefix> bash scripts/setup_libero.sh <ws>/LIBERO-plus \
#       -r <ws>/LIBERO-plus/extra_requirements.txt
#
# Pins, and why:
#   torch 2.6.0+cu124  -- cu124 runs on driver 535. >=2.6 flipped torch.load's default to
#                         weights_only, and LIBERO's init states are pickles, which is why the
#                         .pth below sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1.
#   numpy 1.26.4       -- the harness needs >=1.23.5,<2 and opencv 4.10-4.11, so the repos'
#   opencv (harness)      own numpy==1.22.4 / opencv==4.6 pins are dropped rather than fought.
#   mujoco 3.3.2       -- what these venvs settled on; keep the three repos on one version so
#                         a state restored in one renders identically in another.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # <repo>/scripts
SH="${SHOWHARNESS_ROOT:-$(dirname "$HERE")}"                  # Show-Harness itself

[ $# -ge 1 ] || { sed -n '2,10p' "$0" | sed 's/^# \?//' >&2; exit 1; }
[ -d "$1" ] || { echo "ERROR: not a directory: $1" >&2; exit 1; }
REPO="$(realpath "$1")"; shift
[ -f "${REPO}/requirements.txt" ] || {
  echo "ERROR: ${REPO} has no requirements.txt -- is it a LIBERO-family repo?" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || {
  echo "ERROR: uv is required. https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1; }

# Default the uv cache next to the checkouts rather than under $HOME: on a cluster $HOME is
# usually the slow shared volume, and this cache is written once per package and read often.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "${SH}")/.cache/uv}"
VENV="${REPO}/.venv"
PY="${VENV}/bin/python"

uv venv --managed-python --python 3.10 --seed "${VENV}"
uv pip install --python "${PY}" torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

grep -v -E '^\s*(numpy|opencv-python)\b' "${REPO}/requirements.txt" > "${VENV}/requirements.relaxed.txt"
uv pip install --python "${PY}" \
    -r "${VENV}/requirements.relaxed.txt" \
    -r "${SH}/requirements/requirements.txt" \
    numpy==1.26.4 mujoco==3.3.2 h5py "$@"
# compat: libero/ ships no __init__.py, so the default editable finder maps nothing and the
# import silently resolves to whatever else is on the path.
uv pip install --python "${PY}" -e "${REPO}" --config-settings editable_mode=compat

# robosuite logs to a hard-coded /tmp/robosuite.log. On a shared machine another user owns
# that file and `import robosuite` dies with PermissionError before anything of ours runs.
# A private macros file = the defaults with file logging off.
RS="$("${PY}" -c 'import importlib.util as u; print(u.find_spec("robosuite").submodule_search_locations[0])')"
sed 's/^FILE_LOGGING_LEVEL = .*/FILE_LOGGING_LEVEL = None  # \/tmp\/robosuite.log is not ours on a shared machine/' \
    "${RS}/macros.py" > "${RS}/macros_private.py"

# Per-repo LIBERO config. NEVER ~/.libero: on a cluster that is one NFS file shared with every
# other machine and every other checkout, so whichever ran last decides where all three repos
# think their assets live.
mkdir -p "${REPO}/.libero"
ROOT="${REPO}/libero/libero"
cat > "${REPO}/.libero/config.yaml" <<EOF
assets: ${ROOT}/assets
bddl_files: ${ROOT}/bddl_files
benchmark_root: ${ROOT}
datasets: ${LIBERO_DATASETS:-${REPO}/datasets}
init_states: ${ROOT}/init_files
EOF

# Set at interpreter start-up so nothing depends on `activate` -- these venvs are normally
# invoked by absolute path (<repo>/.venv/bin/python) from a runner living elsewhere.
SITE="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NAME="$(basename "${REPO}" | tr 'A-Z-' 'a-z_')"
{
  printf 'import os, ctypes; '
  printf 'os.environ.setdefault("LIBERO_CONFIG_PATH", "%s"); ' "${REPO}/.libero"
  # wand dlopens MagickWand by absolute path and those libs carry no RPATH, so MagickCore and
  # its two dependencies are preloaded, in order. (conda-forge ImageMagick is a dead end here:
  # its libheif wants a newer libstdc++ than the one torch has already loaded, and its
  # libgobject a newer glib than mujoco's -- both surface only as wand's unhelpful
  # "MagickWand shared library not found".)
  if [ -n "${MAGICK_HOME:-}" ]; then
    printf 'os.environ.setdefault("MAGICK_HOME", "%s"); ' "${MAGICK_HOME}"
    MODULES="$(ls -d "${MAGICK_HOME}"/lib/ImageMagick-* 2>/dev/null | head -1)"
    [ -n "${MODULES}" ] && printf 'os.environ.setdefault("MAGICK_CODER_MODULE_PATH", "%s/modules-Q16/coders"); ' "${MODULES}"
    # One statement per library, never a comprehension: a .pth line runs under site.py's
    # exec(), where names imported on that same line are function locals -- and a
    # comprehension's own scope cannot see them. It raises NameError, which site.py reports
    # as "Remainder of file ignored", so the rest of the line silently never runs.
    for _lib in libfftw3.so.3 liblqr-1.so.0 libMagickCore-6.Q16.so.6; do
      printf 'ctypes.CDLL("%s/lib/%s", mode=ctypes.RTLD_GLOBAL); ' "${MAGICK_HOME}" "${_lib}"
    done
  fi
  printf 'os.environ.setdefault("MUJOCO_GL", "egl"); '
  printf 'os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")\n'
} > "${SITE}/${NAME}_local.pth"

cat <<EOF

LIBERO environment ready.

  venv   : ${VENV}
  config : ${REPO}/.libero/config.yaml
  .pth   : ${SITE}/${NAME}_local.pth

Run the harness inside it, from the Show-Harness checkout:

  cd ${SH} && PYTHONPATH=\$PWD MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \\
    ${PY} scripts/run_libero_mvtoken.py --help
EOF
