#!/bin/bash

# Stop on error
set -e

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [VENV_DIR]"
    echo "Example: $0 .venv"
    exit 1
fi

VENV_DIR="${1:-${VIRTUAL_ENV:-.venv}}"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Error: could not find executable python at '${PYTHON_BIN}'."
    echo "Pass your venv folder as the first argument, e.g. '$0 /venv/main'."
    exit 1
fi

echo "Using virtual environment: ${VENV_DIR}"

ensure_pkg_resources() {
    if "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
        return 0
    fi

    echo "pkg_resources not found; attempting compatibility installs..."
    if ! uv pip install --python "${PYTHON_BIN}" -U pkg_resources; then
        echo "pkg_resources package unavailable; pinning setuptools<81 for bundled pkg_resources..."
        uv pip install --python "${PYTHON_BIN}" -U "setuptools<81"
    fi

    if "${PYTHON_BIN}" -c "import pkg_resources" >/dev/null 2>&1; then
        return 0
    fi

    echo "pkg_resources still unavailable; writing compatibility shim (pkg_resources -> packaging)..."
    SITE_PACKAGES_DIR="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
    cat <<'PY' > "${SITE_PACKAGES_DIR}/pkg_resources.py"
"""Compatibility shim for packages that only need `from pkg_resources import packaging`."""
import packaging as packaging
__all__ = ["packaging"]
PY

    if "${PYTHON_BIN}" -c "import pkg_resources; from pkg_resources import packaging" >/dev/null 2>&1; then
        return 0
    fi

    echo "Error: pkg_resources is still unavailable in '${VENV_DIR}'."
    echo "Please install setuptools/pkg_resources support in this environment and retry."
    exit 1
}

ensure_timm_builder() {
    if "${PYTHON_BIN}" -c "from timm.models._builder import build_model_with_cfg" >/dev/null 2>&1; then
        return 0
    fi

    echo "timm.models._builder not found; upgrading timm to a compatible version..."
    uv pip install --python "${PYTHON_BIN}" -U "timm>=0.9.16"

    if "${PYTHON_BIN}" -c "from timm.models._builder import build_model_with_cfg" >/dev/null 2>&1; then
        return 0
    fi

    echo "Error: timm is installed but does not provide timm.models._builder."
    echo "Please check for environment pinning that forces an incompatible timm version."
    exit 1
}

ensure_hpsv2_bpe_asset() {
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("hpsv2")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit(0)
pkg_dir = Path(next(iter(spec.submodule_search_locations)))
target = pkg_dir / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
raise SystemExit(0 if target.is_file() else 1)
PY
    then
        return 0
    fi

    echo "hpsv2 tokenizer asset missing; attempting repair..."
    if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
import shutil
import sys
import urllib.request
from pathlib import Path

spec = importlib.util.find_spec("hpsv2")
if spec is None or not spec.submodule_search_locations:
    print("hpsv2 not installed; skipping tokenizer asset repair.")
    raise SystemExit(0)

pkg_dir = Path(next(iter(spec.submodule_search_locations)))
target = pkg_dir / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
if target.is_file():
    raise SystemExit(0)
target.parent.mkdir(parents=True, exist_ok=True)

site_packages = pkg_dir.parent
for candidate in site_packages.rglob("bpe_simple_vocab_16e6.txt.gz"):
    if candidate == target or not candidate.is_file():
        continue
    shutil.copyfile(candidate, target)
    print(f"Installed missing hpsv2 tokenizer asset from {candidate}")
    raise SystemExit(0)

url = "https://openaipublic.blob.core.windows.net/clip/bpe_simple_vocab_16e6.txt.gz"
try:
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    target.write_bytes(data)
    print(f"Downloaded missing hpsv2 tokenizer asset to {target}")
except Exception as exc:
    print(f"Failed to repair hpsv2 tokenizer asset: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
    then
        echo "Error: hpsv2 tokenizer asset is missing and could not be repaired."
        echo "Expected file: <venv>/lib/python*/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
        exit 1
    fi
}

echo "1. Creating dependency overrides for Python 3.12 and ComfyUI compatibility..."
# This fixes the llvmlite/numba build errors on Py3.12
# And fixes the OpenCV/Numpy 2.0 conflict
cat <<EOF > temp_overrides.txt
numba>=0.59.0
llvmlite>=0.42.0
numpy<2.0.0
opencv-python<4.11
opencv-python-headless<4.11
timm>=0.9.16
EOF

echo "2. Installing packaging/runtime tools..."
# Fixes 'No module named pkg_resources' from clip/pyiqa and image-reward install issues.
uv pip install --python "${PYTHON_BIN}" -U setuptools packaging
ensure_pkg_resources

echo "3. Resolving dependencies..."
# Compiles the requirements into a temporary file using the overrides
# --no-build-isolation keeps build-time tooling visible to dependency builds
uv pip compile pyproject.toml \
    --python "${PYTHON_BIN}" \
    -o temp_reqs.txt \
    --no-build-isolation \
    --override temp_overrides.txt

echo "4. Installing resolved dependencies..."
uv pip install --python "${PYTHON_BIN}" -r temp_reqs.txt
ensure_pkg_resources
ensure_timm_builder
ensure_hpsv2_bpe_asset

echo "4b. Verifying pyiqa import..."
if ! "${PYTHON_BIN}" -c "import pyiqa" >/dev/null 2>&1; then
    echo "Error: pyiqa still cannot be imported in '${VENV_DIR}'."
    echo "Try running: ${PYTHON_BIN} -c 'import timm; print(timm.__version__)'"
    exit 1
fi

echo "5. Cleaning up..."
rm temp_overrides.txt temp_reqs.txt

echo "✅ Success! Dependencies installed."
