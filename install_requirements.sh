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

echo "1. Creating dependency overrides for Python 3.12 and ComfyUI compatibility..."
# This fixes the llvmlite/numba build errors on Py3.12
# And fixes the OpenCV/Numpy 2.0 conflict
cat <<EOF > temp_overrides.txt
numba>=0.59.0
llvmlite>=0.42.0
numpy<2.0.0
opencv-python<4.11
opencv-python-headless<4.11
EOF

echo "2. Installing packaging/runtime tools..."
# Fixes 'No module named pkg_resources' from clip/pyiqa and image-reward install issues.
uv pip install --python "${PYTHON_BIN}" -U setuptools packaging

echo "3. Resolving dependencies..."
# Compiles the requirements into a temporary file using the overrides
# --no-build-isolation allows image-reward to see the packaging tools we just installed
uv pip compile pyproject.toml \
    --python "${PYTHON_BIN}" \
    -o temp_reqs.txt \
    --no-build-isolation \
    --override temp_overrides.txt

echo "4. Installing resolved dependencies..."
uv pip install --python "${PYTHON_BIN}" -r temp_reqs.txt

echo "5. Cleaning up..."
rm temp_overrides.txt temp_reqs.txt

echo "✅ Success! Dependencies installed."
