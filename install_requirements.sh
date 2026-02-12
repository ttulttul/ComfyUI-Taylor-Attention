#!/bin/bash

# Stop on error
set -e

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
uv pip install -U setuptools packaging

echo "3. Resolving dependencies..."
# Compiles the requirements into a temporary file using the overrides
# --no-build-isolation allows image-reward to see the packaging tools we just installed
uv pip compile pyproject.toml \
    -o temp_reqs.txt \
    --no-build-isolation \
    --override temp_overrides.txt

echo "4. Installing resolved dependencies..."
uv pip install -r temp_reqs.txt

echo "5. Cleaning up..."
rm temp_overrides.txt temp_reqs.txt

echo "✅ Success! Dependencies installed."
