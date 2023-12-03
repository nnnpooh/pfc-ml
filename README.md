# General Guide

- https://www.tensorflow.org/install/pip#windows-wsl2

# GPU

## Install NVidia driver on Windows

- https://www.nvidia.com/Download/index.aspx
- Use Game Ready or GeForce Experience Option

## Install Cuda Toolkit in WSL

- https://docs.nvidia.com/cuda/wsl-user-guide/index.html

## Install tensorflow

- `pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1`
- `pip install -U tensorflow[and-cuda]`

## Install additional packages

- `pip3 install pandas jupyterlab matplotlib plotly kaleido`

# CPU Only

- `pip3 install tensorflow pandas jupyterlab matplotlib plotly kaleido`
