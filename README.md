# Setup and Run

Having Nvidia video card (mine is RTX 3050 Laptop GPU) and using Ubuntu WSL

```shell
lsb_release -a
No LSB modules are available.
ID: Ubuntu
Description:    Ubuntu 22.04.4 LTS
Release:        22.04
Codename:       jammy    
```

Works using drivers Nvidia drivers `537.58`. Latest drivers as of April 2024 are `552.22` which do not work, i.e. Python `torch` does not `cuda.is_available()`

```shell
sudo apt update && sudo apt -y upgrade
sudo apt install -y libjpeg8
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
# install miniconda. this step is missing from this script
# reason for python=3.10 on below line. Python newer than 3.10 failed further with deps issues later when installing other packages (April 2024, Python 3.12 was being installed by default)
conda create -n py10 python=3.10
conda activate py10
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fastchan fastai fastbook ipywidgets sentencepiece
conda install jupyterlab
```

conda exported environment.yml also supplied, but it will not install cuda/nvidia related stuff above.

Check if torch uses GPU
```python
import torch
torch.cuda.is_available()
# must yield True
```

Run jupyter lab
```shell
jupyter lab
```

Anytime new shell is opened, activate the correct py env via conda
```shell
conda activate py10
```

To convert notebook to python (extract all python code)
```shell
# https://stackoverflow.com/questions/54350254/get-only-the-code-out-of-jupyter-notebook
jupyter nbconvert --no-prompt --to script notebook_name.ipynb
```

Install following for torchviz (visual graphs):
```shell
sudo apt -y install graphviz
pip install torchviz
```

This will also be necessary down the road:
```shell
conda install -c conda-forge kaggle datasets protobuf accelerate
```

And this:
```shell
conda install -c conda-forge opencv easyocr
sudo apt install -y libgl1-mesa-glx
```

```shell
conda install -c conda-forge imagehash
```

# File structure

Files beginning with `fai22` refer to `https://github.com/fastai/course22` 

Files beginning with `ffc` refer to lessons from Part 1 at `https://course.fast.ai/`. These are mostly kaggle notebooks presented in course youtube videos.

Files beginning with `fb` refer to fastbook at `https://github.com/fastai/fastbook/`