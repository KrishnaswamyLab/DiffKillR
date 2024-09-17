# DiffKillR: Killing and Recreating Diffeomorphisms for Cell Annotation in Dense Microscopy Images

<!-- [![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/DiffusionSpectralEntropy.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/DiffusionSpectralEntropy/) -->



## Usage
Train and test DiffeoInvariantNet. (Remove `--use-wandb` if you don't want to use Weights and Biases.)
```
cd src/
python main_DiffeoInvariantNet.py --dataset-name A28 --dataset-path '$ROOT/data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_96x96/' --DiffeoInvariantNet-model AutoEncoder --use-wandb --wandb-username yale-cl2482
```


Train and test DiffeoMappingNet.
```
cd src/
python main_DiffeoMappingNet.py --dataset-name A28 --dataset-path '$ROOT/data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_96x96/' --DiffeoMappingNet-model VM-Diff --use-wandb --wandb-username yale-cl2482
```

### Comparison
1. First train/infer the models. Can refer to `bash/baseline_medt.sh`, `bash/baseline_medt_intra.sh`, `bash/baseline_psm.sh`, `bash/baseline_psm_intra.sh`, `bash/baseline_sam.sh`, `bash/baseline_sam_intra.sh`.

2. Then, stitch the images and run evaluation.
```
cd comparison/eval/
python stitch_patches.py
python evaluate_monuseg.py
python evaluate_glysac.py
```


## Preparation

#### To use SAM.
```
## under `comparison/SAM/checkpoints/`
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### To Use SAM2.
```
## under `comparison/SAM2/checkpoints/`
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

#### To use MedSAM.
```
## under `comparison/MedSAM/checkpoints/`
download from https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view
```

#### To use SAM-Med2D.
```
## under `comparison/SAM_Med2D/checkpoints/`
download from https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view
```

### External Dataset

#### TissueNet
```
cd external_data/TissueNet
# Download from https://datasets.deepcell.org/data
unzip tissuenet_v1.1.zip

python preprocess_tissuenet.py
```


#### MoNuSeg
```
cd external_data/MoNuSeg

cd src/preprocessing
python preprocess_MoNuSeg.py
```


#### GLySAC
```
cd src/preprocessing
python preprocess_GLySAC.py
```

### Environment
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name cellseg pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -c anaconda -c conda-forge

conda activate cellseg
conda install -c anaconda scikit-image scikit-learn pillow matplotlib seaborn tqdm
# conda install -c conda-forge libstdcxx-ng=12
python -m pip install antspyx
# python -m pip install dipy
python -m pip install opencv-python
python -m pip install python-dotenv

# MoNuSeg
python -m pip install xmltodict

# PSM
python -m pip install tensorboardX
python -m pip install shapely
python -m pip install ml_collections
python -m pip install ttach

## LACSS
#python -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install jax==0.4.24
python -m pip install lacss
#python -m pip install ml_dtypes==0.2.0

## StarDist
python -m pip install stardist
python -m pip install tensorflow

# For SAM
python -m pip install git+https://github.com/facebookresearch/segment-anything.git

# For SAM2
python -m pip install git+https://github.com/facebookresearch/segment-anything-2.git

# For MedSAM
python -m pip install git+https://github.com/bowang-lab/MedSAM.git

# For SAM-Med2D
python -m pip install albumentations
python -m pip install scikit-learn==1.1.3  # need to downgrade to 1.1.3


# Export CuDNN
# echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
