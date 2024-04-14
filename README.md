# Breast Ultrasound Image Segmentation Project

## Dataset Download
The dataset can be found here: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
For convienience, the dataset is included into this `inputs/` folder of this repository

## Environment setup
The following packages were used:
```
- pytorch >= 2.0
- torchvision
- numpy
- scikit-learn
- pillow
- matplotlib
- tqdm
- icecream
- jupyter (for the notebooks)
```

## Checkpoints
Additional checkpoints too large for this repository can be found in: https://drive.google.com/drive/folders/1xmHW979pR3a5vmzKegH904Yle15j1a0G?usp=sharing

## Running the models
Each model is located in the notebooks:
- `Final_Model_2000epochs.ipynb`: Our model
- `UNet_SOTA.ipynb`: UNET model
- `UNetPlusPlus.ipynb`: UNET++ model

Each notebook is self contained, and will train & output the metric graphs for each model.
