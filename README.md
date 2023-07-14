LAED <img src="https://github.com/ZhangChenLab/LAED/blob/main/README/OIG.png?raw=true" width="280px" align="right" />
===========
***Leukemia Assessment via End-to-end Deep Learning** 
LAED is an end-to-end deep learning approach for predicting leukemia type. It helps hematologists diagnose leukemia more accurately and efficiently by automatically identifying and classifying the different types of cells in marrow smears.

## Requirements: 
* Windows on 64-bit x86 
* NVIDIA GPU (Tested on Nvidia GeForce RTX 3090)
* Python 3.10 

## Quick start: 
To reproduce the experiments in our paper, please down the dataset from [here](https://figshare.com/articles/dataset/single_cell_dataset/19787371]). 
The following example data are stored under a folder named DATA_DIRECTORY
```bash
DATA_DIRECTORY/train/
	├── ALL
	├── AML
	├── APL
	├── CLL
	└── CML
DATA_DIRECTORY/test/
	├── ALL
	├── AML
	├── APL
	├── CLL
	└── CML
```
Data in one hospital are used to train model, three other hospitals are used to test model's performance. Moreover, the microscopy instruments used in these hospitals differed as well.
