# DeepIRES
DeepIRES: a hybird deep learning model for indentifying internal ribosome entry site in mRNA
## EXPLANATION
This repository contains four folders: model, dataset, weights, data, and result.
### Model
This folder contains python code files for constructing model.
### Dataset
This folder contains orginal data, traing dataet and testing dataset we constructed.
### Weights
This folder saves the model weights we trained.
### Data
You can put your input data in .fa format  in this folder to run prediction program. It already contains our testing sets
### Result
This folder is used to save prediction output file
## Installation of DeepIRES and environment
Download the repository and create corresponding environment.

```
git clone https://github.com/zjupgx/DeepCIP.git
cd ./DeepIRES
conda env create -f environment.yml
``` 
