# Fine-tuning a Pre-trained Model ( Part B)
## Overview 
This part of the assignment focuses on leveraging pre-trained CNN architectures (e.g., ResNet50, VGG16, EfficientNet, etc.) to perform transfer learning on a subset of the iNaturalist dataset. The goal is to adapt an ImageNet-trained model for our 10-class classification problem.

## Project Structure (Part A)
```
Assignment_2/
├── data/
|  ├── train                         # Raw iNaturalist train data (to be split into train/val) 
|  └── test                          # iNaturalist test set (used only in final evaluation)
|
|
├── top1.ckpt                         #best model trained weights  saved
├── top2.ckpt
|
├── b_model.py                         # CNN model definition (flexible params)
│                          
│
├── b_helper.py                        # data handling  and logging into wandb 
│   
│
├── b_train.py.py                      # train ans test model and class wise predction grid 
│   
│   
│
├── README.md                         
├── requirements.txt                 # All dependencies
```
## Dataset Details
*  Train folder: Used for both training and validation (80/20 stratified split)
*  Test folder: Treated as test set (only evaluated at the end)
*  Dataset: Subset of iNaturalist classification dataset with 10 animal classes

## Libraries Used 
*  ```torch```, ```torchvision```, ```pytorch-lightning``` for model training
*  ```wandb``` for experiment tracking and sweeps
*  ```matplotlib```, ```numpy``` for visualization
*  ```sklearn``` for stratified splitting and metrics

  

## How to Run ?
### Step 1 : Install Dependencies
Run the coomand : ```pip install -r requirements.txt```

### Step 2 : Evaluate the best model on test data and generate class-wise prediction grid
RUn the command : ```b_train.py```



## Experiment Tracking with wandb
The experiments were tracked using Weights & Biases, and detailed visualizations and insights can be found at the following report: 
https://wandb.ai/cs24m037-iit-madras/DA6401_A2/reports/DA6401-Assignment-2--VmlldzoxMjM0MTczMw?accessToken=xbk3gqrjb1d51zy7fg0mingbtte79xdn9cwer3idoii6lawmrn13pj21piqiw6iq


## Key Observations
*  ResNet50 performed very well and achieved validation accuracy above 85.
*  Layer freezing helps significantly on small datasets.
*  Adam and Nadam generally performed better than SGD.
*  Wandb sweeps helped in exploring a large hyperparameter space efficiently.

## Final Results
*  Best Validation Accuracy : ~ 86%
*  Highest Test Accuracy on Test Dataset : ~ 85% 

## Final Note
All the experiments related to hyper parameter tuning and metric evaluation have been perfomed in kaggle jupyter environment only.
All the experiments are reprocible but make sure to them in environment with powerful computational resources (15gb+ CPU and 10gb+ GPU i.e. Kaggle,Colab). 
