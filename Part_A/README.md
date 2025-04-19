## Project Structure (Part A)
```
Assignment_2/
├── data/
|  ├── train                         # Raw iNaturalist train data (to be split into train/val) 
|  └── val                           # iNaturalist test set (used only in final evaluation)
| 
├── models/
│   └── cnn.py                       # CNN model definition (flexible params)
│
├── dataloaders/
│   └── dataloader.py                # Training/val/test data handling
│
├── trainer/
│   └── lightning_wrapper.py         # LightningModule for training
│
├── sweeps/
│   ├── sweep_config.py              # wandb sweep config
│   └── sweep_runner.py              # script to run sweeps
│
├── evaluate_best_model.py           # Train on full train set + test eval
|
├── Prediction Sample Images  
│   ├── classwise sample images of prediction
│
├── README.md                         
├── requirements.txt                 # All dependencies
```
## Dataset Details
*  Train folder: Used for both training and validation (80/20 stratified split)
*  Val folder: Treated as test set (only evaluated at the end)
*  Dataset: Subset of iNaturalist classification dataset with 10 animal classes

## Libraries Used 
*  ```torch```, ```torchvision```, ```pytorch-lightning``` for model training
*  ```wandb``` for experiment tracking and sweeps
*  ```matplotlib```, ```numpy``` for visualization
*  ```sklearn``` for stratified splitting and metrics

  

## How to Run ?
### Step 1 : Install Dependencies
Run the coomand : ```pip install -r requirements.txt```

### Step 2 : Run the sweeps ( Hyper parameter tuning - Q2 )
Run the command : ```python sweeps/sweep_runner.py```

### Step 3 : Evaluate the best model on test data
RUn the command : ```python eval_best_model.py```

### Step 4 : Generate Class-wise Prediction Grid (10x3)
The above program it self will generate the prediction grid for all classes ( 3 images per class ) with True and Predicted label

## Experiment Tracking with wandb
The experiments were tracked using Weights & Biases, and detailed visualizations and insights can be found at the following report: 
https://wandb.ai/da24m020-iit-madras/DA6401_A2/reports/DA6401-Assignment-2-Report--VmlldzoxMjI5NTQ1Nw?accessToken=oc3vijucgg3gitvesrogrwsr64gg68mkl2luos938nw4oqprmpx46b57x0psjcyq

## Some Key Observation 
*  SiLU consistently performed better than ReLU/GELU
*  Nadam and RMSProp outperformed Adam in many configurations
*  Using batchnorm=True and data_aug=True improved accuracy by 5-10%
*  Dropout=0.3, Dense=128, and Filters=64 or 128 gave consistently strong results
*  A focused second phase sweep improved validation accuracy from ~41% to ~50.45%


## Final Results 
* Best Validation Accuracy : ~ 50%
* Highest Test Accuracy on Best Model : ~ 47%

# Fine-tuning a Pre-trained Model ( Part B)
## Overview 
This part of the assignment focuses on leveraging pre-trained CNN architectures (e.g., ResNet50, VGG16, EfficientNet, etc.) to perform transfer learning on a subset of the iNaturalist dataset. The goal is to adapt an ImageNet-trained model for our 10-class classification problem.

## Project Structure (Part B)
```
PartB/
├── data
|  ├── train
|  └── val 
├── train_model.py             # Main training script for fine-tuning
├── sweep_config.py            # Configuration file for wandb hyperparameter sweeps
├── sweep_runner.py            # Script to execute sweep using wandb
└── eval_best_model.py         # Evaluation of the best model on test dataset
```

## Task Detail (Part B)
Fine-tune a model pre-trained on ImageNet for the iNaturalist dataset:
*  Replace the final layer (1000 classes → 10 classes).
*  Resize input images (ImageNet uses 224×224 resolution).
*  Freeze some layers to prevent overfitting and reduce computation.
*  Tune relevant hyperparameters for optimal performance.

## Dataset and Libraries Used
Same as what has been mentioned in Part A 

## Task Performed
**Question 1:**
*  Loaded pre-trained models (e.g., ResNet50, VGG16).
*  Modified the final layer from 1000 → 10 output neurons.
*  Resized input to 224×224 using transforms.
*  Explained how to deal with shape mismatches.

**Question 2:**
Tried multiple fine-tuning strategies:
*  Freezing all layers except layer4
*  Using different optimizers: adam, rmsprop, nadam, sgd
*  Varied batch sizes, learning rates, augmentation strategies
Hyperparameter tuning via Weights & Biases (wandb) sweep.

**Question 3:**
*  Selected the best strategy from sweeps
*  Reported test performance and comparison with training from scratch.

## How to Run ?
### Step 1 : Performing sweeps using wandb
Run the command : ```python Part_B/sweep_runner.py```

### Step 2 : Evaluation of the best model on test data
Run the command : ```python Part_B/eval_best_model.py```

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
