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


