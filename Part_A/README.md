## Task details (Part A)
*  Build a configurable CNN from scratch using PyTorch
*  Perform hyperparameter tuning using wandb sweeps
*  Evaluate the best model on the test set
*  Visualize predictions in a creative 10x3 grid format
  
## Project Structure (Part A)
```
Assignment_2/
├── data/
|  ├── train                         # Raw iNaturalist train data (to be split into train/val) 
|  └── test                          # iNaturalist test set (used only in final evaluation)
|
|
├── best_model.ckpt                  #best model trained weights  saved  
|
├── model.py                         # CNN model definition (flexible params)
│                          
│
├── helper.py                        # data handling  and logging into wandb 
│   
│
├── train.py.py                      # train ans test model and class wise predction grid 
│   
│
|
├── Final_prediction.png             # Test Reulst Sample Image
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
RUn the command : ```train.py```



## Experiment Tracking with wandb
The experiments were tracked using Weights & Biases, and detailed visualizations and insights can be found at the following report: 
https://wandb.ai/cs24m037-iit-madras/DA6401_A2/reports/DA6401-Assignment-2--VmlldzoxMjM0MTczMw?accessToken=xbk3gqrjb1d51zy7fg0mingbtte79xdn9cwer3idoii6lawmrn13pj21piqiw6iq

## Some Key Observation 
*  SiLU consistently performed better than ReLU/GELU
*  Nadam and RMSProp outperformed Adam in many configurations
*  Using batchnorm=True and data_aug=True improved accuracy by 5-10%
*  Dropout=0.3, Dense=128, and Filters=64 or 128 gave consistently strong results
*  A focused second phase sweep improved validation accuracy from ~41% to ~50.45%


## Final Results 
* Best Validation Accuracy : ~ 50%
* Highest Test Accuracy on Best Model : ~ 47%


