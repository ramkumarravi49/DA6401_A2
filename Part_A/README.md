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

We used two sweeps:

Sweep A (v1): A wide-net exploratory sweep with 105 runs (10 epochs each). This sweep helped identify key hyperparameters.

Sweep A (v2): A zoom-in and focused sweep with 32 runs (20 epochs + early stopping), guided by the learnings from v1.

## Observations from sweeps:

Activation Function: Mish consistently outperformed ReLU, GELU, and SiLU. All top-performing configs used Mish.

Dropout: 0.2 emerged as an optimal value; 0.5 was poor and dropped in v2.

Dense Neurons: 256 and 512 gave better accuracy; 256 was more stable.

Filter Organizations: "32_512" and "pyramid_*" structures worked best. Avoided shallow configurations like "16_*".

Learning Rate: Best performance observed in [1.5e-4, 4e-4], peak at ~1.77e-4.

Data Augmentation & BatchNorm: Always enabled in best configs.

One-Cycle Learning Rate Schedule: Added in v2 for final config, boosted val acc by +1.4%.

Batch Size: 64 gave better convergence

Final sweep config that produced the best results:
```
{
 "activation_fn": "mish",
 "filter_organization": "32_512",
 "data_aug": true,
 "batchnorm": true,
 "dropout": 0.4,
 "dense_neurons": 256,
 "learning_rate": 0.000177,
 "filter_size": 3,
 "batch_size": 64
}
```
## Final Results
Best Validation Accuracy: 46.3%
(achieved after adding One-Cycle LR + better augmentation on top sweep config)

Test Accuracy (on best model):~ 47%

Model Size: ~3.7M parameters

Prediction Grid: A 10x3 creative visualization of predictions from the test set is included as Final_prediction.png


## Final Results 
* Best Validation Accuracy : ~ 46%
* Highest Test Accuracy on Best Model : ~ 47%


