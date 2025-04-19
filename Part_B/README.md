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
├── b_model.py                         # CNN model definition (
│                          
│
├── b_helper.py                        # data handling  and logging into wandb 
│   
│
├── b_train.py.py                      # train ans test model 
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


Strategies:

- Freeze All: Only the classifier layer was trained.

- Freeze Early Layers: The last convolutional block (stage 4) + classifier layer were trained.

- Gradual Unfreezing: The classifier layer was trained initially; later, deeper layers (starting from stage 4) were progressively unfrozen.

Sweep 1 – Grid Search: Explored freeze type (all, gradual), image resolution (224, 299), batch_size (32, 64, 128), and dropout (0.0, 0.2, 0.4).  Found that gradual unfreezing, 299×299 input resolution, and batch size of 64 performed best.

Sweep 2 – Fine-tuning Optimization: Introduced separate learning rates for classifier (head_lr) and stage-4 layers (layer4_lr). Added new hyperparameters like unfreeze_epoch, optimizer, label_smoothing, and weight_decay.

Best Sweep Configuration:
```
{
  "freeze": "gradual",
  "resolution": 299,
  "batch_size": 64,
  "dropout": 0.2,
  "data_aug": true,
  "head_lr": 0.0031,
  "layer4_lr": 0.00014,
  "weight_decay": 1.05e-4,
  "label_smoothing": 0.0645,
  "unfreeze_epoch": 5,
  "optimizer": "Adam"
}
```
This configuration yielded a Test Accuracy of 80.85%.

## Final Results

- Final Model: ResNet50 with gradual unfreezing and learning rate scheduling between layers.

- Best Val Accuracy ~ 81% .

- Best Test Accuracy: ~ 80%

- Trainable Parameters: ~20.5K

- Model Checkpoint: Saved as top1.ckpt
