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
