# SimCLR in PCA

This project applies Principal Component Masking to SimCLR, essentially changing the spatial augmentations used with variance-based augmentations.

![SimCLR-PCA Pipeline](model.png)


## Files

```
data_aug/contrastive_learning_dataset: dataset object creation
data_aug/view_generator: create two views using PCAAugmentor
models/resnet_simclr: architectures
pca.py: apply pca to dataset
PCAAugmentorSimCLR: data augmentation strategies applied (shuffling, interpolation etc)
run: main function
utils: helper functions
```

## Running the model


In order to reproduce the results for SimCLR-PCA, you must first download the dataset you would like to work with into a folder named data.
The second step is to create a relevant PCA basis for the dataset. In **pca.py**, all bases can be computed in one run; however, you can comment out the unwanted datasets to save time.
Additionally, 32*32 bases for Tiny ImageNet are already computed under the outputs folder.
After the PCA basis is computed and saved, simply use **python run.py** with the desired configurations.
This will return a log file including the training and validation contrastive accuracies, followed by a linear classification task.

For patch PCA (position-specific and position-agnostic), firstly run **patch_pca_position_specific.py** and **patch_pca.py** respectively.

## Installing Dependencies

Please run **pip install requirements.txt** for installing the necessary libraries, or install them manually.

