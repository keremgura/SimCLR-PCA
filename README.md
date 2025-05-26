# SimCLR in PCA

![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)


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

