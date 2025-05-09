# SimCLR: Contrastive Representation Learning on STL-10

This repository implements **SimCLR**, a contrastive self-supervised learning framework, applied to the **STL-10 dataset**. The notebook(`notebooks/SimCLR.ipynb`) provides a complete pipeline: from data augmentation and contrastive pretraining, to feature extraction and linear evaluation using logistic regression.

## Overview

**Objective**: Learn transferable image representations without labels using the SimCLR framework, then evaluate them through downstream classification.

**Main Components**:
- Custom contrastive data augmentations.
- SimCLR architecture with ResNet-18 as base encoder.
- InfoNCE contrastive loss implementation.
- Evaluation using logistic regression and full fine-tuning.
- STL-10 dataset: high-resolution images from 10 object categories.



## SimCLR Pretraining

The encoder (`ResNet-18`) is trained on the **unlabeled split** of STL-10 using the **InfoNCE loss**. Each image is randomly augmented twice to form positive pairs.

### Loss Function

We use the InfoNCE loss:
```math
\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \ne i} \exp(\text{sim}(z_i, z_k)/\tau)}
```
where sim is cosine similarity and $\tau$ is the temperature.

### Data Augmentations

Following SimCLR’s design, each image undergoes:
- Random horizontal flips  
- Random resized cropping (96×96)  
- Color jitter (0.5 for brightness/contrast/saturation, 0.1 for hue)  
- Grayscale with p=0.2  
- Gaussian blur  
- Normalization

### Training Details

- Optimizer: Adam  
- Epochs: 5  
- Batch size: 1024  
- Temperature: 0.07

**Final Loss (Epoch 5/5)**:  
- **Validation Loss**: `2.9145`



## Linear Evaluation via Logistic Regression

A linear classifier is trained on frozen representations from the pretrained encoder using STL-10’s labeled split.

### Results (Epoch 100/100)

- **Train Loss**: `1.0051`, Accuracy: `63.70%`  
- **Validation Loss**: `1.1136`, Accuracy: `59.42%`

This demonstrates the model's ability to learn linearly separable features without supervision.


## End-to-End Supervised Training Baseline

For comparison, a ResNet-18 classifier is trained **from scratch** with labels.

### Results (Epoch 40/40)

- **Train Loss**: `0.0005`, Accuracy: `99.84%`  
- **Validation Loss**: `1.4441`, Accuracy: `63.62%`

While the model fits the training data almost perfectly, its **generalization** is only slightly better than the SimCLR features + logistic regression pipeline—showing the robustness of contrastive pretraining.



## References

- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (ICML 2020)
- STL-10 Dataset: https://cs.stanford.edu/~acoates/stl10/
