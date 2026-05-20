# PyTorch Learning Lab

A collection of hands-on PyTorch projects covering deep learning fundamentals, training workflows, computer vision, custom datasets, and neural network experimentation.

# Tech Stack

`PyTorch` `torchvision` `NumPy` `Pandas` `Matplotlib` `PIL` `scikit-learn` `Jupyter Notebook`

# Projects

## 04 — Custom Datasets and DataLoaders

### Overview

Built an end-to-end computer vision pipeline using custom PyTorch Dataset classes, DataLoaders, image augmentation, and a TinyVGG convolutional neural network.

The project demonstrates scalable image preprocessing and training workflows commonly used in real-world deep learning projects.

### Key Concepts

* Custom Dataset API
* DataLoader batching
* torchvision transforms
* Image augmentation
* CNN training workflow
* Model inference
* TinyVGG architecture

### Technologies Used

`PyTorch` `torchvision` `Matplotlib` `PIL` `torchinfo`

### Model 0: TinyVGG without data augmentation
<details open>
  <summary>Model 0 architecture</summary>
  
```
TinyVGG(
  (conv_block_1): Sequential(
    (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=1690, out_features=3, bias=True)
  )
)
```

</details>

### Model 1: TinyVGG with Data Augmentation

<details open>
  <summary>Model 1 architecture</summary>
  
```
TinyVGG(
  (conv_block_1): Sequential(
    (0): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=1690, out_features=3, bias=True)
  )
)
```

</details>

### Compare model results

<details open>
  <summary>Model plots</summary>
  
<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/de595064-9642-4448-b6a3-74c59d1aa74d" />

<details>

| Model                  | Train Accuracy | Test Accuracy |
| ---------------------- | -------------- | ------------- |
| TinyVGG Baseline       | 47.66%         | 57.03%        |
| TinyVGG + Augmentation | 42.58%         | 42.19%        |

Training Time:
* Baseline Model → ~7.48 seconds
* Augmented Model → ~7.35 seconds
