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

### Data transforming

<details open>
  <summary>Image</summary>

<img width="389" height="276" alt="image" src="https://github.com/user-attachments/assets/2ae64109-3d8b-4364-abcc-dc2fb8897b45" />

</details>

### Data classification

<details open>
  <summary>Image</summary>

<img width="1283" height="276" alt="image" src="https://github.com/user-attachments/assets/17636f78-d0f4-475c-aca4-4493b3edce35" />

</details>

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

</details>
  
| Model                  | Train Accuracy | Test Accuracy |
| ---------------------- | -------------- | ------------- |
| TinyVGG Baseline       | 47.66%         | 57.03%        |
| TinyVGG + Augmentation | 42.58%         | 42.19%        |

Training Time:
* Baseline Model → ~7.48 seconds
* Augmented Model → ~7.35 seconds

</details>

## 03 — Computer Vision with PyTorch

### Overview

Built multiple computer vision models using PyTorch and the FashionMNIST dataset, progressing from a linear baseline model to a convolutional neural network (CNN).

The project demonstrates the complete deep learning workflow for image classification, including data loading, visualization, training loops, evaluation, and model saving/loading.

### Key Concepts

* FashionMNIST dataset
* DataLoader batching
* Training and testing loops
* Model evaluation
* CNN architecture
* GPU training
* Confusion matrix analysis
* Model saving and loading

### Technologies Used

`PyTorch` `torchvision` `Matplotlib` `torchmetrics` `mlxtend`

### Dataset Visualization

<details open>
  <summary>Dataset samples</summary>

<img width="716" height="735" alt="image" src="https://github.com/user-attachments/assets/60ac7e5b-f585-4464-93f4-5319756780e8" />

</details>

### Model 0: Baseline Linear Model

<details open>
  <summary>Model 0 architecture</summary>

```text
FashionMNISTModelV0(
  (layer_stack): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=10, bias=True)
  )
)
```

</details>

### Model 1: Non-Linear Neural Network

<details open>
  <summary>Model 1 architecture</summary>

```text
FashionMNISTModelV1(
  (layer_stack): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=10, bias=True)
    (2): ReLU()
    (3): Linear(in_features=10, out_features=10, bias=True)
    (4): ReLU()
    (5): Linear(in_features=10, out_features=10, bias=True)
  )
)
```

</details>

### Model 2: Convolutional Neural Network (CNN)

<details open>
  <summary>Model 2 architecture</summary>

```text
FashionMNISTModelV2(
  (conv_block_1): Sequential(
    (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2)
  )

  (conv_block_2): Sequential(
    (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2)
  )

  (classifier): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=490, out_features=10, bias=True)
  )
)
```

</details>

### Model Predictions

<details open>
  <summary>Predictions</summary>

<img width="733" height="732" alt="image" src="https://github.com/user-attachments/assets/3169c62e-0a88-4fc5-848e-3356cf13f291" />

</details>

### Confusion Matrix

<details open>
  <summary>Prediction evaluation</summary>

<img width="662" height="650" alt="image" src="https://github.com/user-attachments/assets/ccec3ae0-3c34-4ac2-9997-074199fd2bfc" />

</details>

### Compare Model Results

<details open>
  <summary>Training results and plots</summary>

<img width="701" height="432" alt="image" src="https://github.com/user-attachments/assets/b20d2f10-72e2-430c-aedd-32af6d37d905" />

</details>

| Model                     | Test Accuracy |
| ------------------------- | ------------- |
| Baseline Linear Model     | 83.43%        |
| Non-Linear Neural Network | 75.02%        |
| CNN Model                 | 88.24%        |

### Best Performing Model

`FashionMNISTModelV2 (CNN)`

Metrics:

* Test Accuracy → 88.24%
* Test Loss → 0.3247

## 02 — PyTorch Classification Models

### Overview

Built multiple binary c<img width="833" height="582" alt="image" src="https://github.com/user-attachments/assets/c821cc91-2cb9-4eb5-b12d-410a565f2403" />
lassification models using PyTorch, covering the complete machine learning workflow from synthetic dataset generation to model evaluation and visualization.

The project demonstrates how neural networks learn decision boundaries using linear and non-linear architectures.

### Key Concepts

* Binary classification
* Synthetic dataset generation
* Neural network training loops
* Loss functions
* Gradient descent optimization
* Decision boundary visualization
* Linear vs non-linear models
* Model evaluation

### Technologies Used

`PyTorch` `Matplotlib` `scikit-learn` `NumPy`

### DataFrame of circle data

<details open>
  <summary>Circle data</summary>

<img width="559" height="413" alt="image" src="https://github.com/user-attachments/assets/eeeff65b-cbe5-4e53-9796-d6364e64b781" />

</details>

### Model 0: Linear Classification Model

<details open>
  <summary>Model 0 architecture</summary>

```text id="9egp0n"
CircleModelV0(
  (layer_1): Linear(in_features=2, out_features=5, bias=True)
  (layer_2): Linear(in_features=5, out_features=1, bias=True)
)
```

</details>

### Linear Decision Boundary

<details open>
  <summary>Prediction boundaries</summary>

<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/7d2ed589-bfc1-47fc-bdf0-597a8867ccd8" />

</details>

### Model 1: Improved Non-Linear Classification Model

<details open>
  <summary>Model 1 architecture</summary>

```text id="rj8u7m"
CircleModelV1(
  (layer_stack): Sequential(
    (0): Linear(in_features=2, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=10, bias=True)
    (3): ReLU()
    (4): Linear(in_features=10, out_features=1, bias=True)
  )
)
```

</details>

### Non-Linear Decision Boundary

<details open>
  <summary>Prediction boundaries</summary>

<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/0d7da015-5ece-4d0c-8c31-fbf955dd5e2d" />

</details>

### Compare Model Results

| Model                           | Test Accuracy |
| ------------------------------- | ------------- |
| Linear Classification Model     | 50.00%        |
| Non-Linear Classification Model | 99.50%        |

### Best Performing Model

`CircleModelV1`

Metrics:

* Test Accuracy → 99.50%
* Binary Classification Performance → Excellent separation of non-linear classes

### Multi-class Dataset

<details open>
  <summary>Circle data</summary>

<img width="833" height="582" alt="image" src="https://github.com/user-attachments/assets/738a5407-fbc4-48cb-9775-01dd7ee6c393" />

</details>

### Multi-class Decision Boundary

<details open>
  <summary>Circle data</summary>

<img width="999" height="528" alt="image" src="https://github.com/user-attachments/assets/28c4bd91-3a06-4e3a-b50b-920f0661a1cb" />

</details>

| Train Loss     | Test Loss      | Train Accuracy | Test Accuracy  |
| -------------- | -------------- | -------------- | -------------- |
| 0.0352         | 0.0266         | 99.25%         | 99.50%         |
