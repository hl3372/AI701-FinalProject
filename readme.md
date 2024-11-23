# AI701 Final Project Code
This repository contains the implementation and experiments of a hybrid optimization approach combining Adam with Newton steps. The experiments are conducted on three different tasks: MNIST classification, CIFAR-10 classification, and a sine wave binary classification problem.

## Repository Structure
```
.
├── train_mnist.py     # MNIST training script
├── train_cifar.py     # CIFAR-10 training script
├── train_sine.py      # Sine wave classification training script
├── eval_mnist.py      # MNIST evaluation script
├── eval_cifar.py      # CIFAR-10 evaluation script
├── eval_sine.py       # Sine wave evaluation script
├── trained_weights/   # Saved model weights
└── plots/            # Training results and visualizations
```

## Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU support)
- Matplotlib
- Numpy

### OPTAMI Installation
The hybrid optimizer requires the OPTAMI package. Set it up by:
```bash
git clone https://github.com/OPTAMI/OPTAMI
cd OPTAMI
cp -r OPTAMI/utils .
```

## Running the Experiments

### Training

Each training script can be run directly using Python. The scripts will automatically:
- Download and prepare the datasets (for MNIST and CIFAR-10)
- Train models using different optimizers (Adam, SGD, and Hybrid Adam-Newton)
- Save the trained models in the trained_weights directory
- Generate performance plots

```bash
# Train MNIST
python train_mnist.py

# Train CIFAR-10
python train_cifar.py

# Train Sine Wave Classification
python train_sine.py
```

### Evaluation

The evaluation scripts take a model path as an argument and evaluate the model's performance on the test set.

```bash
# Evaluate MNIST model
python eval_mnist.py --model_path trained_weights/best_hybrid_model_mnist.pth

# Evaluate CIFAR-10 model
python eval_cifar.py --model_path trained_weights/best_hybrid_model_cifar.pth

# Evaluate Sine Wave model
python eval_sine.py --model_path trained_weights/best_hybrid_model_sine.pth
```

## Evaluation Results
All evaluation scripts will output:
- Model accuracy on the test set
- Test loss

### Additional Notes
- For MNIST and CIFAR-10, the data will be automatically downloaded to a `data` directory when you run the training scripts
- The training scripts save three models each:
  - Best hybrid optimizer model
  - Adam baseline model
  - SGD baseline model
- Training progress plots are automatically saved in the working directory

If you have issues with the OPTAMI package, ensure you're in the correct directory when copying the utils folder
