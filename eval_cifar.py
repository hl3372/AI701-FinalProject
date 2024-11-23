import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import argparse

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Linear(4 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def prepare_test_dataset():
    """Prepare the CIFAR-10 test dataset"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return DataLoader(test_dataset, batch_size=1000, shuffle=False)

def evaluate_model(model, dataloader, device):
    """Evaluate the model and return metrics"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Evaluating batch {batch_idx}/{len(dataloader)}...', end='\r')
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'total_correct': correct,
        'total_samples': total
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained CIFAR-10 model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved model')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(43)
    np.random.seed(43)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and load weights
    model = SimpleCNN().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare dataset and evaluate
    print("Preparing test dataset...")
    test_loader = prepare_test_dataset()
    
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Test Accuracy: {results['accuracy']:.2f}% ({results['total_correct']}/{results['total_samples']})")
    print(f"Test Loss: {results['loss']:.4f}")

if __name__ == "__main__":
    main()