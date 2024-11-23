import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, classification_report

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 10)  # 10 classes for MNIST
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x

def prepare_test_dataset():
    """Prepare the MNIST test dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=1000, shuffle=False)

def evaluate_model(model, dataloader, device):
    """Evaluate the model and return detailed metrics"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained MNIST model')
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
    model = SimpleDNN().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare dataset and evaluate
    test_loader = prepare_test_dataset()
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Test Loss: {results['loss']:.4f}")

if __name__ == "__main__":
    main()