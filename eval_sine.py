import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Define the model architecture (same as training)
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.fc4 = nn.Linear(2, 2)
        self.fc5 = nn.Linear(2, 2)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.fc5(x)
        return x

def prepare_dataset():
    """Prepare the dataset (same as training)"""
    xd = np.linspace(0, 1, 50)
    yd = np.sin(xd * np.pi * 2) + 0.5
    ypd = np.sin(xd * np.pi * 2) - 0.5
    features = torch.tensor(np.concatenate([np.stack([xd,yd]).T, np.stack([xd,ypd]).T])).float()
    labels = torch.tensor(np.concatenate([np.zeros([50]), np.ones([50])])).long()
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=50, shuffle=False)  # Note: shuffle=False for evaluation

def evaluate_model(model, dataloader):
    """Evaluate the model and return detailed metrics"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'true_labels': all_labels
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved model')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(43)
    np.random.seed(43)
    
    # Initialize model and load weights
    model = SimpleDNN()
    try:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare dataset and evaluate
    dataloader = prepare_dataset()
    results = evaluate_model(model, dataloader)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Loss: {results['loss']:.4f}")
    

if __name__ == "__main__":
    main()