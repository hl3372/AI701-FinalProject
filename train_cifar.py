import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from torch.optim import Optimizer
from torchvision import datasets, transforms
from OPTAMI.OPTAMI.second_order.damped_newton import DampedNewton

class AdamWithNewtonReset(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, n=10, newton_options=None, beta_newton=0.6):
        """Combines Adam optimizer with periodic Newton updates."""
        if newton_options is None:
            newton_options = {}
        self.n = n
        self.step_count = 0
        self.beta_newton = beta_newton  # Weighting factor for Newton step
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params_list = list(params)
        super(AdamWithNewtonReset, self).__init__(params_list, defaults)

        # Create an instance of DampedNewton optimizer
        self.newton_optimizer = DampedNewton(params_list, **newton_options)

        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['m'] = torch.zeros_like(p.data)
                self.state[p]['v'] = torch.zeros_like(p.data)
                self.state[p]['prev_p'] = p.data.clone()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        if closure is None:
            raise ValueError("Closure is required for this optimizer.")

        with torch.enable_grad():
            loss = closure()

        self.step_count += 1

        if self.step_count % self.n == 0 and self.n != 0:
            # Perform Newton step
            self.newton_optimizer.step(closure)

            # Update moment estimates with Newton direction
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    delta_p = p.data - param_state['prev_p']

                    # Update first moment estimate using weighted combination
                    m = param_state['m']
                    m.mul_(1 - self.beta_newton).add_(delta_p, alpha=self.beta_newton)
                    param_state['m'] = m

                    # Optionally update second moment estimate with squared Newton step
                    v = param_state['v']
                    v.mul_(group['betas'][1]).addcmul_(delta_p, delta_p, value=1 - group['betas'][1])
                    param_state['v'] = v

                    # Update previous parameters
                    param_state['prev_p'] = p.data.clone()
        else:
            # Perform standard Adam update
            loss.backward()

            for group in self.param_groups:
                lr = group['lr']
                betas = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                beta1, beta2 = betas

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    if weight_decay != 0:
                        grad = grad.add(p.data, alpha=weight_decay)

                    param_state = self.state[p]
                    m = param_state['m']
                    v = param_state['v']

                    # Update biased first moment estimate
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    param_state['m'] = m

                    # Update biased second moment estimate
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    param_state['v'] = v

                    # We forego bias correction
                    bias_correction1 = 1# - beta1 ** self.step_count
                    m_hat = m / bias_correction1

                    # We forego bias correction
                    bias_correction2 = 1# - beta2 ** self.step_count
                    v_hat = v / bias_correction2

                    # Update parameters
                    step_size = lr / (v_hat.sqrt().add(eps))
                    p.data.add_(-step_size * m_hat)

            self.zero_grad()

        return loss

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # First conv layer: (32x32x3) -> (16x16x4)
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Second conv layer: (16x16x4) -> (8x8x4)
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Third conv layer: (8x8x4) -> (4x4x4)
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Linear(4 * 4 * 4, 10)  # 64 -> 10

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


def train_model(model, optimizer, train_loader, test_loader, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        i = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            i += 1
            if isinstance(optimizer, (AdamWithNewtonReset,)):
                def closure():
                    optimizer.zero_grad()
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    return loss
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)

        accuracy = calculate_accuracy(model, test_loader)
        accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
        if np.isnan(avg_epoch_loss):
            return losses, accuracies
    return losses, accuracies

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


n_values = [20, 30, 50, 75, 100, 150]
L_value = 1.0
beta_newton = 0.15


all_losses = {}
all_accuracies = {}
baseline_losses = {}
baseline_accuracies = {}


model_orig = SimpleCNN().to(device)


print("\nTraining with Adam...")
torch.manual_seed(43)
np.random.seed(43)
model_adam = deepcopy(model_orig).to(device)
adam_opt = torch.optim.Adam(model_adam.parameters(), lr=0.001)
adam_losses, adam_accuracies = train_model(model_adam, adam_opt, train_loader, test_loader, num_epochs=5)
baseline_losses['Adam'] = adam_losses
baseline_accuracies['Adam'] = adam_accuracies

best_model = None
best_acc = 0


for idx, n in enumerate(n_values):
    torch.manual_seed(43)
    np.random.seed(43)
    print(f"\nExperiment {idx+1}/{len(n_values)}: n = {n}")
    model = deepcopy(model_orig).to(device)

    hybrid_opt = AdamWithNewtonReset(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        n=n,
        newton_options={
            'variant': 'GradReg',
            'alpha': 1.0,
            'L': L_value,
            'reg': 0.0,
            'CG_subsolver': True
        },
        beta_newton=beta_newton
    )

    losses, accuracies = train_model(model, hybrid_opt, train_loader, test_loader, num_epochs=5)
    label = f"n={n}"
    all_losses[label] = losses
    all_accuracies[label] = accuracies

    if accuracies[-1] > best_acc:
        best_model = deepcopy(model)
        best_acc = accuracies[-1]


print("\nTraining with SGD...")
torch.manual_seed(43)
np.random.seed(43)
model_sgd = deepcopy(model_orig).to(device)
sgd_opt = torch.optim.SGD(model_sgd.parameters(), lr=0.001, momentum=0.9)
sgd_losses, sgd_accuracies = train_model(model_sgd, sgd_opt, train_loader, test_loader, num_epochs=5)
baseline_losses['SGD'] = sgd_losses
baseline_accuracies['SGD'] = sgd_accuracies



print("\nFinal Losses and Accuracies:")
for label in all_losses:
    final_loss = all_losses[label][-1]
    final_acc = all_accuracies[label][-1]
    print(f"Adam+Newton ({label}): Loss = {final_loss:.4f}, Accuracy = {final_acc:.2f}%")
print("\nBaseline Results:")
for label in baseline_losses:
    final_loss = baseline_losses[label][-1]
    final_acc = baseline_accuracies[label][-1]
    print(f"{label}: Loss = {final_loss:.4f}, Accuracy = {final_acc:.2f}%")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


best_hybrid_label = max(all_accuracies, key=lambda k: all_accuracies[k][-1])


save_model(model_adam, "adam_model_cifar.pth")
save_model(model_sgd, "sgd_model_cifar.pth")
save_model(best_model, f"best_hybrid_model_cifar.pth")


plt.figure(figsize=(20, 10))  

# Training Loss Plot
plt.subplot(1, 2, 1)
for label, losses in all_losses.items():
    plt.plot(losses, label=f'Adam+Newton ({label})')
for label, losses in baseline_losses.items():
    plt.plot(losses, label=label)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.ylim(0, 5)  # Restrict y-axis to range [0, 5]
plt.legend(fontsize='small')
plt.grid(True)

# Test Accuracy Plot
plt.subplot(1, 2, 2)
for label, accuracies in all_accuracies.items():
    plt.plot(accuracies, label=f'Adam+Newton ({label})')
for label, accuracies in baseline_accuracies.items():
    plt.plot(accuracies, label=label)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend(fontsize='small')
plt.grid(True)

plt.savefig("training_results_cifar.png", dpi=300)
print("Plot saved as training_results_cifar.png")
plt.show()