import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from torch.optim import Optimizer
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

        if self.step_count % self.n == 0 and n!=0:
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

                    # Compute bias-corrected first moment estimate
                    bias_correction1 = 1 - beta1 ** self.step_count
                    m_hat = m / bias_correction1

                    # Compute bias-corrected second moment estimate
                    bias_correction2 = 1 - beta2 ** self.step_count
                    v_hat = v / bias_correction2

                    # Update parameters
                    step_size = lr / (v_hat.sqrt().add(eps))
                    p.data.add_(-step_size * m_hat)

            self.zero_grad()

        return loss

# Prepare the data
xd = np.linspace(0,1,50)
yd = np.sin(xd*np.pi*2)+0.5
ypd = np.sin(xd*np.pi*2)-0.5
features = torch.tensor(np.concatenate([np.stack([xd,yd]).T,np.stack([xd,ypd]).T])).float()
labels = torch.tensor(np.concatenate([np.zeros([50]),np.ones([50])])).long()
dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)


torch.manual_seed(43)
np.random.seed(43)


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


def train_model(model, optimizer, dataloader, num_epochs=3000):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        for batch_features, batch_labels in dataloader:
            def closure():
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                return loss

            loss = optimizer.step(closure)
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)

        accuracy = calculate_accuracy(model, dataloader)
        accuracies.append(accuracy)

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        if np.isnan(avg_epoch_loss):
            return losses, accuracies
    return losses, accuracies

def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


n_values = [2, 3, 5, 10, 20]
all_losses = {}
all_accuracies = {}
baseline_losses = {}
baseline_accuracies = {}

model_orig = SimpleDNN()

best_model = None
best_acc = 0


for n in n_values:
    print(f"\nExperiment with n={n}")
    model = deepcopy(model_orig)
    
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
            'L': 1.0,
            'reg': 0.0,
            'CG_subsolver': True
        }
    )

    losses, accuracies = train_model(model, hybrid_opt, dataloader, num_epochs=6000)

    if accuracies[-1] > best_acc:
        best_model = deepcopy(model)
        best_acc = accuracies[-1]
    label = f"n={n}"
    all_losses[label] = losses
    all_accuracies[label] = accuracies

# Train with baseline optimizers
print("\nTraining with SGD+Momentum...")
model_sgd = deepcopy(model_orig)
sgd_opt = torch.optim.SGD(model_sgd.parameters(), lr=0.001, momentum=0.9)
sgd_losses, sgd_accuracies = train_model(model_sgd, sgd_opt, dataloader, num_epochs=6000)
baseline_losses['SGD+Momentum'] = sgd_losses
baseline_accuracies['SGD+Momentum'] = sgd_accuracies

print("\nTraining with Adam...")
model_adam = deepcopy(model_orig)
adam_opt = torch.optim.Adam(model_adam.parameters(), lr=0.001)
adam_losses, adam_accuracies = train_model(model_adam, adam_opt, dataloader, num_epochs=6000)
baseline_losses['Adam'] = adam_losses
baseline_accuracies['Adam'] = adam_accuracies

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


best_hybrid_label = max(all_accuracies, key=lambda k: all_accuracies[k][-1])


save_model(model_adam, "adam_model_sine.pth")
save_model(model_sgd, "sgd_model_sine.pth")
save_model(best_model, f"best_hybrid_model_sine.pth")


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


plt.savefig("training_results_sine.png", dpi=300)
print("Plot saved as training_results_sine.png")
plt.show()