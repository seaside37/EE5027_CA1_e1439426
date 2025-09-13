import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

# load data
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

X = np.vstack((X1, X2))
T = np.array([0]*len(X1) + [1]*len(X2))

# prepare data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
T_tensor = torch.tensor(T, dtype=torch.float32).view(-1,1)

# MLP model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.hidden = nn.Linear(2, 3)   # hidden layer
        self.output = nn.Linear(3, 1)   # output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
    
model = SimpleMLP()

# Dcision Boundary
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)
with torch.no_grad():
    probs = model(grid_tensor).numpy().ravel()

Z = (probs > 0.5).astype(int).reshape(xx.shape)


# Accuracy, Precision, Recall
with torch.no_grad():
    y_prob = model(X_tensor).numpy().ravel()
y_pred = (y_prob > 0.5).astype(int)

accuracy = accuracy_score(T, y_pred)
precision = precision_score(T, y_pred)
recall = recall_score(T, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(T, y_prob)
roc_auc = auc(fpr, tpr)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12,6))

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
axes[0].scatter(X[:,0], X[:,1], c=T, cmap=plt.cm.coolwarm, edgecolors='k')
axes[0].set_title('Initial PyTorch MLP Decision Boundary')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')

axes[1].plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (AUC = %0.2f)' % roc_auc)
axes[1].plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0,1.0])
axes[1].set_ylim([0.0,1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve of Initial PyTorch MLP')
axes[1].legend(loc="lower right")
axes[1].grid(True)

plt.tight_layout()
plt.show()