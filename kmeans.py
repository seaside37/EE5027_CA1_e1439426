import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

# load data
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

X = np.vstack((X1, X2))
T = np.array([0]*len(X1) + [1]*len(X2))

# prepare data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
T_tensor = torch.tensor(T, dtype=torch.float32).view(-1,1)

# RBF model
class RBFNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=3, sigma=1.0):
        super(RBFNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        
        kmeans = KMeans(n_clusters=hidden_dim, n_init=10)
        kmeans.fit(X)  # X: numpy array of shape [N,2]
        centers_np = kmeans.cluster_centers_

        self.centers = nn.Parameter(torch.tensor(centers_np, dtype=torch.float32),
                                    requires_grad=False)  # centers fixed
    
    def rbf_layer(self, x):
        # x: [N, input_dim], centers: [hidden_dim, input_dim]
        # Expand dims to broadcast
        x_expanded = x.unsqueeze(1)  # [N,1,2]
        centers_expanded = self.centers.unsqueeze(0)  # [1,hidden_dim,2]
        dist_sq = torch.sum((x_expanded - centers_expanded)**2, dim=2)  # [N,hidden_dim]
        phi = torch.exp(-dist_sq / (2*self.sigma**2))  # Gaussian
        return phi
    
model = RBFNetwork(input_dim=2, hidden_dim=9, sigma=1.0)
phi = model.rbf_layer(X_tensor)
# Training parameters
N = phi.shape[0]
phi_aug = torch.cat([phi, torch.ones(N,1)], dim=1)  # [N, hidden_dim+1]
# W: [hidden_dim+1, 1]
W = torch.linalg.pinv(phi_aug) @ T_tensor

# Resulting Metrics
y_pred_prob = phi_aug @ W
y_pred_label = (y_pred_prob >= 0.5).int()

accuracy = accuracy_score(T, y_pred_label.numpy())
precision = precision_score(T, y_pred_label.numpy())
recall = recall_score(T, y_pred_label.numpy())
print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
# Decision Boundary
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)
phi_grid = model.rbf_layer(grid_tensor)
phi_grid_aug = torch.cat([phi_grid, torch.ones(phi_grid.shape[0],1)], dim=1)
probs_grid = (phi_grid_aug @ W).numpy().ravel()
Z = (probs_grid > 0.5).astype(int).reshape(xx.shape)

# ROC Curve
fpr, tpr, thresholds = roc_curve(T, y_pred_prob.numpy())
roc_auc = auc(fpr, tpr)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12,6))

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
axes[0].scatter(X[:,0], X[:,1], c=T, cmap=plt.cm.coolwarm, edgecolors='k')
axes[0].scatter(model.centers[:,0].numpy(), model.centers[:,1].numpy(),
                c='yellow', edgecolors='black', s=200, marker='X', label='Centers')
axes[0].set_title('RBF Decision Boundary with Centers')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')

axes[1].plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (AUC = %0.2f)' % roc_auc)
axes[1].plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0,1.0])
axes[1].set_ylim([0.0,1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve of RBF')
axes[1].legend(loc="lower right")
axes[1].grid(True)

plt.tight_layout()
plt.show()