#!/usr/bin/env python
# coding: utf-8

# In[78]:


from __future__ import annotations
import torch

from torch import Tensor, Generator, rand, nn, randn, full, dot, square, randperm, device, sum
from typing import Optional, Tuple
import torch.nn.functional as F
import plotly.graph_objects as go
from torch.optim import Adam



# In[79]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install nbformat>=4.2.0')
import plotly
import nbformat


# In[80]:


features1 = 20
latent1 = 5

features2 = 5
latent2 = 2

features3 = 80
latent3 = 20

# Grid configuration
feature_latent_pairs = [
    # (features1, latent1),
    (features2, latent2),
    # (features3, latent3),
]

p_actives = [0.99, 0.7, 0.5, 0.3, 0.1, 0.01]


# In[81]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_uniform_distribution(
    p_active: float,
    n_features: int,
    n_samples: int = 1,
    norm: float = 1.0,
    generator: Optional[Generator] = None,
    device: Optional[torch.device] = device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    x = torch.rand((n_samples, n_features), device=device, dtype=dtype, generator=generator)
    mask = torch.rand((n_samples, n_features), device=device, dtype=dtype, generator=generator) > p_active
    x[mask] = 0.0
    # # Calculate L2 norm for each row, avoid division by zero
    # row_norms = x.norm(dim=1, keepdim=True)
    # row_norms = row_norms.where(row_norms != 0, torch.ones_like(row_norms))  # avoid division by zero
    # x = x / row_norms * norm
    return x


# In[82]:


def make_importance(uniform: bool = True, base: float = 1.0, shape: Tuple = (features1, ), dtype: torch.dtype = torch.float32, *args, **kwargs) -> Tensor:
    importance = full(size=shape, fill_value=base, dtype=dtype)
    if kwargs.get("exponential", float(0.0)):
        exponential: float = kwargs.get("exponential", 1.0)
        importance = [base ** (exponential * i) for i in range(1, 1 + len(importance))]
    return Tensor(importance).to(device)


# In[83]:


class AutoEncoder(nn.Module):
    def __init__(self, num_features: int = 20, latent_size: int = 5):
        super(AutoEncoder, self).__init__()
        self.W = nn.Parameter(randn(latent_size, num_features, device=device))
        self.b = nn.Parameter(randn(num_features, device=device))

    def encode(self, x: Tensor):
        return self.W @ x.T  

    def decode(self, h: Tensor):
        return F.relu(self.W.T @ h + self.b.unsqueeze(1)).T

    def forward(self, x):
        return self.decode(self.encode(x))


# In[84]:


class ToyModel():
    def __init__(self, name: str, autoencoder: AutoEncoder, distribution: Tensor):
        self.name = name
        self.ae: AutoEncoder = autoencoder
        self.dist: Tensor = distribution
        self.importance: Tensor = make_importance(uniform=True, dist=self.dist, base=1)

    def criterion(self, y_gt: Tensor, y_hat: Tensor, importance: Tensor) -> float:        
        loss = sum(square(y_gt - y_hat) @ importance.unsqueeze(1))
        return loss

    def fit(self, lr: float = 0.01, batch_size: int = 64, epochs: int = 100, importance: Optional[Tensor] = None):
        if importance is not None:
            self.importance = importance

        optimizer = Adam(self.ae.parameters(), lr)
        self.ae.train()

        num_samples = self.dist.size(0)

        for epoch in range(epochs):
            permutation = randperm(num_samples, device=device)
            epoch_loss = 0.0

            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i+batch_size]
                batch = self.dist[indices]

                optimizer.zero_grad()
                y_hat = self.ae.forward(batch)
                # importance = self.importance[indices] if hasattr(self.importance, "__getitem__") else full(batch_size, 1.0)

                loss = self.criterion(batch, y_hat, self.importance)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.size(0)

            avg_loss = epoch_loss / num_samples
            
            if(epoch==0 or epoch==epochs-1):
                print(f"{self.name} Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self.ae.eval()


    def get_latent(self):
        pass


# In[85]:


# Training loop across configurations
results = {}

for (feature_dim, latent_dim) in feature_latent_pairs:
    for p_active in p_actives:
        # Fresh AutoEncoder instance for each (feature_dim, latent_dim, p_active)
        ae = AutoEncoder(feature_dim, latent_dim)

        dist = sparse_uniform_distribution(
            norm=1.0,
            p_active=p_active,
            n_features=feature_dim,
            n_samples=200,
            generator=Generator(device=device)
        )
        key = (feature_dim, latent_dim, p_active)
        importance = make_importance(uniform=False, shape=dist.shape[1:], base=0.7, exponential=1.0)
        # importance = make_importance(uniform=True, shape=dist.shape[1:], base=1.0)
        tm = ToyModel(
            f"tm_ae{feature_dim}_{p_active:.2f}", 
            ae, 
            dist 
        )

        tm.fit(epochs=1000, importance=importance)
        results[key] = {
            'model': tm,
            'ae': ae,
            'dist': dist,
            'imp': importance
        }

        # Inspect learned W for this particular model
        (name_W, W_param), (name_b, b_param) = ae.named_parameters()
        W_print = W_param.detach().cpu().T  # shape (feature_dim, latent_dim)
        print(f"{tm.name}, key={key}, W=\n{W_print}\n")

# Access results via results.keys() or e.g. results[(features1, latent1, 0.99)]['model']


# In[86]:


from plotly.subplots import make_subplots

# Collect all (key, W) from results, ensuring each W is taken from its own trained AutoEncoder
Ws = []
for key, value in results.items():
    ae = value['ae']
    (name_W, W_param), (name_b, b_param) = ae.named_parameters()
    W = W_param.detach().cpu().T  # shape (feature_dim, latent_dim) -> here (5, 2)
    print(f"key={key}, W=\n{W}\n")
    Ws.append((key, W))

if not Ws:
    raise RuntimeError("Could not find any W in results.")

n_models = len(Ws)
fig = make_subplots(
    rows=1,
    cols=n_models,
    subplot_titles=[f"p_active={key[2]:.2f}" for key, _ in Ws],
)

# Plot each model's 5 vectors in its own subplot, all on the same plane
for col_idx, (key, W) in enumerate(Ws, start=1):
    n_vec = W.shape[0]  # should be 5
    for i in range(n_vec):
        vx = W[i, 0].item()
        vy = W[i, 1].item()

        # Arrow from origin to (vx, vy)
        fig.add_trace(
            go.Scatter(
                x=[0.0, vx],
                y=[0.0, vy],
                mode="lines+markers",
                line=dict(width=3, color="royalblue"),
                marker=dict(size=6),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

        # Label the vector index at the head
        fig.add_trace(
            go.Scatter(
                x=[vx],
                y=[vy],
                mode="text",
                text=[f"{i}"],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col_idx,
        )

    # Same plane for all 5 vectors of this model
    fig.update_xaxes(range=[-1.3, 1.3], zeroline=True, row=1, col=col_idx)
    fig.update_yaxes(
        range=[-1.3, 1.3],
        zeroline=True,
        scaleanchor=f"x{col_idx}",
        scaleratio=1,
        row=1,
        col=col_idx,
    )

fig.update_layout(
    title="Encoder W vectors (5 per model) in 2D",
    height=450,
    width=350 * n_models,
    template="plotly_white",
    margin=dict(t=60, l=10, r=10, b=10),
)

fig.show()


# In[87]:


x = sparse_uniform_distribution(n_features=10, n_samples=50, norm=1, generator=Generator(device=device), device=device, p_active=0.20)
print(x)

