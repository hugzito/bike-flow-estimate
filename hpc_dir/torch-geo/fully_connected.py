# Fully Connected Neural Network (MLP) Training Notebook for Graph Node Features
# Adapted to match the structure of your GNN training notebook

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch_geometric.data import Data
import wandb

# --- Configurations ---
epochs = int(os.getenv("EPOCHS", 10))
learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
hidden_c = int(os.getenv("HIDDEN_C", 64))
random_seed = int(os.getenv("RANDOM_SEED", 42))
bins = [int(i) for i in os.getenv("BINS", "1000 5000 10000").split()]
dropout_p = float(os.getenv("DROPOUT", 0.5))
graph_num = os.getenv("GRAPH_NUM", 2)
gat = 'MLP'
api_key = os.getenv("API_KEY", None)
num_layers = int(os.getenv("NUM_LAYERS", 0))

# --- WandB Initialization ---
wandb.login(key=api_key)
run = wandb.init(
    project="Thesis-Final-Benchmarks",
    entity="christian-hugo-rasmussen-it-universitetet-i-k-benhavn",
    config={
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_c": hidden_c,
        "random_seed": random_seed,
        "bins": bins,
        "dropout": dropout_p,
        "graph_num": graph_num,
        "num_layers" : num_layers,
        "gat" : gat,
    },
    settings=wandb.Settings(init_timeout=300)
)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)

bins = torch.tensor(bins, device=device)

# --- Load Graph Data ---
with open(f'data/graphs/{graph_num}/linegraph_tg.pkl', 'rb') as f:
    data = pickle.load(f)

data.edge_index = data.edge_index.contiguous()
data.x = data.x.contiguous()
data.y = data.y.contiguous()
print(data.x.shape, data.edge_index.shape, data.y.shape, flush=True)

# --- Data Split ---
def stratified_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    positive_mask = data.y > 0
    positive_indices = positive_mask.nonzero(as_tuple=False).squeeze()
    negative_indices = (~positive_mask).nonzero(as_tuple=False).squeeze()

    pos_train, pos_temp = train_test_split(positive_indices, train_size=train_ratio, random_state=random_seed)
    pos_val, pos_test = train_test_split(pos_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)

    neg_train, neg_temp = train_test_split(negative_indices, train_size=train_ratio, random_state=random_seed)
    neg_val, neg_test = train_test_split(neg_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_seed)

    train_idx = torch.cat([pos_train, neg_train])
    val_idx = torch.cat([pos_val, neg_val])
    test_idx = torch.cat([pos_test, neg_test])

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data

data = stratified_split(data)

# Move data to device
data.x = data.x.to(device)
data.y = data.y.to(device)
data.edge_index = data.edge_index.to(device)
data.train_mask = data.train_mask.to(device)
data.val_mask = data.val_mask.to(device)
data.test_mask = data.test_mask.to(device)

# --- Fully Connected Network Definition ---
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        torch.manual_seed(random_seed)
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=dropout_p, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=dropout_p, training=self.training)
        x = self.fc3(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, num_heads = None):
        super().__init__()
        torch.manual_seed(random_seed)
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(data.num_features, hidden_channels))
        for _ in range(num_layers):
            self.fcs(torch.nn.Linear(hidden_channels, hidden_channels))
        if bins != 'regression':
            self.fcs.append(torch.nn.Linear(hidden_channels, len(bins) +1 ))
        else:
            self.fcs.append(torch.nn.Linear(hidden_channels, 1))
    def forward(self, x, edge_index = None):
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = F.elu(x)
            x = F.dropout(x, p = dropout_p, training = self.training)
        x = self.fcs[-1](x)
        return x

# --- Instantiate Model ---
model = MLP(hidden_c, num_layers).to(device)
print(model, flush=True)
torch.save(model, f"data/graphs/{graph_num}/models/{run.name}_mlp.pt")

# --- Move data to device ---
data.x = data.x.to(device)
data.y = data.y.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss()
if bins == 'regression':
    criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epochs//100)

def train():
    model.train()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.test_mask = data.test_mask.to(device)
    data.y = data.y.to(device)
    optimizer.zero_grad()  # Clear gradients.
    mask = data.train_mask.squeeze().to(device) & (data.y > 0).squeeze().to(device)
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    # Convert target to 1D tensor with dtype=torch.long
    #target = torch.bucketize(data.y[mask], bins).squeeze()
    if bins == 'regression':
        target = data.y[mask].squeeze()
        loss = criterion(out[mask].squeeze(), target)  # Ensure target is 1D and long
    else:
        target = torch.bucketize(data.y[mask], bins).squeeze()
        loss = criterion(out[mask], target.long())
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def test():
    model.eval()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.val_mask = data.val_mask.to(device)
    data.y = data.y.to(device)
    mask = data.val_mask.squeeze() & (data.y > 0).squeeze()
    out = model(data.x, data.edge_index)
    #target = torch.bucketize(data.y[mask], bins).squeeze()
    if bins == 'regression':
        target = data.y[mask].squeeze()
        loss = criterion(out[mask].squeeze(), target)
        accuracy = r2_score(target.detach().cpu().numpy(), out[mask].detach().cpu().numpy())
    else:
        target = torch.bucketize(data.y[mask], bins).squeeze()
        loss = criterion(out[mask], target.long())  # Ensure target is 1D and long
        correct_preds = out[mask].argmax(dim=1)
        correct = (correct_preds == target).sum()
        accuracy = correct.item() / mask.sum().item()
    return accuracy, out, loss

for epoch in range(1, epochs + 1):
    best_val_acc = 0
    best_val_loss = 100
    loss = train()
    if epoch % 5 == 0:
        acc, val_out, val_loss = test()
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), f'data/graphs/{graph_num}/models/{run.name}_best_accuracy.pt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'data/graphs/{graph_num}/models/{run.name}_best_loss.pt')
        if bins == 'regression':
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        else:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_loss: {val_loss:.4f}, Validation Accuracy: {acc}', flush = True)
        torch.save(model.state_dict(), f'data/graphs/{graph_num}/models/{run.name}_latest.pt')
        run.log({"training_loss": loss, "val_loss": val_loss, "val_acc": acc, "epoch": epoch})
    scheduler.step(loss)
run.finish()
