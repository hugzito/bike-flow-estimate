import os, torch
from sklearn.model_selection import train_test_split
import pickle
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.nn.models import Node2Vec
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import wandb

epochs = int(os.getenv("EPOCHS", 10))  # Default to 10 if not provided
learning_rate = float(os.getenv("LEARNING_RATE", 0.001))  # Default to 0.001
hidden_c = int(os.getenv("HIDDEN_C", 16))  # Default to 16
random_seed = int(os.getenv("RANDOM_SEED", 42))  # Default to 42
bins = [int(i) for i in os.getenv("BINS", "1000 5000 10000").split(' ')]  # Default to [1000, 3000, 5000]
num_layers = int(os.getenv("NUM_LAYERS", 5))  # Default to 5
nh = int(os.getenv("NUM_HEADS", 10))
gat = int(os.getenv("GAT", 0))
api_key = os.getenv("API_KEY", None)
graph_num = os.getenv("GRAPH_NUM", 2)
dropout_p = float(os.getenv("DROPOUT", 0.5))

wandb.login(key=api_key)

run = wandb.init(project="Thesis-project", entity="christian-hugo-rasmussen-it-universitetet-i-k-benhavn", config={
    "epochs": epochs,
    "learning_rate": learning_rate,
    "hidden_c": hidden_c,
    "random_seed": random_seed,
    "bins": bins,
    "num_layers": num_layers,
    'num_heads' : nh,
    "gat" : gat, 
    "graph_num" : graph_num, 
    "dropout" : dropout_p}, 
    settings=wandb.Settings(init_timeout=300))
# Check for CUDA availability and set device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}", flush = True)
else:
    device = torch.device('cpu')
    print("Using CPU", flush = True)

bins = torch.tensor(bins, device=device)

### load graph data

with open(f'data/graphs/{graph_num}/linegraph_tg.pkl', 'rb') as f:
    data = pickle.load(f)

def stratified_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Splits data into train, validation, and test sets, stratifying by y > 0."""

    # Create a boolean mask for nodes where y > 0
    positive_mask = data.y > 0

    # Get indices of positive and negative nodes
    positive_indices = positive_mask.nonzero(as_tuple=False).squeeze()
    negative_indices = (~positive_mask).nonzero(as_tuple=False).squeeze()

    # Split positive indices
    pos_train_idx, pos_temp_idx = train_test_split(positive_indices, train_size=train_ratio, random_state=random_seed)  # Adjust random_state for consistent splits
    pos_val_idx, pos_test_idx = train_test_split(pos_temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed)

    # Split negative indices
    neg_train_idx, neg_temp_idx = train_test_split(negative_indices, train_size=train_ratio, random_state=random_seed)
    neg_val_idx, neg_test_idx = train_test_split(neg_temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed)

    # Combine indices
    train_idx = torch.cat([pos_train_idx, neg_train_idx])
    val_idx = torch.cat([pos_val_idx, neg_val_idx])
    test_idx = torch.cat([pos_test_idx, neg_test_idx])

    # Create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

data.edge_index = data.edge_index.contiguous()
data.x = data.x.contiguous()
data.y = data.y.contiguous()

data = stratified_split(data)
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(random_seed)
        self.conv1 = GCNConv(data.num_features, hidden_channels, improved = True, cached = True)
        conv2_list = []
        hc = hidden_channels
        for _ in range(num_layers):
            conv2_list.append(
                GCNConv(hc, hc//2)
            )
            hc //= 2
        self.conv2 = torch.nn.ModuleList(conv2_list)
        self.conv3 = GCNConv(hc, len(bins) + 1, cached = True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=dropout_p, training=self.training)
        for conv in self.conv2:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=dropout_p, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class mygat(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super().__init__()
        torch.manual_seed(random_seed)
        gat_list = []
        for _ in range(num_layers):
            if _ == num_layers-1:
                layer = GATConv(-1, len(bins)+1, num_heads)
            else:
                layer = GATConv(-1, hidden_channels, num_heads)
            gat_list.append(layer)
        self.gat1 = torch.nn.ModuleList(gat_list)

    def forward(self, x, edge_index):
        for gat in self.gat1:
            x = gat(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=dropout_p, training=self.training)
        return x


if gat==1:
    model = mygat(hidden_channels=hidden_c, num_heads=nh).to(device)
else:
    model = GCN(hidden_channels=hidden_c).to(device) # Move model to device

print(model, flush=True)
torch.save(model, f"data/graphs/{graph_num}/models/{run.name}.pt")

# Move data to device
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

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
    target = torch.bucketize(data.y[mask], bins).squeeze()
    loss = criterion(out[mask], target.long())  # Ensure target is 1D and long
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
    if epoch % (epochs/1000) == 0:
        acc, val_out, val_loss = test()
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), f'data/graphs/{graph_num}/models/{run.name}_best_accuracy.pt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'data/graphs/{graph_num}/models/{run.name}_best_loss.pt')
        run.log({"training_loss": loss, "val_loss": val_loss, "val_acc": acc, "epoch": epoch})
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_loss: {val_loss:.4f}, Validation Accuracy: {acc}', flush = True)
        torch.save(model.state_dict(), f'data/graphs/{graph_num}/models/{run.name}_latest.pt')
run.finish()
