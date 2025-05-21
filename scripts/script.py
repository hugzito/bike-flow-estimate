import os
from models import *
from tg_functions import *
import pickle
import torch
import wandb

# --- Configurations ---
epochs = int(os.getenv("EPOCHS", 50000))
learning_rate = float(os.getenv("LEARNING_RATE", 0.0001))
hidden_c = int(os.getenv("HIDDEN_C", 350))
random_seed = int(os.getenv("RANDOM_SEED", 100))
bins = [try_int(i) for i in os.getenv("BINS", "400 800 1300 2100 3000 3700 4700 7020 9660").split()]
num_layers = int(os.getenv("NUM_LAYERS", 2))
nh = int(os.getenv("NUM_HEADS", 1))
use_gat = try_int(os.getenv("GAT", 1))
api_key = os.getenv("API_KEY", None)
graph_num = os.getenv("GRAPH_NUM", 17)
dropout_p = float(os.getenv("DROPOUT", 0.5))

if bins[0] == 'REGRESSION':
    bins = 'regression'
if use_gat in[0, 1]:
    use_gat = bool(use_gat)

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
        "num_layers": num_layers,
        "num_heads": nh,
        "gat": use_gat,
        "graph_num": graph_num,
        "dropout": dropout_p
    },
    settings=wandb.Settings(init_timeout=300)
)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)

if bins != 'regression':
    bins = torch.tensor(bins, device=device)

# --- Load Graph Data ---
with open(f'data/graphs/{graph_num}/linegraph_tg.pkl', 'rb') as f:
    data = pickle.load(f)

data.edge_index = data.edge_index.contiguous()
data.x = data.x.contiguous()
data.y = data.y.contiguous()
print(data.x.shape, data.edge_index.shape, data.y.shape, flush=True)
    
# --- Model Instantiation ---
model = GAT(hidden_channels=hidden_c, num_heads=nh, num_layers=num_layers).to(device) if use_gat else GCN(hidden_channels=hidden_c, num_layers=num_layers).to(device)

if use_gat == 'MLP':
    model = MLP(hidden_channels=hidden_c, num_layers=num_layers).to(device)

print(model, flush=True)
torch.save(model, f"data/graphs/{graph_num}/models/{run.name}.pt")

# Move data to device
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss()
if bins == 'regression':
    criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=epochs//100)

best_val_acc = 0
best_val_loss = 100
for epoch in range(1, epochs + 1):
    loss = train(model, data, optimizer, criterion, device, bins)
    if epoch % 5 == 0:
        acc, val_out, val_loss = test(model, data, criterion, device, bins)
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
