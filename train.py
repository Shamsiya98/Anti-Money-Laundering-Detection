import torch
from dataset import AMLtoGraph
from GNN import GAT
from torch_geometric.loader import NeighborLoader
import os
import csv
import torch_geometric.transforms as T


# Define the dataset path
dataset_path = r"C:\Users\HP\Desktop\AntiMoney\data"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = AMLtoGraph(dataset_path)
data = dataset[0]
epochs = 100

model = GAT(in_channels=data.num_features, hidden_channels=16, out_channels=1, heads=8).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# Split data into train and test sets
split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0)
data = split(data)

train_loader = NeighborLoader(data, num_neighbors=[30] * 2, batch_size=256, input_nodes=data.train_mask)
test_loader = NeighborLoader(data, num_neighbors=[30] * 2, batch_size=256, input_nodes=data.val_mask)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# File to save GNN results
gnn_results_file = os.path.join(results_dir, 'gnn_results.csv')

# Initialize the CSV file
with open(gnn_results_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Loss', 'Accuracy']) 

# Early stopping parameters
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        output, _ = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(output, batch.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output, _ = model(batch.x, batch.edge_index, batch.edge_attr)
            val_loss += criterion(output, batch.y.unsqueeze(1)).item()
            pred = (output >= 0.5).squeeze().int()
            val_correct += (pred == batch.y).sum().item()
            val_total += batch.y.size(0)

    val_loss /= len(test_loader)
    val_accuracy = val_correct / val_total

    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # Save results to CSV
    with open(gnn_results_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_loss, val_accuracy])

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping after epoch {epoch}')
            break