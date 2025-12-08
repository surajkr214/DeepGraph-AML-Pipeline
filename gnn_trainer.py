import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from data_transformer import load_and_transform_data # Import your data function

# 1. Define the GNN Model Architecture
class GNN_AML_Model(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        # GCNConv is the core layer for Graph Convolutional Networks
        # Layer 1: Takes input features and outputs 16 intermediate features
        self.conv1 = GCNConv(num_node_features, 16)
        # Layer 2: Takes 16 intermediate features and outputs the number of classes (2: Normal/Fraud)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        # x is the node feature matrix (data.x), edge_index is the connectivity matrix (data.edge_index)
        x, edge_index = data.x, data.edge_index

        # First Convolution + ReLU Activation (non-linearity)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Optional: Add Dropout for regularization (important in real models)
        x = F.dropout(x, training=self.training)

        # Second Convolution (Outputs the final prediction scores)
        x = self.conv2(x, edge_index)

        # Return the raw logits (scores)
        return x

# 2. Main Training Function
def train_gnn():
    # Load the graph data created in Phase 1
    data = load_and_transform_data()

    # Determine model parameters
    num_node_features = data.num_node_features  # Should be 1 from mock data
    num_classes = 2 # Normal (0) or Fraud (1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN_AML_Model(num_node_features, num_classes).to(device)
    data = data.to(device)

    # Use the Adam optimizer and Cross Entropy Loss (standard for multi-class classification)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train() # Set model to training mode

    print("--- GNN Training Started ---")
    for epoch in range(1, 50001): # Train for 100 epochs
        optimizer.zero_grad()

        # Forward pass: Get prediction scores (logits)
        out = model(data)

        # Loss Calculation: Compare prediction scores (out) with true labels (data.y)
        loss = criterion(out, data.y)

        # Backpropagation: Calculate gradients and update weights
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    # 3. Save the Trained Model Artifact
    torch.save(model.state_dict(), 'gnn_aml_model.pt')
    print("--- Model Saved Successfully: gnn_aml_model.pt ---")

if __name__ == '__main__':
    train_gnn()