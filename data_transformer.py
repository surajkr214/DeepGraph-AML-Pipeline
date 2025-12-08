import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

def load_and_transform_data():
    # 1. Load Data
    df = pd.read_csv('raw_transactions.csv')

    # 2. Identify Unique Nodes (Accounts)
    # Find all unique accounts from source and target columns
    all_accounts = pd.concat([df['source_account'], df['target_account']]).unique()

    # Create a mapping from account name to a numerical ID (Node Index)
    # This is essential for PyTorch Geometric
    node_map = {name: i for i, name in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    # 3. Create Edge List (Connections)
    # Convert account names to their numerical IDs
    source_nodes = df['source_account'].map(node_map).tolist()
    target_nodes = df['target_account'].map(node_map).tolist()

    # The edge list is a 2xN tensor where N is the number of edges
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # 4. Create Node Features (Simple Mock Features)
    # For a real project, this would be complex (e.g., account age, daily variance)
    # Here, we create a tensor of 1s (mock features) for each node
    node_features = torch.ones((num_nodes, 1), dtype=torch.float)

    # 5. Create Mock Labels (Fraud Labels)
    # Mark the suspicious loop accounts (A, B, C) as 1 (Fraud)
    suspicious_accounts = ['Account_A', 'Account_B', 'Account_C']
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for acc in suspicious_accounts:
        if acc in node_map:
            labels[node_map[acc]] = 1

    # 6. Create the final PyTorch Geometric Data Object
    data = Data(x=node_features, edge_index=edge_index, y=labels)

    print(f"--- Data Transformation Complete ---")
    print(f"Total Nodes (Accounts): {data.num_nodes}")
    print(f"Total Edges (Transactions): {data.num_edges}")
    print(f"Node Feature Shape (x): {data.x.shape}")
    print(f"Edge Index Shape: {data.edge_index.shape}")
    print(f"Labels (y): {data.y}")

    return data

if __name__ == '__main__':
    graph_data = load_and_transform_data()