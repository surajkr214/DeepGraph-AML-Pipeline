from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from fastapi.middleware.cors import CORSMiddleware

# --- Import necessary classes from your existing scripts ---
# We redefine the GNN_AML_Model class exactly as it was defined in gnn_trainer.py
# This is necessary because Python needs the class definition to load the weights.
from torch_geometric.nn import GCNConv

class GNN_AML_Model(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- API Setup ---
app = FastAPI(
    title="AML GNN Prediction API",
    description="Real-time GNN scoring endpoint for Anti-Money Laundering.",
    version="1.0.0"
)

app = FastAPI(
    title="AML GNN Prediction API",
    # ... description ...
)

# --- PASTE THIS BLOCK HERE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows connections from ANY website
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET)
    allow_headers=["*"],
)
# -----------------------------

# --- Global Variables for Model & Data Context ---
MODEL_PATH = "gnn_aml_model.pt"
MODEL = None
# In a real system, you would load ALL historical account data here for context.
# For the POC, we simulate the original 17 nodes.

# Pydantic schema for input validation (Crucial for production APIs)
class TransactionInput(BaseModel):
    source_account: str
    target_account: str
    transaction_amount: float
    # In a real model, you'd send complex features. Here, we keep it simple.


# Function to load model and mock a graph context (runs once when API starts)
@app.on_event("startup")
async def load_model_and_context():
    """Loads the trained model and sets up the graph context."""
    global MODEL

    # NOTE: For this POC, we are mocking the context of the 17 accounts from the CSV.
    # In a real scenario, this would load data from a database (Redshift).

    num_node_features = 1
    num_classes = 2

    MODEL = GNN_AML_Model(num_node_features, num_classes)

    # Load the saved state dict
    try:
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        MODEL.eval()  # Set model to evaluation mode
        print(f"--- Model {MODEL_PATH} loaded successfully! ---")
    except FileNotFoundError:
        print(f"ERROR: Model file {MODEL_PATH} not found. Ensure Phase 2 was successful.")
        raise

# --- Prediction Endpoint ---
@app.post("/predict_aml_risk")
async def predict_risk(transaction: TransactionInput):
    """Calculates AML risk score for a new transaction."""

    # --- 1. Graph Context Mockup (The hardest part of graph deployment) ---
    # For a real GNN prediction, you need the neighborhood data.
    # Here, we SIMULATE that a large graph exists in memory.

    # We need the numerical IDs of the source and target accounts.
    # Since we don't have the full graph in memory, we mock the IDs 1 and 2.
    # This part of the code would be complex in a real bank environment.

    # --- 2. Create Inference Data ---
    # Mock the input graph structure for the inference
    # In a real deployment, you would query the surrounding 3-5 accounts.
    mock_features = torch.ones((2, 1), dtype=torch.float) # Source and Target nodes
    mock_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) # Mock connection
    inference_data = Data(x=mock_features, edge_index=mock_edge_index)

    # --- 3. Run Prediction ---
    with torch.no_grad():
        output = MODEL(inference_data)
        # Use softmax to convert raw logits to probabilities
        probabilities = F.softmax(output, dim=1)
        # The risk score is the probability of class 1 (Fraud)
        risk_score = probabilities[0, 1].item() * 100 # Convert to percentage

    # --- 4. Return Results ---
    return {
        "source_account": transaction.source_account,
        "target_account": transaction.target_account,
        "aml_risk_score_percent": f"{risk_score:.2f}%",
        "prediction_status": "HIGH RISK" if risk_score > 50 else "NORMAL"
    }

# Optional root for health check
@app.get("/")
async def health_check():
    return {"status": "GNN AML Predictor is running and model loaded."}