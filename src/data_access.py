import pandas as pd
import os 
import numpy as np
from pathlib import Path
from graph_features import StockGraph
import torch
from torch_geometric.data import Data

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR/'data'/'processed'/'nifty50_features.csv'
file_path_return = BASE_DIR/'data'/'processed'/'Return1.csv'

stockGNN = StockGraph(file_path_return,3)
ticker_to_id, edge_index,nodes = stockGNN.get_graph()
print("stockGNN edge index",edge_index)
print("stockGNN ticker_to_id",ticker_to_id)
print("stockGNN nodes",nodes)


if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("----- Success! Dataset loaded perfectly.")
    print(f"Dataset Shape: {data.shape}")
    print(data.head(10))
else:
    print(f"---  Error: Could not find the file at:\n{file_path}")


# Keep stocks in the same order as graph node IDs
data["node_id"] = data["Ticker"].map(ticker_to_id)
data = data.sort_values("node_id")
data.drop(columns=['Ticker'],inplace=True)
print(data.head())

# creating the feature for the graph
features = [i for i in data.columns if i not in ['Ticker','node_id','Target']]
print("features : ",features)


# graph features
x = torch.tensor(
    data[features].values,
    dtype=torch.float
)

#graph targets 
y = torch.tensor(
    data['Target'],
    dtype=torch.int
    )

# ingesting the  data into graph
graph_data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

print(graph_data)
print("x shape:", graph_data.x.shape)
print("edge_index shape:", graph_data.edge_index.shape)
print("y shape:", graph_data.y.shape)






