import pandas as pd
import os 
from pathlib import Path
from graph_features import StockGraph
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

class GraphData:
    def __init__(self):

        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.file_path = self.BASE_DIR/'data'/'processed'/'nifty50_features.csv'
        self.file_path_return = self.BASE_DIR/'data'/'processed'/'Return1.csv'

    def data_import(self):
        if os.path.exists(self.file_path):
                data = pd.read_csv(self.file_path)
                print("----- Success! Dataset loaded perfectly.")
                print(f"Dataset Shape: {data.shape}")
                print(data.head(10))
        else:
                print(f"---  Error: Could not find the file at:\n{self.file_path}")

        # graph feature extraction
        stockGNN = StockGraph(self.file_path_return,3)
        ticker_to_id, edge_index,nodes = stockGNN.get_graph()
        print("Graph feature extracted successfully.............")
        


        # Keep stocks in the same order as graph node IDs
        data["node_id"] = data["Ticker"].map(ticker_to_id)
        data = data.sort_values("node_id")
        data.index = data['node_id']
        data.drop(columns=['node_id'],inplace=True)
        data.drop(columns=['Ticker'],inplace=True)
        print(data.head())

        # creating the feature for the graph
        features = [i for i in data.columns if i not in ['Ticker','node_id','Target']]
        print("features : ",features)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[features])


        # graph features
        x = torch.tensor(
            scaled_features,
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
        return graph_data
    def get_data(self):
        return self.data_import()



G = GraphData()
graph_data = G.get_data()

print(graph_data)
print("x shape:", graph_data.x.shape)
print("edge_index shape:", graph_data.edge_index.shape)
print("y shape:", graph_data.y.shape)







