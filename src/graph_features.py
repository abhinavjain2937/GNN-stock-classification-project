import pandas as pd
import os 
import torch
import numpy as np
from pathlib import Path


class StockGraph:

    def __init__(self, file_path, top_k=3):

        self.file_path = file_path
        self.top_k = top_k

        # Load data
        self.data = pd.read_csv(self.file_path)

        # Fill missing values
        self.returns_matrix = self.data.ffill().bfill()

        # Remove Date because correlation should only use stock returns
        self.returns_matrix = self.returns_matrix.drop(
            columns=["Date"]
        )

        # Calculate Pearson correlation
        self.stock_corr = self.returns_matrix.corr(
            method="pearson"
        )

        # Create ticker -> node ID mapping
        self.ticker_to_id = {
            ticker: i
            for i, ticker in enumerate(self.stock_corr.columns)
        }

        # Create edge_index
        self.edge_index = self.create_edges()

        # Create nodes
        self.nodes = torch.arange(
            len(self.ticker_to_id),
            dtype=torch.long
        )

    def create_edges(self):

        edge_list = []

        # For every stock
        for stock in self.stock_corr.columns:

            # Remove correlation of stock with itself
            correlations = self.stock_corr[stock].drop(stock)

            # Select top-k correlated stocks
            top_k_stocks = correlations.nlargest(self.top_k)

            # Source node ID
            source = self.ticker_to_id[stock]

            # Create edges
            for neighbor in top_k_stocks.index:

                target = self.ticker_to_id[neighbor]

                edge_list.append([
                    source,
                    target
                ])

        # Convert list to PyTorch Geometric format
        edge_index = torch.tensor(
            edge_list,
            dtype=torch.long
        ).t().contiguous()

        return edge_index

    def get_graph(self):

        return (
            self.ticker_to_id,
            self.edge_index,
            self.nodes
        )


BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR/"data"/"processed"/"Return1.csv"

StockG = StockGraph(file_path=file_path,top_k=3)
ticker_to_id, edge_index,nodes = StockG.get_graph()
print("\nTicker to ID:")
print(ticker_to_id)

print("\nEdge index:")
print(edge_index)

print("\nEdge index shape:")
print(edge_index.shape)

print("\nNodes:")
print(nodes)

print("\nNumber of nodes:")
print(nodes.shape)