import torch
import torch.nn as nn
import plotly.graph_objects as go
import networkx as nx

def visualize_model(model):
    """
    Visualizes the given PyTorch model using Plotly.
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track node positions
    positions = {}
    layers = []
    
    # Extract model layers
    for i, (name, layer) in enumerate(model.named_children()):
        layers.append((name, layer))
        G.add_node(name, label=name)  # Add node to graph
        positions[name] = (i, 0)  # Set node position

    # Connect layers based on sequential flow
    for i in range(len(layers) - 1):
        G.add_edge(layers[i][0], layers[i+1][0])

    # Create node labels
    node_labels = {node: G.nodes[node]['label'] for node in G.nodes}
    
    # Generate positions for plotly (scatter plot)
    pos = nx.spring_layout(G, seed=42)  # Auto position using NetworkX layout
    edge_x, edge_y = [], []
    
    # Extract edges for visualization
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create edge traces (arrows)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='black'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces (layers)
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_labels[node])  # Layer names

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(size=20, color='blue', line=dict(width=2))
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Neural Network Architecture",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Show the plot
    fig.show()


# Instantiate the model
model = HybridCNNSoftmaxModel(num_classes=10, window_size=128)

# Visualize the model
visualize_model(model)
