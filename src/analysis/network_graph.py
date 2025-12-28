import matplotlib
matplotlib.use('Agg') # <--- OBRIGATÓRIO PARA NÃO TRAVAR O SERVIDOR
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pathlib import Path
from termcolor import colored
import itertools

# --- CONFIG ---
INPUT_FILE = "data/processed/chat_history.parquet"
OUTPUT_DIR = "data/reports"
MIN_MESSAGES_FILTER = 50 

def generate_network_graph():
    print(colored("🕸️  Iniciando Mapeamento de Rede...", "cyan"))
    
    if not Path(INPUT_FILE).exists():
        print(colored("❌ Arquivo não encontrado.", "red"))
        return

    df = pd.read_parquet(INPUT_FILE)
    author_counts = df['author'].value_counts()
    valid_authors = author_counts[author_counts > MIN_MESSAGES_FILTER].index
    df = df[df['author'].isin(valid_authors)].copy()
    
    interactions = []
    df['next_author'] = df['author'].shift(-1)
    transitions = df[df['author'] != df['next_author']].dropna()
    
    for _, row in transitions.iterrows():
        pair = sorted([row['author'], row['next_author']])
        interactions.append(tuple(pair))

    G = nx.Graph()
    for author in valid_authors:
        G.add_node(author, size=author_counts[author])

    interaction_counts = pd.Series(interactions).value_counts()
    for (source, target), count in interaction_counts.items():
        G.add_edge(source, target, weight=count)

    # Plot
    plt.figure(figsize=(16, 12))
    # Limpa figura anterior para não acumular memória
    plt.clf() 
    plt.style.use('dark_background')
    
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    node_sizes = [G.nodes[n]['size'] * 1.5 for n in G.nodes]
    edge_widths = [G.edges[u, v]['weight'] * 0.05 for u, v in G.edges]
    edge_colors = [G.edges[u, v]['weight'] for u, v in G.edges]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#00d4ff', alpha=0.9)
    edges = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.plasma, alpha=0.6)
    
    label_pos = {k: (v[0], v[1] + 0.04) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=10, font_color='white', font_weight='bold')

    plt.title("Grafo de Interações", fontsize=16, color='white')
    plt.axis('off')
    
    # Barra de cores (Check seguro se existem arestas)
    if len(G.edges) > 0:
        cbar = plt.colorbar(edges, shrink=0.8)
        cbar.ax.set_ylabel('Volume', color='white')
        cbar.ax.tick_params(labelcolor='white')

    output_path = f"{OUTPUT_DIR}/interaction_network.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close() # Fecha explicitamente

    print(colored(f"✅ Grafo salvo em: {output_path}", "green"))

if __name__ == "__main__":
    generate_network_graph()
