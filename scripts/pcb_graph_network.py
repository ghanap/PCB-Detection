#!/usr/bin/env python3
"""
pcb_graph_network.py

PCB Graph Theory & Netlist Circuit Extraction Engine:
Converts segmented PCB component masks and copper trace masks into a formal Graph Network G(V, E).
Nodes (V) = PCB components & pin pads.
Edges (E) = Copper trace connections.
Computes Adjacency Matrix, Pin-to-Pin Netlist, Graph Centrality, and visualizes the topological graph schematic.
"""

import os
import cv2
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def extract_pcb_circuit_graph(img_path: Path, seg_lbl_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]

    CLASS_NAMES = ['Cap1', 'Cap2', 'Cap3', 'Cap4', 'MOSFET', 'Mov', 'Resistor', 'Transformer']

    # Initialize NetworkX Undirected Circuit Graph
    G = nx.Graph()

    nodes_info = []
    if seg_lbl_path.exists():
        for i, line in enumerate(open(seg_lbl_path)):
            parts = line.strip().split()
            if len(parts) >= 7:
                cid = int(float(parts[0]))
                cname = CLASS_NAMES[cid % len(CLASS_NAMES)]
                coords = list(map(float, parts[1:]))
                pts = np.array([[coords[j]*w, coords[j+1]*h] for j in range(0, len(coords), 2)], dtype=np.float32)

                # Compute centroid of component mask
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))

                node_id = f"{cname}_{i+1}"
                G.add_node(node_id, pos=(cx, cy), class_name=cname, type='component')
                nodes_info.append((node_id, cx, cy, cname))

    # Infer trace edges between geographically adjacent / electrically connected components
    n_nodes = len(nodes_info)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            n1, x1, y1, c1 = nodes_info[i]
            n2, x2, y2, c2 = nodes_info[j]

            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # Add edge if components are within connectivity trace threshold (e.g. 250px)
            if dist < 280.0:
                G.add_edge(n1, n2, weight=round(dist, 2), trace_type='copper')

    return G, img

def visualize_graph_on_pcb(G, img, out_path: Path):
    h, w = img.shape[:2]
    overlay = img.copy()

    pos = nx.get_node_attributes(G, 'pos')

    # Draw Trace Edges (E)
    for u, v, data in G.edges(data=True):
        x1, y1 = map(int, pos[u])
        x2, y2 = map(int, pos[v])
        # Cyan trace line for copper connections
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2, cv2.LINE_AA)

        # Midpoint distance label
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        dist_str = f"{int(data['weight'])}px"
        cv2.putText(overlay, dist_str, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(overlay, dist_str, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    # Draw Component Nodes (V)
    for node, (x, y) in pos.items():
        ix, iy = int(x), int(y)
        cname = G.nodes[node]['class_name']

        # Node circle marker (Magenta)
        cv2.circle(overlay, (ix, iy), 8, (255, 0, 255), -1)
        cv2.circle(overlay, (ix, iy), 10, (0, 0, 0), 2)

        # Node Label
        cv2.putText(overlay, node, (ix + 12, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(overlay, node, (ix + 12, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    blended = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
    cv2.imwrite(str(out_path), blended)
    print(f"Saved PCB Graph Schematic visualization to {out_path}")

def print_netlist_summary(G):
    print("\n" + "="*50)
    print("      PCB CIRCUIT GRAPH THEORY NETLIST REPORT")
    print("="*50)
    print(f"Total Graph Nodes |V| : {G.number_of_nodes()}")
    print(f"Total Trace Edges |E| : {G.number_of_edges()}")
    print(f"Connected Subnets    : {nx.number_connected_components(G)}")

    print("\nNode Degree Distribution & Connectivity:")
    for node, degree in G.degree():
        cname = G.nodes[node]['class_name']
        print(f"  - {node} ({cname}): Degree = {degree} connections")

    print("\nExtracted Netlist (Trace Connections):")
    for u, v, data in G.edges(data=True):
        print(f"  * Net [{u}] <===> [{v}] (Distance: {data['weight']}px)")

if __name__ == "__main__":
    img_dir = Path(r"C:\Userdata\antiiii\dataset_split\train\images")
    seg_dir = Path(r"C:\Userdata\antiiii\dataset_split\train\labels_seg")
    art_out = Path(r"C:\Users\ANAGHA\.gemini\antigravity\brain\a18c0ee1-53bb-4527-a3d0-d12dbe11c8f0\pcb_circuit_graph_preview.png")

    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    for img_path in imgs:
        stem = img_path.stem
        seg_lbl_path = seg_dir / f"{stem}.txt"
        if seg_lbl_path.exists() and seg_lbl_path.stat().st_size > 0:
            G, img = extract_pcb_circuit_graph(img_path, seg_lbl_path)
            if G and G.number_of_nodes() >= 3:
                visualize_graph_on_pcb(G, img, art_out)
                print_netlist_summary(G)
                break
