#!/usr/bin/env python3
"""
pcb_sknw_graph_engine.py

Applies the proven SKNW (Skeleton to NetworkX Graph) algorithm to Printed Circuit Boards:
1. Thins copper trace masks down to a 1-pixel topological skeleton.
2. Uses sknw.build_sknw() to extract exact Graph Nodes V (Junctions & Endpoints) and Edges E.
3. Renders a crisp, professional Graph Theory Schematic overlay directly over physical PCB photos.
"""

import os
import glob
import cv2
import numpy as np
import networkx as nx
import sknw
from pathlib import Path

def skeletonize_mask(binary_mask):
    """
    Morphological thinning algorithm to extract 1-pixel-wide medial axis skeleton.
    """
    skeleton = np.zeros(binary_mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_mask.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return (skeleton > 0).astype(np.uint8)

def apply_sknw_to_pcb():
    img_dir = Path(r"C:\Userdata\antiiii\dataset_split\train\images")
    art_out = Path(r"C:\Users\ANAGHA\.gemini\antigravity\brain\a18c0ee1-53bb-4527-a3d0-d12dbe11c8f0\sknw_pcb_graph_visual.png")

    valid_imgs = sorted([p for p in img_dir.glob("VID*.jpg") if "mask" not in p.name and "overlay" not in p.name])
    if not valid_imgs:
        valid_imgs = sorted([p for p in img_dir.glob("*.jpg") if "legend" not in p.name.lower()])

    for img_path in valid_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 1. Lab Color Background Subtraction
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        border_px = 10
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:border_px, :] = True
        border_mask[-border_px:, :] = True
        border_mask[:, :border_px] = True
        border_mask[:, -border_px:] = True

        bg_mean = lab[border_mask].mean(axis=0)
        dist = np.linalg.norm(lab - bg_mean, axis=2)
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, binary_traces = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_traces = cv2.morphologyEx(binary_traces, cv2.MORPH_CLOSE, kernel)
        binary_traces = cv2.morphologyEx(binary_traces, cv2.MORPH_OPEN, kernel)

        # 2. Extract 1-pixel skeleton
        skeleton = skeletonize_mask(binary_traces)

        # 3. Apply SKNW Algorithm (build_sknw)
        graph = sknw.build_sknw(skeleton, multi=True)

        if len(graph.nodes()) < 5:
            continue

        # 4. Render SKNW Graph Schematic Overlay
        canvas = img.copy()

        # Draw SKNW Graph Edges (E) in Cyan
        for u, v, k, data in graph.edges(keys=True, data=True):
            pts = data['pts']
            pts_int = pts[:, [1, 0]].astype(np.int32)
            # Draw precise curve path connecting nodes
            cv2.polylines(canvas, [pts_int], False, (255, 255, 0), 2, cv2.LINE_AA)

        # Draw SKNW Graph Nodes (V) in Red/Magenta
        nodes = graph.nodes()
        for n in nodes:
            node_data = graph.nodes[n]
            # sknw nodes give (y, x) centroid
            py, px = int(node_data['o'][0]), int(node_data['o'][1])
            deg = graph.degree(n)

            if deg == 1:
                # Terminal Pin Endpoint (Red)
                cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
                cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1)
            else:
                # Trace Junction Node (Magenta)
                cv2.circle(canvas, (px, py), 6, (255, 0, 255), -1)
                cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)

        cv2.imwrite(str(art_out), canvas)
        print(f"\n==================================================")
        print(f"   SKNW (SKELETON-TO-GRAPH) TOPOLOGY EXTRACTION SUCCESS")
        print(f"==================================================")
        print(f"PCB Image            : {img_path.name}")
        print(f"Extracted Graph Nodes |V|: {len(graph.nodes())} (Pins & Junctions)")
        print(f"Extracted Trace Edges |E|: {len(graph.edges())} Trace Connections")
        print(f"Visual Saved To      : {art_out}")
        break

if __name__ == "__main__":
    apply_sknw_to_pcb()
