# PCB Graph Theory & Netlist Circuit Extraction Engine

We have successfully built the **PCB Graph Theory Engine** (`pcb_graph_network.py`) that converts computer vision segmentation masks into a formal **Graph Network $G(V, E)$** and **Electrical Circuit Netlist**.

![PCB Graph Network Circuit Schematic Preview](C:\Users\ANAGHA\.gemini\antigravity\brain\a18c0ee1-53bb-4527-a3d0-d12dbe11c8f0\pcb_circuit_graph_preview.png)

---

## 1. Graph Theory Formulation $G(V, E)$

1. **Vertices / Nodes ($V$)**:
   - Each detected PCB component mask (`Resistor_1`, `MOSFET_2`, `Cap3_5`, `Transformer_3`, etc.) is mapped to a graph node vertex $v_i \in V$.
   - Node features: Centroid coordinates $(x_i, y_i)$, component classification label, bounding box geometry.

2. **Edges ($E$)**:
   - Copper trace paths and physical wiring interconnections between component pins/pads form edges $e_{ij} = (v_i, v_j) \in E$.
   - Edge weight $w_{ij}$: Euclidean trace distance $\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$.

---

## 2. Extracted Circuit Netlist Report

```text
==================================================
      PCB CIRCUIT GRAPH THEORY NETLIST REPORT
==================================================
Total Graph Nodes |V| : 10 component vertices
Total Trace Edges |E| : 22 trace connections
Connected Subnets    : 1 unified circuit net

Node Degree Centrality & Connectivity:
  - MOSFET_2 (MOSFET): Degree = 6 trace connections
  - Transformer_3 (Transformer): Degree = 6 trace connections
  - Mov_10 (Mov): Degree = 6 trace connections
  - Cap3_5 (Cap3): Degree = 5 trace connections
  - Transformer_9 (Transformer): Degree = 4 trace connections
  - Cap4_6 (Cap4): Degree = 4 trace connections
  - Resistor_1 (Resistor): Degree = 3 trace connections
  - Cap2_7 (Cap2): Degree = 2 trace connections
  - Cap1_8 (Cap1): Degree = 2 trace connections

Extracted Netlist Connections:
  * Net [Resistor_1] <===> [Transformer_3] (Trace Length: 132px)
  * Net [Resistor_1] <===> [Transformer_4] (Trace Length: 156px)
  * Net [MOSFET_2]   <===> [Cap3_5]         (Trace Length: 157px)
  * Net [Cap3_5]     <===> [Cap4_6]         (Trace Length: 98px)
  * Net [Cap2_7]     <===> [Cap1_8]         (Trace Length: 100px)
```

---

## 3. Deliverable Script

- **PCB Graph Theory & Netlist Script**: [pcb_graph_network.py](file:///c:/Userdata/antiiii/pcb_graph_network.py)
- **Detailed Graph Report**: [PCB Graph Circuit Theory Report](file:///C:/Users/ANAGHA/.gemini/antigravity/brain/a18c0ee1-53bb-4527-a3d0-d12dbe11c8f0/pcb_graph_circuit_theory.md)
