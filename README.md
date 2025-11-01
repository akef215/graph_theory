# 🧠 Graph Project – Implementation of Graph Algorithms in Python

This project provides a simple and modular implementation of key graph algorithms in Python.  
It allows handling directed or undirected graphs, weighted or unweighted, and applying several classical graph theory algorithms such as **Dijkstra**, **Bellman-Ford**, **Tarjan**, **Floyd-Warshall**, etc.

---

## 📚 Available Algorithms

| Algorithm | Description | Approximate Complexity |
|-----------|-------------|----------------------|
| **Dijkstra** | Finds the shortest path from a source in a weighted graph (without negative weights). | O(V²) |
| **Bellman-Ford** | Computes shortest paths even with negative weights. | O(VE) |
| **Floyd-Warshall** | Finds all-pairs shortest paths. | O(V³) |
| **Roy-Warshall** | Computes the transitive closure (reachability between vertices). | O(V³) |
| **Tarjan** | Detects strongly connected components in a directed graph. | O(V+E) |
| **DFS (Depth-First Search)** | Performs a depth-first traversal of the graph. | O(V+E) |
| **BFS (Breadth-First Search)** | Performs a breadth-first traversal of the graph. | O(V+E) |
| **Topological Sort** | Orders the vertices of a directed acyclic graph (DAG). | O(V+E) |
| **Union-Find** | Finds connected components in the graph. | O(V+E) |

---

## ⚙️ Installation

1. **Clone the project**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   
2. **Create a virtual environment** (optional but recommended)   
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install dependencies**
  ```bash
  pip install -r requirements.txt 
