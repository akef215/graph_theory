# 🧭 Travelling Salesman Problem (TSP)

Cette branche contient l’implémentation et l’analyse de l’algorithme du **Problème du Voyageur de Commerce (TSP)**, dans le cadre du module **Algorithmes des graphes**.

---

## 📘 Objectif

Le **TSP** consiste à trouver le **plus court cycle** passant une seule fois par chaque ville (sommet) et revenant à la ville de départ.  
Ce problème est un classique de l’optimisation combinatoire et illustre les méthodes de recherche de chemin optimal.

---

## ⚙️ Contenu de la branche

- `tsp.py` — Implémentation principale de l’algorithme du TSP  
- `tsp_demo.ipynb` — Notebook de démonstration et visualisation des résultats  
- `graph.py` — Classe d’aide pour la représentation du graphe  
- `utils.py` — Fonctions utilitaires (distance, génération aléatoire de points, etc.)

---

## 🧩 Algorithmes implémentés

- **Nearest Neighbor (Plus proche voisin)** — Heuristique gloutonne simple  
- **2-opt** — Amélioration locale du chemin trouvé  

---

## 📊 Visualisation

Une partie du code permet de **visualiser le graphe et le chemin optimal** à l’aide de `matplotlib`.  
Chaque exécution peut générer une figure illustrant le circuit minimal trouvé.

---

## 🚀 Exécution

```bash
# Cloner le dépôt
git clone <url-du-repo>

# Se placer sur la branche TSP
git checkout TSP

# Exécuter le script principal
python tsp.py
