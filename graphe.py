import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

class Graph:
    """
        Graph class,

        Let V be a finite set (in this implementation we
        will consider it as a subset of N including zero)

        Let E be a set of couples of V, i.e
        E = {(u, v)/ u $in$ V and v $in$ V)}

        Let w be a function, i.e
        w : E -> R
            (u, v) |-> w(u, v)

        so,
        G = (V, E, w) is a graph and V, E and w are called ,resp.
        Vertices set, Edges set and the weight of the graph     

        Attributs:
            vertices (set) : the vertices set of G
            edges (list) : the edges set of G
            w (func) : the weight function of G
            oriented (boolean) : True if the Graph is oriented.
                False, otherwisr.
                Default is False
            valued (boolean) : True if the Graph is valued.
                False, otherwisr.
                Default is False
    """
    
    # Constructor
    def __init__():
        vertices_ = {} 
        edges_ = []
        w_ = None
        oriend_ = False
        valued_ = False

    @staticmethod
    def check_pre_condition(index, n):
        """
            check if an index is in range (0, n) 

            raise:
                ValueError if index out of range or negative range
        """
        if n <= 0:
            raise ValueError("Negative length")
        
        if index < 0 or index >= n:
            raise ValueError("index out of range")    

    def plot_graph(self, show=True):
        """
        Display the graph

        Args:
            show (bool, optional) : show the graph          
        """
        # Création du graphe
        G = nx.Graph()
        G.add_edges_from(self.edges_)

        # Dispatch nodes pseudo-randomly from seed
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

        nx.draw_networkx_edges(G, pos, width=2)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
        plt.title("Graphe View")
        plt.axis('off')
        if show:
            plt.show()

    def find_path(self, P, i, j):
        '''
            Find a path from i to j using the predecessors matrix

            Args:
                P (ndarray(n, n)) : The predecessors matrix
                i (int) : The origin node
                j (int) : The destination node

            Raise:
                ValueError if i or j is out of the domain of nodes

            Returns:
                path (array) : The path from i to j if it exists. 
                Otherwise it returns []
        '''
        P = P.copy()
        P -= 1
        i -= 1
        j -= 1
        Graph.check_pre_condition(i, len(P))
        Graph.check_pre_condition(j, len(P))
        
        # We initialise with the destination node
        path = [j+1]
        k = j
        while P[i, k] != -1 and P[i, k] != i:
            # Go back to the predecessor until we reach the origin
            # Or a node with no predecessor (no path case)
            k = P[i, k]
            path.insert(0, int(k)+1)

        # If there is no path
        if P[i, k] == -1:
            return []
        else:
            # We add the origin node to the path
            path.insert(0, i+1)
            return path 

    def next(self, u):
        '''
            The Successors of a given vertex u in the graph

            Args:
                u (int) : The vertex to calculate the successors for

            Raise:
                ValueError if the vertex u doesn't belong to the vertices set

        '''
        Graph.check_pre_condition(u, len(self.vertices_))
        succ = []
        for edge in self.edges_:
            if edge[0] == u:
                succ.append(edge[1])
        return np.array(succ)
    
    def shortest_path_from_origin(self, pi, dest, origin = 0):
        '''
            Find the shortest path from the node origin to the node dest

            Args:
                dest (int) : The node of the end
                pi (ndarray(n)) : The predecessors in the shortest path from the node s
                origin (int, optional) : The node of the origin of the shortest path.
                Default is 0

            Raise:
                ValueError if the origin node origin or the node dest
                don't belong to the nodes set

            Returns:
                path (list) : The shortest path from the node origin to the node dest
                If there is no path from origin to dest return an empty array

        '''
        dest -= 1
        origin -= 1
        pi = pi.copy()
        pi -= 1
        Graph.check_pre_condition(origin, len(pi))
        Graph.check_pre_condition(dest, len(pi))
        if dest == origin:
            return [dest + 1]
        
        pred = pi[dest]
        path = [int(pred) + 1, dest + 1]
        while pred != origin and pred != -1:
            pred = pi[pred]
            path.insert(0, int(pred)+1)

        # The case of no path
        if pred == -1:
            return []

        return path  

    def init_dijkstra(self, s = 0):
        '''
            Initialise the distance and predecessors arrays

            Args:
                n (int) : The number of nodes of the Graph
                s (int, optional) : The node of the origin of the shortest path.
                Default is 0
            
            Raise:
                ValueError if the node of the origin doesn't belong to the nodes set
                            Or if n is not a positive integer

            Returns:
                d (ndarray(1, n)) : The initialised distances, from the node s, array 
                pi (ndarray(1, n)) : The initialised The predecessors in the shortest 
                path from the node s array
        '''
        Graph.check_pre_condition(s, n)
        # The number of vertices
        n = len(self.vertices_)
        d = np.ones(n) * np.inf
        d[s] = 0
        pi = np.full(n, -1)
        return d, pi 

    def relacher(self, MD, u, v, d, pi):
        '''
            The Edge Relaxation procedure
            
            Args:
                MD (ndarray(n, n)) : The distance matrix representing the graph G
                u (int) : The node used in relaxation
                v (int) : The node to be relaxed
                d (ndarray(1, n)) : The distances, from the node s, array 
                pi (ndarray(1, n)) : The array of predecessors in the shortest path 
                from the node s
            
            Raise:
                ValueError if the nodes u and v don't belong to the nodes set
        '''
        Graph.check_pre_condition(u, MD.shape[0])
        Graph.check_pre_condition(v, MD.shape[0])
        
        if d[v] > d[u] + MD[u, v]:
            d[v] = d[u] + MD[u, v]
            pi[v] = u   

    def roy_warshall(self, M, verbose=False):
        '''
            An implementation of the Roy-Warshall algorithm
            to determine the accessibility matrix modified
            to return the matrix of predecessors too

            Args:
                M (ndarray(n, n)) : Adjacency matrix 
                verbose (boolean, optional) : Return the intermediate 
                R^k and P^k for k in {0 ... n-1}
                Default is False
            
            Returns:
                R (ndarray(n, n)) : The accessibility Matrix
                P (ndarray(n, n)) : The predecessors matrix 
        '''

        # Remainder : 
        #   Let k and n be positive integers,
        #   Let i and j be positive integers such that i < j
        #   Let (i, x_0, ..., x_n, j) be a path from i to j
        #   the matrix of paths of level <= k R^k is a matrix of booleans
        #   where R^k[i, j] iff max{x_0, ..., x_n} <= k
        #   the matrix of predecessors of level <= k P^k is a matrix of integers 
        #   where P^k[i, j] = the predecessor of j in a 
        #   path (i, ..., j) of level <= k
        #   -1 if such a path doesn't exist

        # Initialise R^0
        R = M.astype(bool).copy()
        # Trivial paths
        np.fill_diagonal(R, True)

        # Initialise P^0
        P = np.zeros(M.shape, dtype=int)
        for i in range(M.shape[0]):
            for j in range(M.shape[0]):
                # -1 for no predecessor
                P[i, j] = i if R[i, j] else -1

        for k in range(M.shape[0]):
            if verbose:
                print(f"R{k} :")
                print(R.astype(int))
                print(f"P{k} :")
                print(P+1)
                print("\n")
            for i in range(M.shape[0]):
                for j in range(M.shape[0]):
                    # If R^k[i, j] than R^(k+1)[i, j]
                    # The case of updating !R^k[i, j]
                    if not R[i, j] and R[i, k] and R[k, j]:
                        R[i, j] = True
                        P[i, j] = P[k, j] 

        # R^(M.shape[0]) represents the accessibility matrix                
        return R, P+1

    def floyd_warshall(self, M, verbose=False):
        '''
            An implementation of the Floyd-Warshall algorithm
            to determine the shortest distance matrix modified
            to return the matrix of predecessors too

            Args:
                M (ndarray(n, n)) : Weighted adjacency matrix 
                verbose (boolean, optional) : Return the intermediate 
                D^k and P^k for k in {0 ... n-1}
                Default is False
            
            Raise:
                ValueError if a negative weighted cycle is detected
            
            Returns:
                D (ndarray(n, n)) : The shortest distance Matrix
                P (ndarray(n, n)) : The predecessors matrix 
        '''

        # Remainder : 
        #   Let k and n be positive integers,
        #   Let i and j be positive integers such that i < j
        #   Let (i, x_0, ..., x_n, j) be a path from i to j
        #   the matrix of paths of level <= k D^k is a matrix of distances
        #   where D^k[i, j] = min{D[i, j]} where max{x_0, ..., x_n} <= k
        #   the matrix of predecessors of level <= k P^k is a matrix of integers 
        #   where P^k[i, j] = the predecessor of j in a 
        #   path (i, ..., j) of level <= k
        #   -1 if such a path doesn't exist

        # Check if there are any trivial absorbant cycle
        neg = np.where(np.diag(M) < 0)[0]
        if len(neg) != 0:
            index = int(neg[0])+1
            raise ValueError(f"The matrix contains negative weight cycles\n\
                The cycle is : ({index})")
        
        # Initialise D^0
        D = M.copy()
        np.fill_diagonal(D, 0)

        # Initialise P^0
        P = np.full(M.shape, -1, dtype=int)
        for i in range(M.shape[0]):
            for j in range(M.shape[0]):
                # -1 for no predecessor
                if D[i, j] < np.inf: P[i, j] = i

        for k in range(M.shape[0]):
            if verbose:
                print(f"D^{k} :")
                print(D)
                print(f"P^{k} :")
                print(P+1)
                print("\n")
            for i in range(M.shape[0]):
                for j in range(M.shape[0]):
                    if D[i, j] > D[i, k] + D[k, j]:
                        D[i, j] = D[i, k] + D[k, j] 
                        P[i, j] = P[k, j] 
                neg = np.where(np.diag(D) < 0)[0]
                if len(neg) != 0:
                    index = int(neg[0])+1
                    raise ValueError(f"The matrix contains negative weight cycles\n\
                                    The cycle is : {self.find_path(P+1, index, index)}")
        # D^(M.shape[0]) represents the matrix of shortest path distances                
        return D, P+1  

    def dijkstra(self, MD, origin=1, verbose=False):
        '''
            An implementation of the Dijkstra Algorithm

            Args:
                MD (ndarray(n, n)) : The distance matrix representing the graph G
                origin (int, optional) : The node of the origin of the shortest path.
                Default is 1
                verbose (boolean, optional) : Return the intermediate 
                d, pi, O and F arrays
                Default is False

            Raise:
                ValueError if the origin node origin doesn't belong to the nodes set
                AssertionError if the Distance matrix contains negative values

            Returns:
                d (ndarray(n)) : The minimal distances, from the node origin, array 
                pi (ndarray(n)) : The array of predecessors in the shortest path 
                from the node origin

        '''
        # Align with the python array indexation
        origin -= 1
        Graph.check_pre_condition(origin, MD.shape[0])
        assert np.all(MD >= 0), "The Distance Matrix must contains only positive numbers"
        # n represents the number of sommets
        n = MD.shape[0]

        # Initialise the distances and predecessors arrays
        d, pi = self.init_dijkstra(origin)

        # Initialise the Opens and Closed arrays
        O = np.zeros(n).astype(bool)
        O[origin]= True
        F = np.zeros(n).astype(bool)

        if verbose:
            print("iter_0 :")
            print(f"distances array : {d}")
            print(f"predecessors array : {pi+1}")
            print(f"opens array : {O}")
            print(f"closed array : {F}")

        # Main loop
        for i in range(n-1):
            # Find all the nodes u where O[u] and not F[u]
            boucle = np.where(O & ~F)[0]
            if len(boucle) == 0:
                break
            # Take the minimum u, in term of distance d[u]
            u = boucle[np.argmin(d[boucle])]
            succ = self.next(MD, u)
            
            for v in succ:
                self.relacher(MD, u, v, d, pi)
                O[v] = True
            
            # Update the Closed array
            F[u] = True
            if verbose:
                print('_________')
                print(f"iter_{i} :")
                print(f"vertex_{u+1} :")
                print(f"distances array : \n{d}")
                print(f"predecessors array : \n{pi+1}")
                print(f"opens array : \n{O}")
                print(f"closed array : \n{F}")
        
        return d, pi+1   

    def bellman_Ford(self, M, origin=1):
        """
            An implementation of the Bellman-Ford algorithm
            to find the shortest path from the origin node

            Args:
                M (ndarray(n, n)) : The distances matrix
                origin (int, optional) : The origin node
                Default is 1
            
            Returns:
                d (ndarray(n)) : The shortest distances from the origin
                node array
                pi (ndarray(n)) : The predecessor in the shortest path
                from the origin node array
        """
        origin -= 1
        # Initialisation of the distances and predecessors arrays
        d, pi = self.init_dijkstra(M.shape[0], origin)

        for init, final, _ in self.edges_:
            # The relaxation process for each edge of the graph
            self.relacher(M, init, final, d, pi)
            
        return d, pi+1
    
    def visited_init(self):
        '''
            An auxillary function to initialise the visited 
            array in a protected way

            Args:
            
            Returns:
                visited (list) : The initialised visited list 
                s.t. (np.all(visited==False) is np.True_)
        '''
        return np.zeros(len(self.vertices_)).astype(bool)
    
    def depth_first_search(self, adj_dict, origin, visited, out, post=False):
        '''
            The Depth First Traversal of a weighted oriented Graph
            from an origin vertex

            Args:
                adj_dict (dict) : The adjacency list of the graph G
                {vertex : [neighbours_of_vertex]}
                origin (int) : The vertex from where we will start the traversal
                visited (list) : The initialisation of the visited array
                it must be a list of booleans all False
                It is recommanded to use the auxilary function visited_init(adj_dict)
                out (list) : The traversal list, it used for the output
                It must be an empty list
                post (bool, optional) : If True we'll make a post order traversal.
                Default is False
            
            Raise:
                ValueError if the origin is out of range

            Warning:
                Due to the recursive implementation there is no way to check 
                the veracity of the visited and output arguments so respect
                the description above 
        '''
        # Check that the origin belongs to the vertices set
        Graph.check_pre_condition(origin, len(adj_dict))

        # Check the length of the visited array
        assert len(visited) == len(adj_dict), "The length of visited must be equals to\
            the length of adj_dict"
        
        # change the state of the current vertex to visited
        visited[origin] = True
        if not post:        
            # add it to the output list
            out.append(int(origin)) 
        
        # Traverse the neighbours of the current vertex
        for neighbour in adj_dict[origin]:
            if not visited[neighbour]:
                # The recursive call
                self.depth_first_search(adj_dict, neighbour, visited, out, post)
        if post:        
            # add it to the output list
            out.append(int(origin)) 
    
    def dfs(self, adj_dict, post=False):
        '''
            The Depth First Traversal of a weighted oriented Graph

            Args:
                adj_dict (dict) : The adjacency list of the graph G
                {vertex : [neighbours_of_vertex]}
                post (bool, optional) : If True we'll make a post order traversal.
                Default is False
    
            Returns:
                out (list) : The traversal list
        '''
        # Initialise safely The visited and out args
        visited = self.visited_init()
        out = []                           
        
        for origin in range(len(adj_dict)):
            if not visited[origin]:
                self.depth_first_search(adj_dict, origin, visited, out, post)
            if np.all(visited):
                break

        return out
    
    def bredth_first_search(self, adj_dict, origin):
        '''
            The Breadth First Search for a graph G given
            by its adjacency list executed from a vertex origin

            Args:
                adj_dict (dict) : The adjacency list of the graph G
                {vertex : [neighbours_of_vertex]}
                origin (int) : The vertex on which we will start the BFS
            
            Raise:
                ValueError if the origin is out of range

            Returns:
                out (list) : The vertices visited and ordered by the BFS
        '''
        # Check that the origin belongs to the vertices set
        Graph.check_pre_condition(origin, len(self.vertices_))
        
        # Visited boolean array to avoid processing the same 
        # vertex more than one time
        visited = self.visited_init()

        # Initialise the FIFO queue
        pile = deque([origin])

        visited[origin] = True
        # Initialise the output list that contains the BFS
        out = []

        # Repeat until the queue is empty
        while len(pile) != 0:
            # Dequeue the vertex the queue
            vertex = pile.popleft()

            # Add it to the output list
            out.append(vertex)

            # Iterate over the neighbours of the current vertex
            for neighbour in adj_dict[vertex]:
                # Enqueue the unvisited neighbours and set them to true
                if not visited[neighbour]:
                    pile.append(neighbour)
                    visited[neighbour] = True
        return out
    
    def tarjan(self, adj_dict):
        '''
            Tarjan algorithm to find strongly connected components
            of an oriented graph G given by its adjacency list

            Args:
                adj_dict (dict) : The adjacency list of the graph G
                {vertex : [neighbours_of_vertex]}
            
            Returns:
                sccs (list) : a list of strongly connected components 
        '''

        # n represents the number of vertices of the graph
        n = len(self.vertices_)
        # initialise the array of ids ordered by their DFS
        # appearing order
        ids = np.full(n, -1, dtype=int)

        # the low link array, such that: for i in {0 .. n-1},
        # low[i] is the smallest id of vertices id reachable from
        # vertex i - according to ids array -
        low = np.zeros(n, dtype=int)

        # to check wether a vertex i is in the stack
        on_stack = np.zeros(n, dtype=bool)
        # initialise an empty stack
        stack = []

        id_counter = [0]
        # Initialise the Strongly connected components list
        sccs = []

        # A modified DFS for the Tarjan algorithm
        def dfs_tarjan(at):
            # A counter to ennumerate the vertices according
            # to their appearence order in the DFS
            id_counter[0] += 1

            # We set new ids for vertices according to their
            # order in the DFS
            # Initialise the low key value to its id
            # NB : each vertex is reachable from itself
            ids[at] = low[at] = id_counter[0]
            
            # Push the vertex in the stack
            stack.append(at)
            on_stack[at] = True

            for to in adj_dict[at]:
                # recursive call to pursue the DFS
                # if 'to' isn't visited yet
                if ids[to] == -1:
                    dfs_tarjan(to)
                    # Update the low link value
                    # NB: 'to' is reachable from 'at'
                    low[at] = min(low[at], low[to])
                elif on_stack[to]:
                    # Update the low link value if
                    # the vertex is in the stack
                    # NB: 'to' is reachable from 'at'
                    low[at] = min(low[at], ids[to])

            # If we find a root of a SCC
            if ids[at] == low[at]:
                scc = []
                # Pop the stack elements until 
                # reaching the 'at' vertex in the SCC list
                while True:
                    node = stack.pop()
                    on_stack[node] = False
                    scc.append(node + 1)
                    if node == at:
                        break
                
                # Add the SCC of root 'at' to the sccs list
                sccs.append(scc[::-1])

        # Starting the DFS in all vertices
        # so it traverses all the graph
        for i in range(n):
            # if vertex i is unvisited yet
            if ids[i] == -1:
                dfs_tarjan(i)

        return sccs  
    
    def topological_sort(self, adj_dict, random=False):
        '''
            Topological sort of an oriented graph G given by its
            successors list.

            Args:
                adj_dict (dict) : The adjacency list of the graph G
                {vertex : [neighbours_of_vertex]}
                
                random (bool, optional) : if False, procede by the
                ids order in adj_dict. Else, using a random order
                Default is False

            Raise:
                AssertionError if G isn t a directed acycled graph (DAG)

            Returns:
                sorted_vertices (ndarray (n,)) : The topological sort from left to right   
        '''

        # We check that the number of SCC is the number of vertices
        assert len(self.tarjan(adj_dict)) == len(adj_dict), "There's no topological \
            sort in cycled grahes"  
        
        sorted_vertices = []
        # Initialise safely The visited and out args
        visited = self.visited_init()
        out = []  

        # Choose the order of vertices
        vertices = self.vertices_   
        if random:
            np.random.shuffle(vertices)  
                            
        for origin in vertices:
            if not visited[origin]:
                # The visited vertices by the dfs from vertex origin
                self.depth_first_search(adj_dict, origin, visited, out, post=True)

                # Concatenate inversed order results from the left 
                sorted_vertices = out[::-1] + sorted_vertices 

                # re-initialise the out array so we only get the visited vertices
                # from the vertex origin
                out = []   
            # If all vertices visited we stop 
            if np.all(visited):
                break
                
        return np.array(sorted_vertices)
    
    def topological_index(self, adj_dict):
        '''
            Find a topological indexing of a graph 
            - rename the vertices, so they match a topological sort

            Args:
                adj_dict (dict) : The adjacency list of the graph G
                {vertex : [neighbours_of_vertex]}
        '''
        out = {}
        sort = list(self.topological_sort(adj_dict))
        for id in adj_dict:
            neighbours = []
            for succ in adj_dict[id]:
                neighbours.append(sort.index(succ)) 
            out[sort.index(id)] = neighbours
        return out  
    
    def valid_topo_sort(self, adj_dict, sort):
        """
            Check if a given sorting is a topological sort of 
            the graph given by adj_dict

            Args:
                adj_dict (dict) : The adjacency list of the graph G
                {vertex : [neighbours_of_vertex]}

                sort (list) : the sort we are willing to check if
                it is topological
        """
        n = len(self.vertices_)
        for u in range(n-1):
            for v in range(u, n):
                # the index operation of python is considered as a bijection
                # between the vertex in the original graph i and the sorted
                # graph sort[i]

                # let u' and v' be two vertices of the sorted graph,
                # we are checking if (u' < v') and (u' is a successor of v')
                # (translated by : u' in Succ[v']) 
                # which means that it is not a valid topological sort 
                # u' = sort[u] is the image of u by the defined bijection above,
                # without a loss of generality for v'   
                if sort[u] in adj_dict[sort[v]]:
                    print("problem with vertices : ", sort[u], sort[v])
                    return False
                
        return True
    
    def discrete_forest(self):
        '''
            Discrete Forest of a graph G, each vertex is considered as an arborescence

            Args:
            
            Returns:
                pi (ndarray(|G|)) : The array of predecessors of G
        '''
        max_ = max(max(u, v) for u, v in self.edges_)
        return -np.ones(shape=max_, dtype=int)

    def root(self, pi, v):
        """
            Find the root of a node v using the predecessors list
        """
        # align v with the python indexing
        v -= 1 
        # we loop until reaching the root caracterized by pi[root] == -1
        while pi[v] != -1:
            # we keep going up in the arborescence
            v = pi[v]
        return v

    def union(self, pi, r1, r2):
        """
            Attach the node r1 to r2

            Args:
                pi (ndarray(len(self.vertices_))) : The list of predecessors
                of the graph 
                r1 (int) : the id of the node that will get attached 
                r2 (int) : the id of the node that will be a the parent

            Returns:        
        """
        # attach the node r1 to r2
        if r1 != r2:
            pi[r1] = r2

    def union_find(self, verbose=False):
        """
            Union-Find to find the connex components of the graph

            Args:
                verbose (boolean, optional) : Return the intermediate 
                steps of union
                Default is False

            Returns:
                pi (ndarray(len(self.vertices_))) : The list of predecessors
                of the graph        
        """
        # initialise a discrete forest
        pi = self.discrete_forest()

        for v_begin, v_end in self.edges_:
            # Find part:
            # find the roots of the extremeties of the edges
            # in the random forest 
            r1 = self.root(pi, v_begin)
            r2 = self.root(pi, v_end)

            # Union part:
            self.union(pi, r1, r2)

            if verbose:
                print(f"Union {r1} and {r2}: {pi}")
                print("_________")
        
        return pi
    
