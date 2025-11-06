class Heap:
    """
        A python implementation of the heap data structure

        Attributs:
            nodes (list of tuples) : The list of the pairs (value, priority)
            of the heap.

            type (str, optional) : 'min' if the highest priority is the smaller
            value, 'max' otherwise.
            Default 'min'.  
    """
    # Constructor
    def __init__(self, type='min'):
        assert type == 'min' or type == 'max', "The type must be min or max"
        self.nodes_ = [] # [(value, priority)]
        self.positions_ = {} 
        self.type_ = type
        
    def _swap(self, i, j):
        # swap nodes
        self.nodes_[i], self.nodes_[j] = self.nodes_[j], self.nodes_[i]
        # update positions
        v1, _ = self.nodes_[i]
        v2, _ = self.nodes_[j]
        self.positions_[v1] = i
        self.positions_[v2] = j

    def init_heap(self, tas):
        self.nodes_ = list(tas)
        self.positions_ = {v: i for i, (v, _) in enumerate(self.nodes_)}
        for i in reversed(range(len(self.nodes_) // 2)):
            self._heapify_down(i)

    def reset(self):
        self.nodes_ = []
        self.positions_ = {}
             
    def copy(self):
        new_heap = Heap(self.type_)
        new_heap.init_heap(self.nodes_.copy()) 
        new_heap.positions_ = self.positions_.copy()
        return new_heap

    def parent(self, index):
        """
            The index of the parent in self.nodes_
        """
        assert 0 <= index < len(self.nodes_)
        return (index - 1) // 2

    def right_child(self, index):
        """
            The index of the right child in self.nodes_
            if it exits, -1 otherwise
        """
        assert 0 <= index < len(self.nodes_)
        calcul = 2 * (index + 1)
        return calcul if calcul < len(self.nodes_) else -1 

    def left_child(self, index):
        assert 0 <= index < len(self.nodes_)
        """
            The index of the left child in self.nodes_
            if it exits, -1 otherwise
        """
        calcul = 2 * (index + 1) - 1
        return calcul if calcul < len(self.nodes_) else -1   

    def priority(self, index):
        """
            The priority of the node at index
        """
        assert 0 <= index < len(self.nodes_)
        return self.nodes_[index][1]
    
    def value(self, index):
        """
            The value of the node at index
        """
        assert 0 <= index < len(self.nodes_)
        return self.nodes_[index][0]
    
    def _eval(self, x, y):
        """
            check if x and y doesn't respect the
            priority order of the heap
        """
        if self.type_ == 'max':
            return self.priority(x) < self.priority(y)
        else:
            return self.priority(x) > self.priority(y) 

    def _heapify_up(self, index):
        """Move a node up to restore heap property."""
        assert 0 <= index < len(self.nodes_)
        while index > 0:
            p = self.parent(index)
            if self._eval(p, index):
                self._swap(p, index)
                index = p
            else:
                break

    def _heapify_down(self, index):
        """
            Restore heap order from given index downward
        """
        assert 0 <= index < len(self.nodes_)
        while True:
            left = self.left_child(index)
            right = self.right_child(index)

            # No children
            if left == -1 and right == -1:
                break

            # Determine which child to compare with
            if right == -1 or (left != -1 and not self._eval(left, right)):
                best_child = left
            else:
                best_child = right

            if best_child == -1:
                break

            # If current and child are in wrong order -> swap
            if self._eval(index, best_child):
                self._swap(index, best_child)
                index = best_child
            else:
                break

    def enqueue(self, value, priority):
        self.nodes_.append((value, priority))
        i = len(self.nodes_) - 1
        self.positions_[value] = i
        self._heapify_up(i) 

    def dequeue(self):
        if not self.nodes_:
            return None

        self._swap(0, len(self.nodes_) - 1)
        out = self.nodes_.pop()
        del self.positions_[out[0]]

        if self.nodes_:
            self._heapify_down(0)
        return out

    def get_nodes(self):
        return [value for value, _ in self.nodes_]
    
    def get_priorities(self):
        return [priority for _, priority in self.nodes_]

    def show(self):
        print("The Heap :")
        print(f"|-- Values : {self.get_nodes()}")
        print(f"|-- Priorities : {self.get_priorities()}")
        print("_________")

    def update_priority(self, value, new_priority):
        if value not in self.positions_:
            return

        i = self.positions_[value]
        old_priority = self.nodes_[i][1]  # <-- extract priority

        if self.type_ == 'min' and new_priority >= old_priority:
            return
        if self.type_ == 'max' and new_priority <= old_priority:
            return

        self.nodes_[i] = (value, new_priority)

        # Decide heapify direction
        if self.type_ == 'min':
            if new_priority < old_priority:
                self._heapify_up(i)
            else:
                self._heapify_down(i)
        else:  # max-heap
            if new_priority > old_priority:
                self._heapify_up(i)
            else:
                self._heapify_down(i)

    def is_empty(self):
        return len(self.nodes_) == 0
