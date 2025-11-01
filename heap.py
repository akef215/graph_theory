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
        self.type_ = type

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
        if self.type_ == 'max':
            return self.priority(x) < self.priority(y)
        else:
            return self.priority(x) > self.priority(y) 
 
    def enqueue(self, value, priority):
        n = len(self.nodes_)
        self.nodes_.append((value, priority))
        current = n
        pere = self.parent(current)
        while pere >= 0 and self._eval(pere, current):
            # swap pere and current
            self.nodes_[pere], self.nodes_[current] = \
            self.nodes_[current], self.nodes_[pere]
            current = pere
            pere = self.parent(current) 

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
                self.nodes_[index], self.nodes_[best_child] = self.nodes_[best_child], self.nodes_[index]
                index = best_child
            else:
                break 
            
    def dequeue(self):
        """
            Dequeue operation from the heap
        """
        if not self.nodes_:
            return None

        out = self.nodes_[0]
        last = self.nodes_.pop()

        if self.nodes_:
            self.nodes_[0] = last
            self._heapify_down(0)

        return out
