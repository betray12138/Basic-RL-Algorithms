import numpy as np

# Partly From https://github.com/XinJingHao/Prioritized-DQN-DDQN-Pytorch/blob/main/PriorDQN_gym0.1x/utils.py
class SumTree(object):
    """
    Story data with its priority in the tree.
    Tree structure and array storage:

    full-binary-tree structure
    
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions

    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, buffer_capacity: int):
        self.buffer_capacity = buffer_capacity  # replay buffer capacity
        self.tree_capacity = 2 * buffer_capacity - 1 
        self.tree = np.zeros(self.tree_capacity)    # used to save the p_i^\alpha

    def update_priority(self, buffer_index: int, priority: float):
        ''' Update the priority for one transition according to its index in buffer '''
        tree_index = buffer_index + self.buffer_capacity - 1  # convert the buffer_index into the tree_index
        change = priority - self.tree[tree_index]  # the change is used to update the parent node
        self.tree[tree_index] = priority 
        # propagate the change through the tree
        while tree_index != 0:  
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def prioritized_sample(self, N: int, batch_size: int, beta: float):
        ''' sample a batch of index and normlized IS weight according to priorites '''
        batch_index = np.zeros(batch_size, dtype=np.int64)
        IS_weight = np.zeros(batch_size, dtype=np.float64)    # import sampling weights
        # Segment the [0,priority_sum] into `batch_size` areas and sample uniformly according to these areas
        # Priority Proportional Sampling
        segment = self.priority_sum / batch_size  
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            buffer_index, priority = self.__get_index(v)
            batch_index[i] = buffer_index
            prob = priority / self.priority_sum     # get p(i)
            IS_weight[i] = (N * prob) ** (-beta)    
        Normed_IS_weight = IS_weight / IS_weight.max()  # normalize the weight

        return batch_index, Normed_IS_weight

    def __get_index(self, v: float):
        ''' determine the leaf node belonging to v '''
        parent_idx = 0 
        while True:
            child_left_idx = 2 * parent_idx + 1  
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:  # reach bottom, end search
                tree_index = parent_idx 
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  # tree_index->data_index
        return data_index, self.tree[tree_index]  # return the replay buffer index and the corresponding priority

    @property
    def priority_sum(self):
        return self.tree[0]  # root node saves the sum of all the priority

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max() 