import numpy as np

class ProbabilityBag:
    """
    A ProbabilityBag is a collection of items that can provide samples of items,
    where you can define the probability of drawing items from the bag.
    It has the following characteristics:
     * The rows inserted into the bag indexable things (tuples, lists, or numpy arrays)
     * The probability that a given item will be drawn from a bag depends that item's weight.
     * The weight of each item is given when it's inserted into the bag.
     * When the bag gets full we remove oldest items to make room.
    """
    def __init__(self, max_size):
        self.total_weights = 0 # For calculating probability per item. We'll maintain this as we add and remove items.
        self.max_size = max_size # Maximum number of entries in the bag.
        self.weights = np.array([])
        self.items = []
        self.nonzero_epsilon = 0.0001 # Make all probabilities non-zero.
    
    def insert_batch(self, weighted_items):
        """Insert a group of items.  The priority of each item is the first member of the tuple.
        weighted_items: a sequence like [(weight1, item1), (weight2, item2)...]

        Example:
            bag = ProbabilityBag(max_size=10)
            items = [(7, 'seven'), (9, 'nine')]
            probability_bag.push_batch(items)
        """
        # Make room for new items if necessary.
        if not hasattr(weighted_items, '__len__'):
            weighted_items = list(weighted_items) # So we can check length.
        overage = len(self.items) + len(weighted_items) - self.max_size
        if overage > 0:
            self._delete_first_items(overage)

        for weight, item in weighted_items:
            self.total_weights += weight
            self.weights = np.append(self.weights, weight + self.nonzero_epsilon)
            self.items.append(item)

    def remove_batch(self, desired_count):
        """Remove a randomly selected batch of n items from the bag.  The probability
        of an item being selected is proportional to its priority.
        """
        batch_count = min(desired_count, len(self.items))
        if batch_count == 0:
            return []
        # I could keep a list of probabilities stable/updated instead of calculating it here.
        # But there's no point: whenever we calculate the probabilities we're going to
        # update the list anyway.
        probabilities = self.weights / np.sum(self.weights)
        # Generate n indecies, chosen with the given probabilities, no duplicates.
        indexes = np.random.choice(len(probabilities), batch_count, p=probabilities, replace=False)
        
        # We're going to remove the items from the items list.  To keep our indexes stable,
        # we need to remove the highest indexes first.
        indexes_descending = sorted(list(indexes), reverse=True) # Theres probably a way to do this in numpy, but this is fine.

        # Build the batch, and remove the items from the bag.
        batch = []
        for index in indexes_descending:
            weight, item = self._pop(index)
            batch.append(item)
        
        return batch

    def _delete_first_items(self, n):
        self.total_weights -= np.sum(self.weights[:n])
        self.weights = np.delete(self.weights, np.s_[:n]) # Delete first n items from the front.
        self.items = self.items[n:] # Delete first n items.

    def _pop(self, index):
        weight = self.weights[index]
        self.weights = np.delete(self.weights, index)
        self.total_weights -= weight
        item = self.items.pop(index)        
        return weight, item

    def __len__(self):
        return len(self.items)
