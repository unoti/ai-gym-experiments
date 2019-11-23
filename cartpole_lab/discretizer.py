import numpy as np

class Discretizer:
    """Discretizer takes continuous state features and maps them to discrete buckets.
    This implementation is linear; we might try a curved one using sigmoid later.
    ranges: A list of (min,max) for each feature.
    Example: d = Discretizer([(-4, 4), (-4, 4), (-1, 1), (-4, 4)])
    """
    def __init__(self, ranges, bins=10):
        self.bins = []
        for (start, end) in ranges:
            self.bins.append(self._make_bins(start, end, bins))
    
    def get(self, observation):
        """Returns a discrete state for an observation, suitable for using as a dictionary key."""
        result = []
        for buckets, feature in zip(self.bins, observation):
            x = np.digitize(feature, buckets)
            result.append(x)
        return tuple(result) # Tuples are immutable so they can be used as dictionary keys.

    def _make_bins(self, start, end, steps):
        step_size = (end - start) / steps
        bins = []
        x = start
        for i in range(steps):
            bins.append(x)
            x += step_size
        return np.stack(bins)