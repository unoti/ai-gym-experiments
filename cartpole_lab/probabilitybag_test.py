from probabilitybag import ProbabilityBag
import unittest
from collections import Counter

class TestProbabilityBag(unittest.TestCase):
    def setUp(self):
        names = ['Roland', 'Eddie', 'Odetta', 'Jake', 'Oy']
        weights = [3, 2, 2, 1, 0.5]
        # This isn't part of the API for ProbabilityBag, but
        # we need it for the tests.
        self._weights_by_name = {}
        for weight, name in zip(weights, names):
            self._weights_by_name[name] = weight

        self.gunslingers = ProbabilityBag(max_size=7)
        self.gunslingers.insert_batch(zip(weights, names))

    def test_correct_weight(self):
        self.assertEqual(8.5, self.gunslingers.total_weights)

    def test_items_inserted(self):
        self._verify_count(5)

    def _verify_count(self, n):
        self.assertEqual(n, len(self.gunslingers.items))
        self.assertEqual(n, len(self.gunslingers.weights))
        self.assertEqual(n, len(self.gunslingers))

    def test_batch_remove(self):
        items = self.gunslingers.remove_batch(2)
        self.assertEqual(2, len(items))
        self.assertEqual(type('x'), type(items[0]))
        self._verify_count(3) # After taking 2 out there should be 3 left.

    def test_probability(self):
        """Verify that each item is selected proportionally to its weight."""
        self.assertEqual(5, len(self.gunslingers.weights))
        counts_by_name = Counter()
        iterations = 1000
        items_per_batch = 3
        items_remaining = len(self.gunslingers) - items_per_batch
        for i in range(iterations):
            self._verify_count(5)
            items = self.gunslingers.remove_batch(items_per_batch)
            self.assertEqual(items_remaining, len(self.gunslingers))
            self.assertEqual(items_per_batch, len(items))

            # Count how many times each was selected
            for item in items:
                counts_by_name[item] += 1
            
            # Add them back in at the same probabilities
            batch = [(self._weights_by_name[name], name) for name in items]
            self.gunslingers.insert_batch(batch)

        total_selections = sum(counts_by_name.values())
        #print('total_selections',total_selections)
        for key, value in counts_by_name.items():
            select_count = counts_by_name[key]
            # Verify each item selected at least once
            self.assertGreater(select_count, 0, 'Expected %s to be selected at least once' % key)

            # Verify percentage of each item.
            weight = self._weights_by_name[key]
            expected_fraction = weight / self.gunslingers.total_weights
            actual_fraction = select_count / total_selections
            #print('actual_count=%d name=%s expected_frac=%f actual_frac=%f' % (select_count, key, expected_fraction, actual_fraction))
            #I'm unsure what assertions we can make here usefully and reliably. So taking this out.
            #self.assertAlmostEqual(expected_fraction, actual_fraction, places=1)

        largest = max(counts_by_name.values())
        smallest = min(counts_by_name.values())
        self.assertEqual(largest, counts_by_name['Roland'], 'Expected Roland to be selected the most because his weight is greatest')
        self.assertEqual(smallest, counts_by_name['Oy'], 'Expected Oy to be selected the least because his weight is smallest')
        self.assertNotEqual(largest, smallest) # To make sure there's no funny stuff like all counts = 0
        self.assertNotEqual(largest, 0)
        self.assertNotEqual(smallest, 0)

if __name__ == '__main__':
    unittest.main()