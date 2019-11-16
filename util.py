import matplotlib.pyplot as plt
import numpy as np

def histogram(seq, bins=10, title='Histogram'):
    plt.hist(seq, edgecolor = 'black', bins = bins)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_fn(f, start, end, point_count = 100):
    """Example: plot_fn(sigmoid, -5, 5)"""
    
    x = np.linspace(start, end, point_count)
    y = f(x)
    plt.plot(x, y)
    plt.show()