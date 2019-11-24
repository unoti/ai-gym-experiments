import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

def histogram(seq, bins=10, title='Histogram'):
    plt.hist(seq, edgecolor = 'black', bins = bins)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_fn(f, start, end, point_count = 100):
    """Example: plot_fn(sigmoid, -5, 5)"""
    
    x = np.linspace(start, end, point_count)
    y = f(x)
    plt.plot(x, y)
    plt.show()

def plot_episode_lengths(episode_lengths, epsilons, save=None):
    """Plot the episode lengths over time.
    save: Filename to save plot such as 'image.png'
    Example:
        episode_lengths = [9, 19, 17, 24, 32, 54, 76, 12, 92]
        epsilons = [0.99 ** i for i in range(len(episode_lengths))]
        plot_episode_lengths(episode_lengths, epsilons)
    """
    # 
    # https://matplotlib.org/gallery/api/two_scales.html
    if save:
        clear_output(wait=True)

    fig, ax1 = plt.subplots(figsize=(10,5))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps', color=color)
    if len(episode_lengths) > 1000: # https://matplotlib.org/api/markers_api.html
        dotstyle = ',' # pixels
    else:
        dotstyle = '.' # points
    ax1.plot(episode_lengths, dotstyle, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    avg_width = int(len(episode_lengths)/4)
    ax1.plot(moving_average(episode_lengths, window_width=avg_width), color='black', linestyle='--')
    #handles, labels = ax1.get_legend_handles_labels()

    color = 'tab:orange'
    ax2 = ax1.twinx() # Instantiate a second set of axes that share the same x axis
    ax2.set_label('Epsilon')
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(epsilons, linestyle='dotted', alpha=0.5, color=color)
    ax2.locator_params(nbins=8)
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() # Otherwise the right y-label is slightly clipped
    plt.title("Episode Length over Time")

    if save:
        plt.savefig(save)
    plt.show()

def test_episode_lengths():
    episode_lengths = [9, 19, 17, 24, 32, 54, 76, 12, 92]
    epsilons = [0.99 ** i for i in range(len(episode_lengths))]
    plot_episode_lengths(episode_lengths, epsilons)

def moving_average(data, window_width=10):
    # https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    # Maybe we'd be better off using an exponentially weighted moving average. But here's this.
    w = window_width
    cumulative = np.cumsum(np.insert(data, 0, 0))
    avg_vec = (cumulative[w:] - cumulative[:-w]) / w
    return avg_vec