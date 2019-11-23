import matplotlib.pyplot as plt

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

def plot_episode_lengths(episode_lengths, epsilons):
    # Plot the episode length over time
    # https://matplotlib.org/gallery/api/two_scales.html
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('episode')
    ax1.set_ylabel('steps', color=color)
    ax1.plot(episode_lengths, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx() # Instantiate a second set of axes that share the same x axis

    color = 'tab:orange'
    ax2.set_ylabel('epsilon', color=color)
    ax2.plot(epsilons, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() # Otherwise the right y-label is slightly clipped
    plt.title("Episode Length over Time")
    plt.show()