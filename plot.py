from IPython.core.display_functions import display
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np


def slider2d(xts, lims=None, title='', c='r', flow_plot_n=1000):
    from ipywidgets import interact, FloatSlider
    from matplotlib import pyplot as plt
    import io
    from PIL import Image
    import numpy as np  # Import numpy
    start_t = 0  # Define start_t if it's not already defined
    # Pre-generate all frames
    time_steps = np.linspace(start_t, 1, xts.shape[-1])
    frames = []
    for i in range(xts.shape[-1]):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
        if not isinstance(c, np.ndarray) and not isinstance(c, list):
            ax.plot(xts[:flow_plot_n, 0].T, xts[:flow_plot_n, 1].T, c='grey', alpha=0.1, zorder=0, linewidth=0.3)
        else:
            for unique_c in np.unique(c):
                mask = np.array(c[:flow_plot_n]) == unique_c
                ax.plot(xts[:flow_plot_n][mask][:, 0].T, xts[:flow_plot_n][mask][:, 1].T, alpha=0.1, zorder=0,
                        linewidth=0.3, c=unique_c)
        ax.scatter(xts[:flow_plot_n, 0, i], xts[:flow_plot_n, 1, i],
                   c=np.full(flow_plot_n, c) if not isinstance(c, np.ndarray) and not isinstance(c, list) else c[
                                                                                                               :flow_plot_n],
                   alpha=0.1 + 0.9 * time_steps[i],
                   zorder=1, edgecolor='w', linewidth=1)
        ax.set_title(title + "\n" * bool(title) + f'Flow at t={time_steps[i]:.2f}')
        if lims:
            plt.xlim(lims[0])
            plt.ylim(lims[1])

        # Save the figure to a buffer and load as a PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).copy()
        frames.append(image)
        plt.close(fig)

    def plot_flow_at_t(t):
        # Find the index corresponding to the current time t
        closest_t_index = np.argmin(np.abs(time_steps - t))
        # Use display to show the image in the output cell
        display(frames[closest_t_index])

    return interact(plot_flow_at_t,
                    t=FloatSlider(min=start_t, max=1.0, step=time_steps[1] - time_steps[0], value=start_t,
                                  description='Time (t):'))


def time_animation(xts, lims=None, title='', c='r', flow_plot_n=1000, interval=100, last_extended=10):
    """
    Creates a matplotlib animation from a 3D numpy array by displaying scatter plots over the last dimension (time).

    Args:
        xts (np.ndarray): The input array with shape (num_samples, dimensions, time_steps).
        lims (tuple, optional): A tuple of two lists defining the x and y limits for the plot. Defaults to None.
        title (str, optional): The title of the plot. Defaults to ''.
        c (str or list or np.ndarray, optional): The color(s) for the scatter plot. Defaults to 'r'.
        flow_plot_n (int, optional): The number of samples to plot for the flow lines and scatter points. Defaults to 1000.
        interval (int, optional): Delay between frames in milliseconds. Defaults to 100.

    Returns:
        IPython.display.HTML: The HTML object containing the generated animation.
    """
    start_t = 0
    time_steps = np.linspace(start_t, 1, xts.shape[-1])
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)

    if lims is None:
        min_x, max_x = np.min(xts[:, 0]), np.max(xts[:, 0])
        min_y, max_y = np.min(xts[:, 1]), np.max(xts[:, 1])
        lims = ([min_x, max_x], [min_y, max_y])

    # Duplicate the last frame to make it last longer
    xts_extended = np.concatenate([xts] + [xts[:, :, -1:]] * last_extended, axis=-1)
    time_steps_extended = np.concatenate((time_steps, np.ones(last_extended)))

    # Initialize the plot elements that will be updated in each frame
    # Plot all flow lines at once
    flow_lines = ax.plot(xts[:flow_plot_n, 0].T, xts[:flow_plot_n, 1].T, alpha=0.1, zorder=0, linewidth=0.3)
    if isinstance(c, np.ndarray) or isinstance(c, list):
        for i, line in enumerate(flow_lines):
            line.set_color(c[i])

    scatter_plot = ax.scatter(xts_extended[:flow_plot_n, 0, 0], xts_extended[:flow_plot_n, 1, 0],
                              c=c[:flow_plot_n] if (isinstance(c, np.ndarray) or isinstance(c, list)) else c,
                              alpha=0.1 + 0.9 * time_steps_extended[0], zorder=1, edgecolor='w', linewidth=1)

    # Collect all artists to be updated
    artists_to_update = [scatter_plot]

    ax.set_title(title + "\n" * bool(title) + f'Flow at t={time_steps_extended[0]:.2f}')
    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])

    def init():
        scatter_plot.set_offsets(xts_extended[:flow_plot_n, :, 0])
        scatter_plot.set_alpha(0.1 + 0.9 * time_steps_extended[0])
        if isinstance(c, np.ndarray) or isinstance(c, list):
            scatter_plot.set_color(c[:flow_plot_n])
            scatter_plot.set_edgecolor('w')

        return artists_to_update

    def update(frame):
        current_t = time_steps_extended[frame]
        ax.set_title(title + "\n" * bool(title) + f'Flow at t={current_t:.2f}')

        scatter_plot.set_offsets(xts_extended[:flow_plot_n, :, frame])
        scatter_plot.set_alpha(0.1 + 0.9 * time_steps_extended[frame])
        if isinstance(c, np.ndarray) or isinstance(c, list):
            scatter_plot.set_color(c[:flow_plot_n])
            scatter_plot.set_edgecolor('w')

        return artists_to_update

    ani = animation.FuncAnimation(fig, update, frames=xts_extended.shape[-1],
                                  init_func=init,
                                  blit=True, interval=interval)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def calculate_square_rows_cols(n):
    """
    Given n, calculates ceil(sqrt(n)) and ceil(n/ceil(sqrt(n)),
    to choose the number of rows and columns to make the figure as square as possible
    :param n: number of subplots
    :return: (rows, cols)
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols
