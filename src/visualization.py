import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import colorsys


def generate_distinct_colors(n):
    """Generate n distinct colors with good separation.

    Args:
        n (int): Number of colors to generate.

    Returns:
        list: List of RGB color tuples.
    """
    colors = []
    for i in range(n):
        # Use HSV color space to generate evenly distributed hues
        hue = i / n
        # Lower saturation and value for softer, more pastel colors (similar to Set3)
        saturation = 0.4 + (i % 3) * 0.05  # 0.4, 0.45, 0.5
        value = 0.95 - (i // 3) * 0.05  # 0.95, 0.9, 0.85, ...
        rgb = colorsys.hsv_to_rgb(hue, saturation, min(value, 1.0))
        colors.append(rgb)
    return colors


def plotGANTT(actions, FJSPinfo, save_path=None):
    """Plot Gantt chart for a given sequence of actions.

    Args:
        actions (list): List of actions.
        FJSPinfo (dict): Dictionary containing information about the FJSP.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    n, m = FJSPinfo["n"], FJSPinfo["m"]

    sns.set_style("white")
    colors = generate_distinct_colors(n)

    jobs_state = np.zeros((n,))
    machines_state = np.zeros((m,))
    operations = FJSPinfo["o"]

    for action in actions:

        opt = action // m
        machine = action % m

        time = operations["time"][0][opt, machine]
        job = operations["job"][0][opt]

        start = max(jobs_state[job], machines_state[machine])
        end = start + time

        jobs_state[job] = end
        machines_state[machine] = end

        plt.barh(machine + 1, time, left=start, color=colors[job])

    for i in range(n):

        plt.barh(0, 0, left=0, color=colors[i], label=f"Job {i+1}")

    makespan = jobs_state.max()
    plt.plot([makespan, makespan], [0, m + 1], color="red", linestyle="--", linewidth=1)

    plt.legend(loc="lower right", ncol=2, bbox_to_anchor=(1.45, 0.0))

    plt.xlabel("Time")
    plt.ylabel("Machine")

    plt.ylim(2.5 / m, m + 1 - 2.5 / m)
    plt.yticks(np.arange(1, m + 1))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=900)

    plt.show()
