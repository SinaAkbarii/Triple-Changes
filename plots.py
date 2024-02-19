import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


class CustomFormatter(ScalarFormatter):
    def __init__(self, useMathText=True, *args, **kwargs):
        super().__init__(useMathText=useMathText, *args, **kwargs)

    def _formatSciNotation(self, powerlimits, useMathText):
        formatted = super()._formatSciNotation(powerlimits, useMathText)
        if useMathText:
            formatted = formatted.replace('e', r'\times 10^')
        return formatted


def init_plot(lim_x):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Major and minor ticks setup
    xmajor_ticks = np.arange(0, 8001, 1000)
    xminor_ticks = np.arange(0, 8001, 250)

    ymajor_ticks = np.arange(0, 3, 1)
    yminor_ticks = np.arange(0, 3, .5)

    ax.set_xticks(xmajor_ticks)
    ax.set_xticks(xminor_ticks, minor=True)
    ax.set_yticks(ymajor_ticks)
    ax.set_yticks(yminor_ticks, minor=True)


    # Grid settings
    ax.xaxis.grid(which='minor', alpha=0.4)
    ax.yaxis.grid(which='minor', alpha=0.4)
    ax.xaxis.grid(which='major', alpha=0.6)
    ax.yaxis.grid(which='major', alpha=0.6)

    # Tick label font size
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    # Use scientific notation for y-axis ticks
    formatter = CustomFormatter(useMathText=True)
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    font = {'family': 'arial',
            'color':  'darkblue',
            'weight': 'normal',
            'size': 16,
            }
    plt.xlabel(r'# Samples', fontdict=font)
    plt.ylabel(r'Relative abs. bias', fontdict=font)
    # plt.legend()
    plt.xlim([0, lim_x])
    plt.ylim([0, 2])
    # Draw a solid box around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
        spine.set_clip_path(ax.patch)
    return ax


def plot_data(ax, sample_choices, data, label, color, marker):
    ax.plot(sample_choices, data, label=label, color=color, linewidth=2.5, marker=marker)