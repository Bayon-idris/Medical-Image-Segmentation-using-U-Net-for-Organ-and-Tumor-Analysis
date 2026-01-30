import os
import matplotlib.pyplot as plt


def plot_and_save(x, ys, labels, title, xlabel, ylabel, save_path):
    plt.figure()

    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
