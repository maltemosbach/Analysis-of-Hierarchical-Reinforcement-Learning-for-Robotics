import os
import shutil
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import iqr
import time




# Import arrays
path = Path("./data")
paths = []
for x in path.iterdir():
    paths.append(x)
    print (x)

#print("len(paths):", len(paths))
dates = []
for m in range(len(paths)):
    dates.append(str(paths[m])[5:])

#print("dates:", dates)

assert len(paths) > 0, "No data found"

path_to_graphs = paths[0]

for j in range(len(paths)):
    path_to_graphs = paths[j]
    f=open(str(path_to_graphs) + "/title.txt", "r")
    title =f.read()

    tmp2 = Path(path_to_graphs)
    graphs = []
    for x in tmp2.iterdir():
        x = str(x)
        if x[-4:] == ".npy":
            graphs.append(x)

    test_graph = np.load(graphs[0])

    #print("test_graph:", test_graph)
    #print("test_graph.shape:", test_graph.shape)

    plots = np.empty((len(graphs), test_graph.shape[0], test_graph.shape[1]))

    for i in range(len(graphs)):
        plots[i, :, :] = np.load(graphs[i])
        
    #print("plots:", plots)
    #print("plots.shape:", plots.shape)


    # Creating graphs for the average testing success rate

    colors = [(0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.7, 0.5, 0.85, 1.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.0, 1.0), (0.5, 0.0, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0)]

    x = np.arange(0, test_graph.shape[1], 1)
    print("x:", x)
    fig, ax = plt.subplots()

    # Calculate interquartile range
    intq_range = np.empty((plots.shape[0], plots.shape[2]), dtype=float)
    for i in range(plots.shape[0]):
        intq_range[i] = iqr(plots[i, :, :], axis=0)

    print("intq_range:", intq_range)
    # Calculate average success rate
    average = np.empty((plots.shape[0], plots.shape[2]), dtype=float)
    for i in range(plots.shape[0]):
        average[i] = np.mean(plots[i, :, :], axis=0)
    print("average:", average)


    yerr = intq_range


    for k in range(plots.shape[0]):
        y = average[k]
        yerr = intq_range[k]
        ax.plot(x, y, color=colors[k])
        plt.fill_between(x, y-yerr, y+yerr, facecolor=colors[k], alpha=0.2)

    plt.title(title)
    plt.ylabel('Median Test Success Rate')
    ax.set_xlim((0, plots.shape[2]-1))
    ax.set_ylim((0.0, 1.2))
    plt.xlabel('Epoch')
    plt.grid(True)
    fig.set_size_inches(8, 4)
    Path("./figures").mkdir(parents=True, exist_ok=True)

    plt.savefig("./figures/" + "success_rate_plot_" + dates[j] + ".jpg", dpi=400, facecolor='w', edgecolor='w',
        orientation='landscape',transparent=False, bbox_inches='tight')




