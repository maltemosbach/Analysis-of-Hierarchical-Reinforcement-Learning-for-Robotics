import os
import shutil
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import iqr
import time




# Import arrays
datadir = Path("./data")
dates = []
date_paths = []
for x in datadir.iterdir():
    date_paths.append(x)
    dates.append(str(x)[5:])


assert len(dates) > 0, "No data found"


for j in range(len(date_paths)):
    current_path = date_paths[j]
    paths_to_graph = []

    for x in current_path.iterdir():
        if str(x)[-7:-1] == "graph_":
            paths_to_graph.append(x)
    f=open(str(paths_to_graph[0]) + "/title.txt", "r")  
    title =f.read()  

    plots = []

    for k in range(len(paths_to_graph)):
        current_path_to_graph = paths_to_graph[k]

        current_sr_list = []
        current_Q_val_list = []
        for x in current_path_to_graph.iterdir():
            if str(x)[-12:-5] == "sr_run_":
                current_sr_list.append(x)
            elif str(x)[-21:-5] == "Q_val_table_run_":
                current_Q_val_list.append(x)


        time.sleep(100)

        test_graph = np.load(current_sr_list[0])

        current_sr_array = np.empty((len(current_sr_list), test_graph.shape[0]))

        #print("current_sr_array:", current_sr_array)

        for i in range(len(current_sr_list)):
            current_sr_array[i, :] = np.load(current_sr_list[i])

        #print("current_sr_array:", current_sr_array)
        plots.append(current_sr_array)

    # Creating graphs for the average testing success rate

    colors = [(0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.7, 0.5, 0.85, 1.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.0, 1.0), (0.5, 0.0, 0.5, 1.0), (0.0, 0.5, 0.5, 1.0)]

    x = np.arange(0, test_graph.shape[0], 1)
    #print("x:", x)
    fig, ax = plt.subplots()

    # Calculate interquartile range
    intq_range = np.empty((len(plots), plots[0].shape[1]), dtype=float)
    for i in range(len(plots)):
        intq_range[i] = iqr(plots[i], axis=0)

    #print("intq_range:", intq_range)
    # Calculate average success rate
    average = np.empty((len(plots), plots[0].shape[1]), dtype=float)
    for i in range(len(plots)):
        average[i] = np.mean(plots[i], axis=0)
    #print("average:", average)

    yerr = intq_range


    for k in range(len(plots)):
        y = average[k]
        yerr = intq_range[k]
        ax.plot(x, y, color=colors[k])
        plt.fill_between(x, y-yerr, y+yerr, facecolor=colors[k], alpha=0.2)

    plt.title(title)
    plt.ylabel('Median Test Success Rate')
    ax.set_xlim((0, plots[0].shape[1]-1))
    ax.set_ylim((0.0, 1.2))
    plt.xlabel('Epoch')
    plt.grid(True)
    fig.set_size_inches(8, 4)
    Path("./figures").mkdir(parents=True, exist_ok=True)

    plt.savefig("./figures/" + "success_rate_plot_" + dates[j] + ".jpg", dpi=400, facecolor='w', edgecolor='w',
        orientation='landscape',transparent=False, bbox_inches='tight')









































