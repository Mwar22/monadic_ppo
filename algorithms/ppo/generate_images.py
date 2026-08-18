# -*- coding:utf-8 -*-
###
# File:  generate_images.py
# Created Date: 06/08/2026 06:12:59
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 17/08/2026 09:15:29
# Modified By: Lucas de Jesus 
# -----
# Copyright (c) 2026
# 
# This file is subject to the terms and conditions defined in
# the 'LICENSE.txt' file found in the root of this source tree.
# Please read LICENSE.txt for full copyright and licensing details.
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from scipy.signal import savgol_filter
from typing import List, Tuple


plots = [
    {
        "title" : "Average PPO Loss",
        "save_name": "avg_loss.png",
        "data_list": [
            {
                "metric": "losses",
                "mean": 1,
                "rightside":False,
                "style": "r",
                "ylabel": "Loss"
            },
        ],
    },
    {
        "title" : "Average PPO Entropy",
        "save_name": "avg_entropy.png",
        "data_list": [
            {
                "metric": "entropy",
                "mean": 1,
                "rightside":False,
                "style": "b",
                "ylabel": "Entropy"
            },
        ],
    },
    {
        "title" : "Success Rate",
        "save_name": "success_rate.png",
        "loc": "center right",
        "data_list": [
            {
                "metric": "success_rate",
                "label": "Success rate",
                "rightside":False,
                "style": "g",
                "smooth": {
                    "window_length":9, 
                    "polyorder": 3
                },
                "ylabel": "Rate"
            },
            {
                "metric": "error_tol",
                "label": "Error_tolerance",
                "rightside":True,
                "style": "r--",
                "ylabel": "Tol"
            },
        ],
    },
]


def left_right_split(data_list: List)->Tuple[List, List]:
    right_list = []
    left_list= []

    for data in data_list:
        if data["rightside"] is True:
            right_list.append(data)
        else:
            left_list.append(data)

    return left_list, right_list


metrics  = np.load("training_metrics.npz")
#data = np.mean(metrics[metric_name], axis=1) if is_mean else metrics[metric_name]
# data = savgol_filter(data, window_length=13, polyorder=3)

def plot_datalist(axis, data_list):
    for data in data_list:
        metric = metrics[data["metric"]]
        if "mean" in data:
            metric = np.mean(metric, axis=data["mean"])

        if "smooth" in data:
            spar = data["smooth"]
            metric = savgol_filter(metric, window_length=spar["window_length"], polyorder=spar["polyorder"])

        if "ylabel" in data:
            axis.set_ylabel(data["ylabel"])

        if "label" in data:
            axis.plot(metric, data["style"], label= data["label"])
        else:
            axis.plot(metric, data["style"])
        

for plot in plots:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(plot["title"])
    plot_loc = "center right" if "loc" in plot else "best"
    ax.set_xlabel("Update")

    left_list, right_list = left_right_split(plot["data_list"])

    plot_datalist(ax, left_list)
    if right_list:
        ax2 = ax.twinx()
        plot_datalist(ax2, right_list)

        # Get labels/handles from both axes
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc=plot_loc
        )


        ax.yaxis.set_major_locator(LinearLocator(6))
        ax2.yaxis.set_major_locator(LinearLocator(6))
        ax2.grid(False)
        
    else:
        ax.legend(loc=plot_loc)
    
    ax.grid(True)
    fig.savefig(plot["save_name"])
    print(f"\nSaved <<{plot["save_name"]}>>")