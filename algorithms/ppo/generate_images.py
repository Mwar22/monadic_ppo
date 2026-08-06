# -*- coding:utf-8 -*-
###
# File:  generate_images.py
# Created Date: 06/08/2026 06:12:59
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 06/08/2026 06:30:36
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


metric_name = "entropy"
save_name = "avg_entropy"
title = "Avg Training Entropy"
is_mean = True

metrics  = np.load("training_metrics.npz") 
data = np.mean(metrics[metric_name], axis=1) if is_mean else metrics[metric_name]
plt.plot(data)
plt.title("Avg Training PPO loss")
plt.xlabel("updates")
plt.savefig(save_name)
print(f"\nTraining plots saved to {save_name}.png")