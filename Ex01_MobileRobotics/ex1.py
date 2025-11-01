#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


# def motion_model(action, belief):
    # add code here
    
def sensor_model(observation, belief, world):
    """
    Sensor update step for Bayes filter.
    observation: 0 = blue, 1 = white
    world: array of true colors (0=blue, 1=white)
    belief: current belief before incorporating observation
    """

    # Sensor correctness probabilities
    p_blue_correct = 0.85
    p_white_correct = 0.75

    # Likelihood for each cell
    likelihood = np.zeros_like(belief)

    for i, tile in enumerate(world):
        if tile == 0:  # true color = blue
            if observation == 0:       # sensed blue
                likelihood[i] = p_blue_correct
            else:                      # sensed white
                likelihood[i] = 1 - p_blue_correct
        else:  # true color = white
            if observation == 1:       # sensed white
                likelihood[i] = p_white_correct
            else:                      # sensed blue
                likelihood[i] = 1 - p_white_correct

    # Apply Bayes rule: posterior ∝ likelihood × prior belief
    posterior = belief * likelihood

    # Normalize so total probability = 1
    posterior /= np.sum(posterior)

    return posterior


# def recursive_bayes_filter(actions, observations, belief, world):
    # add code here
