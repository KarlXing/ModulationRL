import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
from os import listdir
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

def get_event_files(direc):
    events = []
    for f in listdir(direc):
        print(f)
        if "commit" in f:
            continue
        for file in listdir(join(direc,f)):
            if "event" in file:
                events.append(join(join(direc,f),file))
    return events

def get_log(event_path):
    # log information: evaluations, pderror, rewards, gs, entropies, dones
    evaluations = dict()
    pderrors = dict()
    dones = dict()
    rewards = dict()
    entropies = dict()
    gs = dict()
    episode_rewards = dict()
    dicts = [evaluations, pderrors, rewards, gs, entropies, dones, episode_rewards ]
    tags = ["analysis/evaluations", "analysis/pd_error", "analysis/reward", "analysis/g", "analysis/entropy",  "analysis/done", "data/reward"]
    assert(len(dicts) == len(tags))
    for event in tf.train.summary_iterator(event_path):
        if len(event.summary.value)>0:
            for i in range(len(dicts)):
                match = add2dict(tags[i], event, dicts[i])
                if match:
                    break
    return dicts

def add2dict(tag, event, dic):
    if tag in event.summary.value[0].tag:
        assert(event.step not in dic)
        dic[event.step] = event.summary.value[0].simple_value
        return True
    else:
        return False


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def get_episode_steps(done, n):
    if type(done) is dict:
        done_steps = [i for i in done.keys()]
    elif type(done) is list:
        done_steps = done
    else:
        raise Exception("invalid type")
    return done_steps[2::3]


def limit(ls, ma, mi):
    return [max(min(i,ma),mi) for i in ls]


def wrap_test(env):
