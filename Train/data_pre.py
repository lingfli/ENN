import numpy as np
import torch

def node_input(path):
    node = np.load(path + '/node.npy', allow_pickle=True)
    return node


def element_input(path):
    element = np.load(path + '/element.npy', allow_pickle=True)
    return element


def strain_input(path):
    e11 = np.load(path + '/e11.npy', allow_pickle=True)
    e22 = np.load(path + '/e22.npy', allow_pickle=True)
    e12 = np.load(path + '/e12.npy', allow_pickle=True)
    e33 = np.load(path + '/e33.npy', allow_pickle=True)
    return e11, e22, e12, e33

def stress_input(path):
    s11 = np.load(path + '/s11.npy', allow_pickle=True)
    s22 = np.load(path + '/s22.npy', allow_pickle=True)
    s12 = np.load(path + '/s12.npy', allow_pickle=True)
    s33 = np.load(path + '/s33.npy', allow_pickle=True)
    return s11, s22, s12, s33


def boundary_input(path):
    bottom = np.load(path + '/node_bottom.npy', allow_pickle=True)
    top = np.load(path + '/node_top.npy', allow_pickle=True)
    left = np.load(path + '/node_left.npy', allow_pickle=True)
    right = np.load(path + '/node_right.npy', allow_pickle=True)
    return bottom, top, left, right


def steptime_input(path):
    steptime = np.load(path + '/steptime.npy', allow_pickle=True)
    steptime = steptime.shape[0]
    return steptime


def rf_input(path):
    bottom = np.load(path + '/rf_bottom.npy', allow_pickle=True)
    top = np.load(path + '/rf_top.npy', allow_pickle=True)
    left = np.load(path + '/rf_left.npy', allow_pickle=True)
    right = np.load(path + '/rf_right.npy', allow_pickle=True)
    bottom = bottom.reshape(-1, 1)
    top = top.reshape(-1, 1)
    left = left.reshape(-1, 1)
    right = right.reshape(-1, 1)
    rf_out = np.zeros((6, bottom.shape[0], 1))
    rf_out[0, :, :] = top
    rf_out[1, :, :] = bottom
    rf_out[2, :, :] = left
    rf_out[3, :, :] = right
    return rf_out

def Ngrad_input(path):
    N1_grad = np.load(path + '/N1_grad.npy', allow_pickle=True)
    N2_grad = np.load(path + '/N2_grad.npy', allow_pickle=True)
    N3_grad = np.load(path + '/N3_grad.npy', allow_pickle=True)

    return N1_grad, N2_grad, N3_grad
