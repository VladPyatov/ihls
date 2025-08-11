# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:04:43 2021

@author: Stamatis Lefkimmiatis
"""

import torch as th


def F_compute(E):
    F = E.unsqueeze(-2) - E.unsqueeze(-1)
    F.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    F.pow_(-1)
    return F


def F_approximation_(E, order=10, descending=False):
    # It has the same problem as the original F_compute when two elements of
    # E are the same.
    K = E.unsqueeze(-1).div(E.unsqueeze(-2))
    L = th.ones_like(K)
    R = L - K.pow(order+1)
    R = R.div(L-K)
    R = R.mul(E.unsqueeze(-2).pow(-1))
    R = th.tril(R) if descending else th.triu(R)
    F = R - R.transpose(-1, -2)
    F.diagonal(dim1=-2, dim2=-1).fill_(0.)
    return F


def F_approximation(E, order=10, descending=False):

    K = E.unsqueeze(-1).div(E.unsqueeze(-2))
    R = th.zeros_like(K)
    for i in range(order+1):
        R = R + K.pow(i)

    R = R.mul(E.unsqueeze(-2).pow(-1))
    R = th.tril(R) if descending else th.triu(R)
    F = R - R.transpose(-1, -2)
    return F
