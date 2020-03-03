# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 20:00:31 2016 by emin
"""
import numpy as np
import torch

class LeInit():
    """Initialize weights with diagonal + off-diagonal random matrix.
    Parameters
    ----------
     diag_val : float  (diagonal scale)
     offdiag_val : float  (off-diagonal scale)
    """
    def __init__(self, diag_val=1.0, offdiag_val=0.0):
        self.diag_val = diag_val
        self.offdiag_val = offdiag_val

    def sample(self, shape):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('LeInit initializer can only be used for 2D square matrices.')
            
        off_diag_part = self.offdiag_val * np.random.randn(shape[0], shape[1])
        return torch.Tensor(np.eye(shape[0]) * self.diag_val + off_diag_part - np.diag(np.diag(off_diag_part)))
