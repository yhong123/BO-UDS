#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:50:51 2021

@author: Yean Hoon Ong
"""

# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

#from .base import AcquisitionBase
#from ..util.general import get_quantiles



import torch
from torch import Tensor
from ..models.base_model import BaseModel
from hebo.acquisitions.acq import SingleObjectiveAcq

import numpy as np


import math
import scipy
from scipy.stats import norm




class UDS(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, model_rw : BaseModel, model_av : BaseModel, space, 
                 eps_type='uds', 
                 folder='results', **conf):
    
        super().__init__(model, **conf)
        self.model_rw = model_rw
        self.model_av = model_av
        self.eps_type = eps_type  
        self.result_folder = folder
        assert(model.num_out == 1)
        
        
            
    
    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        
        
        py_rw, ps2_rw = self.model_rw.predict(x, xe)
        ps_rw = ps2_rw.sqrt()
        py_av, ps2_av = self.model_av.predict(x, xe)
        ps_av = ps2_av.sqrt()
        
        
        
        hedonic = torch.zeros(py_rw.shape) 
        for i in range(py_rw.shape[0]):
            
            bfgs_rw_m = np.ravel(py_rw[i,0]) 
            bfgs_rw_s = np.ravel(ps_rw[i,0])
            bfgs_av_m = np.ravel(py_av[i,0]) 
            bfgs_av_s = np.ravel(ps_av[i,0])
             
            if isinstance(bfgs_rw_s, np.ndarray):
                bfgs_rw_s[bfgs_rw_s<1e-10] = 1e-10
            elif bfgs_rw_s < 1e-10:
                bfgs_rw_s = 1e-10
                
            if isinstance(bfgs_av_s, np.ndarray):
                bfgs_av_s[bfgs_av_s<1e-10] = 1e-10
            elif bfgs_av_s < 1e-10:
                bfgs_av_s = 1e-10
                
            
                
            if (self.eps_type == 'uds'):
                
                gap = 0
                if (bfgs_av_m > bfgs_rw_m):
                    gap = (bfgs_av_m - bfgs_av_s) - (bfgs_rw_m + bfgs_rw_s)
                else:
                    gap = (bfgs_rw_m - bfgs_rw_s) - (bfgs_av_m + bfgs_av_s)
                if (gap <= 0):
                    pleasure_y = (bfgs_rw_m + bfgs_av_m) / 2
                    
                    if (bfgs_rw_m < bfgs_av_m):
                        pleasure_y = ((bfgs_rw_m + bfgs_rw_s) + (bfgs_av_m - bfgs_av_s)) / 2
                    if (bfgs_rw_m > bfgs_av_m):
                        pleasure_y = ((bfgs_rw_m - bfgs_rw_s) + (bfgs_av_m + bfgs_av_s)) / 2
                    
                else:
                
                    if (bfgs_rw_s < bfgs_av_s):
                        initial_guess = bfgs_rw_m
                    else:
                        initial_guess = bfgs_av_m
                    optimized_y, pleasure, _ = scipy.optimize.fmin_l_bfgs_b(_func_pleasure_bfgs, 
                                                                        x0=initial_guess, 
                                                                        args=(bfgs_rw_m, bfgs_rw_s, bfgs_av_m, bfgs_av_s), 
                                                                        approx_grad=True, maxiter=10)
                    pleasure_y = optimized_y[0]
                        
                if (bfgs_rw_m <= bfgs_av_m):
                    if (pleasure_y > bfgs_av_m):
                        pleasure_y = bfgs_av_m
                    if (pleasure_y < bfgs_rw_m):
                        pleasure_y = bfgs_rw_m
                
                if (bfgs_rw_m > bfgs_av_m):
                    if (pleasure_y < bfgs_av_m):
                        pleasure_y = bfgs_av_m
                    if (pleasure_y > bfgs_rw_m):
                        pleasure_y = bfgs_rw_m
                        
                 
                #--------------- predictive sigma
                
                if (True):
                    
                    
                    rw_weight = 0.5
                    av_weight = 0.5 
                    if (abs(bfgs_av_m - bfgs_rw_m) > 0.3):
                        
                        
                        if (bfgs_av_m > bfgs_rw_m):
                            # (w)m_A + (1-w)m_R = pleasure
                            av_weight = (pleasure_y - bfgs_rw_m) / (bfgs_av_m - bfgs_rw_m)
                            rw_weight = 1 - av_weight
                            
                        if (bfgs_rw_m > bfgs_av_m):
                            # (w)m_R + (1-w)m_A = pleasure
                            rw_weight = (pleasure_y - bfgs_av_m) / (bfgs_rw_m - bfgs_av_m)
                            av_weight = 1 - rw_weight
                        
                        rw_av_m = pleasure_y
                        rw_av_s = ( ((av_weight) * bfgs_av_s) + ((rw_weight) * bfgs_rw_s) ) #+ ((bfgs_rw_s + bfgs_av_s) * 0.5)
                        
                        
                    else:
                        rw_av_m = pleasure_y
                        if (bfgs_rw_s <= bfgs_av_s):
                            rw_av_s = bfgs_rw_s
                        else:
                            rw_av_s = bfgs_av_s
                        
                    
                rw_av_m = pleasure_y
                #diff_score = 1 - ( abs(bfgs_rw_s - bfgs_av_s) / (bfgs_rw_s + bfgs_av_s) )
                diff_score_tanh = 1 - math.tanh( abs(bfgs_rw_s - bfgs_av_s) )
                rw_av_explore = diff_score_tanh * (bfgs_rw_s + bfgs_av_s)
                rw_av_s = rw_av_s + (rw_av_explore * 0.5)
                
                if (rw_av_s < 1e-10) or (math.isnan(rw_av_s)):
                    rw_av_s = np.array([1e-10])
                
                hedonic[i, 0] = torch.from_numpy( rw_av_m - rw_av_s) 
                
            
        
        
        return hedonic
    
    

    
def _func_pleasure_bfgs(params, *args):
        
    # https://stackoverflow.com/questions/8672005/correct-usage-of-fmin-l-bfgs-b-for-fitting-model-parameters
    
    rw_m = args[0]
    rw_s = args[1]
    av_m = args[2]
    av_s = args[3]
    
    
    y = params
    
    if isinstance(rw_s, np.ndarray):
        rw_s[rw_s<1e-10] = 1e-10
    elif rw_s < 1e-10:
        rw_s = 1e-10
        
    if isinstance(av_s, np.ndarray):
        av_s[av_s<1e-10] = 1e-10
    elif av_s < 1e-10:
        av_s = 1e-10
        
    
    rw_Phi = norm.cdf(y, loc=rw_m, scale=rw_s)
    av_Phi = norm.cdf(y, loc=av_m, scale=av_s)
    
    if (rw_m < av_m):
        pleasure = -(rw_Phi - av_Phi)  # pleasure is an inverted U curve # maximise pleasure, -ve coz bfgs is minimise 
    else:
        pleasure = (rw_Phi - av_Phi) # pleasure is an  U curve
    
    return pleasure


