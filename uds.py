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
from torch.distributions import Normal

from ..models.base_model import BaseModel
from hebo.acquisitions.acq import SingleObjectiveAcq

import numpy as np
from matplotlib import pylab
import pylab as plt


import math
import scipy
import statistics
from scipy.special import erf, erfc
from scipy.stats import norm
from torch.quasirandom import SobolEngine
from scipy.stats import pearsonr
from scipy.optimize import bisect



class EPS(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, model_rw : BaseModel, model_av : BaseModel, space, 
                 epsilon, kappa = 2.0, eps_type='eps-ms', 
                 rw_py_best=0, rw_ps_best=0, av_py_worst=0,  av_ps_worst=0, single_model=False,
                 folder='results', **conf):
    
        super().__init__(model, **conf)
        self.kappa = kappa
        self.model_rw = model_rw
        self.model_av = model_av
        self.space = space
        self.rw_py_best = rw_py_best
        self.rw_ps_best = rw_ps_best
        self.av_py_worst = av_py_worst
        self.av_ps_worst = av_ps_worst
        self.epsilon = epsilon
        self.setting = 'min'  # min / max
        self.eps_type = eps_type  # eps-m, eps-ms, eps-bfgs
        self.result_folder = folder
        self.single_model = single_model
        assert(model.num_out == 1)
        
        self.sobol       = SobolEngine(self.space.num_paras, scramble = False)
        
        if (self.eps_type == 'eps-bfgs-est') or (self.eps_type == 'eps-est'):
            
            self.est_bound = self._est_grid_sample(rw_py_best)
            print("rw_best_y: ", self.rw_py_best, ", est_bound: ", self.est_bound)
        
        if (self.eps_type == 'eps-mes'):
            self.num_samples_mes = 100
            self.samples_mes = np.zeros(self.num_samples_mes)
            self._mes_grid_sample_rw()
            
            
        
        
        #print("eps - reset bufs")
        self.buf_rw_m = []
        self.buf_rw_ub = []
        self.buf_rw_lb = []
        self.buf_av_m = []
        self.buf_av_ub = []
        self.buf_av_lb = []
        self.buf_acq = []
            
    
    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        
        
        if (self.single_model):
            
            py, ps2 = self.model_rw.predict(x, xe)
            return py - 2 * ps2.sqrt()
            #return py - self.kappa * ps2.sqrt()
         
            
        
        py_rw, ps2_rw = self.model_rw.predict(x, xe)
        ps_rw = ps2_rw.sqrt()
        py_av, ps2_av = self.model_av.predict(x, xe)
        ps_av = ps2_av.sqrt()
        
        
        
        rw_s_min = 0 #self.rw_ps_best  #min(np.ravel(ps_rw))
        rw_s_max = max(np.ravel(ps_rw))
        if (rw_s_min == rw_s_max): 
            rw_s_min = 0 #1e-10
        av_s_min = 0 #self.av_ps_worst #min(np.ravel(ps_av))
        av_s_max = max(np.ravel(ps_av))
        if (av_s_min == av_s_max): 
            av_s_min = 0 #1e-10
        rw_av_s_min = min(rw_s_min, av_s_min)
        rw_av_s_max = max(rw_s_max, av_s_max)
        
        
        rw_s2_min = min(np.ravel(ps2_rw))
        rw_s2_max = max(np.ravel(ps2_rw))
        if (rw_s2_min == rw_s2_max): 
            rw_s2_min = 0 #1e-10
        av_s2_min = min(np.ravel(ps2_av))
        av_s2_max = max(np.ravel(ps2_av))
        if (av_s2_min == av_s2_max): 
            av_s2_min = 0 #1e-10
        #print("rw_s2_min: ", rw_s2_min, ", rw_s2_max: ", rw_s2_max, ", av_s2_min: ", av_s2_min, ", av_s2_max: ", av_s2_max)
        
        ctr_rw = 0
        ctr_av = 0
        hedonic = torch.zeros(py_rw.shape) #np.zeros(py_rw.shape)
        for i in range(py_rw.shape[0]):
            
            bfgs_rw_m = np.ravel(py_rw[i,0]) 
            bfgs_rw_s = np.ravel(ps_rw[i,0])
            bfgs_rw_s2 = np.ravel(ps2_rw[i,0])
            bfgs_av_m = np.ravel(py_av[i,0]) 
            bfgs_av_s = np.ravel(ps_av[i,0])
            bfgs_av_s2 = np.ravel(ps2_av[i,0])
            
            if isinstance(bfgs_rw_s, np.ndarray):
                bfgs_rw_s[bfgs_rw_s<1e-10] = 1e-10
            elif bfgs_rw_s < 1e-10:
                bfgs_rw_s = 1e-10
                
            if isinstance(bfgs_av_s, np.ndarray):
                bfgs_av_s[bfgs_av_s<1e-10] = 1e-10
            elif bfgs_av_s < 1e-10:
                bfgs_av_s = 1e-10
                
            
            
            if (py_rw[i,0] > 0):
                ctr_rw += 1
                
            if (py_av[i,0] < 0):
                ctr_av += 1
                
            if (self.eps_type == 'eps-lcb'):
                
                #print("eps-bfgs-lcb")
                bfgs_rw_s_norm = (bfgs_rw_s - rw_s_min) / (rw_s_max - rw_s_min)
                bfgs_av_s_norm = (bfgs_av_s - av_s_min) / (av_s_max - av_s_min)   # it could happen that av_s_max = av_s_min
                
                
                gap = 0
                if (bfgs_av_m > bfgs_rw_m):
                    #gap = (bfgs_av_m - bfgs_av_s_norm) - (bfgs_rw_m + bfgs_rw_s_norm)
                    gap = (bfgs_av_m - bfgs_av_s) - (bfgs_rw_m + bfgs_rw_s)
                else:
                    #gap = (bfgs_rw_m - bfgs_rw_s_norm) - (bfgs_av_m + bfgs_av_s_norm)
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
                    #initial_guess = (bfgs_rw_m + bfgs_av_m) / 2
                    optimized_y, pleasure, _ = scipy.optimize.fmin_l_bfgs_b(_func_pleasure_bfgs, 
                                                                        x0=initial_guess, 
                                                                        #args=(bfgs_rw_m, bfgs_rw_s_norm, bfgs_av_m, bfgs_av_s_norm, "min"), 
                                                                        args=(bfgs_rw_m, bfgs_rw_s, bfgs_av_m, bfgs_av_s, "min"), 
                                                                        #bounds=mybounds, 
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
                    if (bfgs_rw_m < bfgs_av_m):
                        rw_z = (pleasure_y - bfgs_rw_m) / bfgs_rw_s
                        av_z = (bfgs_av_m - pleasure_y) / bfgs_av_s
                    else:
                        rw_z = (bfgs_rw_m - pleasure_y) / bfgs_rw_s
                        av_z = (pleasure_y - bfgs_av_m) / bfgs_av_s
                    
                    rw_weight = 0.5
                    av_weight = 0.5 
                    #if (abs(bfgs_av_m - bfgs_rw_m) > 1e-10):
                    if (abs(bfgs_av_m - bfgs_rw_m) > 0.3):
                        # (w)m_A + (1-w)m_R = pleasure
                        av_weight = (pleasure_y - bfgs_rw_m) / (bfgs_av_m - bfgs_rw_m)
                        rw_weight = 1 - av_weight
                        
                        if (bfgs_av_m > bfgs_rw_m):
                            # (w)m_A + (1-w)m_R = pleasure
                            av_weight = (pleasure_y - bfgs_rw_m) / (bfgs_av_m - bfgs_rw_m)
                            rw_weight = 1 - av_weight
                            
                        if (bfgs_rw_m > bfgs_av_m):
                            # (w)m_R + (1-w)m_A = pleasure
                            rw_weight = (pleasure_y - bfgs_av_m) / (bfgs_rw_m - bfgs_av_m)
                            av_weight = 1 - rw_weight
                        
                        
                        rw_av_m = (av_weight * bfgs_av_m) + (rw_weight * bfgs_rw_m)
                        rw_av_m = pleasure_y
                        
                        
                        rw_av_s = ( ((av_weight) * bfgs_av_s) + ((rw_weight) * bfgs_rw_s) ) #+ ((bfgs_rw_s + bfgs_av_s) * 0.5)
                        diff_score = 1 - ( abs(bfgs_rw_s - bfgs_av_s) / (bfgs_rw_s + bfgs_av_s) )
                        rw_av_explore = diff_score * (bfgs_rw_s + bfgs_av_s)
                        
                        
                    else:
                        rw_weight = 0.5
                        av_weight = 0.5 
                        rw_av_m = (bfgs_rw_m + bfgs_av_m) / 2
                        rw_av_m = pleasure_y
                        #rw_av_s = (bfgs_rw_s + bfgs_av_s) / 2
                        if (bfgs_rw_s <= bfgs_av_s):
                            rw_av_s = bfgs_rw_s
                        else:
                            rw_av_s = bfgs_av_s
                        rw_av_explore = 0
                        #rw_av_s = rw_av_s * 2 #self.kappa
                        #pscore = 0
                        diff_score = 0
                        diff_score_s = 0
                        diff_score_m = 0
                    
                rw_av_m = pleasure_y
                diff_score = 1 - ( abs(bfgs_rw_s - bfgs_av_s) / (bfgs_rw_s + bfgs_av_s) )
                diff_score_tanh = 1 - math.tanh( abs(bfgs_rw_s - bfgs_av_s) )
                rw_av_explore = diff_score_tanh * (bfgs_rw_s + bfgs_av_s)
                rw_av_s = rw_av_s + (rw_av_explore * 0.5)
                
                
                
                
                if (rw_av_s < 1e-10) or (math.isnan(rw_av_s)):
                    rw_av_s = np.array([1e-10])
                  
                
                hedonic[i, 0] = torch.from_numpy( rw_av_m - rw_av_s) # - pscore)
                
            
        
        
        return hedonic
    
    
    
    
    
    
    
    
    

    
    
def _func_pleasure_bfgs(params, *args):
        
    # https://stackoverflow.com/questions/8672005/correct-usage-of-fmin-l-bfgs-b-for-fitting-model-parameters
    
    rw_m = args[0]
    rw_s = args[1]
    av_m = args[2]
    av_s = args[3]
    setting = args[4]
    
    y = params
    
    #rw_m = abs(rw_m)
    #av_m = abs(av_m)
    
    if isinstance(rw_s, np.ndarray):
        rw_s[rw_s<1e-10] = 1e-10
    elif rw_s < 1e-10:
        rw_s = 1e-10
        
    if isinstance(av_s, np.ndarray):
        av_s[av_s<1e-10] = 1e-10
    elif av_s < 1e-10:
        av_s = 1e-10
        
    '''
    rw_u = (y - rw_m) / rw_s
    av_u = (y - av_m) / av_s
    rw_Phi = (0.5 * (1 + erf(rw_u / np.sqrt(2))))
    av_Phi = (0.5 * (1 + erf(av_u / np.sqrt(2))))
    '''
    rw_Phi = norm.cdf(y, loc=rw_m, scale=rw_s)
    av_Phi = norm.cdf(y, loc=av_m, scale=av_s)
    
    if (rw_m < av_m):
        pleasure = -(rw_Phi - av_Phi)  # pleasure is an inverted U curve # maximise pleasure, -ve coz bfgs is minimise 
    else:
        pleasure = (rw_Phi - av_Phi) # pleasure is an  U curve
    
    return pleasure


