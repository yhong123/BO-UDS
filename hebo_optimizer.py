#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:36:20 2021

@author: Yean Hoon Ong
"""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import numpy  as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
from sklearn.preprocessing import power_transform

from hebo.design_space.design_space import DesignSpace
from hebo.models.model_factory import get_model
from hebo.acquisitions.acq import MACE, Mean, Sigma, LCB, PI, EI
from hebo.acquisitions.eps import EPS
from hebo.acquisitions.eps_mace import EPS_MACE
from hebo.acquisitions.est_mes import EST, MES
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt

#from .abstract_optimizer import AbstractOptimizer
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace


import statistics
from matplotlib import pylab
import pylab as plt
import scipy
from scipy.special import erf, erfc
import csv
from scipy.stats import norm
import math
import random

torch.set_num_threads(min(1, torch.get_num_threads()))


class HeboOptimizer(AbstractOptimizer):
    
    primary_import = "scikit-optimize"
    
    support_parallel_opt  = True
    support_combinatorial = True
    support_contextual    = True
    def __init__(self, api_config, model_name = 'gpy', rand_sample = None, acq_cls = None, es = 'nsga2', 
                 acq_type = 'mace', eps_th_type = 'q50', folder=''):
        """
        model_name : surrogate model to be used
        rand_iter  : iterations to perform random sampling
        """
        #super().__init__(space)
        AbstractOptimizer.__init__(self, api_config)

        
        space_config =  HeboOptimizer.get_sk_dimensions(api_config) #JointSpace(api_config).spaces
        self.dimensions_list = tuple(dd['name'] for dd in space_config)
        
        self.space = DesignSpace().parse(space_config)
        
        
        self.es          = es
        self.X           = pd.DataFrame(columns = self.space.para_names)
        self.y           = np.zeros((0, 1))
        self.model_name  = model_name
        self.rand_sample = 4
        self.sobol       = SobolEngine(self.space.num_paras, scramble = False)
        
        
        self.acq_type     = acq_type
        if (self.acq_type == 'mace'):
            self.acq_cls = MACE
            print(" HEBO MACE")
        elif (self.acq_type == 'eps-lcb'):
            self.acq_cls = EPS
            print("\n HEBO EPS-lcb")
            
        elif (self.acq_type == 'pi'):
            self.acq_cls = PI
            print(" HEBO PI")
        elif (self.acq_type == 'ei'):
            self.acq_cls = EI
            print(" HEBO EI")
        elif (self.acq_type == 'lcb'):
            self.acq_cls = LCB
            print(" HEBO LCB")
        elif (self.acq_type == 'est'):
            self.acq_cls = EST
            print(" HEBO EST")
        elif (self.acq_type == 'mes'):
            self.acq_cls = MES
            print(" HEBO MES")
        
        self.eps_th_type = eps_th_type
        self.result_folder = folder
        

    @staticmethod
    def get_sk_dimensions(api_config, transform="normalize"):
        """Help routine to setup skopt search space in constructor.

        Take api_config as argument so this can be static.
        """
        # The ordering of iteration prob makes no difference, but just to be
        # safe and consistnent with space.py, I will make sorted.
        param_list = sorted(api_config.keys())
        
        
        sk_dims = []
        #round_to_values = {}
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]

            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            # Some setup for case that whitelist of values is provided:
            values_only_type = param_type in ("cat", "ordinal")
            if (param_values is not None) and (not values_only_type):
                assert param_range is None
                param_values = np.unique(param_values)
                param_range = (param_values[0], param_values[-1])
                #round_to_values[param_name] = interp1d(
                #    param_values, param_values, kind="nearest", fill_value="extrapolate"
                #)
            
            if param_type == "int":
                # Integer space in sklearn does not support any warping => Need
                # to leave the warping as linear in skopt.
                dict1 = {}
                dict1['name'] = param_name
                dict1['type'] = 'int'
                dict1['lb'] = param_range[0]
                dict1['ub'] = param_range[-1]
                sk_dims.append(dict1)
                
            elif param_type == "bool":
                assert param_range is None
                assert param_values is None
                dict1 = {}
                dict1['name'] = param_name
                dict1['type'] = 'bool'
                sk_dims.append(dict1)
                
            elif param_type in ("cat", "ordinal"):
                assert param_range is None
                # Leave x-form to one-hot as per skopt default
                dict1 = {}
                dict1['name'] = param_name
                dict1['type'] = 'cat'
                dict1['categories'] = param_values
                sk_dims.append(dict1)
            
            elif param_type == "real":
                dict1 = {}
                dict1['name'] = param_name
                dict1['type'] = 'num'
                dict1['lb'] = param_range[0]
                dict1['ub'] = param_range[-1]
                sk_dims.append(dict1)
            else:
                assert False, "type %s not handled in API" % param_type
        return sk_dims
    
    
    def quasi_sample(self, n, fix_input = None): 
        samp    = self.sobol.draw(n)
        samp    = samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x       = samp[:, :self.space.num_numeric]
        xe      = samp[:, self.space.num_numeric:]
        df_samp = self.space.inverse_transform(x, xe)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_samp[k] = v
        return df_samp

    @property
    def model_config(self):
        if self.model_name == 'gp':
            cfg = {
                    'lr'           : 0.01,
                    'num_epochs'   : 100,
                    'verbose'      : False,
                    'noise_lb'     : 8e-4, 
                    'pred_likeli'  : False
                    }
        elif self.model_name == 'gpy':
            cfg = {
                    'verbose' : False,
                    'warp'    : True,
                    'space'   : self.space
                    }
        elif self.model_name == 'gpy_mlp':
            cfg = {
                    'verbose' : False
                    }
        elif self.model_name == 'rf':
            cfg =  {
                    'n_estimators' : 20
                    }
        else:
            cfg = {}
        if self.space.num_categorical > 0:
            cfg['num_uniqs'] = [len(self.space.paras[name].categories) for name in self.space.enum_names]
        return cfg
            
    
    def convert_df_to_dict(self, df):
        
        df_len = len(df)
        dict1 = df.to_dict('list')  
        ls_dict = []
        for k in range(df_len):
            dict2 = {}
            for i in range(len(self.dimensions_list)):
                dict2[self.dimensions_list[i]] = dict1[self.dimensions_list[i]][k]  #.tolist()
            ls_dict.append(dict2)
        return ls_dict
    
    
    def suggest(self, n_suggestions=1, fix_input = None):
        
        
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            ls_dict = self.convert_df_to_dict(sample)
            return ls_dict
        else:
            X, Xe = self.space.transform(self.X)
            
            #if (self.acq_cls != EPS):
                #if (self.acq_cls == MACE):
            try:
                if self.y.min() <= 0:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                else:
                    y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'box-cox'))
                    if y.std() < 0.5:
                        y = torch.FloatTensor(power_transform(self.y / self.y.std(), method = 'yeo-johnson'))
                if y.std() < 0.5:
                    raise RuntimeError('Power transformation failed')
                if (self.acq_cls != EPS):
                    model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                    model.fit(X, Xe, y)
            except:
                print('*** Power transformation failed')
                y     = torch.FloatTensor(self.y).clone()
                if (self.acq_cls != EPS):
                    model = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                    model.fit(X, Xe, y)
                
            
            best_id = np.argmin(self.y.squeeze())
            best_x  = self.X.iloc[[best_id]]
            
            
            if (self.acq_cls != EPS):
                py_best, ps2_best = model.predict(*self.space.transform(best_x))
                py_best = py_best.detach().numpy().squeeze()
                ps_best = ps2_best.sqrt().detach().numpy().squeeze()

            iter  = max(1, self.X.shape[0] // n_suggestions)
            upsi  = 0.5
            delta = 0.01
            kappa = np.sqrt(upsi * 2 * ((2.0 + self.X.shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi**2 / (3 * delta))))
            

            #----------------------------------------
            
            if (self.acq_cls == MACE):
                acq = self.acq_cls(model, py_best, kappa = kappa) # LCB < py_best
                assert acq.num_obj > 1
            
            elif (self.acq_cls == PI):
                acq = self.acq_cls(model, py_best)
                
            elif (self.acq_cls == EI):
                acq = self.acq_cls(model, py_best)
            
            elif (self.acq_cls == LCB):
                acq = self.acq_cls(model, kappa = kappa)
                
            elif (self.acq_cls == EST):    
                acq = self.acq_cls(model, self.space, py_best)
                
            elif (self.acq_cls == MES):    
                acq = self.acq_cls(model, self.space)
            
            
            # partition rw & av & apply y transformation for each GP
            elif (self.acq_cls == EPS): # and (False):
                
                 if y.std() < 0.5:
                    print('Power transformation failed')
                    y1 = torch.FloatTensor(self.y).clone()
                    y1_norm = y1 - y1.mean()
                    std = y1.std()
                    if std > 0:
                        y1_norm /= std
                else:
                    y1 = torch.FloatTensor(y).clone()
                    y1_norm = y1
                    
                
                ls_y = np.ravel(y1_norm)
                
                epsilon = statistics.median(ls_y)
                
                
                
                if (self.eps_th_type == 'range'):
                    
                    y_median = sum(ls_y) / len(ls_y)
                    print("\ny_mean: ", y_median)
                    
                    epsilon = y_median
                    
                    ls_rw =[]
                    ls_rw_idx = []
                    ls_rw_med =[]
                    ls_rw_temp =[]
                    ls_rw_med_idx = []
                    ls_av =[]
                    ls_av_idx = []
                    ls_av_med = []
                    ls_av_temp = []
                    ls_av_med_idx = []
                    ls_med = []
                    ls_med_idx = []
                    ls_y_sort = np.sort(ls_y)  # sort from low to high
                    ls_y_argsort = np.argsort(ls_y)
                    
                    
                    
                    for i in range(len(ls_y_sort)):
                        if (ls_y_sort[i] <= y_median):
                            ls_rw_med.append(ls_y_sort[i])
                            ls_rw_med_idx.append(ls_y_argsort[i])
                        if (ls_y_sort[i] < y_median):
                            ls_rw.append(ls_y_sort[i])
                            ls_rw_idx.append(ls_y_argsort[i])
                            ls_rw_temp.append(ls_y_sort[i])
                        if (ls_y_sort[i] >= y_median):
                            ls_av_med.append(ls_y_sort[i])
                            ls_av_med_idx.append(ls_y_argsort[i])
                        if (ls_y_sort[i] > y_median):
                            ls_av.append(ls_y_sort[i])
                            ls_av_idx.append(ls_y_argsort[i])
                            ls_av_temp.append(ls_y_sort[i])
                        if (ls_y_sort[i] == y_median):
                            ls_med.append(ls_y_sort[i])
                            ls_med_idx.append(ls_y_argsort[i])
                            
                    len_ls_med = len(ls_med)
                    #num = math.ceil(len_ls_med / 2)
                    num = int(len_ls_med / 2)
                    if (len_ls_med > 0):
                        for i in range(num):
                            ls_rw_idx.append(ls_med_idx[i])
                            ls_rw.append(ls_med[i])
                            ls_rw_temp.append(ls_med[i])
                        for i in range(num, len_ls_med):
                            ls_av_idx.insert(0, ls_med_idx[i])
                            ls_av.insert(0, ls_med[i])
                            ls_av_temp.insert(0, ls_med[i])
                    
                    
                    rw_add_max = len(ls_av_temp) #- 1 #int(len(ls_av_temp) / 2) #min(3, int(len(ls_av_temp) / 2))
                    num_rw_added = 0
                    ls_av_addto_rw = []
                    ls_av_addto_rw_idx = []
                    
                    av_add_max = len(ls_rw_temp) #- 1 #int(len(ls_rw_temp) / 2) #min(3, int(len(ls_rw_temp) / 2))
                    num_av_added = 0
                    ls_rw_addto_av = []
                    ls_rw_addto_av_idx = []
                    
                    
                    rw_var = np.std(ls_rw_temp, ddof=1)
                    av_var = np.std(ls_av_temp, ddof=1)
                    print("rw_var: ", rw_var, ", av_var: ", av_var)
                    
                    
                    
                    
                    # add middle point to create soft border
                    
                    
                    middle = 0  #int(len(ls_av_idx)/2)
                    ls_av_min = min(ls_av)
                    ls_av_max = max(ls_av)
                    av_gap = ls_av_max - ls_av_min
                    av_len = len(ls_av_idx)
                    if (ls_av_max < 0.5):
                        ls_av_avg1 = statistics.median(ls_av)
                        ls_av_avg2 = statistics.median(ls_av)
                        ls_av_avg3 = statistics.median(ls_av)
                        #ls_av_avg1 = ls_av_min + (av_gap * 0.5)
                        #ls_av_avg2 = ls_av_min + (av_gap * 0.5)
                        middle1 = max(0, int(av_len*0.5) -1)
                        middle2 = max(0, int(av_len*0.5) -1)
                    else:
                        #ls_av_avg1 = ls_av_max * 0.25
                        #ls_av_avg2 = ls_av_max * 0.75
                        ls_av_avg1 = np.quantile(ls_av, 0.25)
                        ls_av_avg2 = np.quantile(ls_av, 0.75)
                        ls_av_avg3 = statistics.median(ls_av)
                        #ls_av_avg1 = ls_av_min + (av_gap * 0.25)
                        #ls_av_avg2 = ls_av_min + (av_gap * 0.75)
                        middle1 = max(0, int(av_len*0.25) -1)
                        if (av_len > 1):
                            middle2 = -2 
                        else:
                            middle2 = max(0, int(av_len*0.9) -1)
                        
                        
                    
                    for i in range(len(ls_av)-1, -1, -1):
                        if (ls_av[i] <= ls_av_avg1):
                            middle = i
                            break
                    
                    
                    middle = middle1 #max(0, int(len(ls_av_idx)*0.5) -1)
                    if (middle1 != middle2):
                        if (not (ls_av_idx[middle] in ls_rw_idx)) and (not (ls_av_idx[middle] in ls_av_addto_rw_idx)): 
                            ls_av_addto_rw_idx.append(ls_av_idx[middle])
                            ls_av_addto_rw.append(ls_av[middle])
                            ls_rw_temp.append(ls_av[middle])
                            num_rw_added += 1
                        
                    print("ls_av_avg1: ", ls_av_avg1, ", idx: ", middle)
                    
                    
                    middle = -1  #int(len(ls_av_idx)/2)
                    for i in range(len(ls_av)):
                        if (ls_av[i] >= ls_av_avg2):
                            middle = i
                            break
                    
                    
                    middle = middle2 #max(0, int(len(ls_av_idx)*0.5) -1)
                    if (not (ls_av_idx[middle] in ls_rw_idx)) and (not (ls_av_idx[middle] in ls_av_addto_rw_idx)): 
                        ls_av_addto_rw_idx.append(ls_av_idx[middle])
                        ls_av_addto_rw.append(ls_av[middle])
                        ls_rw_temp.append(ls_av[middle])
                        num_rw_added += 1
                    
                    print("ls_av_avg2: ", ls_av_avg2, ", idx: ", middle)
                    print("add middle - num_rw_added: ", num_rw_added, ", num_av_added: ", num_av_added)
                    
                    
                    ### augment av
                    
                    #middle = max(0, int(len(ls_rw_idx)/2) - 1)
                    middle = 0 #int(len(ls_rw_idx)/2)
                    ls_rw_min = min(ls_rw)
                    ls_rw_max = max(ls_rw)
                    rw_gap = ls_rw_max - ls_rw_min
                    rw_len = len(ls_rw_idx)
                    if (ls_rw_min > -0.5):
                        ls_rw_avg1 = statistics.median(ls_rw)
                        ls_rw_avg2 = statistics.median(ls_rw)
                        ls_rw_avg3 = statistics.median(ls_rw)
                        #ls_rw_avg1 = ls_rw_min + (rw_gap * 0.5)
                        #ls_rw_avg2 = ls_rw_min + (rw_gap * 0.5)
                        middle1 = max(0, int(rw_len*0.5) )
                        middle2 = max(0, int(rw_len*0.5) )
                    else:
                        #ls_rw_avg1 = ls_rw_min * 0.75
                        #ls_rw_avg2 = ls_rw_min * 0.25
                        ls_rw_avg1 = np.quantile(ls_rw, 0.25)
                        ls_rw_avg2 = np.quantile(ls_rw, 0.75)
                        ls_rw_avg3 = statistics.median(ls_rw)
                        #ls_rw_avg1 = ls_rw_min + (rw_gap * 0.25)
                        #ls_rw_avg2 = ls_rw_min + (rw_gap * 0.75)
                        if (rw_len > 1):
                            middle1 = 1 
                        else:
                            middle1 = max(0, int(rw_len*0.25) )
                        middle2 = max(0, int(rw_len*0.75) )
                        #middle1 = max(0, int(rw_len*0.25) )
                        #if (rw_len > 1):
                        #    middle2 = -2
                        #else:
                        #    middle2 = -1 #max(0, int(rw_len*0.75) )
                    
                    
                    
                    for i in range(len(ls_rw)-1, -1, -1):
                        if (ls_rw[i] <= ls_rw_avg1):
                            middle = i
                            break
                    
                    
                    middle = middle1 #max(0, int(len(ls_rw_idx)*0.5) )
                    if  (not (ls_rw_idx[middle] in ls_av_idx)) and (not (ls_rw_idx[middle] in ls_rw_addto_av_idx)): 
                        ls_rw_addto_av_idx.append(ls_rw_idx[middle])
                        ls_rw_addto_av.append(ls_rw[middle])
                        ls_av_temp.insert(0, ls_rw[middle])
                        num_av_added += 1
                     
                    print("ls_rw_avg1: ", ls_rw_avg1, ", idx: ", middle)
                    
                    
                    middle = -1 #int(len(ls_rw_idx)/2)
                    for i in range(len(ls_rw)):
                        if (ls_rw[i] >= ls_rw_avg2):
                            middle = i
                            break
                    
                    
                    middle = middle2 #max(0, int(len(ls_rw_idx)*0.5) )
                    if (middle1 != middle2):
                        if  (not (ls_rw_idx[middle] in ls_av_idx)) and (not (ls_rw_idx[middle] in ls_rw_addto_av_idx)): 
                            ls_rw_addto_av_idx.append(ls_rw_idx[middle])
                            ls_rw_addto_av.append(ls_rw[middle])
                            ls_av_temp.insert(0, ls_rw[middle])
                            num_av_added += 1
                    
                    print("ls_rw_avg2: ", ls_rw_avg2, ", idx: ", middle)
                    print("add middle - num_rw_added: ", num_rw_added, ", num_av_added: ", num_av_added)
                    
                    
                    
                    
                    # check minimum no of elements
                    min_len = 5
                    total_len = len(ls_rw) + len(ls_av_addto_rw)
                    if  (total_len < min_len): 
                        for i in range(len(ls_av_idx)):
                        #for i in range(len(ls_av_idx)-1, -1, -1):
                            if (not (ls_av_idx[i] in ls_rw_idx)) and (not (ls_av_idx[i] in ls_av_addto_rw_idx)): 
                                ls_av_addto_rw_idx.append(ls_av_idx[i])
                                ls_av_addto_rw.append(ls_av[i])
                                ls_rw_temp.append(ls_av[i])
                                num_rw_added += 1
                                total_len = len(ls_rw) + len(ls_av_addto_rw)
                                if  (total_len >= min_len):
                                    break
                    
                    total_len = len(ls_av) + len(ls_rw_addto_av)
                    if   (total_len < min_len):
                        for i in range(len(ls_rw_idx)-1, -1, -1):
                        #for i in range(len(ls_rw_idx)):
                            if  (not (ls_rw_idx[i] in ls_av_idx)) and (not (ls_rw_idx[i] in ls_rw_addto_av_idx)): 
                                ls_rw_addto_av_idx.append(ls_rw_idx[i])
                                ls_rw_addto_av.append(ls_rw[i])
                                ls_av_temp.insert(0, ls_rw[i])
                                num_av_added += 1
                                total_len = len(ls_av) + len(ls_rw_addto_av)
                                if  (total_len >= min_len):
                                    break
                    
                    print("check len - num_rw_added: ", num_rw_added, ", num_av_added: ", num_av_added)
                    
                    
                    # check variance
                    var_th = 0.5
                    var_th_max = 1.1
                    rw_var = np.std(ls_rw_temp, ddof=1)
                    av_var = np.std(ls_av_temp, ddof=1)
                    rw_av_var_max = max(rw_var, av_var)
                    var_th = rw_av_var_max * 0.25
                    print("rw_var: ", rw_var, ", av_var: ", av_var, ", var_th: ", var_th )
                    if (var_th < 0.2):
                        var_th = 0.2
                    var_th = 0.2
                    if (rw_var < var_th): # or (rw_var > var_th_max):
                        if (num_rw_added < rw_add_max):
                            for i in range(len(ls_av_idx)):
                                if (not (ls_av_idx[i] in ls_rw_idx)) and (not (ls_av_idx[i] in ls_av_addto_rw_idx)): 
                                    ls_av_addto_rw_idx.append(ls_av_idx[i])
                                    ls_av_addto_rw.append(ls_av[i])
                                    ls_rw_temp.append(ls_av[i])
                                    num_rw_added += 1
                                    #rw_var = np.var(ls_rw_temp, ddof=1)
                                    rw_var = np.std(ls_rw_temp, ddof=1)
                                    #if (rw_var >= var_th): # and (rw_var <= var_th_max):
                                    if (rw_var >= var_th) or (num_rw_added >= rw_add_max): 
                                        break
                    print("rw_var: ", rw_var)
                    
                    if (av_var < var_th): # or (av_var > var_th_max):
                        if (num_av_added < av_add_max):
                            for i in range(len(ls_rw_idx)-1, -1, -1):
                                if  (not (ls_rw_idx[i] in ls_av_idx)) and (not (ls_rw_idx[i] in ls_rw_addto_av_idx)): 
                                    ls_rw_addto_av_idx.append(ls_rw_idx[i])
                                    ls_rw_addto_av.append(ls_rw[i])
                                    ls_av_temp.insert(0, ls_rw[i])
                                    num_av_added += 1
                                    #av_var = np.var(ls_av_temp, ddof=1)
                                    av_var = np.std(ls_av_temp, ddof=1)
                                    #if (av_var >= var_th): # and (av_var <= var_th_max):
                                    if (av_var >= var_th) or (num_av_added >= av_add_max):
                                        break
                    print("av_var: ", av_var)    
                    print("check std < 0.3 - num_rw_added: ", num_rw_added, ", num_av_added: ", num_av_added)
                    
                    
                    
                    
                    for i in range(len(ls_av_addto_rw_idx)):
                        ls_rw_idx.append(ls_av_addto_rw_idx[i]) 
                    for i in range(len(ls_rw_addto_av_idx)):
                        ls_av_idx.insert(0, ls_rw_addto_av_idx[i])   
                     
                    
                    
                    idx_rw = ls_rw_idx
                    idx_av = ls_av_idx
                    
                    
                
                
                    
                
                y_rw = y1_norm[idx_rw]
                X_rw = X[idx_rw]
                Xe_rw = Xe[idx_rw]
                #y_rw = y1_norm[idx_rw]
                rw_best_y  = y_rw.min()
                
                
                y_av = y1_norm[idx_av]
                X_av = X[idx_av]
                Xe_av = Xe[idx_av]
                #y_av = y1_norm[idx_av]
                av_best_y  = y_av.min()
                av_best_y = av_best_y.detach().numpy().squeeze()
                
                
                print("y_rw: ", y_rw)
                print("y_av: ", y_av)
                print("y len: ", len(ls_y), ", y_rw len: ", len(y_rw), ", y_av len: ", len(y_av))
                print("y1 std: ", y1.std(), ", y1_norm std: ", y1_norm.std(), ", y_rw std: ", y_rw.std(), ", y_av std: ", y_av.std())
                print("y1 var: ", y1.var(), ", y1_norm var: ", y1_norm.var(), ", y_rw var: ", y_rw.var(), ", y_av var: ", y_av.var())
                
                
                
                y_rw_t     = torch.FloatTensor(y_rw).clone()
                model_rw = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                model_rw.fit(X_rw, Xe_rw, y_rw_t)
                
                y_av_t    = torch.FloatTensor(y_av).clone()
                model_av = get_model(self.model_name, self.space.num_numeric, self.space.num_categorical, 1, **self.model_config)
                model_av.fit(X_av, Xe_av, y_av_t)
                
                
                rw_py_best, rw_ps2_best = model_rw.predict(*self.space.transform(best_x))
                rw_py_best = rw_py_best.detach().numpy().squeeze()
                worst_id = np.argmax(self.y.squeeze())
                worst_x  = self.X.iloc[[worst_id]]
                av_py_worst, av_ps2_worst = model_av.predict(*self.space.transform(worst_x))
                av_py_worst = av_py_worst.detach().numpy().squeeze()
                
                
                acq = self.acq_cls(model_rw, model_rw, model_av, self.space, epsilon, eps_type=self.acq_type, folder=self.result_folder, 
                                   kappa=kappa, rw_py_best=rw_py_best,  av_py_worst=av_py_worst)
                
                
           
            #----------------------------------------
            
            
            #mu  = Mean(model)
            #sig = Sigma(model, linear_a = -1.)
            opt = EvolutionOpt(self.space, acq, pop = 100, iters = 10, verbose = False, es=self.es)
            '''
            if (self.acq_cls == MACE):
                opt = EvolutionOpt(self.space, acq, pop = 100, iters = 100, verbose = False, es=self.es)
            else:
                opt = EvolutionOpt(self.space, acq, pop = 100, iters = 1, verbose = False, es=self.es)
            '''
            rec = opt.optimize(initial_suggest = best_x, fix_input = fix_input).drop_duplicates()
            rec = rec[self.check_unique(rec)]
            #print("rec.shape[0]: ", rec.shape[0], ", n_suggestions: ", n_suggestions)

            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec      = rec.append(rand_rec, ignore_index = True)
                cnt +=  1
                if cnt > 3:
                    # sometimes the design space is so small that duplicated sampling is unavoidable
                    break 
            if rec.shape[0] < n_suggestions:
                print("----- hebo - rec num = 0 -> rand sample")
                rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
                rec      = rec.append(rand_rec, ignore_index = True)

            select_id = np.random.choice(rec.shape[0], n_suggestions, replace = False).tolist()
            '''
            x_guess   = []
            with torch.no_grad():
                py_all       = mu(*self.space.transform(rec)).squeeze().numpy()
                ps_all       = -1 * sig(*self.space.transform(rec)).squeeze().numpy()
                best_pred_id = np.argmin(py_all)
                best_unce_id = np.argmax(ps_all)
                if best_unce_id not in select_id and n_suggestions > 2:
                    select_id[0]= best_unce_id
                if best_pred_id not in select_id and n_suggestions > 2:
                    select_id[1]= best_pred_id
                rec_selected = rec.iloc[select_id].copy()
            '''
            rec_selected = rec.iloc[select_id].copy()
            
            # Then convert to list of dicts
            #print("\nrec_selected: ", rec_selected)
            #ls_rec_selected = rec_selected.squeeze().numpy()
            #next_guess = [dict(zip(self.dimensions_list, x)) for x in ls_rec_selected]
            ls_dict = self.convert_df_to_dict(rec_selected)
            
            return ls_dict #rec_selected


    def check_unique(self, rec : pd.DataFrame) -> [bool]:
        return (~pd.concat([self.X, rec], axis = 0).duplicated().tail(rec.shape[0]).values).tolist()


    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : pandas DataFrame
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,1)
            Corresponding values where objective has been evaluated
        """
        
        # convert list of dict to pandas dataframe
        #print("X: ", X)
        dict1 = {}
        for i in range(len(self.dimensions_list)):
            dict1[self.dimensions_list[i]] = []
        for k in range (len(X)):
            dict2 = X[k]
            for i in range(len(self.dimensions_list)):
                dict1[self.dimensions_list[i]].append(dict2[self.dimensions_list[i]])
        X_df = pd.DataFrame.from_dict(dict1)
        #print("X_df: ", X_df)
            
        y_arr = np.asarray(y)
        
        valid_id = np.where(np.isfinite(y_arr.reshape(-1)))[0].tolist()
        XX       = X_df.iloc[valid_id]
        yy       = y_arr[valid_id].reshape(-1, 1)
        self.X   = self.X.append(XX, ignore_index = True)
        self.y   = np.vstack([self.y, yy])
        
        print("**** y min: ", self.y.min())

    
    
        

    def save_report(self, report_file= None):
        """
        Saves a report with the main results of the optimization.

        :param report_file: name of the file in which the results of the optimization are saved.
        """

        with open(report_file,'w') as file:
            import time

            file.write('-----------------------------' + ' Hebo Report file ' + '-----------------------------------\n')
            file.write('Date and time:               ' + time.strftime("%c")+'\n')
            file.write('Optimization completed:      ' +'YES, ' + str(self.X.shape[0]).strip('[]') + ' samples collected.\n')
            file.write('Number initial samples:      ' + str(self.rand_sample) +' \n')
            #file.write('Optimization time:           ' + str(self.cum_time).strip('[]') +' seconds.\n')

            file.write('\n')
            file.write('--------------------------------' + ' Problem set up ' + '------------------------------------\n')
            #file.write('Problem name:                ' + self.objective_name +'\n')
            file.write('Problem dimension:           ' + str(self.space.num_paras) +'\n')
            #file.write('Number continuous variables  ' + str(len(self.space.get_continuous_dims()) ) +'\n')
            #file.write('Number discrete variables    ' + str(len(self.space.get_discrete_dims())) +'\n')
            #file.write('Number bandits               ' + str(self.space.get_bandit().shape[0]) +'\n')
            #file.write('Noiseless evaluations:       ' + str(self.exact_feval) +'\n')
            #file.write('Cost used:                   ' + self.cost.cost_type +'\n')
            #file.write('Constraints:                  ' + str(self.constraints==True) +'\n')

            file.write('\n')
            file.write('------------------------------' + ' Optimization set up ' + '---------------------------------\n')
            #file.write('Normalized outputs:          ' + str(self.normalize_Y) + '\n')
            #file.write('Model type:                  ' + str(self.model_type).strip('[]') + '\n')
            #file.write('Model update interval:       ' + str(self.model_update_interval) + '\n')
            file.write('Acquisition type:            ' + str(self.acq_type).strip('[]') + '\n')
            #file.write('Acquisition optimizer:       ' + str(self.acquisition_optimizer.optimizer_name).strip('[]') + '\n')

            #if hasattr(self, 'acquisition_optimizer') and hasattr(self.acquisition_optimizer, 'optimizer_name'):
            #    file.write('Acquisition optimizer:       ' + str(self.acquisition_optimizer.optimizer_name).strip('[]') + '\n')
            #else:
            #    file.write('Acquisition optimizer:       None\n')
            #file.write('Evaluator type (batch size): ' + str(self.evaluator_type).strip('[]') + ' (' + str(self.batch_size) + ')' + '\n')
            #file.write('Cores used:                  ' + str(self.num_cores) + '\n')

            file.write('\n')
            file.write('---------------------------------' + ' Summary ' + '------------------------------------------\n')
            file.write('Value at minimum:            ' + str(min(self.y)).strip('[]') +'\n')
            file.write('Best found minimum location: ' + str(self.X.iloc[np.argmin(self.y),:]).strip('[]') +'\n')

            file.write('----------------------------------------------------------------------------------------------\n')
            file.close()
            


    def _write_csv(self, filename, data):
        with open(filename, 'w') as csv_file:
           writer = csv.writer(csv_file, delimiter='\t')
           writer.writerows(data)

    def save_evaluations(self, evaluations_file = None):
        """
        Saves  evaluations at each iteration of the optimization

        :param evaluations_file: name of the file in which the results are saved.
        """
        iterations = np.array(range(1, self.y.shape[0] + 1))[:, None]
        results = np.hstack((iterations, self.y)) #np.hstack((iterations, self.Y, self.X))
        header = ['Iteration', 'Y'] #+ ['var_' + str(k) for k in range(1, self.X.shape[1] + 1)]

        data = [header] + results.tolist()
        self._write_csv(evaluations_file, data)
        
        
    
        



        

if __name__ == "__main__":
    experiment_main(HeboOptimizer)