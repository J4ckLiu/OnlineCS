import os
import sys
import numpy as np
import pandas as pd 
import random

current_file = os.path.abspath(__file__)
simulation_dir = os.path.dirname(current_file)
project_root = os.path.dirname(simulation_dir)
sys.path.append(project_root)


from algorithms.selection_algorithms import online_BH, online_split
from utils.utils import gen_data, get_regressor, calc_pval, get_metrics
from plots.plot_tools import plot_fdp
from plots.plot_tools import plot_power

# num of train and cal
n1 = 1000
n2 = 2000


    
if __name__ == "__main__":

    ntest = int(sys.argv[1])
    set_id = int(sys.argv[2])  
    q = int(sys.argv[3]) / 10 
    repeat = int(sys.argv[4])
    sig_seq = np.linspace(0.1, 1, num=10)
    reg_names = ['gbr', 'rf', 'svm']
    gamma_r = 0.995


    all_res = pd.DataFrame()
    
    for reg_name in reg_names:
        all_metrics = {}
        for sig_idx in [0, 4]:
            sig = sig_seq[sig_idx]
            metrics_accum = {n: {
                'BH_res_fdp': [], 'BH_res_power': [], 'BH_res_nsel': [],
                'BH_rel_fdp': [], 'BH_rel_power': [], 'BH_rel_nsel': [],
                'BH_2clip_fdp': [], 'BH_2clip_power': [], 'BH_2clip_nsel': [],
                'BH_2clip_split_fdp': [], 'BH_2clip_split_power': [], 'BH_2clip_split_nsel': []
            } for n in range(1, ntest+1)}

            for seed in range(1, repeat+1):
                random.seed(seed)
                
                Xtrain, Ytrain, mu_train = gen_data(set_id, n1, sig)
                Xcalib, Ycalib, mu_calib = gen_data(set_id, n2, sig)
                Xtest, Ytest, mu_test = gen_data(set_id, ntest, sig)
                

                reg = get_regressor(reg_name)
                reg.fit(Xtrain, 1*(Ytrain>0))
                
                pred_calib = reg.predict(Xcalib)
                calib_scores = Ycalib - pred_calib
                calib_scores0 = -pred_calib
                calib_scores_2clip = 1000 * (Ycalib > 0) - pred_calib
                test_scores = -reg.predict(Xtest)
                
                pval_a = calc_pval(calib_scores, test_scores, len(calib_scores))
                pval_b = calc_pval(calib_scores0[Ycalib <= 0], test_scores, len(calib_scores0[Ycalib <= 0]))
                pval_c = calc_pval(calib_scores_2clip, test_scores, len(calib_scores_2clip))
                
                for n_inc in range(1, ntest+1):
                    ts_sub = test_scores[:n_inc]
                    y_sub = Ytest[:n_inc]
                    pa_sub = pval_a[:n_inc]
                    pb_sub = pval_b[:n_inc]
                    pc_sub = pval_c[:n_inc]
                    
                    gamma = (gamma_r ** np.arange(1, n_inc+1)) * (1 - gamma_r) / gamma_r
                    

                    bh_res = online_BH(q, gamma, pa_sub, n_inc)
                    res_fdp, res_power, res_nsel = get_metrics(bh_res, y_sub)
                    bh_rel = online_BH(q, gamma, pb_sub, n_inc)
                    rel_fdp, rel_power, rel_nsel = get_metrics(bh_rel, y_sub)
                    bh_2clip = online_BH(q, gamma, pc_sub, n_inc)
                    clip_fdp, clip_power, clip_nsel = get_metrics(bh_2clip, y_sub)
                    bh_split = online_split(q, gamma, pc_sub, n_inc)
                    split_fdp, split_power, split_nsel = get_metrics(bh_split, y_sub)
                    
                    metrics_accum[n_inc]['BH_res_fdp'].append(res_fdp)
                    metrics_accum[n_inc]['BH_res_power'].append(res_power)
                    metrics_accum[n_inc]['BH_res_nsel'].append(res_nsel)
                    metrics_accum[n_inc]['BH_rel_fdp'].append(rel_fdp)
                    metrics_accum[n_inc]['BH_rel_power'].append(rel_power)
                    metrics_accum[n_inc]['BH_rel_nsel'].append(rel_nsel)
                    metrics_accum[n_inc]['BH_2clip_fdp'].append(clip_fdp)
                    metrics_accum[n_inc]['BH_2clip_power'].append(clip_power)
                    metrics_accum[n_inc]['BH_2clip_nsel'].append(clip_nsel)
                    metrics_accum[n_inc]['BH_2clip_split_fdp'].append(split_fdp)
                    metrics_accum[n_inc]['BH_2clip_split_power'].append(split_power)
                    metrics_accum[n_inc]['BH_2clip_split_nsel'].append(split_nsel)
            
            all_metrics[sig] = metrics_accum
        
        plot_power(all_metrics[sig_seq[0]], all_metrics[sig_seq[4]], reg_name, set_id)  
        plot_fdp(all_metrics[sig_seq[0]], all_metrics[sig_seq[4]], reg_name, set_id)