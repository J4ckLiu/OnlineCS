import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_context("notebook")

PLOT_CONFIG = {
    "temp": 601,
    "figsize": (4, 4),
    "linewidth": 1.75,
    "fontsize_label": 14,
    "fontsize_tick": 12,
    "fontsize_legend": 12,
    "dpi": 600,
    "fdr_threshold": 0.1004,
    "fdr_ylim": (-0.01, 0.12),
    "power_ylim": (-0.01, None),
    "line_styles": {
        "CLIP": {"color": "#82B0D2", "label": r"CLIP"},
        "RES": {"color": "#BEB8DC", "label": r"RES"},
        "OB": {"color": "#FA7F6F", "label": r"OB"},
        "FDR_LINE": {
            "color": "#999999",
            "linewidth": 2,
            "alpha": 1,
            "linestyle": "--",
            "dash_capstyle": "round",
            "zorder": 3,
            "label": "FDR=0.1"
        }
    }
}

def _prepare_plot_data(dic, metric_type):
    n_increments = sorted(dic.keys())
    clip_vals = np.array([np.mean(dic[n][f'BH_2clip_{metric_type}']) for n in n_increments])
    res_vals = np.array([np.mean(dic[n][f'BH_rel_{metric_type}']) for n in n_increments])
    clip_split_vals = np.array([np.mean(dic[n][f'BH_2clip_split_{metric_type}']) for n in n_increments])
    return clip_vals, res_vals, clip_split_vals

def _setup_ax_style(ax, xlim, ylim, xlabel, ylabel, add_fdr_line=False):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG["fontsize_label"])
    ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG["fontsize_label"])
    ax.grid(True, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG["fontsize_tick"])
    
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    if add_fdr_line:
        temp = PLOT_CONFIG["temp"]
        fdr_cfg = PLOT_CONFIG["line_styles"]["FDR_LINE"]
        ax.plot([1, temp-1], [PLOT_CONFIG["fdr_threshold"]]*2, **fdr_cfg)

def _plot_single_figure(x, data_list, save_path, ylim, ylabel, add_fdr_line=False, add_legend=False):
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])
    
    line_types = ["CLIP", "RES", "OB"]
    for idx, line_type in enumerate(line_types):
        cfg = PLOT_CONFIG["line_styles"][line_type]
        ax.plot(x, data_list[idx], color=cfg["color"], label=cfg["label"], 
                linewidth=PLOT_CONFIG["linewidth"])
    
    _setup_ax_style(ax, [0, PLOT_CONFIG["temp"]-1], ylim, 
                    xlabel='Time t', ylabel=ylabel, add_fdr_line=add_fdr_line)
    
    if add_legend:
        ax.legend(fontsize=PLOT_CONFIG["fontsize_legend"], framealpha=0.75, loc='lower right')
    
    plt.savefig(save_path, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    plt.close(fig)

def plot_power(dic_1, dic_5, reg_name, set_id):
    temp = PLOT_CONFIG["temp"]
    x = np.arange(1, temp, 1)
    
    clip_1, res_1, ob_1 = _prepare_plot_data(dic_1, "power")
    clip_5, res_5, ob_5 = _prepare_plot_data(dic_5, "power")
    
    _plot_single_figure(
        x=x,
        data_list=[clip_1, res_1, ob_1],
        save_path=f'results/{reg_name}_{set_id}_power_n01.pdf',
        ylim=PLOT_CONFIG["power_ylim"],
        ylabel=r'Power$_\text{t}$',
        add_fdr_line=False,
        add_legend=(set_id == 1)
    )
    
    _plot_single_figure(
        x=x,
        data_list=[clip_5, res_5, ob_5],
        save_path=f'results/{reg_name}_{set_id}_power_n05.pdf',
        ylim=PLOT_CONFIG["power_ylim"],
        ylabel=r'Power$_\text{t}$',
        add_fdr_line=False,
        add_legend=False
    )

def plot_fdp(dic_1, dic_5, reg_name, set_id):
    temp = PLOT_CONFIG["temp"]
    x = np.arange(1, temp, 1)
    
    clip_1, res_1, ob_1 = _prepare_plot_data(dic_1, "fdp")
    clip_5, res_5, ob_5 = _prepare_plot_data(dic_5, "fdp")
    
    _plot_single_figure(
        x=x,
        data_list=[clip_1, res_1, ob_1],
        save_path=f'results/{reg_name}_{set_id}_fdp_n01.pdf',
        ylim=PLOT_CONFIG["fdr_ylim"],
        ylabel=r'FDR$_\text{t}$',
        add_fdr_line=True,
        add_legend=(set_id == 1)
    )
    
    _plot_single_figure(
        x=x,
        data_list=[clip_5, res_5, ob_5],
        save_path=f'results/{reg_name}_{set_id}_fdp_n05.pdf',
        ylim=PLOT_CONFIG["fdr_ylim"],
        ylabel=r'FDR$_\text{t}$',
        add_fdr_line=True,
        add_legend=False
    )