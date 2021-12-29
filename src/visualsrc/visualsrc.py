# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 11:34 2021

@author: au558899

Source codes for visualization-related codes for main extractor of newsFluxus

"""

import os
from icecream import ic
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from icecream import ic

import sys
sys.path.insert(1, r'/home/commando/marislab/newsFluxus/src/')
import saffine.detrending_method as dm

mpl_size = 10000

class baseVisualsrc:
    @staticmethod
    def normalize(x, lower=-1, upper=1):
        """ transform x to x_ab in range [a, b]
        x: list of values to normalize
        lower: int lower bound
        upper: int upper bound
        """
        x_norm = (upper - lower)*((x - np.min(x)) / (np.max(x) - np.min(x))) + lower
        return x_norm

    @staticmethod
    def adaptive_filter(y, span=56):
        """
        y: list
        span: int
        """
        w = int(4 * np.floor(len(y)/span) + 1)
        y_dt = np.mat([float(j) for j in y])
        y_dt = np.float32(y_dt)
        _, y_smooth = dm.detrending_method(y_dt, w, 1)
        return y_smooth.T

class plotVisualsrc:
    @staticmethod
    def plot_ci_manual(
                       t,
                       s_err,
                       n,
                       x,
                       x2,
                       y2,
                       ax=None):
        """Return an axes of confidence bands using a simple approach.
        t: 
        s_err: 
        n:
        x: 
        x2: 
        y2: 
        ax: 
        """
        if ax is None:
            ax = plt.gca()
        ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")
        return ax

    @staticmethod
    def plot_ci_bootstrap(
                          xs,
                          ys,
                          resid,
                          nboot=500,
                          ax=None):
        """Return an axes of confidence bands using a bootstrap approach.
        xs: 
        ys: 
        resid: 
        nboot: 
        ax: 
        
        Returns
        -------
        ax : axes
            - Cluster of lines
            - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers
        """ 
        if ax is None:
            ax = plt.gca()

        bootindex = sp.random.randint

        for _ in range(nboot):
            resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
            # Make coeffs of for polys
            pc = np.polyfit(xs, ys + resamp_resid, 1)                   
            # Plot bootstrap cluster
            ax.plot(xs, np.polyval(pc, xs), "r-", linewidth=2, alpha=3.0 / float(nboot))
        return ax

    @staticmethod
    def adaptiveline(
                     x1,
                     x2, 
                     fname="adaptline.png"):
        """
        x1: 
        x2: 
        fname: filename for saving the figure
        """
        bV = baseVisualsrc()
        mpl.rcParams['agg.path.chunksize'] = mpl_size
        _, ax = plt.subplots(2,1,figsize=(14,6),dpi=300)
        c = ["g", "r", "b"]
        ax[0].plot(bV.normalize(x1, lower=0), c="gray")
        for i, span in enumerate([128, 56, 32]):
            n_smooth = bV.normalize(bV.adaptive_filter(x1, span=span), lower=0)
            ax[0].plot(n_smooth,c=c[i])
        ax[0].set_ylabel("$\\mathbb{N}ovelty$", fontsize=14)
        
        ax[1].plot(bV.normalize(x2, lower=-1),c="gray")
        for i, span in enumerate([128, 56, 32]):
            r_smooth = bV.normalize(bV.adaptive_filter(x2, span=span), lower=-1)
            ax[1].plot(r_smooth,c=c[i])
        ax[1].set_ylabel("$\\mathbb{R}esonance$", fontsize=14)
        mpl.rcParams['agg.path.chunksize'] = mpl_size
        plt.tight_layout()
        plt.show()
        plt.savefig(fname)
        plt.close()
        
    @staticmethod
    def adaptiveline_toptimes(
                            x1, 
                            x2, 
                            x, 
                            y, 
                            cond, 
                            fname="adaptline_top.png"):
        """
        x1: 
        x2: 
        x: 
        y: 
        cond: 
        fname: filename for saving the figure
        """
        bV = baseVisualsrc()
        mpl.rcParams['agg.path.chunksize'] = mpl_size
        fig, ax = plt.subplots(2,1,figsize=(14,6),dpi=300)
        c = ["g", "r", "b"]
        ax[0].plot(bV.normalize(x1, lower=0),c="gray")
        for i, span in enumerate([128, 56, 32]):
            n_smooth = bV.normalize(bV.adaptive_filter(x1, span=span), lower=0)
            ax[0].plot(n_smooth,c=c[i])
        ax[0].set_ylabel("$\\mathbb{N}ovelty$", fontsize=14)
        
        ax[1].plot(bV.normalize(x2, lower=-1),c="gray")
        for i, span in enumerate([128, 56, 32]):
            r_smooth = bV.normalize(bV.adaptive_filter(x2, span=span), lower=-1)
            ax[1].plot(r_smooth,c=c[i])
        ax[1].set_ylabel("$\\mathbb{R}esonance$", fontsize=14)
        
        ax[1].scatter(x[cond == True], y[cond == True], c='r') 
        y2 = y+1
        ax[0].scatter(x[cond == True], y2[cond == True], c='r')
        mpl.rcParams['agg.path.chunksize'] = mpl_size
        plt.tight_layout()
        plt.show()
        plt.savefig(fname)
        plt.close()
        del fig

    @staticmethod
    def regline(
                x, 
                y, 
                bootstap=True, 
                fname="regline.png"):
        """
        x: 
        y: 
        bootstap: boolean, to bootrstrap or not
        fname: filename for saving the figure
        """
        pV = plotVisualsrc
        mpl.rcParams['agg.path.chunksize'] = mpl_size
        p, _ = np.polyfit(x, y, 1, cov=True)
        y_model = np.polyval(p, x)
        # statistics
        n = y.size
        m = p.size
        dof = n - m
        t = stats.t.ppf(0.975, n - m)
        # estimates of error
        resid = y - y_model                           
        #chi2 = np.sum((resid / y_model)**2) 
        #chi2_red = chi2 / dof
        s_err = np.sqrt(np.sum(resid**2) / dof)    
        # plot
        fig, ax = plt.subplots(figsize=(8, 7.5),dpi=300)
        ax.plot(x, y, ".", color="#b9cfe7", markersize=8,markeredgewidth=1, markeredgecolor="r", markerfacecolor="None")
        ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="$\\beta_1 = {}$".format(round(p[0], 2)))
        x2 = np.linspace(np.min(x), np.max(x), 100)
        y2 = np.polyval(p, x2)
        # confidence interval option
        if bootstap:
            pV.plot_ci_bootstrap(x, y, resid, ax=ax)
        else:
            pV.plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
        # prediction interval
        pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))   
        ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
        ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
        ax.plot(x2, y2 + pi, "--", color="0.5")
        # borders
        ax.spines["top"].set_color("0.5")
        ax.spines["bottom"].set_color("0.5")
        ax.spines["left"].set_color("0.5")
        ax.spines["right"].set_color("0.5")
        ax.get_xaxis().set_tick_params(direction="out")
        ax.get_yaxis().set_tick_params(direction="out")
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        # labels
        plt.title("Classification of Uncertainty State", fontsize="14", fontweight="bold")
        plt.xlabel("$\\mathbb{N}ovelty_z$", fontsize="14", fontweight="bold")
        plt.ylabel("$\\mathbb{R}esonance_z$", fontsize="14", fontweight="bold")
        plt.xlim(np.min(x) - .25, np.max(x) + .25)
        # custom legend
        handles, labels = ax.get_legend_handles_labels()
        display = (0, 1)
        anyArtist = plt.Line2D((0, 1), (0, 0), color="#ea5752")
        legend = plt.legend(
            [handle for i, handle in enumerate(handles) if i in display] + [anyArtist],
            [label for i, label in enumerate(labels) if i in display] + ["95% Confidence Limits"],
            loc=9, bbox_to_anchor=(0, -0.21, 1., 0.102), ncol=3, mode="expand"
        )  
        frame = legend.get_frame().set_edgecolor("0.5")
        mpl.rcParams['axes.linewidth'] = 1
        # save figure
        plt.tight_layout()
        plt.savefig(fname, bbox_extra_artists=(legend,), bbox_inches="tight")
        plt.close()
        del fig, frame
        
    @staticmethod
    def regline_without_figure(
                            x, 
                            y):
        """
        x: 
        y: 
        """
        p, _ = np.polyfit(x, y, 1, cov=True)
        beta1 = round(p[0], 2)
        return beta1

    @staticmethod
    def extract_adjusted_main_parameters(
                                         df,
                                         window):
        """
        df: pandas DataFrame with ["date", "novelty", "resonance"]
        window: int size of sliding window
        x: 
        y: 
        """
        pV = plotVisualsrc
        time = df['date'].tolist()
        novelty = df['novelty'].tolist()
        resonance = df['resonance'].tolist()
        # remove window start-end      #Ida: had to move window removal above plotting to avoid error messages
        time = time[window:-window]
        novelty = novelty[window:-window]
        resonance = resonance[window:-window]
        # Handle and remove NaNs
        if np.argwhere(np.isnan(novelty)) == np.argwhere(np.isnan(resonance)):
            pop_id = np.argwhere(np.isnan(novelty))[0][0]
            novelty.pop(pop_id)
            resonance.pop(pop_id)
        # classification based on z-scores
        xz = stats.zscore(novelty)
        yz = stats.zscore(resonance)
        beta1 = pV.regline_without_figure(xz,yz)
        return time, novelty, resonance, beta1, xz, yz

    @staticmethod
    def test_windows_extract_adjusted_main_parameters(
                                         df,
                                         windows:list):
        """
        df: pandas DataFrame with ["date", "novelty", "resonance"]
        windows: list of int size of sliding windows
        """
        pV = plotVisualsrc
        time = df['date'].tolist()

        out = {}
        for window in windows:
            window = str(window)
            out["window"] = window
            novelty = df[f"novelty{window}"].tolist()
            resonance = df[f"resonance{window}"].tolist()
            # remove window start-end      #Ida: had to move window removal above plotting to avoid error messages
            window = int(window)

            time = time[window:-window]
            novelty = novelty[window:-window]
            resonance = resonance[window:-window]
            # Handle and remove NaNs
            if np.argwhere(np.isnan(novelty)) == np.argwhere(np.isnan(resonance)):
                pop_id = np.argwhere(np.isnan(novelty))[0][0]
                novelty.pop(pop_id)
                resonance.pop(pop_id)
            # classification based on z-scores
            xz = stats.zscore(novelty)
            yz = stats.zscore(resonance)
            beta1 = pV.regline_without_figure(xz,yz)

            out["window"][window]["time"] = time
            out["window"][window]["novelty"] = novelty
            out["window"][window]["resonance"] = resonance
            out["window"][window]["beta1"] = beta1
            out["window"][window]["xz"] = xz
            out["window"][window]["yz"] = yz

        return out

    @staticmethod
    def plot_initial_figures(
                            novelty: list,
                            resonance: list,
                            xz,
                            yz,
                            OUT_PATH: str,
                            IN_DATA: str):
        """
        novelty: list of novelty values
        resonance: list of resonance values
        xz: zscore
        yz: zscore
        OUT_PATH: path for where the output is saved to
        IN_DATA: specifying the name of the output dependent on dataset name
        """
        pV = plotVisualsrc
        # Trend detection
        if not os.path.exists(os.path.join(OUT_PATH, "fig")):
            os.mkdir(os.path.join(OUT_PATH, "fig"))
        figname0 = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_adaptline.png")
        ic("[PLOT] Adaptiveline")
        pV.adaptiveline(novelty, resonance, fname=figname0)
        figname1 = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_regline.png")
        ic("[PLOT] Regline")
        pV.regline(xz, yz, fname=figname1)
        return 0

    @staticmethod
    def plot_initial_figures_facebook(
                            novelty: list,
                            resonance: list,
                            xz,
                            yz,
                            OUT_PATH: str,
                            group_id: str,
                            datatype: str,
                            window: str):
        """
        novelty: list of novelty values
        resonance: list of resonance values
        xz: zscore
        yz: zscore
        OUT_PATH: path for where the output is saved to
        filename: specifying the name of the output
        """
        pV = plotVisualsrc
        # Trend detection
        if not os.path.exists(os.path.join(OUT_PATH, "fig")):
            os.mkdir(os.path.join(OUT_PATH, "fig"))
        figname0 = os.path.join(OUT_PATH, "fig", group_id + "_" + datatype + "_" + window + "_adaptline.png")
        ic("[PLOT] Adaptiveline")
        pV.adaptiveline(novelty, resonance, fname=figname0)
        figname1 = os.path.join(OUT_PATH, "fig", group_id + "_" + datatype + "_" + window + "_regline.png")
        ic("[PLOT] Regline")
        pV.regline(xz, yz, fname=figname1)
        return 0
