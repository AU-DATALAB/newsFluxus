# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 2021

@author: au558899

Source codes for beta timeseries-related codes for main extractor of newsFluxus

"""

import os
import math
from itertools import islice
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from icecream import ic
import pandas as pd
import sys
sys.path.insert(1, r'/home/commando/marislab/newsFluxus/src/')
from visualsrc.visualsrc import plotVisualsrc

pV = plotVisualsrc

def sliding_window(seq, n=21):
    """
    seq: 
    n: int
        
    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   
        
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
    
def beta_time_series(
                        time: list, 
                        novelty: list, 
                        resonance: list, 
                        window: int, 
                        OUT_PATH: str, 
                        IN_DATA: str):
    """
    time: list of dates
    novelty: list of novelty values
    resonance: list of resonance values
    window: int size of the window
    OUT_PATH: path for where the output is saved to
    IN_DATA: specifying the name of the output dependent on dataset name
    """
    if not os.path.exists(os.path.join(OUT_PATH, "fig")):
        os.mkdir(os.path.join(OUT_PATH, "fig"))
    #convert time series into windows
    time_w = list()
    for w in sliding_window(time, window):
        time_w.append(w)  
    novelty_w = list()
    for w in sliding_window(novelty, window):
        novelty_w.append(w)
    resonance_w = list()
    for w in sliding_window(resonance, window):
        resonance_w.append(w)    
        
    #loop over window
    beta_w = list()
    for i in range(len(time_w)):
        # classification based on z-scores
        xz = stats.zscore(novelty_w[i])
        yz = stats.zscore(resonance_w[i])
        #get beta without generating a figure for each window
        beta = pV.regline_without_figure(xz, yz)
        beta_w.append(beta)
        
    #choose middle time point for plot
    #later: maybe average instead, as time points are not spaced evenly
    time_middle = list()
    middle = round((len(time_w[0]) - 1)/2)
    for i in range(len(time_w)):
        time_middle.append(time_w[i][middle])
    time_middle_days = list()
    for i in range(len(time_middle)):
        time_middle_days.append(time_middle[i][0:10])
        
    #save beta timeseries
    ic("[INFO] Saving beta timeseries")
    output = pd.DataFrame({'beta': beta_w, 'time middle': time_middle, 'time': time_w})
    output.to_csv(os.path.join(OUT_PATH, "{}_beta_timeseries.csv".format(IN_DATA.split(".")[0])), index=False, encoding='utf-8-sig', sep=';')  
        
    return beta_w, time_w, time_middle, time_middle_days
    
    
def plot_beta_time_series(
                            time_middle, 
                            beta_w,
                            time_middle_days,
                            OUT_PATH,
                            IN_DATA):
    #plot beta over time    
    #(execute as block)
    plt.scatter(time_middle, beta_w)
    plt.xticks([])  
    plt.ylabel('beta')
    plt.xlabel('time')
    plt.box(False)
    xlabels = list()
    for i in range(0, len(time_middle_days)-1, math.floor((len(time_middle_days)-1)/4)):
        xlabels.append(time_middle_days[i])
    plt.xticks(range(0, len(time_middle_days)-1, math.floor((len(time_middle_days)-1)/4)), xlabels)  #, rotation = 45
    #save figure
    fname = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_beta_timeseries.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    
def timepoints_beta_top(
                            beta_w, 
                            time_w, 
                            percentage):
    """
    beta_w: 
    time_w: 
    percentage: 
    """
    #Find treshold    
    beta_index = [0]*len(beta_w)
    beta_ranked = [0]*len(beta_w)
    for i, x in enumerate(sorted(range(len(beta_w)), key=lambda y: beta_w[y], reverse=True)):       #sort descending
        beta_ranked[i] = beta_w[x]
        beta_index[i] = x
        
    threshold = beta_ranked[round(percentage*len(beta_w))]
    #treshold_idx = beta_index[round(percentage*len(beta_w))]
    list_top_idx = beta_index[0:round(percentage*len(beta_w))]
        
    #find time points according to top beta values
    time_top = list(time_w[i] for i in list_top_idx)
    #put all windowed time points back into one long list
    time_top_unpacked = [item for sublist in time_top for item in sublist]
    #remove all duplicates
    time_top_unpacked = list(set(time_top_unpacked))
    time_top_unpacked = sorted(time_top_unpacked)
    #NB: 557/847 time points end up in top 20 list due to windowing (66%)! For top 10% it is 384/847 (45%)
    return time_top_unpacked, threshold, list_top_idx

def line_top_time(
                    size_df: int, 
                    idx_top: list, 
                    WINDOW: int):
    """
    size_df: size of the pandas DataFrame
    idx_top: ids of the top posts
    WINDOW: int, size of the window
    """
    #preparation to plot selected time points
    idx_bin = [0]*size_df  #convert top time points into binary array
    for i in range(size_df):
        for j in idx_top:
            if i == j:
                idx_bin[i] = 1
    idx_bin = idx_bin[WINDOW:-WINDOW]           #shorten by removing first and last window to fit length of novelty and resonace arrays
        
    cond = np.array(idx_bin) == 1
    x = np.array(range(len(idx_bin)))
    y = np.array([-1]*len(idx_bin))
    #plt.scatter(x[cond == True], y[cond == True], c='r')
    return x, y, cond

def plot_beta_top_time(
                        time_middle, 
                        beta_w, 
                        time_middle_days, 
                        time_top, 
                        threshold, 
                        list_top_idx, 
                        OUT_PATH: str, 
                        IN_DATA: str):
    """
    time_middle: 
    beta_w: 
    time_middle_days:
    time_top:
    threshold: 
    list_top_idx: 
    OUT_PATH:
    IN_DATA:
    """
    #plot beta over time    
    #(execute as block)
    plt.scatter(time_middle, beta_w)
    plt.xticks([])  
    plt.ylabel('beta')
    plt.xlabel('time')
    plt.box(False)
    xlabels = list()
    for i in range(0, len(time_middle_days)-1, math.floor((len(time_middle_days)-1)/4)):
        xlabels.append(time_middle_days[i])
    plt.xticks(range(0, len(time_middle_days)-1, math.floor((len(time_middle_days)-1)/4)), xlabels)  #, rotation = 45
    #add threshold line
    plt.axhline(y=threshold, color='k')
        
    top_beta = [0]*len(beta_w)  #convert top time points into binary array
    for i in range(len(list_top_idx)):
        top_beta[list_top_idx[i]] = threshold
        
    cond2 = np.array(top_beta) == threshold
    plt.scatter(np.array(time_middle)[cond2 == True], np.array(top_beta)[cond2 == True], c='r')
        
    #save figure
    fname = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_beta_timeseries_top.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def main_beta_plotting_with_top_tokens(
                                       window: int, 
                                       percentage: float, 
                                       size_df: int, 
                                       df, 
                                       tokens: list, 
                                       time: list, 
                                       novelty: list, 
                                       resonance: list, 
                                       OUT_PATH: str, 
                                       IN_DATA: str,
                                       WINDOW: int):
    """
    window: window size
    percentage: percentage for calculating the top posts based on beta values
    size_df: size of the df
    df: pandas DataFrame
    tokens: list of tokens
    time: list of dates
    novelty: list of novelty scores
    resonance: list of resonance scores
    OUT_PATH: path for where the output is saved to
    IN_DATA: specifying the name of the output dependent on dataset name
    """
    #Analyse posts with top beta values
    ic("[INFO] Calculate beta timeseries")
    beta_w, time_w, time_middle, time_middle_days = beta_time_series(time, novelty, resonance, window, OUT_PATH, IN_DATA) # takes time
    #ic("[PLOT] Beta time series")
    #plot_beta_time_series(time_middle, beta_w, time_middle_days, OUT_PATH, IN_DATA)
    del time
    #find time points according to top beta values
    ic("[INFO] Find top timepoints with beta") # This takes a hot minute
    time_top, threshold, list_top_idx = timepoints_beta_top(beta_w, time_w, percentage)
    #ic("[PLOT] Beta timeseries toptimes")
    #plot_beta_top_time(time_middle, beta_w, time_middle_days, time_top, threshold, list_top_idx, OUT_PATH, IN_DATA)
    del time_w, time_middle, beta_w, time_middle_days, threshold, list_top_idx
        
    #find indices of those time points using df     NB: idx_top follows length of df
    idx_top = list(df['date'].index[df['date'] == time_top[i]].tolist() for i in range(len(time_top)))
    del df, time_top
    idx_top = list(idx_top[i][0] for i in range(len(idx_top)))
        
    #select the tokens
    tokens_top = list(tokens[i] for i in idx_top)
    #save top tokens
    ic("[INFO] Save top tokens")
    with open(os.path.join(OUT_PATH, "mdl", "{}_toptokens.txt".format(IN_DATA.split(".")[0])), "w") as f:
        for element in tokens_top:
            f.write("{}\n".format(element))
    del tokens_top
        
    #resonance novelty timeseries plot
    figname = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_adaptline_top.png")
    x, y, cond = line_top_time(size_df, idx_top, WINDOW)
    del idx_top
    #ic("[PLOT] Adaptiveline toptimes")
    #pV.adaptiveline_toptimes(novelty, resonance, x, y, cond, figname)
    del novelty, resonance#, x, y, cond

