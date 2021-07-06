# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:29:39 2021

@author: au685355
"""

"""
1. Lemmatize texts
2. Train mallet model
3. Get topic model distribution per document
4. Extract novelty/resonance

#To run script in shell console:
python3 src/main_extractor.py
"""

import os
import pickle
import glob
import nolds

import pandas as pd
import spacy
import re
from itertools import islice
import math
from icecream import ic
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, r'/home/commando/marislab/newsFluxus/src/')
from tekisuto.preprocessing import CaseFolder
from tekisuto.preprocessing import RegxFilter
from tekisuto.preprocessing import StopWordFilter
from tekisuto.preprocessing import Tokenizer
from tekisuto.models import TopicModel
from tekisuto.models import InfoDynamics
from tekisuto.metrics import jsd
from tekisuto.models import LatentSemantics
from import_ndjson_files_incl_comments import import_ndjson_files
import saffine.detrending_method as dm

#########################################################################################################
### PREPARE TEXTS FOR TOPIC MODELING
#########################################################################################################

def spacy_lemmatize(texts, nlp, **kwargs):
    
    docs = nlp.pipe(texts, **kwargs)
    
    def __lemmatize(doc):
        lemmas = []
        for sent in doc.sents:
            for token in sent:
                lemmas.append(token.lemma_)
        return lemmas

    return [__lemmatize(doc) for doc in docs]

def make_lemmas(df):
    
    url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    remove_more =  [r"\d+", r"\W+", r"[^A-z]", r"_", r"\s+"]  #first remove urls, then numbers and other characters, then non-english letters (e.g. chinese), then underscores (from user names for example), and last excess spaces    
    pre_lemma = [url_pattern.sub(' ',x) for x in df["text"].tolist()]
    for i in range(len(remove_more)):
        remove_more_pattern = re.compile(remove_more[i])
        pre_lemma = [remove_more_pattern.sub(' ',x) for x in pre_lemma]

    lemmas = spacy_lemmatize(pre_lemma, nlp=nlp)
    lemmas = [' '.join(doc) for doc in lemmas]
    #Ida: remove -PRON- from text (white spaces are removed in the next step preprocess_for_topic_models)
    lemmas = [re.sub('-PRON-', '', lemmas[x]) for x in range(len(lemmas))]
    
    return lemmas

def preprocess_for_topic_models(lemmas: list, lang="da"):
    
    cf = CaseFolder(lower=True)
    re0 = RegxFilter(pattern=r"\W+")
    re1 = RegxFilter(pattern=r"\d+")
    sw = StopWordFilter(path=os.path.join(ROOT_PATH, "res", f"stopwords-{lang}.txt"))
    processors = [cf, re0, re1, sw]
    for processor in processors:
        lemmas = [processor.preprocess(t) for t in lemmas]
        
    return lemmas

#########################################################################################################
### TOPIC MODELING
#########################################################################################################

def train_topic_model_mallet(tokens, 
                      estimate_topics: bool,
                      tune_topic_range=[10,30,50],
                      plot_topics=False,
                      **kwargs):
    """
    tokens: list of strings (document)
    estimate topics: whether to search a range of topics
    tune_topic_range: number of topics to fit
    plot_topics: quality check, plot coherence by topics
    **kwargs: other arguments to LDAmulticore
    """
    
    if estimate_topics:
        ls = LatentSemantics(tokens)
        n, n_cohers = ls.coherence_k(
            krange=tune_topic_range)
        print(f"\n[INFO] Optimal number of topics is {n}")
        ls = LatentSemantics(tokens, k=n)
        ls.fit()
    else:
        ls = LatentSemantics(tokens, k=10)# change to your preferred default value
        n = 10
        ls.fit()
        
    return ls, n

#########################################################################################################
### NOVELTY RESONANCE
#########################################################################################################

def extract_novelty_resonance(df, theta, dates, window):
    
    idmdl = InfoDynamics(data = theta, time = dates, window = window)
    idmdl.novelty(meas = jsd)
    idmdl.transience(meas = jsd)
    idmdl.resonance(meas = jsd)

    df["novelty"] = idmdl.nsignal
    df["transience"] = idmdl.tsignal
    df["resonance"] = idmdl.rsignal
    df["nsigma"] = idmdl.nsigma
    df["tsigma"] = idmdl.tsigma
    df["rsigma"] = idmdl.rsigma
    
    return df

#########################################################################################################
### PREPARE FOR PLOTTING - BASE
#########################################################################################################

def normalize(x, lower=-1, upper=1):
    """ transform x to x_ab in range [a, b]
    """
    
    x_norm = (upper - lower)*((x - np.min(x)) / (np.max(x) - np.min(x))) + lower
    return x_norm


def adaptive_filter(y, span=56):
    
    w = int(4 * np.floor(len(y)/span) + 1)
    y_dt = np.mat([float(j) for j in y])
    _, y_smooth = dm.detrending_method(y_dt, w, 1)
    
    return y_smooth.T

#########################################################################################################
### PLOTTING FUNCTIONS
#########################################################################################################

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.
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

def adaptiveline(x1, x2, fname="adaptline.png"):
    
    _, ax = plt.subplots(2,1,figsize=(14,6),dpi=300)
    c = ["g", "r", "b"]
    ax[0].plot(normalize(x1, lower=0),c="gray")
    for i, span in enumerate([128, 56, 32]):
        n_smooth = normalize(adaptive_filter(x1, span=span), lower=0)
        ax[0].plot(n_smooth,c=c[i])
    ax[0].set_ylabel("$\\mathbb{N}ovelty$", fontsize=14)
    
    ax[1].plot(normalize(x2, lower=-1),c="gray")
    for i, span in enumerate([128, 56, 32]):
        r_smooth = normalize(adaptive_filter(x2, span=span), lower=-1)
        ax[1].plot(r_smooth,c=c[i])
    ax[1].set_ylabel("$\\mathbb{R}esonance$", fontsize=14)
    plt.tight_layout()
    plt.savefig(fname)
    #plt.close()
    

def adaptiveline_toptimes(x1, x2, x, y, cond, fname="adaptline_top.png"):
    
    #_, ax = plt.subplots(2,1,figsize=(14,6),dpi=300)
    fig, ax = plt.subplots(2,1,figsize=(14,6),dpi=300)
    c = ["g", "r", "b"]
    ax[0].plot(normalize(x1, lower=0),c="gray")
    for i, span in enumerate([128, 56, 32]):
        n_smooth = normalize(adaptive_filter(x1, span=span), lower=0)
        ax[0].plot(n_smooth,c=c[i])
    ax[0].set_ylabel("$\\mathbb{N}ovelty$", fontsize=14)
    
    ax[1].plot(normalize(x2, lower=-1),c="gray")
    for i, span in enumerate([128, 56, 32]):
        r_smooth = normalize(adaptive_filter(x2, span=span), lower=-1)
        ax[1].plot(r_smooth,c=c[i])
    ax[1].set_ylabel("$\\mathbb{R}esonance$", fontsize=14)
    
    ax[1].scatter(x[cond == True], y[cond == True], c='r') 
    y2 = y+1
    ax[0].scatter(x[cond == True], y2[cond == True], c='r')
    
    plt.tight_layout()
    plt.savefig(fname)
    #plt.close()


def regline(x, y, bootstap=True, fname="regline.png"):
    p, _ = np.polyfit(x, y, 1, cov=True)
    y_model = np.polyval(p, x)
    # statistics
    n = y.size
    m = p.size
    dof = n - m
    t = stats.t.ppf(0.975, n - m)
    # estimates of error
    resid = y - y_model                           
    chi2 = np.sum((resid / y_model)**2) 
    chi2_red = chi2 / dof
    s_err = np.sqrt(np.sum(resid**2) / dof)    
    # plot
    fig, ax = plt.subplots(figsize=(8, 7.5),dpi=300)
    ax.plot(x, y, ".", color="#b9cfe7", markersize=8,markeredgewidth=1, markeredgecolor="r", markerfacecolor="None")
    ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="$\\beta_1 = {}$".format(round(p[0], 2)))
    x2 = np.linspace(np.min(x), np.max(x), 100)
    y2 = np.polyval(p, x2)
    # confidence interval option
    if bootstap:
        plot_ci_bootstrap(x, y, resid, ax=ax)
    else:
        plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
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
    #Ida added
    beta1 = round(p[0], 2)
    return beta1


def regline_without_figure(x, y):
    p, _ = np.polyfit(x, y, 1, cov=True)
    #Ida added
    beta1 = round(p[0], 2)
    return beta1


def plot_figures(df, OUT_PATH, IN_DATA, window):
    time = df['date'].tolist()
    novelty = df['novelty'].tolist()
    resonance = df['resonance'].tolist()
    
    # remove window start-end      #Ida: had to move window removal above plotting to avoid error messages
    time = time[window:-window]
    novelty = novelty[window:-window]
    resonance = resonance[window:-window]
    # trend detection
    if not os.path.exists(os.path.join(OUT_PATH, "fig")):
        os.mkdir(os.path.join(OUT_PATH, "fig"))
    figname0 = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_adaptline.png")
    #with open(figname0, "wb") as f:
    #    adaptiveline(novelty, resonance, fname=figname0)
    #or (both do not show figure in plot window, but save it)
    adaptiveline(novelty, resonance, fname=figname0)
    # classification based on z-scores
    xz = stats.zscore(novelty)
    yz = stats.zscore(resonance)
    figname1 = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_regline.png")
    regline(xz, yz, fname=figname1)
    beta1 = regline(xz, yz, fname=figname1)
    return time, novelty, resonance, beta1

#########################################################################################################
### HURST EXPONENT
#########################################################################################################

def hurst_exp(resonance, OUT_PATH):
    
    #hurst_n = nolds.hurst_rs(novelty, nvals=None, fit='poly', debug_plot=True, plot_file=None, corrected=True, unbiased=True)
    #show figure
    nolds.hurst_rs(resonance, nvals=None, fit='poly', debug_plot=True, plot_file=None, corrected=True, unbiased=True)
    #save figure
    fignameH = os.path.join(OUT_PATH, "fig", "H_plot.png")
    hurst_r = nolds.hurst_rs(resonance, nvals=None, fit='poly', debug_plot=True, plot_file=fignameH, corrected=True, unbiased=True)
    
    return hurst_r

#########################################################################################################
### BETA TIME SERIES
#########################################################################################################

def sliding_window(seq, n=21):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def beta_time_series(time, novelty, resonance, window, OUT_PATH, IN_DATA):
    
    if not os.path.exists(os.path.join(OUT_PATH, "fig")):
        os.mkdir(os.path.join(OUT_PATH, "fig"))
    
    #convert time series into windows
    time_w = list()
    for w in sliding_window(time, window):
        time_w.append(w)
        #print(w)    
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
        beta = regline_without_figure(xz, yz)
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
    plt.show()
    
    
    
    #save beta timeseries
    output = pd.DataFrame({'beta': beta_w, 'time middle': time_middle, 'time': time_w})
    output.to_csv(os.path.join(OUT_PATH, "{}_beta_timeseries.csv".format(IN_DATA.split(".")[0])), index=False, encoding='utf-8-sig', sep=';')  
    
    return beta_w, time_w, time_middle, time_middle_days
    

def timepoints_beta_top(beta_w, time_w, percentage):
    
    #Find treshold    
    beta_index = [0]*len(beta_w)
    beta_ranked = [0]*len(beta_w)
    for i, x in enumerate(sorted(range(len(beta_w)), key=lambda y: beta_w[y], reverse=True)):       #sort descending
        beta_ranked[i] = beta_w[x]
        beta_index[i] = x
    
    threshold = beta_ranked[round(percentage*len(beta_w))]
    treshold_idx = beta_index[round(percentage*len(beta_w))]
    
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


def line_top_time(size_df, idx_top, WINDOW):
    
    #preparation to plot selected time points
    idx_bin = [0]*size_df  #convert top time points into binary array
    #for i in range(size_df):
        #idx_bin[idx_top[i]] = 1
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


def plot_beta_top_time(time_middle, beta_w, time_middle_days, time_top, threshold, list_top_idx, OUT_PATH, IN_DATA):
    
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
    plt.show()
    
#########################################################################################################
### MAIN MODULE
#########################################################################################################

def main_module(SUBREDDIT_NAME, ROOT_PATH, REDDIT_DATA):
    
    ### MAIN VARIABLES ###
    IN_DATA= SUBREDDIT_NAME + "_incl_comments.csv"
    OUT_PATH = os.path.join(ROOT_PATH, "dat/subreddits_incl_comments/output")
    OUT_FILE = os.path.join(OUT_PATH, IN_DATA.split(".")[0] + "_theta.csv")
    ESTIMATE_TOPIPCS = True # whether to tune multiple topic model sizes
    TOPIC_TUNE = [20, 30, 50, 80] # number of topics to tune over in topic model   #Ida: do not set to only one, then set estimate_topipcs = False.
    PLOT_TOPICS = True # plot a topic of coherence by number of topics
    SAVE_SEMANTIC_TOPICS = True
    WINDOW = 3 # window for novelty/resonance
    LANG = "en"

    ### CHECK PATH ###
    if not os.path.exists(OUT_PATH):
        ic(os.path.exists(OUT_PATH))
        ic(OUT_PATH)
        os.makedirs(OUT_PATH)
    if not os.path.exists(os.path.join(OUT_PATH, "mdl")):
        os.mkdir(os.path.join(OUT_PATH, "mdl"))
        
    ### GET DATA ###
    df = import_ndjson_files(SUBREDDIT_NAME, REDDIT_DATA)
    print(df.head())
    
    ### CHECK SUFFICIENCY OF DATAPOINTS IN DATA ###
    length_df = len(df)
    print('length of datapoints in subreddit: ', length_df)
    if length_df < 120:
        print('not sufficient datapoints')
        with open(os.path.join(OUT_PATH, "{}_not_executed.txt".format(IN_DATA.split(".")[0])), "w") as f:
            f.write("not sufficient datapoints (no datapoints = {})".format(length_df))
        
        #do not execute subreddit, return zeros for indicator variables
        beta1 = 0
        hurst_r = 0
        return df, OUT_PATH, IN_DATA, beta1, hurst_r
      
    # sorting date in descending order for correct calculation
    # of novelty and resonance
    df = df.sort_values("date")
    
    ### LEMMATIZING AND TOKENIZING ###
    print("\n[INFO] lemmatizing...\n")
    lemmas = make_lemmas(df)
    # preprocess
    lemmas = preprocess_for_topic_models(lemmas, lang=LANG)
    # model training
    print("\n[INFO] training model...\n")
    to = Tokenizer()
    tokens = to.doctokenizer(lemmas)
    #take out empty lists
    invalid_entries = [index for index in range(len(tokens)) if tokens[index] == []]
    print(f'Invalid entries removed at {invalid_entries}: {df.iloc[invalid_entries,0]}')
    tokens = [x for x in tokens if x]
    #also remove line from df
    df_orig = df
    df = df.drop(labels=invalid_entries)
    df = df.reset_index(drop=True)
    #raise SystemExit
    
    ### TRAIN TOPIC MODEL ###
    tm, n = train_topic_model_mallet(tokens,
                            ESTIMATE_TOPIPCS,
                            TOPIC_TUNE,
                            PLOT_TOPICS)
    ic(tm, n)
    
    
    if SAVE_SEMANTIC_TOPICS:
        # From bow_mdl.py
        # static semantic content for model summary
        print("\n[INFO] writing content to file...\n")
        with open(os.path.join(OUT_PATH, "mdl", "{}_content.txt".format(IN_DATA.split(".")[0])), "w") as f:
            for topic in tm.model.show_topics(num_topics=-1, num_words=10):
                f.write("{}\n\n".format(topic))
    
    # Get topic representation for each document
    print("\n[INFO] Getting topic distribution per document...")
    print("subreddit = ", SUBREDDIT_NAME)
    #theta = tm.get_topic_distribution()
    #takes a lot of time
    theta = list()
    for i, doc in enumerate(tm.corpus):
        vector = [x[1] for x in tm.model[doc]]
        theta.append(vector)
        #print("[INFO] processed {}/{}".format(i + 1, len(lemmas)))
        print("[INFO] processed {}/{}".format(i + 1, len(tokens)))  #Ida: needs to be length of tokens as empty entries were removed
    
    print("[INFO] exporting model...")
    out = dict()
    out["model"] = tm.model
    out["id2word"] = tm.id2word
    out["corpus"] = tm.corpus
    #out["tokenlists"] = tm.tokenlists
    out["tokens"] = tokens   #add?
    out["theta"] = theta
    out["dates"] = df['date'].tolist()
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(IN_DATA.split(".")[0])), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    df['theta'] = theta
    ### Extract novelty and resonance
    dates = df["date"].tolist()
    # instantiate and call
    print("[INFO] extracting novelty and resonance...")
    df = extract_novelty_resonance(df, theta, dates, WINDOW)
    
    df.to_csv(OUT_FILE, index=False)    
    
    #%% Print figures
    time, novelty, resonance, beta1 = plot_figures(df, OUT_PATH, IN_DATA, WINDOW)
    
    #%% Hurst exponent
    hurst_r = hurst_exp(resonance, OUT_PATH)
    
    #%% Save further output
    output = pd.DataFrame({'hurst': [hurst_r], 'beta1': [beta1]})
    output.to_csv(os.path.join(OUT_PATH, "{}_hurst_beta.csv".format(IN_DATA.split(".")[0])), index=False, encoding='utf-8-sig', sep=';')
    
    
    #%% Beta over time
    #generate beta time series with sliding window
    window = 21
    beta_w, time_w, time_middle, time_middle_days = beta_time_series(time, novelty, resonance, window, OUT_PATH, IN_DATA)

    #Analyse posts with top beta values
    percentage = 0.1
    #find time points according to top beta values
    time_top, threshold, list_top_idx = timepoints_beta_top(beta_w, time_w, percentage)
    #find indices of those time points using df     NB: idx_top follows length of df
    idx_top = list(df['date'].index[df['date'] == time_top[i]].tolist() for i in range(len(time_top)))
    idx_top = list(idx_top[i][0] for i in range(len(idx_top)))
    
    #select the tokens
    tokens_top = list(tokens[i] for i in idx_top)
    #save top tokens
    with open(os.path.join(OUT_PATH, "mdl", "{}_toptokens.txt".format(IN_DATA.split(".")[0])), "w") as f:
        for element in tokens_top:
            f.write("{}\n".format(element))
    
    #%%plot top time points onto other figures
    #resonance novelty timeseries plot
    figname = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_adaptline_top.png")
    size_df = len(df)
    x, y, cond = line_top_time(size_df, idx_top, WINDOW)
    adaptiveline_toptimes(novelty, resonance, x, y, cond, figname)
    
    #beta timeseries plot
    plot_beta_top_time(time_middle, beta_w, time_middle_days, time_top, threshold, list_top_idx, OUT_PATH, IN_DATA)
    
    return df, OUT_PATH, IN_DATA, beta1, hurst_r

#########################################################################################################
### MAIN
#########################################################################################################

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_lg")
    
    ROOT_PATH = r"/home/commando/marislab/newsFluxus/"
    REDDIT_DATA = r'/data/datalab/reddit-sample-hv/comments/*.ndjson'
    WINDOW=3 # window for novelty/resonance
    
    slope_all = []
    hurst_all = []
    #ex: file = 'U:\\Python\\Newsfluxus\\newsFluxus-master_Lasse\\newsFluxus-master\\dat\\subreddits\\subreddit_ACTA.csv'
    for file in glob.glob(REDDIT_DATA):
        print("Start the loop")
        ic(file)
        fname = file.split("/")
        SUBREDDIT_NAME = fname[-1]
        SUBREDDIT_NAME = SUBREDDIT_NAME.split(".")
        SUBREDDIT_NAME = SUBREDDIT_NAME[0]
        
        ic(SUBREDDIT_NAME)
        
        #check for previous runs if subreddit run through
        if os.path.isfile(os.path.join(ROOT_PATH, "dat", "subreddits_incl_comments", "output", (SUBREDDIT_NAME + '_incl_comments_not_executed.txt'))):
            print("True")
            continue
        if os.path.isfile(os.path.join(ROOT_PATH, "dat", "subreddits_incl_comments", "output", (SUBREDDIT_NAME + '__incl_comments_beta_timeseries.csv'))): # Change this since '_finished.txt' does not exist
            print("True")
            continue
        
        #wait with the large subreddits
        if not(SUBREDDIT_NAME == 'Bitcoin' or SUBREDDIT_NAME == 'technology' or SUBREDDIT_NAME  == 'conspiracy' or SUBREDDIT_NAME == 'ComputerSecurity' or SUBREDDIT_NAME == 'netsec' or SUBREDDIT_NAME == 'privacy' or SUBREDDIT_NAME == 'privacytools' or SUBREDDIT_NAME == 'privacytoolsIO' or SUBREDDIT_NAME == 'Stellar'):
            
            print("NOT large subreddit")
            #novelty, resonance and indicator variables (beta and hurst)
            df, OUT_PATH, IN_DATA, beta1, hurst_r = main_module(SUBREDDIT_NAME, ROOT_PATH, REDDIT_DATA)
    
            #store indicator variables
            slope_all.append(beta1)
            hurst_all.append(hurst_r)
    
    ic("Outside of the for loop")
    dict = {'hurst': hurst_all, 'slope': slope_all}
    df_ind_var = pd.DataFrame(dict)
    df_ind_var.to_csv(os.path.join(OUT_PATH, 'all_subreddits_indicator_variables.csv'), index=False, encoding='utf-8-sig', sep=';')
    
    with open(os.path.join(OUT_PATH, "{}_finished.txt".format(SUBREDDIT_NAME)), "w") as f:
        f.write("finished successfully")
        