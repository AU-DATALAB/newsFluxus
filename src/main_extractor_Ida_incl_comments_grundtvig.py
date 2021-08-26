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
import scipy.stats as stats
import matplotlib.pyplot as plt
import traceback

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
from visualsrc.visualsrc import regline_without_figure, adaptiveline_toptimes, extract_adjusted_main_parameters, plot_initial_figures

mpl_size = 10000

#########################################################################################################
### PREPARE TEXTS FOR TOPIC MODELING
#########################################################################################################

def downsample(df,
               frequency='1T'):
    """
    df: pandas DataFrame with columns "date" and "text"
    frequency: time interval for downsampling, default 1T - creates 1min timebins

    Returns downsampled pandas DataFrame
    """
    df.index = pd.to_datetime(df["date"])
    df = df.drop("date", axis = 1)
    df = df.resample(frequency).agg({"text": ' '.join})
    df = df.reset_index(col_fill = "date")
    df = df[["text", "date"]]
    df["text"] = df["text"].astype(str)
    df["date"] = df["date"].astype(str)
    return df

def spacy_lemmatize(texts: list, 
                    nlp, 
                    **kwargs):
    """
    texts: input texts as list
    nlp: specifies spacy language model
    **kwargs: other arguments to spacy NLP pipe

    Returns lemmas for all documents in a list
    """
    docs = nlp.pipe(texts, **kwargs)
    
    def __lemmatize(doc):
        lemmas = []
        for sent in doc.sents:
            for token in sent:
                lemmas.append(token.lemma_)
        return lemmas

    return [__lemmatize(doc) for doc in docs]

def make_lemmas(df):
    """
    df: pandas DataFrame with column "text"

    Returns lemmas as list
    """
    df["text"] = df["text"].astype(str)

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

def preprocess_for_topic_models(lemmas: list, 
                                lang="da"):
    """
    lemmas: list of lemmas
    lang: string specification of language, default da (danish)

    Returns lemmas as list
    """
    cf = CaseFolder(lower=True)
    re0 = RegxFilter(pattern=r"\W+")
    re1 = RegxFilter(pattern=r"\d+")
    sw = StopWordFilter(path=os.path.join(ROOT_PATH, "res", f"stopwords-{lang}.txt"))
    
    processors = [cf, re0, re1, sw]
    for processor in processors:
        lemmas = [processor.preprocess(t) for t in lemmas]
        
    return lemmas

#########################################################################################################
### TOPIC MODELING - GENSIM
#########################################################################################################

def train_topic_model(tokens, 
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

    Returns topic model tm and number of topics n
    """
    if estimate_topics:
        tm = TopicModel(tokens)
        n, n_coher = tm.tune_topic_range(
            ntopics=tune_topic_range,
            plot_topics=plot_topics)
        del n_coher
        print(f"\n[INFO] Optimal number of topics is {n}")
        tm = TopicModel(tokens)
        tm.fit(n, **kwargs)
    else:
        tm = TopicModel(tokens)
        n = 10
        tm.fit(10, **kwargs)
    return tm, n

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

    Returns topic model ls and number of topics n
    """
    
    if estimate_topics:
        ls = LatentSemantics(tokens)
        n, n_coher = ls.coherence_k(
            krange=tune_topic_range)
        del n_coher
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

def extract_novelty_resonance(df, 
                             theta: list, 
                             dates: list, 
                             window: int):
    """
    df: pandas DataFrame
    theta: list of theta values (list of lists)
    dates: list of dates
    window: int of the window size

    Returns pandas DataFrame with novelty-resonance values
    """
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
### HURST EXPONENT
#########################################################################################################

def hurst_exp(resonance: list, 
              OUT_PATH: str):
    """
    resonance:  list of resonance values
    OUT_PATH: path for where the output is saved to

    Returns hurst exponent hurst_r
    """
    nolds.hurst_rs(resonance, nvals=None, fit='poly', debug_plot=True, plot_file=None, corrected=True, unbiased=True)
    fignameH = os.path.join(OUT_PATH, "fig", "H_plot.png")
    hurst_r = nolds.hurst_rs(resonance, nvals=None, fit='poly', debug_plot=True, plot_file=fignameH, corrected=True, unbiased=True)
    return hurst_r

#########################################################################################################
### BETA TIME SERIES
#########################################################################################################

def sliding_window(seq, 
                   n=21):
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

def beta_time_series(time: list, 
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
    
    #save beta timeseries
    print("[INFO] Saving beta timeseries")
    output = pd.DataFrame({'beta': beta_w, 'time middle': time_middle, 'time': time_w})
    output.to_csv(os.path.join(OUT_PATH, "{}_beta_timeseries.csv".format(IN_DATA.split(".")[0])), index=False, encoding='utf-8-sig', sep=';')  
    
    return beta_w, time_w, time_middle, time_middle_days
    
def plot_beta_time_series(time_middle, 
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

def timepoints_beta_top(beta_w, 
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


def line_top_time(size_df: int, 
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


def plot_beta_top_time(time_middle, 
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

#########################################################################################################
### MAINTENANCE FUNCTIONS
#########################################################################################################

def exception():
    print("Exception in user code:")
    print("-"*60)
    traceback.print_exc(file=sys.stdout)
    print("-"*60)

def check_OUT_PATH(OUT_PATH:str):
    """
    OUT_PATH: str

    Checks whether OUT_PATH exists, creates it if it does not, with folder mdl
    """
    if not os.path.exists(OUT_PATH):
        ic(os.path.exists(OUT_PATH))
        ic(OUT_PATH)
        os.makedirs(OUT_PATH)
    if not os.path.exists(os.path.join(OUT_PATH, "mdl")):
        os.mkdir(os.path.join(OUT_PATH, "mdl"))

def check_sufficiency_of_datapoints(df,
                                    size_df: int,
                                    OUT_PATH: str,
                                    IN_DATA: str):
    """
    size_df: int,
    OUT_PATH: str,
    IN_DATA: str
    """
    print('[INFO] Not sufficient datapoints')
    with open(os.path.join(OUT_PATH, "{}_not_executed.txt".format(IN_DATA.split(".")[0])), "w") as f:
        f.write("not sufficient datapoints (no datapoints = {})".format(size_df))
        
    beta1 = 0
    hurst_r = 0
    return df, OUT_PATH, IN_DATA, beta1, hurst_r

def remove_invalid_entries(tokens: list, 
                           df):
    """
    Removes lines from dataset where tokens are empty/invalid

    tokens: list of tokens
    df: pandas DataFrame
    """
    invalid_entries = [index for index in range(len(tokens)) if tokens[index] == []]
    print(f'Invalid entries removed at {invalid_entries}: {df.iloc[invalid_entries,0]}')
    tokens = [x for x in tokens if x]
    df = df[~df.index.isin(invalid_entries)].reset_index(drop=True)
    return df, tokens

def export_model_and_tokens(tm, 
                            n, 
                            tokens, 
                            theta, 
                            dates, 
                            OUT_PATH, 
                            IN_DATA):
    out = {}
    out["model"] = tm.model
    out["nr_of_topics"] = n
    #out["id2word"] = tm.id2word
    #out["corpus"] = tm.corpus
    out["tokenlists"] = tm.tokenlists
    out["tokens"] = tokens   #add?
    out["theta"] = theta
    out["dates"] = dates
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(IN_DATA.split(".")[0])), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    del out, tm



def load_from_premade_model(OUT_PATH, IN_DATA):
    print("[INFO] Loading theta values...")
    df = pd.read_csv(os.path.join(OUT_PATH, IN_DATA.split(".")[0] + "_theta.csv"))
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df = df.sort_values("date")
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(IN_DATA.split(".")[0])), 'rb') as f:
        out = pickle.load(f)
    tokens = out["tokens"]
    return df, tokens

def load_from_premade_model_generate_thetas(OUT_PATH, IN_DATA, WINDOW, OUT_FILE):
    print("[INFO] Generating theta values from saved tokens...")
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(IN_DATA.split(".")[0])), 'rb') as f:
        out = pickle.load(f)
    tokens = out["tokens"]
    #tm = out["model"]
    dates = out["dates"]
    theta = out["theta"]
            
    df = pd.DataFrame()
    df["date"] = dates
    df["theta"] = theta

    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df = df.sort_values("date")
        
    #print("[INFO] exporting model...")
    #export_model_and_tokens(tm, n, tokens, theta, dates, OUT_PATH, IN_DATA)

    print("[INFO] Extracting novelty and resonance...")
    df = extract_novelty_resonance(df, theta, dates, WINDOW)
    del theta, dates
    size_df = len(df)
    df.to_csv(OUT_FILE, index=False)
    return df, tokens, size_df

#########################################################################################################
### MAIN MODULES
#########################################################################################################

def main_beta_plotting_with_top_tokens(window: int, 
                                       percentage: float, 
                                       size_df: int, 
                                       df, 
                                       tokens: list, 
                                       time: list, 
                                       novelty: list, 
                                       resonance: list, 
                                       OUT_PATH: str, 
                                       IN_DATA: str):
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
    print("[INFO] Calculate beta timeseries")
    beta_w, time_w, time_middle, time_middle_days = beta_time_series(time, novelty, resonance, window, OUT_PATH, IN_DATA) # takes time
    #print("[PLOT] Beta time series")
    #plot_beta_time_series(time_middle, beta_w, time_middle_days, OUT_PATH, IN_DATA)
    del time
    #find time points according to top beta values
    print("[INFO] Find top timepoints with beta") # This takes a hot minute
    time_top, threshold, list_top_idx = timepoints_beta_top(beta_w, time_w, percentage)
    #print("[PLOT] Beta timeseries toptimes")
    #plot_beta_top_time(time_middle, beta_w, time_middle_days, time_top, threshold, list_top_idx, OUT_PATH, IN_DATA)
    del time_w, time_middle, beta_w, time_middle_days, threshold, list_top_idx
    
    #find indices of those time points using df     NB: idx_top follows length of df
    idx_top = list(df['date'].index[df['date'] == time_top[i]].tolist() for i in range(len(time_top)))
    del df, time_top
    idx_top = list(idx_top[i][0] for i in range(len(idx_top)))
    
    #select the tokens
    tokens_top = list(tokens[i] for i in idx_top)
    #save top tokens
    print("[INFO] Save top tokens")
    with open(os.path.join(OUT_PATH, "mdl", "{}_toptokens.txt".format(IN_DATA.split(".")[0])), "w") as f:
        for element in tokens_top:
            f.write("{}\n".format(element))
    del tokens_top
    
    #resonance novelty timeseries plot
    #figname = os.path.join(OUT_PATH, "fig", IN_DATA.split(".")[0] + "_adaptline_top.png")
    #x, y, cond = line_top_time(size_df, idx_top, WINDOW)
    del idx_top
    #print("[PLOT] Adaptiveline toptimes")
    #adaptiveline_toptimes(novelty, resonance, x, y, cond, figname)
    del novelty, resonance#, x, y, cond


def main_module(if_reload: bool,
                downsample_frequency: str,
                SUBREDDIT_NAME: str, 
                ROOT_PATH: str, 
                REDDIT_DATA: str):
    """
    if_reload: bool
    downsample_frequency: frequency at which the data is downsampled (time)
    SUBREDDIT_NAME: str, 
    ROOT_PATH: str, 
    REDDIT_DATA: path to the Reddit data (ndjson files)
    """
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

    check_OUT_PATH(OUT_PATH)
    
    ### GET DATA ###
    if if_reload:
        print("[INFO] Importing previously made model")
        if os.path.isfile(os.path.join(OUT_PATH, IN_DATA.split(".")[0] + "_theta.csv")):
            df, tokens = load_from_premade_model(OUT_PATH, IN_DATA)
        else:
            df, tokens, size_df = load_from_premade_model_generate_thetas(OUT_PATH, IN_DATA, WINDOW, OUT_FILE)
    else:
        df = import_ndjson_files(SUBREDDIT_NAME, REDDIT_DATA)
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df["text"] = df["text"].astype(str)
        df = df.sort_values("date")
        ic(len(df))

        if len(df) >= 10000:
            print("[INFO] Downsampling")
            df = downsample(df, downsample_frequency)
            print(df.head())
            ic(len(df))
            TOPIC_TUNE = [100, 500, 1000, 1500, 3000]

        size_df = len(df)
        print('Length of datapoints in subreddit: ', size_df)
        if size_df < 120:
            df, OUT_PATH, IN_DATA, beta1, hurst_r = check_sufficiency_of_datapoints(df, size_df, OUT_PATH, IN_DATA)
        
        print("\n[INFO] Lemmatizing...\n")
        lemmas = make_lemmas(df)
        lemmas = preprocess_for_topic_models(lemmas, lang=LANG)
        print("\n[INFO] Training model...\n")
        to = Tokenizer()
        tokens = to.doctokenizer(lemmas)
        del lemmas
        
        df, tokens = remove_invalid_entries(tokens, df)
        ic(len(df))
        ic(len(tokens))

        # save tokens
        out = {}
        out["tokens"] = tokens
        with open(os.path.join(OUT_PATH, "mdl", "tokens_{}.pcl".format(IN_DATA.split(".")[0])), "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        del out

        tm, n = train_topic_model(tokens,
                                ESTIMATE_TOPIPCS,
                                TOPIC_TUNE,
                                PLOT_TOPICS)
        ic(tm, n)
        
        if SAVE_SEMANTIC_TOPICS:
            # From bow_mdl.py
            # static semantic content for model summary
            print("\n[INFO] Writing content to file...\n")
            with open(os.path.join(OUT_PATH, "mdl", "{}_content.txt".format(IN_DATA.split(".")[0])), "w") as f:
                for topic in tm.model.show_topics(num_topics=-1, num_words=10):
                    f.write("{}\n\n".format(topic))
        
        print("\n[INFO] Getting topic distribution per document...")
        print("Subreddit = ", SUBREDDIT_NAME)
        
        theta = tm.get_topic_distribution()
        df["theta"] = theta
        dates = df["date"].tolist()
        #takes a lot of time
        #theta = list()
        #for i, doc in enumerate(tm.corpus):
        #    vector = [x[1] for x in tm.model[doc]]
        #    theta.append(vector)
        #    print("[INFO] processed {}/{}".format(i + 1, len(tokens)))  #Ida: needs to be length of tokens as empty entries were removed
   
        print("[INFO] exporting model...")
        export_model_and_tokens(tm, n, tokens, theta, dates, OUT_PATH, IN_DATA)

        print("[INFO] extracting novelty and resonance...")
        df = extract_novelty_resonance(df, theta, dates, WINDOW)
        del theta, dates
        df.to_csv(OUT_FILE, index=False)

    if_rerun_topic_model = False
    if if_rerun_topic_model:
        if len(df) >= 10000:
            print("[INFO] Downsampling")
            df = downsample(df, downsample_frequency)
            print(df.head())
            ic(len(df))
            #TOPIC_TUNE = [100, 120, 180, 220, 300]
        tm, n = train_topic_model(tokens,
                                ESTIMATE_TOPIPCS,
                                TOPIC_TUNE,
                                PLOT_TOPICS)
        ic(tm, n)
        
        if SAVE_SEMANTIC_TOPICS:
            # From bow_mdl.py
            # static semantic content for model summary
            print("\n[INFO] Writing content to file...\n")
            with open(os.path.join(OUT_PATH, "mdl", "{}_content.txt".format(IN_DATA.split(".")[0])), "w") as f:
                for topic in tm.model.show_topics(num_topics=-1, num_words=10):
                    f.write("{}\n\n".format(topic))
        
        print("\n[INFO] Getting topic distribution per document...")
        print("Subreddit = ", SUBREDDIT_NAME)
        
        theta = tm.get_topic_distribution()
        df["theta"] = theta
        dates = df["date"].tolist()
   
        print("[INFO] exporting model...")
        export_model_and_tokens(tm, n, tokens, theta, dates, OUT_PATH, IN_DATA)

        print("[INFO] extracting novelty and resonance...")
        df = extract_novelty_resonance(df, theta, dates, WINDOW)
        del theta, dates
        df.to_csv(OUT_FILE, index=False)
    
    print("[INFO] Get novelty, resonance, beta1")
    time, novelty, resonance, beta1, xz, yz = extract_adjusted_main_parameters(df, WINDOW)
    
    #print("[PLOT] Initial adaptiveline and regplot")
    #plot_initial_figures(novelty, resonance, xz, yz, OUT_PATH, IN_DATA)

    print("[INFO] Hurst exponent")
    hurst_r = hurst_exp(resonance, OUT_PATH)
    
    print("[INFO] Saving hurst and beta")
    output = pd.DataFrame({'hurst': [hurst_r], 'beta1': [beta1]})
    output.to_csv(os.path.join(OUT_PATH, "{}_hurst_beta.csv".format(IN_DATA.split(".")[0])), index=False, encoding='utf-8-sig', sep=';')
    
    print("[INFO] Plot beta timeseries figures...")
    window = 21
    percentage = 0.1
    size_df = len(df)
    main_beta_plotting_with_top_tokens(window, percentage, size_df, df, tokens, time, novelty, resonance, OUT_PATH, IN_DATA)
    
    return OUT_PATH, IN_DATA, beta1, hurst_r

#########################################################################################################
### MAIN
#########################################################################################################

if __name__ == '__main__':
    activated = spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_lg")
    
    ROOT_PATH = r"/home/commando/marislab/newsFluxus/"
    REDDIT_DATA = r'/data/datalab/reddit-sample-hv/comments/*.ndjson'
    WINDOW=3 # window for novelty/resonance
    OUT_PATH = os.path.join(ROOT_PATH, "dat/subreddits_incl_comments/output")
    
    slope_all = []
    hurst_all = []
    for file in glob.glob(REDDIT_DATA):
        ic(file)
        fname = file.split("/")
        SUBREDDIT_NAME = fname[-1]
        SUBREDDIT_NAME = SUBREDDIT_NAME.split(".")
        SUBREDDIT_NAME = SUBREDDIT_NAME[0]
        IN_DATA = SUBREDDIT_NAME + "_incl_comments.csv"
        ic(SUBREDDIT_NAME)
        if os.path.isfile(os.path.join(ROOT_PATH, "dat", "subreddits_incl_comments", "output", (SUBREDDIT_NAME + '_incl_comments_not_executed.txt'))):
            ic("Not excecuted = True")
            continue
        if os.path.isfile(os.path.join(OUT_PATH, "{}_finished.txt".format(SUBREDDIT_NAME))): # Change this since '_finished.txt' does not exist
            ic("Already processed = True")
            #continue
        ic("No processing has occurred on this subreddit")
        
        large_subreddit = ['technology', 'conspiracy'] #['privacytoolsIO', 'Bitcoin', 
                           #'privacy', 'Stellar', 'netsec']
                           
        if SUBREDDIT_NAME not in large_subreddit:
            continue

        try:
            if os.path.isfile(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(IN_DATA.split(".")[0]))):
                if_reload = False ## CHANGE THIS BACK TO TRUE
            else:
                if_reload = False
            downsample_frequency = "60T" #10T is 10 minutes
            OUT_PATH, IN_DATA, beta1, hurst_r = main_module(if_reload, downsample_frequency, SUBREDDIT_NAME, ROOT_PATH, REDDIT_DATA)
                    
            slope_all.append(beta1)
            hurst_all.append(hurst_r)
                    
            with open(os.path.join(OUT_PATH, "{}_finished.txt".format(SUBREDDIT_NAME)), "w") as f:
                f.write("finished successfully")
            print("[INFO] PIPELINE PER FILE FINISHED\n")
        except Exception:
            exception()
            continue
    
    dict = {'hurst': hurst_all, 'slope': slope_all}
    df_ind_var = pd.DataFrame(dict)
    df_ind_var.to_csv(os.path.join(OUT_PATH, 'all_subreddits_indicator_variables.csv'), index=False, encoding='utf-8-sig', sep=';')

