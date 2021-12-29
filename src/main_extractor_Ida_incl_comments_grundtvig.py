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
from icecream import ic
import numpy as np
import traceback

import sys
sys.path.insert(1, r'/home/commando/marislab/newsFluxus/src/')
from tekisuto.preprocessing import Tokenizer
from tekisuto.models import TopicModel
from tekisuto.models import InfoDynamics
from tekisuto.metrics import jsd
from tekisuto.models import LatentSemantics
from import_ndjson_files_incl_comments import import_ndjson_files
from visualsrc.visualsrc import plotVisualsrc
from preparations.preptopicmodeling import prepareTopicModeling
from preparations.betatimeseries import main_beta_plotting_with_top_tokens

mpl_size = 10000
pV = plotVisualsrc
preTM = prepareTopicModeling

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
        ic(f"[INFO] Optimal number of topics is {n}")
        tm = TopicModel(tokens)
        tm.fit(n, **kwargs)
    else:
        tm = TopicModel(tokens)
        n = 10
        n = tune_topic_range[0]
        tm.fit(n, **kwargs)
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
        ic(f"[INFO] Optimal number of topics is {n}")
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
### MAINTENANCE FUNCTIONS
#########################################################################################################

def exception():
    ic("Exception in user code:")
    ic("-"*60)
    traceback.print_exc(file=sys.stdout)
    ic("-"*60)

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
    ic('[INFO] Not sufficient datapoints')
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
    invalid_entries = [index for index in range(len(tokens)) if (tokens[index] == [] or len(tokens[index]) < 3)]
    # remove invalid entries that are not in the index of the df
    #indices = df.index.to_list()
    #invalid_entries = [x for x in invalid_entries if x in set(indices)]
    #ic(len(invalid_entries))
    #keep_entries = set(indices).difference(set(invalid_entries))
    ic(len(invalid_entries))
    #ic(len(indices))
    #ic(len(keep_entries))

    ic(f'Invalid entries removed at {invalid_entries}')#': {df.loc[invalid_entries,0]}')
    df["tokens"] = tokens
    df = df[df["tokens"].apply(lambda x: len(x)) >= 3].reset_index(drop=True)
    tokens = df["tokens"]
    #tokens = [x for x in tokens if x] # fun line of code :)
    #tokens = [x for x in tokens if len(x)>=3]
    #df = df[~df.index.isin(invalid_entries)].reset_index(drop=True)
    #df = df[df.index.isin(keep_entries)].reset_index(drop=True)
    
    return tokens, df

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
    ic("[INFO] Loading theta values...")
    df = pd.read_csv(os.path.join(OUT_PATH, IN_DATA.split(".")[0] + "_theta.csv"))
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df = df.sort_values("date")
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(IN_DATA.split(".")[0])), 'rb') as f:
        out = pickle.load(f)
    tokens = out["tokens"]
    return df, tokens

def load_from_premade_model_generate_thetas(OUT_PATH, IN_DATA, WINDOW, OUT_FILE):
    ic("[INFO] Generating theta values from saved tokens...")
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
        
    #ic("[INFO] exporting model...")
    #export_model_and_tokens(tm, n, tokens, theta, dates, OUT_PATH, IN_DATA)

    ic("[INFO] Extracting novelty and resonance...")
    df = extract_novelty_resonance(df, theta, dates, WINDOW)
    del theta, dates
    size_df = len(df)
    df.to_csv(OUT_FILE, index=False)
    return df, tokens, size_df

def generate_thetas_from_start(SUBREDDIT_NAME, 
                               REDDIT_DATA,
                               downsample_frequency,
                               OUT_PATH,
                               IN_DATA,
                               LANG,
                               TOPIC_TUNE,
                               ESTIMATE_TOPIPCS,
                               PLOT_TOPICS,
                               SAVE_SEMANTIC_TOPICS,
                               WINDOW,
                               OUT_FILE):
    if_sample = True
    sample_percent = 0.5
    
    if SUBREDDIT_NAME in ["technology", "conspiracy"]:
        if if_sample:
            if os.path.isfile(os.path.join(OUT_PATH, "extra", "sampled_{}.csv".format(IN_DATA.split(".")[0]))):
                ic("[INFO] Loading previous sample")
                df = pd.read_csv(os.path.join(OUT_PATH, "extra", "sampled_{}.csv".format(IN_DATA.split(".")[0])))
                ic(len(df))
            else:
                df = import_ndjson_files(SUBREDDIT_NAME, REDDIT_DATA)
                df = df.dropna().drop_duplicates().reset_index(drop=True)
                df["text"] = df["text"].astype(str)
                if SUBREDDIT_NAME in ["conspiracy"]:
                    ic("[INFO] Subreddit conspiracy - removing bot '[meta] sticky comment'")
                    df = df[~df.text.str.contains(r"[meta] sticky comment", case=False, regex=False)].reset_index(drop=True)
                df = df.sort_values("date")
                ic("[INFO] Sampling a fraction")
                df["date"] = pd.to_datetime(df["date"])
                df['just_date'] = df['date'].dt.date
                # take 0.5 randomly per date
                df = df.groupby("just_date").sample(frac=sample_percent, random_state = 1984)
                df["date"] = df["date"].astype(str)
                df = df.drop(["just_date"], axis=1)
                df.to_csv(os.path.join(OUT_PATH, "extra", "sampled_{}.csv".format(IN_DATA.split(".")[0])), index=False)
    else:
        df = import_ndjson_files(SUBREDDIT_NAME, REDDIT_DATA)
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df["text"] = df["text"].astype(str)
        df = df.sort_values("date")
        ic(len(df))
    
    df["text"] = df["text"].astype(str)
    df = df.sort_values("date")
    ic(len(df))

    size_df = len(df)
    if size_df < 120:
        df, OUT_PATH, IN_DATA, beta1, hurst_r = check_sufficiency_of_datapoints(df, size_df, OUT_PATH, IN_DATA)
    
    if_nah = False
    if if_nah:#os.path.isfile(os.path.join(OUT_PATH, "mdl", "tokens_{}.pcl".format(IN_DATA.split(".")[0]))):
        ic("[INFO] Reloading tokens...")
        with open(os.path.join(OUT_PATH, "mdl", "tokens_{}.pcl".format(IN_DATA.split(".")[0])), 'rb') as f:
            out = pickle.load(f)
        tokens = out["tokens"]
    else:
        ic("[INFO] Lemmatizing...")
        lemmas = preTM.make_lemmas(df, nlp)
        lemmas = preTM.preprocess_for_topic_models(lemmas, ROOT_PATH, lang=LANG)    
        ic("[INFO] Tokenizing...")
        to = Tokenizer()
        tokens = to.doctokenizer(lemmas)
        del lemmas

        # save tokens
        out = {}
        out["tokens"] = tokens
        with open(os.path.join(OUT_PATH, "mdl", "tokens_{}.pcl".format(IN_DATA.split(".")[0])), "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        del out

    tokens, df = remove_invalid_entries(tokens, df)
    ic(len(df))
    ic(len(tokens))

    if len(df) >= 10000:
        ic("[INFO] Downsampling")
        df["text_list"] = tokens
        df["text"] = [' '.join(map(str, l)) for l in df['text_list']]
        df["text"] = df["text"].astype(str)
        df = preTM.downsample(df, downsample_frequency, if_list=False)
        ic(df.head())
        ic(len(df))

        def str_to_list(s):
            return s.split(' ')

        tokens = df['text'].apply(str_to_list).to_list()
        tokens, df = remove_invalid_entries(tokens, df)
        ic(len(df))
        ic(len(tokens))
        ic(tokens[0:2])
    #    TOPIC_TUNE = [500] #[100, 500, 1000]#, 1500]#, 3000]

    

    ic("[INFO] Training model...")
    tm, n = train_topic_model(tokens,
                                ESTIMATE_TOPIPCS,
                                TOPIC_TUNE,
                                PLOT_TOPICS)
    ic(tm, n)
        
    if SAVE_SEMANTIC_TOPICS:
        # From bow_mdl.py
        # static semantic content for model summary
        ic("[INFO] Writing content to file...")
        with open(os.path.join(OUT_PATH, "mdl", "{}_content.txt".format(IN_DATA.split(".")[0])), "w") as f:
            for topic in tm.model.show_topics(num_topics=-1, num_words=10):
                f.write("{}\n\n".format(topic))
        
    ic("[INFO] Getting topic distribution per document...")
    ic(SUBREDDIT_NAME)
        
    theta = tm.get_topic_distribution()
    df["theta"] = theta
    dates = df["date"].tolist()
   
    ic("[INFO] exporting model...")
    export_model_and_tokens(tm, n, tokens, theta, dates, OUT_PATH, IN_DATA)

    ic("[INFO] extracting novelty and resonance...")
    df = extract_novelty_resonance(df, theta, dates, WINDOW)
    del theta, dates
    df.to_csv(OUT_FILE, index=False)
    return df, tokens, size_df

#########################################################################################################
### MAIN MODULES
#########################################################################################################

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
        ic("[INFO] Importing previously made model")
        if os.path.isfile(os.path.join(OUT_PATH, IN_DATA.split(".")[0] + "_theta.csv")):
            df, tokens = load_from_premade_model(OUT_PATH, IN_DATA)
        else:
            df, tokens, size_df = load_from_premade_model_generate_thetas(OUT_PATH, IN_DATA, WINDOW, OUT_FILE)
    else:
        df, tokens, size_df = generate_thetas_from_start(SUBREDDIT_NAME, REDDIT_DATA, downsample_frequency, OUT_PATH, IN_DATA, LANG, TOPIC_TUNE, ESTIMATE_TOPIPCS, PLOT_TOPICS, SAVE_SEMANTIC_TOPICS, WINDOW, OUT_FILE)

    if_rerun_topic_model = False
    if if_rerun_topic_model:
        if len(df) >= 10000:
            ic("[INFO] Downsampling")
            df["text"] = tokens
            df["text"] = df["text"].astype(str)
            df = preTM.downsample(df, downsample_frequency)
            ic(df.head())
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
            ic("[INFO] Writing content to file...")
            with open(os.path.join(OUT_PATH, "mdl", "{}_content.txt".format(IN_DATA.split(".")[0])), "w") as f:
                for topic in tm.model.show_topics(num_topics=-1, num_words=10):
                    f.write("{}\n\n".format(topic))
        
        ic("[INFO] Getting topic distribution per document...")
        ic("Subreddit = ", SUBREDDIT_NAME)
        
        theta = tm.get_topic_distribution()
        df["theta"] = theta
        dates = df["date"].tolist()
   
        ic("[INFO] exporting model...")
        export_model_and_tokens(tm, n, tokens, theta, dates, OUT_PATH, IN_DATA)

        ic("[INFO] extracting novelty and resonance...")
        df = extract_novelty_resonance(df, theta, dates, WINDOW)
        del theta, dates
        df.to_csv(OUT_FILE, index=False)
    
    ic("[INFO] Get novelty, resonance, beta1")
    time, novelty, resonance, beta1, xz, yz = pV.extract_adjusted_main_parameters(df, WINDOW)
    
    #ic("[PLOT] Initial adaptiveline and regplot")
    #pV.plot_initial_figures(novelty, resonance, xz, yz, OUT_PATH, IN_DATA)

    ic("[INFO] Hurst exponent")
    hurst_r = hurst_exp(resonance, OUT_PATH)
    
    ic("[INFO] Saving hurst and beta")
    output = pd.DataFrame({'hurst': [hurst_r], 'beta1': [beta1]})
    output.to_csv(os.path.join(OUT_PATH, "{}_hurst_beta.csv".format(IN_DATA.split(".")[0])), index=False, encoding='utf-8-sig', sep=';')
    
    ic("[INFO] Plot beta timeseries figures...")
    window = 21
    percentage = 0.1
    size_df = len(df)
    main_beta_plotting_with_top_tokens(window, percentage, size_df, df, tokens, time, novelty, resonance, OUT_PATH, IN_DATA, WINDOW)
    
    return OUT_PATH, IN_DATA

#########################################################################################################
### MAIN
#########################################################################################################

if __name__ == '__main__':
    activated = spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_lg")

    SOME_FIXED_SEED = 1984
    np.random.seed(SOME_FIXED_SEED)
    
    ROOT_PATH = r"/home/commando/marislab/newsFluxus/"
    #REDDIT_DATA = r"/home/commando/marislab/wallstreetbets/comments/*.ndjson"
    REDDIT_DATA = r'/data/datalab/reddit-sample-hv/comments/*.ndjson'
    WINDOW=3 # window for novelty/resonance
    OUT_PATH = os.path.join(ROOT_PATH, "dat/subreddits_incl_comments/output")

    for file in glob.glob(REDDIT_DATA):
        #ic(file)
        fname = file.split("/")
        SUBREDDIT_NAME = fname[-1].split(".")[0]
        IN_DATA = SUBREDDIT_NAME + "_incl_comments.csv"
        #ic(SUBREDDIT_NAME)
        if os.path.isfile(os.path.join(ROOT_PATH, "dat", "subreddits_incl_comments", "output", (SUBREDDIT_NAME + '_incl_comments_not_executed.txt'))):
            #ic("Not excecuted = True")
            continue
        if os.path.isfile(os.path.join(OUT_PATH, "{}_finished.txt".format(SUBREDDIT_NAME))): # Change this since '_finished.txt' does not exist
            ic("Already processed = True")
            #continue
        #ic("No processing has occurred on this subreddit")
        
        selected_subreddit = ["conspiracy"] # ['privacytoolsIO', 'Bitcoin', 'privacy', 'Stellar', 'netsec'] #['technology']#, 'conspiracy']# ["InformationPolicy"] #["Stellar"]
                           ### CONSPIRACY AND TECHNOLOGY HAVE A
        if SUBREDDIT_NAME not in selected_subreddit:
            continue

        try:
            if os.path.isfile(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(IN_DATA.split(".")[0]))):
                if_reload = False
            else:
                if_reload = False
            downsample_frequency = "60T" #10T is 10 minutes
            OUT_PATH, IN_DATA = main_module(if_reload, downsample_frequency, SUBREDDIT_NAME, ROOT_PATH, REDDIT_DATA)
                    
                    
            with open(os.path.join(OUT_PATH, "{}_finished.txt".format(SUBREDDIT_NAME)), "w") as f:
                f.write("finished successfully")
            ic("[INFO] PIPELINE PER FILE FINISHED")
        except Exception:
            exception()
            continue
    