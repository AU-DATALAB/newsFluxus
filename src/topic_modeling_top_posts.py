
import ast
import os
import glob
import re
#import pickle
import numpy as np
import pandas as pd

from icecream import ic
import traceback

import sys
sys.path.insert(1, r'/home/commando/marislab/newsFluxus/src/')
from tekisuto.models import TopicModel

SOME_FIXED_SEED = 1984
np.random.seed(SOME_FIXED_SEED)

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
    """
    if estimate_topics:
        tm = TopicModel(tokens)
        n, n_cohers = tm.tune_topic_range(
            ntopics=tune_topic_range,
            plot_topics=plot_topics)
        print(n_cohers)
        print(f"\n[INFO] Optimal number of topics is {n}")
        tm = TopicModel(tokens)
        tm.fit(n, **kwargs)
    else:
        tm = TopicModel(tokens)
        n = 10
        tm.fit(n, **kwargs)
        n_cohers=0
    return tm, n, n_cohers

#########################################################################################################
### MAIN MODULE
#########################################################################################################

def main_module(SUBREDDIT_NAME, ROOT_PATH):
    IN_DATA= SUBREDDIT_NAME + "_incl_comments.csv"
    OUT_PATH = os.path.join(ROOT_PATH, "dat/subreddits_incl_comments/output")
    ESTIMATE_TOPIPCS = True # whether to tune multiple topic model sizes
    TOPIC_TUNE = [20, 30, 50, 80] #number of topics to tune over in topic model   #Ida: do not set to only one, then set estimate_topipcs = False.
    PLOT_TOPICS = True # plot a topic of coherence by number of topics
    
    large_subreddit = ['technology', 'conspiracy', 'privacytoolsIO', 'Bitcoin', 'privacy', 'Stellar', 'netsec']
    if SUBREDDIT_NAME in large_subreddit:
        print("[INFO] SUBREDDIT is a large subreddit, using higher number of topic tune range")
        TOPIC_TUNE = [100, 500, 1000, 1500, 3000]

    print("[INFO] Load top tokens")
    with open(os.path.join(OUT_PATH, "mdl", "{}_toptokens.txt".format(IN_DATA.split(".")[0]))) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    tokens = [ast.literal_eval(lines[i]) for i in range(len(lines))]
    
    print("[INFO] Train topic model")
    tm, n, n_cohers = train_topic_model(tokens,
                            ESTIMATE_TOPIPCS,
                            TOPIC_TUNE,
                            PLOT_TOPICS)

    print("[INFO] Get top posts per topics")
    thetas = tm.get_topic_distribution()
    topics = tm.model.show_topics(num_topics=-1, num_words=15) # used to be 10 but need more meaningful topics
    topic_number = [np.argmax(thetas[post]) for post in range(len(thetas))]
    max_thetas = [max(thetas[post]) for post in range(len(thetas))]

    best_post_per_topic_df = pd.DataFrame({'topic_nr': topic_number, "max_theta": max_thetas}).reset_index().groupby("topic_nr").max("max_theta").reset_index()
    
    if len(best_post_per_topic_df) != n:
        print("[INFO] Estimated nr of topics (", str(n), ") does not follow topics found in text (", str(len(best_post_per_topic_df)), ")")

    id_top_posts = best_post_per_topic_df["index"]
    ori_topic_numbers = best_post_per_topic_df["topic_nr"]
    del best_post_per_topic_df
    top_posts = [tokens[i] for i in id_top_posts]
    topic_words = [re.findall(r'([a-zA-Z]+)', list(topics[i])[1]) for i in ori_topic_numbers]
    
    print("[INFO] Saving output")
    out = pd.DataFrame({'topic_nr': ori_topic_numbers, 'topic_words': topic_words, 'top_post_tokens': top_posts, 
                        'topic_tune': [TOPIC_TUNE]*len(ori_topic_numbers), 
                        'n_cohers': [n_cohers]*len(ori_topic_numbers)})
    out.to_csv(os.path.join(OUT_PATH, "mdl/testing_phase/", "{}_toptokens_tm_per_post.csv".format(IN_DATA.split(".")[0])), index=False)
    del out


#########################################################################################################
### MAIN
#########################################################################################################

if __name__ == '__main__':    
    ROOT_PATH = r"/home/commando/marislab/newsFluxus/"
    REDDIT_DATA = r'/data/datalab/reddit-sample-hv/comments/*.ndjson'
    OUT_PATH = os.path.join(ROOT_PATH, "dat/subreddits_incl_comments/output")
    
    #ex: file = 'U:\\Python\\Newsfluxus\\newsFluxus-master_Lasse\\newsFluxus-master\\dat\\subreddits\\subreddit_ACTA.csv'
    for file in glob.glob(REDDIT_DATA):
        ic(file)
        fname = file.split("/")
        SUBREDDIT_NAME = fname[-1].split(".")[0]
        
        IN_DATA = SUBREDDIT_NAME + "_incl_comments.csv"
        
        ic(SUBREDDIT_NAME)
        if os.path.isfile(os.path.join(ROOT_PATH, "dat", "subreddits_incl_comments", "output", (SUBREDDIT_NAME + '_incl_comments_not_executed.txt'))):
            ic("Not excecuted = True")
            continue
        if os.path.isfile(os.path.join(OUT_PATH, "mdl", "{}_toptokens_topicmodeling.txt".format(IN_DATA.split(".")[0]))):
            ic("Already processed = True")
        #    continue
        ic("No processing has occurred on this subreddit")
        testem = ['Bitcoin'] #['conspiracy']# 'privacy', 'privacytoolsIO','privacy','netsec','Stellar','Bitcoin','technology']

        if SUBREDDIT_NAME in testem:
            try:   
                if os.path.isfile(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(IN_DATA.split(".")[0]))) & os.path.isfile(os.path.join(OUT_PATH, IN_DATA.split(".")[0] + "_theta.csv")):
                    ic(SUBREDDIT_NAME)
                    main_module(SUBREDDIT_NAME, ROOT_PATH)
                    print("[INFO] ----------- ALL DONE -----------\n")
                else:
                    print("[INFO] Processing failed: tokens have not been calculated/saved for subreddit ", SUBREDDIT_NAME)
                    
            except Exception:
                print("Exception in user code:")
                print("-"*60)
                traceback.print_exc(file=sys.stdout)
                print("-"*60)
            
