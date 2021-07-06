"""
Legacy codes from main_extractor_Ida_incl_comments_grundtvig.py
"""

#%% not used
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
        print(f"\n[INFO] Optimal number of topics is {n}")
        tm = TopicModel(tokens)
        tm.fit(n, **kwargs)
    else:
        tm = TopicModel(tokens)
        n = 10
        tm.fit(10, **kwargs)
    return tm, n
#%%