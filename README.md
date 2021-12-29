# NGI: Trend reservoir detection on Reddit
_Main author of the repository: [Maris Sala](https://pure.au.dk/portal/da/persons/maris-sala(e5f64ed3-eb68-4604-9420-48f65b3cb6a5).html)_
_Codes have been developed with [CHCAA](https://github.com/centre-for-humanities-computing/newsFluxus) and [Ida Nissen](https://pure.au.dk/portal/da/persons/ida-anthonj-nissen(92b9b19e-6ca0-4d67-9725-b7a8772ba1f5).html)_

_Done as a project related to [AU DATALAB](https://datalab.au.dk/) under European Union's [Next Generation Internet initiative](https://www.ngi.eu/)_

_Code base has been developed at CHCAA as the newsFluxus model, more information [here](newsFluxus-README.md)_

This codebase includes codes developed to study potential trends in Reddit subreddits.
The analysis uses newsFluxus and infodynamics to calculate novelty, resonance, and the Hurst exponent based on LDA topic modeling of subreddits related to human rights and technology.

## Structure of this repository
    .
    ├── dat/                                                # Output data
    │   └── subreddits_incl_comments/       
    │       └── output/                     
    │           ├── extra/                                  # Sampled subreddits
    │           ├── fig/                                    # Figures: adaptline of top posts, beta timeseries, regline
    │           └── mdl/                                    # Topic models and contents of topic modeling
    │               └── testing_phase/                      # Top tokens per subreddits
    ├── fig/                                                # Example figures of adaptline and regline
    ├── mdl/                                                # Irrelevant: Danish language models
    ├── res/                                                # Resources: stopwords
    ├── src/                                                # Source codes
    │   ├── __pycache__
    │   ├── archive/                                        # Original versions of main codes before they were modified for the project
    │   ├── preparations/                                   # Preparing for topic modeling
    │   │   ├── __init__.py
    │   │   ├── betatimeseries.py
    │   │   └── preptopicmodeling.py
    │   ├── saffine/                                        # Detrending time series
    │   │   ├── detrending_coeff.py
    │   │   ├── detrending_method.py
    │   │   └── multi_detrending.py
    │   ├── tekisuto/                                       # Infodynamics source codes
    │   │   ├── datasets/
    │   │   ├── metrics/
    │   │   ├── models/
    │   │   ├── preprocessing/
    │   │   ├── __init__.py
    │   │   └── tekiutil.py
    │   ├── visualsrc/                                      # Visualizing trends
    │   ├── bow_mdl.py
    │   ├── import_ndjson_files_incl_comments.py            # Main code that loads subreddit posts and comments, joins together
    │   ├── main_extractor_Ida_incl_comments_grundtvig.py   # Main code
    │   ├── main_extractor.py          
    │   ├── news_uncertainty.py               
    │   ├── signal_extraction.py           
    │   ├── spacy_parsing.py          
    │   └── topic_modeling_top_posts.py                     # Topic modeling of top trending posts per subreddit
    ├── .gitignore
    ├── LICENCE
    └── README.md                                           # Main information for this repository