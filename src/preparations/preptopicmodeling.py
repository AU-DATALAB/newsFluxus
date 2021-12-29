# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 2021

@author: au558899

Source codes for preparing topic modeling codes for main extractor of newsFluxus

"""

#########################################################################################################
### PREPARE TEXTS FOR TOPIC MODELING
#########################################################################################################
import pandas as pd
import re
import os

import sys
sys.path.insert(1, r'/home/commando/marislab/newsFluxus/src/')
from tekisuto.preprocessing import CaseFolder
from tekisuto.preprocessing import RegxFilter
from tekisuto.preprocessing import StopWordFilter

class prepareTopicModeling:
    @staticmethod
    def downsample(df,
                frequency='1T',
                if_list=False):
        """
        df: pandas DataFrame with columns "date" and "text"
        frequency: time interval for downsampling, default 1T - creates 1min timebins

        Returns downsampled pandas DataFrame
        """
        df = df.dropna().reset_index(drop=True)
        df["text"] = df["text"].astype(str)
        print(len(df["text"]))
        print(len(df["date"]))
        df["text"] = pd.Series(df["text"])
        df["date"] = pd.Series(df["date"])

        df.index = pd.to_datetime(df["date"])
        df = df.drop("date", axis = 1)
        if if_list:
            df = df.resample(frequency).agg({"text": lambda x: list(x)})
        else:
            df = df.resample(frequency).agg({"text": ' '.join})
        df = df.reset_index(col_fill = "date")
        df = df[["text", "date"]]
        if not if_list:
            df["text"] = df["text"].astype(str)
        df["date"] = df["date"].astype(str)
        return df
    
    @staticmethod
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
    
    @staticmethod
    def make_lemmas(df, nlp, lang):
        """
        df: pandas DataFrame with column "text"

        Returns lemmas as list
        """
        preTM = prepareTopicModeling()
        df["text"] = df["text"].astype(str)

        url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
        if lang == "en":
            remove_more =  [r"&\w{2,}", r"\d+", r"\W+", r"[^A-z]", r"_", r"\s+"]  #first remove urls, then &gt and &amp, then numbers and other characters, then non-english letters (e.g. chinese), then underscores (from user names for example), and last excess spaces    
        else:
            remove_more =  [r"&\w{2,}", r"\d+", r"\W+", r"_", r"\s+"]  #first remove urls, then &gt and &amp, then numbers and other characters, then underscores (from user names for example), and last excess spaces    

        pre_lemma = [url_pattern.sub(' ',x) for x in df["text"].tolist()]
        for i in range(len(remove_more)):
            remove_more_pattern = re.compile(remove_more[i])
            pre_lemma = [remove_more_pattern.sub(' ',x) for x in pre_lemma]

        lemmas = preTM.spacy_lemmatize(pre_lemma, nlp=nlp)
        lemmas = [' '.join(doc) for doc in lemmas]
        #Ida: remove -PRON- from text (white spaces are removed in the next step preprocess_for_topic_models)
        lemmas = [re.sub('-PRON-', '', lemmas[x]) for x in range(len(lemmas))]
        
        return lemmas

    @staticmethod
    def make_lemmas2(row, nlp):
        """
        row: pandas DataFrame with column "text"

        Returns lemmas as list
        """
        preTM = prepareTopicModeling()
        #row["text"] = row["text"].astype(str)

        url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
        remove_more =  [r"&\w{2,}", r"\d+", r"\W+", r"[^A-z]", r"_", r"\s+"]  #first remove urls, then &gt and &amp, then numbers and other characters, then non-english letters (e.g. chinese), then underscores (from user names for example), and last excess spaces    
        
        pre_lemma = [url_pattern.sub(' ', x) for x in [row["text"]]]
        for i in range(len(remove_more)):
            remove_more_pattern = re.compile(remove_more[i])
            pre_lemma = [remove_more_pattern.sub(' ',x) for x in pre_lemma]

        lemmas = preTM.spacy_lemmatize(pre_lemma, nlp=nlp)
        lemmas = [' '.join(doc) for doc in lemmas]
        #Ida: remove -PRON- from text (white spaces are removed in the next step preprocess_for_topic_models)
        lemmas = [re.sub('-PRON-', '', lemmas[x]) for x in range(len(lemmas))]
        
        return lemmas
    
    @staticmethod
    def preprocess_for_topic_models(lemmas: list,
                                    ROOT_PATH: str, 
                                    lang=str):
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