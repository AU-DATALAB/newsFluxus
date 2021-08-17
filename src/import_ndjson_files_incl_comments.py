# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:09:53 2021

@author: au685355
"""


def import_ndjson_files(SUBREDDIT_NAME, REDDIT_DATA):

#%% Combine the json files with posts (submissions) (inclusive their own text body besides the title) and comments into one text file

    import glob
    from datetime import datetime, timezone
    import ndjson
    import json
    import pandas as pd
    import os
    from icecream import ic
    import os, shutil
    import os.path
    from os import path
    
    # Create a directory for these files
    temp_path = "/home/commando/marislab/newsFluxus/tmp_data/"

    try:
        os.mkdir(temp_path)
    except OSError:
        print ("Creation of the directory %s failed" % temp_path)
    else:
        print ("Successfully created the directory %s " % temp_path)
    


#%% First: Submissions
#read all files in folder
    reddit_path = os.path.split(REDDIT_DATA)
    reddit_path = os.path.split(reddit_path[0])
    reddit_path = reddit_path[0]
    
    file = os.path.join(reddit_path, 'submissions', (SUBREDDIT_NAME + '.ndjson'))
    print(file)
    #with open(file, encoding="utf8") as f:
    #    submissions = ndjson.load(f)
    
    submissions = []
    with open(file) as f:
        for line in f:
            submissions.append(json.loads(line))
            
    #Structure: dataframe with two columns
    #column 1: title
    
    list_title = list(submissions[i]['title'] for i in range(len(submissions)))# if ("selftext" in submissions[i]))
    ic(len(list_title))
    
    #add body beneath title (account for that some have missing key 'selftext')
    list_body = []
    for i in range(len(submissions)):
        if 'selftext' in submissions[i]:
            list_body.append(submissions[i]['selftext'])
        else:
            list_body.append('')
    ic(len(list_body))
  
    #Add list_body strings behind list_title strings, after a space
    list_text = list(list_title[i] + ' ' + list_body[i] for i in range(len(submissions)))#len of submissions
    ic(len(list_text))
    del list_title, list_body
    
    #column 2: date - convert unix timestamp to format “2020-03-31T10:01:50+02:00”
    list_date = []
    for i in range(len(submissions)):
        dtobj = datetime.fromtimestamp(submissions[i]['created_utc'], timezone.utc)
        # 2010-09-10T06:51:25+00:00
        dtobj = dtobj.isoformat()
        list_date.append(dtobj)
        
    del submissions
    ic(len(list_date))
    
    #Put into dataframe
    df_subreddit = pd.DataFrame(
        {'text': list_text,
         'date': list_date
         })
    del list_text, list_date

    ic(len(df_subreddit))
    #Export to csv file
    temp_file = temp_path + SUBREDDIT_NAME + ".csv"
    df_subreddit.to_csv(temp_file, index=False, encoding='utf-8-sig', sep=';')
    del df_subreddit
    
###################################################################################################
#%% Second: add comments to csv file
    file = os.path.join(reddit_path, 'comments', (SUBREDDIT_NAME + '.ndjson'))
    print(file)
    #fname = file.split("\\")
    #fname = fname[-1].split(".")[0]
    with open(file, encoding="utf8") as f:
        comments = ndjson.load(f)
    
    #Structure: dataframe with two columns
    #column 1: title
    list_text_comments = list(comments[i]['body'] for i in range(len(comments)))
    ic(len(list_text_comments))
    
    #column 2: date - convert unix timestamp to format “2020-03-31T10:01:50+02:00”
    list_date_comments = []
    for i in range(len(comments)):
        dtobj = datetime.fromtimestamp(comments[i]['created_utc'], timezone.utc)
        # 2010-09-10T06:51:25+00:00
        dtobj = dtobj.isoformat()
        list_date_comments.append(dtobj)    
    
    del comments
    ic(len(list_date_comments))
    
    #Put into dataframe
    df_subreddit_comments = pd.DataFrame(
        {'text': list_text_comments,
          'date': list_date_comments
          })
    del list_text_comments, list_date_comments
    ic(len(df_subreddit_comments))
    
    #Export to csv file
    temp_file_comments = temp_path + SUBREDDIT_NAME + "_comments.csv"
    df_subreddit_comments.to_csv(temp_file_comments, index=False, encoding='utf-8-sig', sep=';')
    del df_subreddit_comments
    
    ################################################################################################
    # Read both files in again and try to join them
    #merge both dataframes into one
    print("Merge submission with comments")
    df = pd.read_csv(temp_file, sep=";", lineterminator='\n')
    df_0 = pd.read_csv(temp_file_comments, sep=";", lineterminator='\n')
    df_subreddit_incl_comments = df.append(df_0)
    del df_0, df
    
    #sort per date
    df_subreddit_incl_comments.sort_values(by=['date'], inplace=True, ignore_index=True)
    
    print("Delete contents of temp/")
    try:
        shutil.rmtree(temp_path)
    except OSError:
        print ("Deletion of the directory %s failed" % temp_path)
    else:
        print ("Successfully deleted the directory %s" % temp_path)
        
    return df_subreddit_incl_comments
    