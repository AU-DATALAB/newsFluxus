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
    import pandas as pd
    import os


#%% First: Submissions
#read all files in folder
#for file in glob.glob(r'U:\NGI\Human_values\Data_subreddits_37\ida_anthonj_nissen_reddit\submissions\*.ndjson'):
    reddit_path = os.path.split(REDDIT_DATA)
    reddit_path = os.path.split(reddit_path[0])
    reddit_path = reddit_path[0]
    
    file = os.path.join(reddit_path, 'submissions', (SUBREDDIT_NAME + '.ndjson'))
    print(file)
    #fname = file.split("\\")
    #fname = fname[-1].split(".")[0]
    with open(file, encoding="utf8") as f:
        submissions = ndjson.load(f)
    
    #Structure: dataframe with two columns
    #column 1: title
    list_title = list(submissions[i]['title'] for i in range(len(submissions)))
    #add body beneath title (account for that some have missing key 'selftext')
    list_body = []
    for i in range(len(submissions)):
        if 'selftext' in submissions[i]:
            list_body.append(submissions[i]['selftext'])
        #else:
        #    list_body.append('gotcha')
    
    
  
    #Add list_body strings behind list_title strings, after a space
    list_text = list(list_title[i] + ' ' + list_body[i] for i in range(len(submissions)))
    
    #column 2: date - convert unix timestamp to format “2020-03-31T10:01:50+02:00”
    list_date = []
    for i in range(len(submissions)):
        dtobj = datetime.fromtimestamp(submissions[i]['created_utc'], timezone.utc)
        # 2010-09-10T06:51:25+00:00
        dtobj = dtobj.isoformat()
        list_date.append(dtobj)    

    #Put into dataframe
    df_subreddit = pd.DataFrame(
        {'text': list_text,
         'date': list_date
         })

    #Export to csv file
    #df_subreddit.to_csv(r'U:\Python\Newsfluxus\newsFluxus-master_Lasse\newsFluxus-master\dat\subreddits_incl_comments\subreddit_{}_incl_comments.csv'.format(fname), index=False, encoding='utf-8-sig', sep=';')


#%% Second: add comments to csv file
#for file in glob.glob(r'U:\NGI\Human_values\Data_subreddits_37\ida_anthonj_nissen_reddit\comments\*.ndjson'):
#for file in glob.glob(r'U:\NGI\Human_values\Data_subreddits_37\ida_anthonj_nissen_reddit\comments\FreeAsInFreedom.ndjson'):
    file = os.path.join(reddit_path, 'comments', (SUBREDDIT_NAME + '.ndjson'))
    print(file)
    #fname = file.split("\\")
    #fname = fname[-1].split(".")[0]
    with open(file, encoding="utf8") as f:
        comments = ndjson.load(f)
    
    #Structure: dataframe with two columns
    #column 1: title
    list_text_comments = list(comments[i]['body'] for i in range(len(comments)))
    
    #column 2: date - convert unix timestamp to format “2020-03-31T10:01:50+02:00”
    list_date_comments = []
    for i in range(len(comments)):
        dtobj = datetime.fromtimestamp(comments[i]['created_utc'], timezone.utc)
        # 2010-09-10T06:51:25+00:00
        dtobj = dtobj.isoformat()
        list_date_comments.append(dtobj)    

    #Put into dataframe
    df_subreddit_comments = pd.DataFrame(
        {'text': list_text_comments,
          'date': list_date_comments
          })

    #append to existing csv file with the posts and add the comments
    #df_subreddit_comments.to_csv(r'U:\Python\Newsfluxus\newsFluxus-master_Lasse\newsFluxus-master\dat\subreddits_incl_comments\subreddit_{}_incl_comments.csv'.format(fname), mode='a', header=False, index=False, encoding='utf-8-sig', sep=';')
    
    #merge both dataframes into one
    df_subreddit_incl_comments = pd.concat([df_subreddit, df_subreddit_comments], ignore_index=True)
    #sort per date
    df_subreddit_incl_comments.sort_values(by=['date'], inplace=True, ignore_index=True)
    
    #df_subreddit_incl_comments.to_csv(r'U:\Python\Newsfluxus\newsFluxus-master_Lasse\newsFluxus-master\dat\subreddits_incl_comments\subreddit_{}_incl_comments.csv'.format(fname), index=False, encoding='utf-8-sig', sep=';')
    
    return df_subreddit_incl_comments
    