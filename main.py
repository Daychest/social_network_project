# Project Group 3 - Introduction to Social Network Analysis
# Ivan Aguilar Garcia - Johannes Hyry - Veikka Päivärinta

# Imports
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import langid as ld
import re
import demoji
import spacy as sp
import en_core_web_sm
import itertools

# Task 1
# Read the Boston Bombings dataset and the timestamp dataset
boston = pd.read_csv('2013_Boston_bombings-tweets_labeled.csv', )
boston_timestamp = pd.read_csv('2013_Boston_bombings-tweetids_entire_period.csv')
boston.rename(columns={"Tweet ID": " Tweet-ID"}, inplace=True)

# Merge the two datasets together
boston_join = pd.merge(boston, boston_timestamp, on=" Tweet-ID")

# Add the event_type column
boston_join.insert(len(boston_join.columns), 'event_type', 'Boston Bombings')

# Read the Alberta Floods dataset and the timestamp dataset
alberta = pd.read_csv('2013_Alberta_floods-tweets_labeled.csv')
alberta_timestamp = pd.read_csv('2013_Alberta_floods-tweetids_entire_period.csv')
alberta.rename(columns={"Tweet ID": " Tweet-ID"}, inplace=True)

# Merge the two datasets together
alberta_join = pd.merge(alberta, alberta_timestamp, on=" Tweet-ID")

# Add the event_type column
alberta_join.insert(len(alberta_join.columns), 'event_type', 'Alberta Floods')

# Read the Queensland Floods dataset and the timestamp dataset
queensland = pd.read_csv('2013_Queensland_floods-tweets_labeled.csv')
queensland_timestamp = pd.read_csv('2013_Queensland_floods-tweetids_entire_period.csv')
queensland.rename(columns={"Tweet ID": " Tweet-ID"}, inplace=True)

# Merge the two datasets together
queensland_join = pd.merge(queensland, queensland_timestamp, on=" Tweet-ID")

# Add the event_type column
queensland_join.insert(len(queensland_join.columns), 'event_type', 'Queensland Floods')

# Read the Spain Train Crash dataset and the timestamp dataset
spain = pd.read_csv('2013_Spain_train_crash-tweets_labeled.csv')
spain_timestamp = pd.read_csv('2013_Spain_train_crash-tweetids_entire_period.csv')
spain.rename(columns={"Tweet ID": " Tweet-ID"}, inplace=True)

# Merge the two datasets together
spain_join = pd.merge(spain, spain_timestamp, on=" Tweet-ID")

# Add the event_type column
spain_join.insert(len(spain_join.columns), 'event_type', 'Spain Train Crash')

# Read the Colorado Wildfires dataset and the timestamp dataset
colorado = pd.read_csv('2012_Colorado_wildfires-tweets_labeled.csv')
colorado.rename(columns={"Tweet ID": " Tweet-ID"}, inplace=True)
colorado_timestamp = pd.read_csv('2012_Colorado_wildfires-tweetids_entire_period.csv')

# Merge the two datasets together
colorado_join = pd.merge(colorado, colorado_timestamp, on=" Tweet-ID")

# Add the event_type column
colorado_join.insert(len(colorado_join.columns), 'event_type', 'Colorado Wildfires')

# Concatenate all the datasets into one
main_dataset = pd.concat([boston_join, alberta_join, queensland_join, spain_join, colorado_join], ignore_index=True)

# Remove the leading spaces from the column names
main_dataset.rename(columns={" Tweet-ID": "Tweet-ID", " Tweet Text": "Tweet Text", " Information Source": "Information Source", " Information Type": "Information Type", " Informativeness": "Informativeness", " Included(Y/N)": "Included(Y/N)"}, inplace=True)

# Convert the Timestamp column to datetime format
main_dataset['Timestamp'] = pd.to_datetime(main_dataset['Timestamp'], format='%a %b %d %H:%M:%S +0000 %Y')

# Task 2
# Remove duplicate tweets
main_dataset.drop_duplicates(inplace=True)

# Remove tweets that are not in English
for index, row in main_dataset.iterrows():
    if ld.classify(row['Tweet Text'])[0] != 'en':
        main_dataset.drop(index, inplace=True)
main_dataset.reset_index(drop=True, inplace=True)

# Remove URLs and emojis, create a new column for the cleaned text
main_dataset.insert(len(main_dataset.columns), 'cleaned_text', main_dataset['Tweet Text'])

for index, row in main_dataset.iterrows():
    main_dataset.at[index, 'cleaned_text'] = re.sub(r'http\S+', '', row['cleaned_text'])

for index, row in main_dataset.iterrows():
    main_dataset.at[index, 'cleaned_text'] = demoji.replace(row['cleaned_text'], '')

# Convert the cleaned text to lowercase
main_dataset['cleaned_text'] = main_dataset['cleaned_text'].str.lower()

# Task 3
# Extract hashtags and mentions from the cleaned text and create new columns for them
main_dataset.insert(len(main_dataset.columns), 'hashtags', "")
main_dataset.insert(len(main_dataset.columns), 'mentions', "")

# Add hashtags to the hashtags column
for index in main_dataset["cleaned_text"].index:
    hashtags = re.findall(r'#\w+', main_dataset.at[index, 'cleaned_text'])
    for hashtag in hashtags:
        main_dataset.at[index, 'hashtags'] += hashtag + " "

# Add mentions to the mentions column
for index in main_dataset["cleaned_text"].index:
    mentions = re.findall(r'@\w+', main_dataset.at[index, 'cleaned_text'])
    for mention in mentions:
        main_dataset.at[index, 'mentions'] += mention + " "

# Applying Named Entity Recognition (NER) using spaCy
nlp = en_core_web_sm.load()
main_dataset.insert(len(main_dataset.columns), 'named_entities', "")

# Save organizations and locations to the named_entities column
for index in main_dataset["cleaned_text"].index:
    doc = nlp(main_dataset.at[index, 'cleaned_text'])
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'GPE']:
            main_dataset.at[index, 'named_entities'] += ent.text + " "

# Task 4
G_users = nx.DiGraph()

# Add nodes for each unique user in the dataset
for index, row in main_dataset.iterrows():
    if row['mentions'] != "":
        mentions = row['mentions'].split()
        for mention in mentions:
            G_users.add_node(mention)

# Extract retweets for processing
main_dataset.insert(len(main_dataset.columns), 'rts', "")
for index, row in main_dataset.iterrows():
    rts = re.findall(r'rt @\w+', row['cleaned_text'])
    for rt in rts:
        main_dataset.at[index, 'rts'] += rt + " "

# Given that we don't know the original user that posted the tweet, we will consider mentions and retweets to be the following:
# Add edges for mentions, we consider a mention to happen when a retweeted tweet from a user includes another user's username. Example: RT @user1: @user2... This means that user1 is mentioning user2.
for index, row in main_dataset.iterrows():
    if row['mentions'] != "":
        mentions = row['mentions'].split()
        rts = row['rts'].split()
        if row['cleaned_text'].startswith('rt @') or row['cleaned_text'].startswith('"rt @'):
            for mention in range(len(mentions) - 1):
                G_users.add_edge(mentions[0], mentions[mention+1], type='mention')

# Add edges for retweets, we consider a retweet to happen when a retweeted tweet from a user includes another retweeted tweet. Example: RT @user1: RT @user2: ... This means that user1 is retweeting user2.
for index, row in main_dataset.iterrows():
    if row['rts'] != "":
        rts = row['rts'].split()
        for rt in range(len(rts) - 1):
            if rts[rt+1] != "rt" and rts[1]!=rts[rt+1]:
                G_users.add_edge(rts[1], rts[rt+1], type='retweet')

# Task 5
G_hashtags = nx.Graph()

# Add nodes for each unique hashtag in the dataset
for index, row in main_dataset.iterrows():
    if row['hashtags'] != "":
        hashtags = row['hashtags'].split()
        for hashtag in hashtags:
            G_hashtags.add_node(hashtag)

# Add edges for co-occurring hashtags. If they only occur once, their weight eill be 0
for index, row in main_dataset.iterrows():
    if row['hashtags'] != "":
        hashtags = row['hashtags'].split()
        combinations = itertools.combinations(hashtags, 2)
        for combination in combinations:
            if not G_hashtags.has_edge(combination[0], combination[1]):
                G_hashtags.add_edge(combination[0], combination[1])
                G_hashtags[combination[0]][combination[1]]['weight'] = 0
            else:
                G_hashtags[combination[0]][combination[1]]['weight'] += 1

# Task 6