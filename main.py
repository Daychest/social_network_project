# Project Group 3 - Introduction to Social Network Analysis
# Ivan Aguilar Garcia - Johannes Hyry - Veikka Päivärinta

# Imports
import random
from collections import Counter
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import langid as ld
import re
import demoji
import spacy as sp
import en_core_web_sm
import itertools
import sklearn
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
                G_users.add_edge(mentions[0], mentions[mention + 1], type='mention')

# Add edges for retweets, we consider a retweet to happen when a retweeted tweet from a user includes another retweeted tweet. Example: RT @user1: RT @user2: ... This means that user1 is retweeting user2.
for index, row in main_dataset.iterrows():
    if row['rts'] != "":
        rts = row['rts'].split()
        for rt in range(len(rts) - 1):
            if rts[rt + 1] != "rt" and rts[1] != rts[rt + 1]:
                G_users.add_edge(rts[1], rts[rt + 1], type='retweet')

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

# Load pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert cleaned tweets to embeddings
cleaned_tweet_list = main_dataset['cleaned_text'].tolist()
tweet_embeddings = model.encode(cleaned_tweet_list)
# Compute cosine similarity between tweet vectors
threshold = 0.7
cs = sklearn.metrics.pairwise.cosine_similarity(tweet_embeddings)
# Create edges only when similarity exceeds the threshold
G_semantic = nx.Graph()
for tweet in cleaned_tweet_list:
    G_semantic.add_node(tweet)

for i in range(len(cs)):
    for j in range(i + 1, len(cs)):
        if cs[i][j] > threshold:
            G_semantic.add_edge(cleaned_tweet_list[i], cleaned_tweet_list[j])
G_semantic.remove_edges_from(nx.selfloop_edges(G_semantic))
plt.figure()
plt.title("cosine similarity")
nx.draw(G_semantic, node_size=10)
plt.show()

# Task 7
user_hashtag_map = {}

# Maps users to hashtags
for _, row in main_dataset.iterrows():
    if row["mentions"] and row["hashtags"]:
        users = row["mentions"].split()
        hashtags = row["hashtags"].split()

        for user in users:
            if user not in user_hashtag_map:
                user_hashtag_map[user] = set()
            user_hashtag_map[user].update(hashtags)

print("USER HASHTAG MAPPING(20): ")
for i, (user, hashtags) in enumerate(user_hashtag_map.items()):
    if i == 20:
        break
    print(user, " :: ", hashtags)
# Nodes = users, Edges = hashtags, it tells which users use which hashtags

# Task 8
# compute centralities
degree_centrality = nx.degree_centrality(G_users)
betweenness_centrality = nx.betweenness_centrality(G_users)
eigenvector_centrality = nx.eigenvector_centrality(G_users)
# Store them in DataFrame and sort them
centralities = pd.DataFrame({
    "node": degree_centrality.keys(),
    "degree": degree_centrality.values(),
    "betweenness": betweenness_centrality.values(),
    "eigenvector": eigenvector_centrality.values()
})
print("TOP DEGREE CENTRALITY:")
top_degree_centrality = centralities.sort_values("degree", ascending=False).head(10)
print(top_degree_centrality)
print("")
print("TOP BETWEENNESS CENTRALITY:")
top_betweenness_centrality = centralities.sort_values("betweenness", ascending=False).head(10)
print(top_betweenness_centrality)
print("")
print("TOP EIGENVECTOR CENTRALITY:")
top_eigenvector_centrality = centralities.sort_values("eigenvector", ascending=False).head(10)
print(top_eigenvector_centrality)
print("")
print("highest degree centrality users are most connected")
print("Highest betweenness centrality users connects different communities the most")
print("Highest eigenvector centrality users are most influential users")
print("As seen in the ranks above, most connected users also usually connects different communities the most")
print("But most influential users arent the most connected users and their betweenness is low")

# Task 9

# remove self loops
G_users.remove_edges_from(nx.selfloop_edges(G_users))
# create k_core
k_core = nx.k_core(G_users)
print("users in k_core")
print(k_core.nodes())
plt.figure()
plt.title("k_core")
nx.draw(k_core, node_size=10)
plt.show()

# Task 10
# split data set into 6h time groups
time_groups = main_dataset.groupby(pd.Grouper(key="Timestamp", freq="6h"))
time_graphs = {}
time_graphs_overtime = {}
metrics = []
metrics_overtime = []
G_overtime = nx.Graph()
for time, group in time_groups:
    G = nx.Graph()

    users = group["mentions"].tolist()

    for user in users:
        us = user.split(" ")
        if not us:
            continue
        elif len(us) == 1:
            if us[0] not in G:
                G.add_node(us[0])
            if us[0] not in G_overtime:
                G_overtime.add_nodes(us[0])
        elif len(us) > 1:
            for i in range(1, len(us)):
                G.add_edge(us[0], us[i])
                G_overtime.add_edge(us[0], us[i])

    time_graphs[time] = G
    time_graphs_overtime[time] = G_overtime.copy()
    degr = nx.degree_centrality(G)
    degr_overtime = nx.degree_centrality(G_overtime)

    metrics.append({
        "time": time,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": sum(degr.values()) / len(degr) if degr else 0
    })

    metrics_overtime.append({
        "time": time,
        "nodes": G_overtime.number_of_nodes(),
        "edges": G_overtime.number_of_edges(),
        "avg_degree": sum(degr_overtime.values()) / len(degr_overtime) if degr_overtime else 0
    })


metrics_df = pd.DataFrame(metrics)
metrics_df_overtime = pd.DataFrame(metrics_overtime)
print("DATASET METRICS IN 6-HOUR TIME WINDOWS")
print(metrics_df.head(100))
print()
print("CENTRALITY AND NETWORK SIZE EVOLUTION:")
print(metrics_df_overtime.head(100))


# Task 11
analyzer = SentimentIntensityAnalyzer()

# Add a new column to the dataset called sentiment_score,
# the value of which is determined by the compound polarity score of the tweet text.
# Compound polarity is a value between [-1, 1], where -1 indicates negative tone and 1 indicates positive tone.
main_dataset['sentiment_score'] = main_dataset['cleaned_text'].apply(lambda input: analyzer.polarity_scores(input)['compound'])

# Grouping the dataset into one day groups. The sentiment score inside a day gets summed together.
groups = main_dataset.groupby(pd.Grouper(key='Timestamp', freq='D'))['sentiment_score'].sum()

if False:  # <- turning off drawing plots in order to speed up execution
    ax = groups.plot(y="sentiment_score")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Sentiment score sum")
    plt.show()


# Task 12
communities = list(nx.community.louvain_communities(G_users))

# Creating a counter for each community, so we can count which hashtags are most popular in each community.
counters = []
for _ in range(len(communities)):
    counters.append(Counter())

# Assigning each community its own random color.
colors = []
for _ in range(len(communities)):
    colors.append("#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

# The node_colors array determines which color gets assigned for each individual node.
node_colors = []
for node in G_users:
    for i, community in enumerate(communities):
        if node in community:
            node_colors.append(colors[i])
            counters[i].update(user_hashtag_map.get(node))
            break

# Draw the plot
if False:
    pos = nx.spring_layout(G_users, k=0.3, iterations=50, seed=2)
    nx.draw(G_users, pos=pos, node_size=100, node_color=node_colors, with_labels=False, font_size=12)

communities.sort(key=len, reverse=True)

print("5 most common hashtags in the 10 largest communities")
for i in range(10):
    S = G_users.subgraph(communities[i])
    print(f"#{i + 1} largest component, size: {len(communities[i])}, hashtags: {counters[i].most_common(5)}")


# Task 13
visitedEdges = dict()
for user in G_users:
    # Conducting a bfs search starting from each user, so that we can see which user has the best reach (cascade size)
    visitedEdges[user] = list(nx.bfs_edges(G_users, user))
# Sorting users based on the amount of other users they reached
visitedEdges = dict(sorted(visitedEdges.items(), key=lambda item: len(item[1]), reverse=True))

print("Top 5 users for cascade size")
for i, (key, value) in enumerate(visitedEdges.items()):
    print(f"User: {key}, Cascade size: {len(value)}, Traversed edges: {value}")
    if i == 5:
        break

# Checking longest path for cyclic graphs is slow, turning the graph into an acyclic one.
G_users_acyclic = G_users.copy()
while not nx.is_directed_acyclic_graph(G_users_acyclic):
    cycle = nx.find_cycle(G_users_acyclic)
    G_users_acyclic.remove_edge(*cycle[0])

# Get the longest path
path = nx.dag_longest_path(G_users_acyclic)
length = nx.dag_longest_path_length(G_users_acyclic)

print("Longest path:", path)
print("Length:", length)


# Task 14
model = ep.IndependentCascadesModel(G_users)
config = mc.Configuration()

# Starting information spread from seed node @buzzfeednews
config.add_model_initial_configuration("Infected", ["@buzzfeednews"])
for e in G_users.edges():
    config.add_edge_configuration("threshold", e, 1.0)  # <- spread always succeeds when treshold is 1

model.set_initial_status(config)

# Calculate iterations
iterations = model.iteration_bunch(10)

# Print iterations
for i, iteration in enumerate(iterations):
    if i != 0:  # Not printing iteration 0 due to verbosity
        print(f"Step {i}")
        print("Active nodes:", iteration["status"])


# Task 15

# Average path length won't work with directed graph
undirected_users = G_users.to_undirected()

nodes_removed = []
connected_component_amounts = []
largest_component_average_path_lengths = []

# Removing 10 most central nodes one by one according to eigenvector_centrality,
# and recalculating amount of connected components and average path length of largest connected component each time
for i, node in enumerate(top_eigenvector_centrality["node"]):
    nodes_removed.append(i)
    undirected_users.remove_node(node)
    connected_component_amounts.append(nx.number_connected_components(undirected_users))
    largest_connected_component = max(nx.connected_components(undirected_users), key=len)
    largest_component_average_path_lengths.append(nx.average_shortest_path_length(undirected_users.subgraph(largest_connected_component)))

# Displaying results as two plots
if True:
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(nodes_removed, connected_component_amounts)
    ax[0].set_xlabel("Central nodes removed")
    ax[0].set_ylabel("Connected components")
    ax[1].plot(nodes_removed, largest_component_average_path_lengths)
    ax[1].set_xlabel("Central nodes removed")
    ax[1].set_ylabel("Average path lengths of largest connected component")
    plt.show()
