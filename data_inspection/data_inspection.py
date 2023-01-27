import numpy as np
import pandas as pd
from collections import Counter

# This file is not really used anymore. Only the plots in the [COLUMN] distribution is used

df = pd.read_json('../data/News_Category_Dataset_v2.json', lines=True)[['category', 'headline']]
pd.set_option('display.max_columns', 100)
df = df.loc[df['headline'].str.len() > 0]
df = df.dropna(axis=0)  # Remove rows with no category or headline
df = df.loc[df['headline'].str.len() <= 120]  # Remove rows where length of headline is above 120
df = df.loc[df['short_description'].str.len() > 0]
df = df.loc[df['short_description'].str.len() <= 300]  # Remove rows where length of headline is above 300
df['category'] = df['category'].replace({"ARTS & CULTURE": "CULTURE & ARTS",
                                                           "HEALTHY LIVING": "WELLNESS",
                                                           "QUEER VOICES": "VOICES",
                                                           "BUSINESS": "BUSINESS & FINANCES",
                                                           "PARENTS": "PARENTING",
                                                           "BLACK VOICES": "VOICES",
                                                           "THE WORLDPOST": "WORLD NEWS",
                                                           "STYLE": "STYLE & BEAUTY",
                                                           "GREEN": "ENVIRONMENT",
                                                           "TASTE": "FOOD & DRINK",
                                                           "WORLDPOST": "WORLD NEWS",
                                                           "SCIENCE": "SCIENCE & TECH",
                                                           "TECH": "SCIENCE & TECH",
                                                           "MONEY": "BUSINESS & FINANCES",
                                                           "ARTS": "CULTURE & ARTS",
                                                           "COLLEGE": "EDUCATION",
                                                           "LATINO VOICES": "VOICES",
                                                           "FIFTY": "MISCELLANEOUS",
                                                           "GOOD NEWS": "MISCELLANEOUS"})  # Group some categories
df['headline'] = df['headline'].str.lower()  # All headlines in lower case
df['short_description'] = df['short_description'].str.lower()  # All headlines in lower case
#print(df.head())

label_occurence = Counter(df['category']).most_common()
amount_of_data = len(df)
amount_of_labels = len(set(df['category']))
list_of_lengths = [len(i) for i in df['headline']]
mean_len_headline = round(np.mean(list_of_lengths), 2)
mean_words_headline = round(np.mean([i.count(" ") + 1 for i in df['headline']]), 2)
max_len_headline = max(list_of_lengths)
min_len_headline = min(list_of_lengths)
label_occurence_ints = [i[1] for i in label_occurence]
print(label_occurence)
print(f'Average label occurence: {np.mean(label_occurence_ints)}')
print(f'Median label occurence: {np.median(label_occurence_ints)}')

print(f"""\nIn this project we are interested in predicting the category of 
online news through their headlines. In our dataset there are {amount_of_data} 
different news articles with in total {amount_of_labels} different news 
categories. The length of the average headline is {mean_len_headline} 
characters, and {mean_words_headline} in terms of average number of words. The 
maximum headline length in terms of characters is {max_len_headline}, while the
minimum is {min_len_headline}.""")
