import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_json('../data/News_Category_Dataset_v2.json', lines=True)[['short_description', 'link']]
pd.set_option('display.max_columns', 100)
df = df.loc[df['short_description'].str.len() > 0]
df = df.dropna(axis=0)  # Remove rows with no category or headline
df = df.loc[((df['short_description'].str.len() <= 119) | (df['short_description'].str.len() > 131)) & (df['short_description'].str.len() < 320)]  # Remove rows where length of short_description is above 300
df['short_description'] = df['short_description'].str.lower()  # All headlines in lower case



pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)
tmp = df.loc[df['short_description'].str.len() == 150]
_ = 5
print(tmp.iloc[_,0])
print(tmp.iloc[_,1])


list_of_lengths = [len(i) for i in df['short_description']]
len(list_of_lengths)
length_occurence = Counter(list_of_lengths).most_common()

plt.rcParams["figure.figsize"] = (18.5, 10.5)
plt.hist(list_of_lengths, bins=318)
plt.xlabel('Length of short_descriptions')
plt.ylabel('Occurrences')
plt.show()
