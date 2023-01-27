import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_json('../data/News_Category_Dataset_v2.json', lines=True)
group_categories = True
if group_categories:
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
label_occurence = Counter(df['category']).most_common()
plt.rcParams["figure.figsize"] = (18.5, 10.5)
plt.barh([i[0] for i in label_occurence], [i[1] for i in label_occurence])
plt.xlabel('Occurrences')
plt.show()
