from torch.utils.data import Dataset
import pandas as pd
import torch


class NewsDataset(Dataset):
    def __init__(self, json_path):
        self.df = pd.read_json(json_path, lines=True)[['category', 'headline', 'short_description']]  # Load data, but only keep columns of interest
        self.df = self.df.dropna(axis=0)  # Remove rows with no category or headline
        self.df = self.df.loc[self.df['headline'].str.len() > 0]  # Remove rows where headline is empty string
        self.df = self.df.loc[self.df['headline'].str.len() <= 120]  # Remove rows where length of headline is above 120
        self.df = self.df.loc[self.df['short_description'].str.len() > 0]  # Remove rows where short_description is empty string
        self.df = self.df.loc[self.df['short_description'].str.len() < 320]  # Remove rows where length of short_description is above 300
        self.df = self.df.loc[(self.df['short_description'].str.len() < 120) | (131 < self.df['short_description'].str.len())]
        self.df['category'] = self.df['category'].replace({"ARTS & CULTURE": "CULTURE & ARTS",
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
        self.df['headline'] = self.df['headline'].str.lower()  # All headlines in lower case
        self.df['short_description'] = self.df['short_description'].str.lower()  # All headlines in lower case
        self.df['concatenation'] = self.df['headline'] + " " + self.df['short_description']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_row = self.df.iloc[idx, :]
        data_point_category = data_row[0]
        #data_point_headline = data_row[1]
        data_point_concatenation = data_row[3]
        return data_point_category, data_point_concatenation


def loader_for_vocab():
    data_path = '../data/News_Category_Dataset_v2.json'
    dataset = NewsDataset(data_path)
    loader_ = torch.utils.data.DataLoader(dataset)
    return loader_
