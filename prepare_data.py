# 1. Split data into train, val (test is same as val)
# 2. Standardize data with train statistics

import os
import json
import glob
import shutil

import torch
import pandas as pd
from torch.utils.data import random_split
from aidsorb.data import get_names
from sklearn.preprocessing import StandardScaler


def split_db(db, split_ratio):
    names = list(pd.read_csv(f'data/{db}/clean.csv', usecols=['name'])['name'])

    train, val = random_split(names, split_ratio, generator=rng)
    test = val  # Test set same as validation (for debugging purposes)

    for split, mode in zip((train, val, test), ('train', 'validation', 'test')):
        names = list(split)

        filename = f'data/{db}/{mode}.json'
        with open(filename, 'w') as fhand:
            json.dump(names, fhand, indent=4)
            print(f'Created file: \033[0;34m{filename}\033[0m')

    print(f'\033[32;1mData splitting for {db} completed!\033[0m')


def standardize_db(db):
    index_col = 'name'
    out_csv = f'data/{db}/standardized_clean.csv'

    df = pd.read_csv(f'data/{db}/clean.csv', index_col=index_col)

    train, val, test = (
            get_names(f'data/{db}/{mode}.json')
            for mode in ['train', 'validation', 'test']
            )

    scaler = StandardScaler()
    scaler.set_output(transform='pandas')
    scaler.fit(df.loc[train, :])  # Fit on train data

    df_scaled = scaler.transform(df)
    df_scaled.to_csv(out_csv, index_label=index_col)

    print(f'Created file: \033[0;34m{out_csv}\033[0m')
    print(f'\033[32;1mData standardization for {db} completed!\033[0m')


if __name__ == '__main__':

    rng = torch.Generator().manual_seed(42)

    db_dict = {
            'UO': [0.99, 0.01],
            'MOFXDB_hMOF': [0.975, 0.025],
            'hCOF': [0.96, 0.04],
            'MOFXDB_Tobacco': [0.9, 0.1],
            'MOFXDB_CoRE2019': [0.9, 0.1],
            }

    print('Prepare pretraining databases')

    for db, split_ratio in db_dict.items():
        split_db(db, split_ratio)
        standardize_db(db)
        print(f'\033[32;1mData preparation for {db} completed!\033[0m')
