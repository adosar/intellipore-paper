#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd

pd.set_option("mode.copy_on_write", True)

df = pd.read_csv('raw.csv', index_col='MOFname')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df = df[df.void_fraction > 0]
df['log_CO2/N2_selectivity'] = np.log(df['CO2/N2_selectivity'])

df = df.select_dtypes(include=['number'])

vox_names = [i.removesuffix('.npy') for i in os.listdir('voxels_data')]
mat_names = list(df.index)
common_names = list(set(vox_names) & set(mat_names))
df = df.loc[common_names, :]

print(f'Size of clean dataset: {len(df)}')

df.to_csv('clean.csv', index_label='name')
