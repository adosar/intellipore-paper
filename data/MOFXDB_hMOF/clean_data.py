import pandas as pd
import numpy as np

index_col = 'name'
df = pd.read_csv('raw.csv', index_col=index_col)

# These columns have a lot of NaNs
df.drop(columns=['Nitrogen_1_298K_0.09bar_kj/mol', 'Nitrogen_1_298K_0.9bar_kj/mol', 'mofid'], inplace=True)

df.dropna(inplace=True)

print(f'Size of clean dataset: {len(df)}')
print(f'Excluding materials of SpbNet benchmark dataset...')

spb_names = []
for ds in ['hmof', 'hmofheat']:
    for mode in ['validate', 'test']:
        names = pd.read_csv(
                f'/home/asarikas/databases/SpbNet/benchmark/{ds}/benchmark.{mode}.csv'
                ).cifid
        spb_names.extend(list(names))

df = df.loc[~df.index.isin(spb_names)]
assert df.index.isin(spb_names).sum() == 0

print(f'Size of final dataset: {len(df)}')

df.to_csv('clean.csv', index_label=index_col)
