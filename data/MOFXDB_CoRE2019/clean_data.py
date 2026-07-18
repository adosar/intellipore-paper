import pandas as pd

index_col = 'name'
df = pd.read_csv('raw.csv', index_col=index_col)

# Keep only Ar properties since others are mostly NaNs
keep_cols = ['pld', 'lcd', 'void_fraction', 'surface_area_m2g', 'surface_area_m2cm3']
keep_cols += [c for c in df.columns if 'Ar' in c]

df = df.loc[:, keep_cols]
df.dropna(inplace=True)

print(f'Size of clean dataset: {len(df)}')
print(f'Excluding materials of SpbNet benchmark dataset...')

spb_names = []
for ds in ['c3h6c3h8coremof', 'ch4n2']:
    for mode in ['validate', 'test']:
        names = pd.read_csv(
                f'/home/asarikas/databases/SpbNet/benchmark/{ds}/benchmark.{mode}.csv'
                ).cifid
        spb_names.extend(list(names))

df = df.loc[~df.index.isin(spb_names)]
assert df.index.isin(spb_names).sum() == 0

print(f'Size of final dataset: {len(df)}')

df.to_csv('clean.csv', index_label=index_col)
