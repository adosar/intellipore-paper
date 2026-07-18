import pandas as pd

index_col = 'name'
df = pd.read_csv('raw.csv', index_col=index_col)

# Remove 1 COF with VF 2e+44 (outlier)
df = df.sort_values(by='void fraction [widom]', ascending=False)[1:]
df = df.select_dtypes(include=['number'])

print(f'Size of clean dataset: {len(df)}')
print(f'Excluding materials of SpbNet benchmark dataset...')

spb_names = []
for mode in ['validate', 'test']:
    names = pd.read_csv(
            f'/home/asarikas/databases/SpbNet/benchmark/cof/benchmark.{mode}.csv'
            ).cifid
    spb_names.extend(list(names))

df = df.loc[~df.index.isin(spb_names)]
assert df.index.isin(spb_names).sum() == 0

print(f'Size of final dataset: {len(df)}')

df.to_csv('clean.csv', index_label='name')
