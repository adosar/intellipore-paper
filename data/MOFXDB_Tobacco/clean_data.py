import pandas as pd

index_col = 'name'

df = pd.read_csv('raw.csv', index_col=index_col)
df.dropna(inplace=True)

# Convert to floats
df.drop(columns=['mofid'], inplace=True)
df.astype('float')

print(f'Size of clean dataset: {len(df)}')

df.to_csv('clean.csv', index_label=index_col)
