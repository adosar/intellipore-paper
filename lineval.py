import os
import json
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_absolute_error as mae
from aidsorb.data import get_names

n_runs = 1
#train_sizes = [50, 100, 300, 500, 1000, 4950]
#train_sizes = [50, 100, 300, 500, 1000, 4950]
train_sizes = [5000]
target_tasks = [
        #'1bar',  '65bar',
        #'unitless_KH', 'heat_of_adsorption',
        'lowbar', 'highbar',
        #'C3H6_Henry_298K',
        #'C3H6_loadings',
        #'C3H8_Henry_298K',
        #'C3H8_loadings',
        #'C3H8_C3H6_Selectivity_1bar',
        #'C3H8_C3H6_Selectivity_infinite',
        #'TSN_S_1Bar',
        #'H2-77-2',
        #'N2-298-0.9',
        #'Xe-273-10',
        #'Kr-273-10',
        #'CO2-298-2.5',
        #'CH4-298-2.5',
        #'CO2-298K-2.5bar', 'H2-77K-2bar',
        #'KCH4',
        #'KN2',
        #'ch4n2ratio-0.1bar',
        #'ch4n2ratio-10bar',
        #'ch4n2ratio-1bar',
        #'ch4uptake-0.1bar',
        #'ch4uptake-10bar',
        #'ch4uptake-1bar',
        #'n2uptake-0.1bar',
        #'n2uptake-10bar',
        #'n2uptake-1bar',
        ]

random.seed(1)

#db = 'ppn'
db = 'cof'
#db = 'hmofheat'
#db = 'hmof'
#db = 'zeolite'
#db = 'ch4n2'
#db = 'c3h6c3h8coremof'
train_names, val_names, test_names = [
    #list(get_names(f'data/{db}/{stg}.json'))
    #list(get_names(f'/home/asarikas/databases/PMTransformer/ZEOs_PPNs_COFs/{db}/{stg}.json'))
    list(get_names(f'/home/asarikas/databases/SpbNet/benchmark/{db}/{stg}.json'))
    #list(get_names(f'/home/asarikas/databases/COFSpace/{stg}.json'))
    for stg in ['train', 'validation', 'test']
]

#df_labels = pd.read_csv(f'data/{db}/clean.csv', index_col='name')
#df_labels = pd.read_csv(f'/home/asarikas/databases/PMTransformer/ZEOs_PPNs_COFs/{db}/all.csv', index_col='name')
df_labels = pd.read_csv(f'/home/asarikas/databases/SpbNet/benchmark/{db}/all.csv', index_col='cifid')
#df_Z = pd.read_csv(f'embeddings/early_pretrain-{db}.csv', index_col='name')
#df_Z = pd.read_csv(f'embeddings/early_pretrain-Mercado.csv', index_col='name')
#df_Z = pd.read_csv(f'embeddings/initial_pretrain-Mercado_hCOF.csv', index_col='name')
#df_Z = pd.read_csv(f'embeddings/early_pretrain-{db}.csv', index_col='name')
#df_Z = pd.read_csv(f'embeddings/initial_pretrain-{db}.csv', index_col='name')

#model_name = 'initial_pretrain'
#model_name = 'hypo-only_early_pretrain'
#model_name = 'hypo-exp_early_pretrain'

for model_name in ['early_pretrain']:

    print(f"\n\n\033[1;31m{model_name}\033[0m\n\n")

    df_Z = pd.read_csv(f'embeddings/{model_name}-{db}.csv', index_col='name')

    #df_labels = pd.read_csv(f'/home/asarikas/databases/COFSpace/csv_data/H2_log_Henry.csv', index_col='ID')
    #df_Z = pd.read_csv(f'embeddings/initial_pretrain-COFSpace.csv', index_col='name')

    for target in (pbar := tqdm(target_tasks)):
        pbar.set_description(target)

        results = {s: {} for s in train_sizes}
        #results = {}

        y_test =  df_labels.loc[test_names, target].dropna()
        Z_test = df_Z.loc[y_test.index]

        y_fit = df_labels.loc[train_names, target].dropna()
        Z_train = df_Z.loc[y_fit.index]

        for size in train_sizes:
            scores = []
            for run in range(n_runs):
                #fit_names = random.sample(train_names, k=size)
                fit_names = random.sample(list(y_fit.index), k=size)

                X_train = Z_train.loc[fit_names]
                y_train = df_labels.loc[fit_names, target]

                reg = Ridge(0.1, solver='cholesky')
                #reg = RF(n_jobs=-1)
                reg.fit(X_train, y_train)
                scores.append(reg.score(Z_test, y_test))
                #scores.append(mae(y_true=y_test, y_pred=reg.predict(Z_test)))

            results[size]['mean'] = np.mean(scores)
            results[size]['std'] = np.std(scores)
                # Match the output returned by Lightning (list of dicts),
                # for easier visualization of the results.
                #results[size][run] = {
                #        'test_R2Score': reg.score(Z_test, y_test),
                #        'test_MeanAbsoluteError': mae(y_true=y_test, y_pred=reg.predict(Z_test))
                #        }

        print(json.dumps(results, indent=4))
        #dirname = f'tl_experiments/from_{source.replace("/", "_")}-to_{target.replace("/", "_")}-lineval/'
        #dirname = f'Tobacco_tl_experiments/from_{source.replace("/", "_")}-to_{target.replace("/", "_")}-lineval/'
        #if not os.path.isdir(dirname):
        #    os.mkdir(dirname)
        #with open(f'{dirname}/results.json', 'w') as fhand:
        #    json.dump(results, fhand, indent=4)
