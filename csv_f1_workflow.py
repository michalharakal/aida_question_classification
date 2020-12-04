import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


#%%


files = glob.glob('report/question_*.csv')

df = pd.concat([pd.read_csv(fp).assign(filename=os.path.basename(fp).split('.')[0]) for fp in files])
df.columns = ['category', 'precision', 'recall', 'f1', 'support', 'filename']

print(df)


#%%
df[['category', 'f1', 'filename']]

df_pv = df.pivot_table(columns=['filename'])

print(df_pv)