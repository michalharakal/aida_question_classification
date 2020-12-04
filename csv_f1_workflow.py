import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import imgkit

#%%

files = glob.glob('report/question_*.csv')
lstm_files = glob.glob('report/LSTM*.csv')
files = files + lstm_files

df = pd.concat([pd.read_csv(fp).assign(filename=os.path.basename(fp).split(',')[0]) for fp in files])

print(df.columns)

df.columns = ['category', 'precision', 'recall', 'f1', 'support', 'filename']

print(df)


#%%

df[['category', 'f1', 'filename']]

df_pv = df.pivot_table(columns=['filename'], index=['category']).round(2)

print(df_pv)

#%%

cm = sns.light_palette("seagreen", as_cmap=True)
styled_table = df_pv.style.background_gradient(cmap=cm)
html = styled_table.render()
imgkit.from_string(html, './report/styled_table.png')

