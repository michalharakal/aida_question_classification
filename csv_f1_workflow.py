import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import imgkit

#%%


svm_files = glob.glob('report/svm_*.csv')
lstm_files = glob.glob('report/LSTM*.csv')

# files = svm_files + lstm_files
files = svm_files

count_files = len(files)

df = pd.concat([pd.read_csv(fp).assign(filename=os.path.basename(fp).split(',')[0]) for fp in files])

print(df.columns)

df.columns = ['category', 'precision', 'recall', 'f1', 'support', 'filename']

print(df)


#%%

df[['category', 'f1', 'filename']]

df_pv = df.pivot_table(columns=['filename'], index=['category']).round(2)


df_f1 = df_pv['f1']

#%%
df_f1.to_csv('f1.csv')

df_f1.to_html('f1.html')

#%%

cm = sns.light_palette("seagreen", as_cmap=True)
styled_table = df_pv.style.background_gradient(cmap=cm)
html = styled_table.render()
imgkit.from_string(html, './report/styled_table.png')

