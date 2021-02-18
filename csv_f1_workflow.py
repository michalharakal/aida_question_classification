import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import imgkit

# %%


svm_files = glob.glob('report/svm_*.csv')
lstm_files = glob.glob('report/LSTM*.csv')

# files = svm_files + lstm_files
files = svm_files

count_files = len(files)

df = pd.concat([pd.read_csv(fp).assign(filename=os.path.basename(fp).split(',')[0]) for fp in files])

print(df.columns)

df.columns = ['Category', 'precision', 'recall', 'f1', 'support', 'filename']

print(df)

# %%

df[['Category', 'f1', 'filename']]

df_pv = df.pivot_table(columns=['filename'], index=['Category']).round(2)

df_f1 = df_pv['f1']
# print(df_f1.columns)
df_f1.columns = ['Original TfidfVect', 'Original ng11',
                 'Original ng12', 'Original ng13',
                 'Original ng22', 'Original ng23',
                 'Original ng33',
                 'cleaned TfidfVect',
                 'cleaned ng11', 'cleaned ng12',
                 'cleaned ng13',
                 'cleaned ng22',
                 'cleaned ng23',
                 'cleaned ng33']

df_f1_transpose = df_f1.T

# %%

df_f1_transpose.plot()
plt.title('F1 Scores over different models and n-gram length')
plt.xlabel('original and cleaned questions')
plt.ylabel('values')
plt.xticks(rotation=15)

plot_name = 'svm_f1_results.png'
plt.savefig(f'./plots/{plot_name}')
plt.show()

# %%
df_f1.to_csv('f1.csv')

df_f1.to_html('f1.html')

# %%

cm = sns.light_palette("seagreen", as_cmap=True)
styled_table = df_f1.T.style.background_gradient(cmap=cm)
html = styled_table.render()
imgkit.from_string(html, './report/styled_table.png')
