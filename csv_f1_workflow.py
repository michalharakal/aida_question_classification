import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


#%%

# df = pd.concat(map(pd.read_csv, glob.glob('report/question_*.csv')))
#df = pd.read_csv('report/question_pipe_cv.csv', )

files = glob.glob('report/question_*.csv')

df = pd.concat([pd.read_csv(fp).assign(New=os.path.basename(fp).split('.')[0]) for fp in files])

print(df)

#%%
