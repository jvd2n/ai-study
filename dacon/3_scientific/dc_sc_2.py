from openbabel import pybel
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

def sdf_load(uid,train=True):
    if train:
        paths = f'./dacon/_data/3_scientific/train_sdf/train_{uid}.sdf'
    else:
        paths = f'./dacon/_data/3_scientific/test_sdf/test_{uid}.sdf'
    return paths

train_df = pd.read_csv(os.path.join('.', 'dacon', '_data', '3_scientific', 'train.csv'))

print(train_df.head())

mols = dict()

for n in tqdm(train_df.index):
    mol = [i for i in pybel.readfile('sdf', sdf_load(n))]
    if len(mol) > 0:
        mols[n] = mol[0]
        
set([i for i in range(train_df.shape[0])]) - set(mols.keys())

mols_df = pd.DataFrame().from_dict({n:v.calcdesc()for n,v in mols.items()}).transpose()
mols_df = mols_df.dropna(axis = 1)
mols_df.loc[:,'uid'] = [f'train_{n}' for n in mols.keys() ]

df = pd.merge(train_df,mols_df,'outer',on='uid').dropna()
df['y'] = df['S1_energy(eV)'] - df['T1_energy(eV)']

df = df.reset_index(drop=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

train_x = df.iloc[:,4:-1]
train_y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(train_x,train_y)
RF = RandomForestRegressor()
RF.fit(X_train,y_train)
y_pred = RF.predict(X_test)


print(f'MSE : {mean_squared_error(y_test,y_pred):.6} / R2 : {r2_score(y_test,y_pred):.6}')

submission = pd.read_csv('./dacon/_data/3_scientific/sample_submission.csv')
submission['ST1_GAP(eV)'] = y_pred
print(submission.head())

submission.to_csv('./dacon/_output/3_scientific/dc_sc_2_out.csv', index=False)