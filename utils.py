import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import mealpy

def get_algorithm_groups():
    optimizers=mealpy.get_all_optimizers()
    optimizer_group={}
    for k,v in optimizers.items():
        optimizer_group[k.replace('Original','Base')]=str(v).split('.')[1]
        optimizer_group[k]=str(v).split('.')[1]
    optimizer_group['ModifiedBA']='swarm_based'
    optimizer_group['SHADE']='evolutionary_based'
    
    return optimizer_group

def calculate_dynamorep_features(df, x_y_columns, id_columns):
    df=df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    grouped = df[x_y_columns + id_columns].groupby(id_columns)
    features = pd.concat([grouped.mean(), grouped.min(), grouped.max(), grouped.std()], axis=1)
    feature_names = [f'{j}_{i}' for j in
                         ['mean', 'min', 'max', 'std'] for i in x_y_columns]
    
    features.columns = feature_names
    return features

def rescale(d,x_y_columns):
    scaled=list(filter(lambda x: x.startswith('scaled_'), d.columns))
    d=d.drop(columns=scaled)
    d[[f'scaled_{x}' for x in x_y_columns]]=MinMaxScaler().fit_transform(d[x_y_columns])
    return d

def get_removed_algorithms():
    all_removed_algorithms=[]
    for dimension in [2,5,10]:
    
        for file in os.listdir(f'../data/removed_algorithms/dim_{dimension}'):
            if os.path.isfile(f'../data/removed_algorithms/dim_{dimension}/{file}'):
                all_removed_algorithms+=[pd.read_csv(f'../data/removed_algorithms/dim_{dimension}/{file}',index_col=0)]
            else:
                print(file)
            
    all_removed_algorithms=pd.concat(all_removed_algorithms)
    all_removed_algorithms.index.name='algorithm'
    all_removed_algorithms=all_removed_algorithms.reset_index().groupby('algorithm').sum()
    return all_removed_algorithms