import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

for dimension in [2,5,10]:
    print("Dimension ", dimension)

    directory=f'../data/clustering_features/cluster_distributions/dim_{dimension}'
    
    trajectory_similarities=[]
    for file in tqdm(os.listdir(directory)):
        file_loc=f'{directory}/{file}'
        if os.path.isfile(file_loc):
            df=pd.read_csv(file_loc,index_col=[0,1,2])
            d_algorithm_run=df.unstack(level=-1).dropna()
            #d_algorithm_run=pd.DataFrame(MinMaxScaler().fit_transform(d_algorithm_run), index=d_algorithm_run.index, columns=d_algorithm_run.columns).dropna()
            cs=pd.DataFrame(cosine_similarity(d_algorithm_run), index=d_algorithm_run.index, columns=d_algorithm_run.index)
            c=cs.reset_index().rename(columns={'algorithm':'algorithm2', 'run':'run2'}).melt(id_vars=[('algorithm2',  ''),('run2',  '')], value_vars=list(cs.columns)).rename(columns={('algorithm2',  ''): 'algorithm2', ('run2',''):'run2'})
            trajectory_similarities+=[c.assign(problem=file.replace('.csv',''))]
    
    trajectory_similarities=pd.concat(trajectory_similarities)
    t=trajectory_similarities.groupby('algorithm').count()['run']
    t_max=t.max()
    algorithms_to_remove=list(t[t<t_max].index)
    trajectory_similarities=trajectory_similarities.query('algorithm not in @algorithms_to_remove and algorithm2 not in @algorithms_to_remove')
    trajectory_similarities['problem_class']=[int(tt.split('_')[0].replace('F','')) for tt in trajectory_similarities['problem']]
    trajectory_similarities['instance']=[int(tt.split('_')[1].replace('I','')) for tt in trajectory_similarities['problem']]
    cross_algorithm_similarity=trajectory_similarities.query('run==run2').groupby(['algorithm','algorithm2'])['value'].mean()
    
    optimizer_group=get_algorithm_groups()
    
    unique_optimizer_groups=list(set(optimizer_group.values()))
    for aggregation in ['mean','median']:
        tt=trajectory_similarities.query('run==run2').groupby(['algorithm','algorithm2'])['value'].mean().dropna() if aggregation=='mean' else trajectory_similarities.query('run==run2').groupby(['algorithm','algorithm2'])['value'].median().dropna()
        sorted_sim=tt.to_frame().reset_index().query('algorithm!=algorithm2').sort_values(by='value',ascending=False)
        sorted_sim['algorithm_group']=sorted_sim['algorithm'].apply(lambda x: optimizer_group[x])
        sorted_sim['algorithm2_group']=sorted_sim['algorithm2'].apply(lambda x: optimizer_group[x])
        sorted_sim.to_csv(f'../data/clustering_features/algorithm_pairwise_similarity/algorithm_{aggregation}_similarity_{dimension}D.csv')