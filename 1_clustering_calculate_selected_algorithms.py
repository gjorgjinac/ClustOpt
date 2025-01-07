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
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import math
def determine_number_of_clusters(X):

    # Instantiate the clustering model
    model = KMeans()
    
    # Use the KElbowVisualizer to find the optimal number of clusters
    cluster_options=[int(math.pow(2,x)) for x in range(2,10)]
    visualizer = KElbowVisualizer(model, k=cluster_options)
    
    # Fit the data to the visualizer
    visualizer.fit(X)
    
    # Render the plot
    visualizer.show()
    return visualizer.elbow_value_


def do_clustering(X):
    number_of_clusters=determine_number_of_clusters(X)
    model=KMeans(number_of_clusters)
    model.fit(X)
    clusters=model.predict(X)
    return clusters, model.cluster_centers_



all_removed_algorithms=get_removed_algorithms()
algorithms_of_interest=["ModifiedAEO",
"EnhancedAEO",
"OriginalAEO",
"AugmentedAEO",
"SHADE",
"SADE",
"JADE",
"BaseDE",
"OriginalWOA",
"HI_WOA"]
data_dir=f'../data/clustering_features_x_only_10_algorithms_kmeans_2pow_no_init/'
print(data_dir)
for dimension in [2,5,10]:
    x_columns=[f'x{i}' for i in range (0,dimension)]
    x_y_columns=x_columns+['raw_y']
    os.makedirs(f'{data_dir}/cluster_centers/dim_{dimension}',exist_ok=True)
    os.makedirs(f'{data_dir}/cluster_distributions/dim_{dimension}',exist_ok=True)
    os.makedirs(f'{data_dir}/clustering_results/dim_{dimension}',exist_ok=True)
    x_columns_scaled=[f'scaled_x{i}' for i in range(0,dimension)]
    x_y_columns_scaled=x_columns_scaled + ['scaled_raw_y']
    for p in tqdm(os.listdir(f'../data/processed/dim_{dimension}')):
        
        print(p)
        d=pd.read_csv(f'../data/processed/dim_{dimension}/{p}',compression='zip',index_col=0).query('evaluations<=500*@dimension')
        scaled=list(filter(lambda x: x.startswith('scaled_'), d.columns))
        d=d.query('algorithm not in @all_removed_algorithms.index and algorithm in @algorithms_of_interest')
        d=d.query('iteration>0')
        print(d['algorithm'].unique())
        d=rescale(d,x_y_columns)
    
        X=d[x_columns]
        print(X.shape)
        cluster_labels, cluster_centers=do_clustering(X)
        d['cluster']=cluster_labels
        cluster_distribution=d.groupby(['algorithm','run','iteration','cluster']).count()['evaluations'].unstack(level=-1).fillna(0)
        pd.DataFrame(cluster_centers, columns= x_columns_scaled).to_csv(f'{data_dir}/cluster_centers/dim_{dimension}/{p}')
        d.to_parquet(f'{data_dir}/clustering_results/dim_{dimension}/{p.replace(".csv",".parquet")}', compression='gzip')
        cluster_distribution.to_csv(f'{data_dir}/cluster_distributions/dim_{dimension}/{p}')
