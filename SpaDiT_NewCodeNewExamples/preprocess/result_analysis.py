import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, \
    normalized_mutual_info_score


def clustering_metrics(adata, target, pred, mode="AMI"):
    if(mode=="AMI"):
        ami = adjusted_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("AMI ",ami)
        return ami
    elif(mode=="ARI"):
        ari = adjusted_rand_score(adata.obs[target], adata.obs[pred])
        print("ARI ",ari)
        return ari
    elif(mode=="Homo"):
        homo = homogeneity_score(adata.obs[target], adata.obs[pred])
        print("Homo ",homo)
        return homo
    elif(mode=="NMI"):
        nmi = normalized_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("NMI ", nmi)
        return nmi

def get_N_clusters(adata, n_cluster, cluster_method='louvain', range_min=0, range_max=3, max_steps=30, tolerance=0):

    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        if cluster_method=='leiden':
            sc.tl.leiden(adata, resolution=this_resolution)
        if cluster_method=='louvain':
            sc.tl.louvain(adata, resolution=this_resolution)
        this_clusters = adata.obs[cluster_method].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f."%(n_cluster, this_resolution))
            return adata
        this_step += 1

    print('Cannot find the number of clusters.')
    return adata

