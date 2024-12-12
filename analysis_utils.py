import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/yufan") 
import scanpy as sc
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from scipy.stats import ttest_ind, zscore, pearsonr


def label_to_float(label):
    label = label.replace("_"," ").replace("-",' ')
    if 'year' in label:
        if "-" in label:
            return float(label.split(' ')[0])
        elif label == "under 1 year old human stage":
            return 1.0
        else:
            return float(label.split(' ')[0])
    elif 'month' in label:
        if "LMP" not in label:
            return float(label.split(' ')[0])/12
        elif label == "eighth LMP month human stage":
            return 8/12
        elif label == "fifth LMP month human stage":
            return 5/12
    elif 'week' in label:
        return float(label.split(" ")[0].replace("th",'').replace("st",""))/52
    return -1.0

def cal_z_score(ground_truth, predictions, cal_r = True):
    r = None
    if cal_r:
        r, _ = pearsonr(ground_truth, predictions)

    age_gap = predictions - ground_truth
    z_scored_age_gap = zscore(age_gap)

    return {
        "ground_truth": ground_truth,
        "prediction": predictions,
        "z_score":z_scored_age_gap,
        "r_value":r
    }

def cal_cosine_sim(matrixA, matrixB=None, format="pairweise"): # choose from pariweise, general
    # calculate pairweise cosine similarity, e.g. one cell's cell type embedding with its own age embedding
    similarity_matrix = cosine_similarity(matrixA, matrixB)
    if matrixB is None and format == "half": # to compute variance innerhalb each age group
        return similarity_matrix[np.tril_indices_from(similarity_matrix, k = -1)]
    elif format == "pairweise": # e.g. for the case to compare cell type embedding corresponding to its age embedding, thus one similarity value for each cell
        assert matrixA.shape == matrixB.shape
        return np.diag(similarity_matrix)
    elif format == "general": # e.g. for the case to get cosine similarity of gene embeddings in each age group, thus similarity values for each sample in A == number of samples in B
        return similarity_matrix.flatten()
    else:
        return

def p_2_sign(p):
    """Convert p-value to significance markers."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def cal_p(lst1, lst2, stars = True):
    t_stat, p_value = ttest_ind(lst1, lst2, equal_var=False) # weich t-test bc unequal sample sizes
    if stars:
        return p_2_sign(p_value)
    else:
        return p_value
    

def cal_sim_between_two(matrix_gt, matrix_ref=None, age_group_list=None, ref_age_group_list = None, # just for gene-gene similarity, diff. age lists for each gene
                        age_order = None, by_age_group = True, format="pairweise", ref_same=False):
    if format == "pairweise":
        assert matrix_gt.shape == matrix_ref.shape
    assert matrix_gt.shape[0] == len(age_group_list)
    if by_age_group:
        assert age_group_list is not None, "need the list of ages"
        sims_by_age = {}
        if ref_same: # always same reference matrix, e.g. for the case compare embeddings of each group always to the youngest group
            for age_group in age_order:
                indices = [index for index, label in enumerate(age_group_list) if label == age_group]
                # if format == "general": geneA to youngest group
                similarities = cal_cosine_sim(matrix_gt[indices], matrix_ref, format=format) 
                assert (matrix_gt[indices].shape[0] * matrix_ref.shape[0]) == len(similarities)
                sims_by_age[age_group] = similarities
            return sims_by_age
        else:
            if age_order is None:
                age_order = set(age_group_list)
            for age_group in age_order:
                indices = [index for index, label in enumerate(age_group_list) if label == age_group]
                if matrix_ref is None:
                    similarities = cal_cosine_sim(matrix_gt[indices], None, format=format) # to cal inner each age group gene embedding variance
                else:
                    # if format == "pairweise": the most used one, gene-age,tissue-age, cell_type-age
                    # if format == "general": geneA to geneB
                    if format == "general":
                        indices2 = [index for index, label in enumerate(ref_age_group_list) if label == age_group]
                        if len(indices) != 0 and len(indices2) != 0:
                            similarities = cal_cosine_sim(matrix_gt[indices], matrix_ref[indices2], format=format) 
                            assert (matrix_gt[indices].shape[0] * matrix_ref[indices2].shape[0]) == len(similarities)
                        else:
                            similarities = None
                    else:
                        similarities = cal_cosine_sim(matrix_gt[indices], matrix_ref[indices], format=format) 
                sims_by_age[age_group] = similarities
            return sims_by_age
    else:
        similarities = cal_cosine_sim(matrix_gt, matrix_ref, format=format) #common use cases
        return similarities
    
def sim_gene_age(matrix_gene, matrix_age, age_group_list, age_order = None):
    assert age_group_list is not None
    return cal_sim_between_two(matrix_gene, matrix_age, age_group_list,age_order=age_order, by_age_group = True, format="pairweise")

def sim_celltype_age(matrix_celltype, matrix_age, age_group_list, age_order = None):
    assert age_group_list is not None
    return cal_sim_between_two(matrix_celltype, matrix_age, age_group_list, age_order=age_order, by_age_group = True, format="pairweise")

def sim_tissue_age(matrix_tissue, matrix_age, age_group_list, age_order = None):
    assert age_group_list is not None
    return cal_sim_between_two(matrix_tissue, matrix_age, age_group_list, age_order=age_order, by_age_group = True, format="pairweise")

def sim_gene_gene(matrix_geneA, matrix_geneB, age_group_list, ref_age_group_list, age_order = None):
    assert age_group_list is not None
    return cal_sim_between_two(matrix_geneA, matrix_geneB, age_group_list, ref_age_group_list, age_order=age_order, by_age_group = True, format="general")

def sim_gene_youngest(matrix_gene, age_group_list, ref_age = None, age_order = None):
    assert age_group_list is not None and ref_age is not None
    indices = [index for index, label in enumerate(age_group_list) if label == ref_age]
    matrix_ref = matrix_gene[indices]
    return cal_sim_between_two(matrix_gene, matrix_ref, age_group_list, age_order=age_order, by_age_group = True, format="general", ref_same=True)

def sim_gene_each_group(matrix_gene, age_group_list, age_order = None):
    assert age_group_list is not None
    return cal_sim_between_two(matrix_gene, matrix_ref=None, age_group_list = age_group_list, age_order=age_order, by_age_group = True, format="half")

def top_n_genes_by_age(sim_gene_age_dict: dict, age_counts_for_tissue: dict, threshold=0.01, top_n=3):
    tissues = sim_gene_age_dict.keys()
    top_n_genes_by_tissue_by_age = {}
    for tissue in tissues:
        ages = sim_gene_age_dict[tissue].keys()
        top_n_genes_by_tissue_by_age[tissue] = {}
        for age in ages:
            genes = sim_gene_age_dict[tissue][age].keys()
            top_n_genes = {}
            for gene in genes:
                sims = sim_gene_age_dict[tissue][age][gene]
                if len(sims) >= (threshold * age_counts_for_tissue[tissue][age]):
                    top_n_genes[gene] = np.mean(sims)
            top_n_genes_by_tissue_by_age[tissue][age] = dict(sorted(top_n_genes.items(), key=lambda item: item[1], reverse=True)[:top_n])
    return top_n_genes_by_tissue_by_age

def plot_clock_tissue(tissue,tissue_gene_dict):
    sizes = [1] * 8  
    labels = ["20y","30y","40y","50y","60y","70y","80y","10y"]
    age_groups = ["10-20","20-30","30-40","40-50","50-60","60-70","70-80",">80"]
    gene_list = []
    for age_group in age_groups:
        if age_group in tissue_gene_dict[tissue]:
            genes = tissue_gene_dict[tissue][age_group]
            gene_list.append('\n'.join(genes))  
        else:
            gene_list.append("")

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0) 
    ax.pie(sizes,  labels = gene_list,startangle=90, counterclock=False, colors=sns.color_palette('Set2'))
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    theta = np.linspace(0, 2 * np.pi, len(sizes) + 1)[1:] 
    for i in range(len(sizes)):
        angle = -(theta[i] - np.pi/2)
        x_label = 0.6 * np.cos(angle) 
        y_label = 0.6 * np.sin(angle)  
        ax.text(x_label, y_label, labels[i], ha='center', va='center', fontsize=12,)
    ax.text(0, 0, tissue, ha='center', va='center', fontsize=16, fontweight='bold')
    return fig

