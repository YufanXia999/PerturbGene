import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("/home/yufan") 
import scanpy as sc
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import pandas as pd
from scipy.stats import ttest_ind

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
    cal_r = None
    if cal_r:
        # Calculate Pearson correlation
        r, _ = pearsonr(ground_truth, predictions)

    # Calculate the z-scored age gap - (age gap - mean of age gap)/sd of age gap
    age_gap = predictions - ground_truth
    mean_age_gap = np.mean(age_gap)
    std_age_gap = np.std(age_gap)
    z_scored_age_gap = (age_gap - mean_age_gap) / std_age_gap

    return {
        "ground_truth": ground_truth,
        "prediction": predictions,
        "z_score":z_scored_age_gap,
        "r_value":r
    }

def plot_z_score(data_dict, fig_name = "plots/age_prediction_z_score.png",show_fig=False):
    ground_truth = data_dict["ground_truth"]
    prediction = data_dict["prediction"]
    z_scored_age_gap = data_dict["z_score"]
    r = data_dict["r_value"]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=ground_truth, y=prediction, hue=z_scored_age_gap, palette='coolwarm', edgecolor='k', alpha=0.6, legend=False)
    plt.plot([ground_truth.min(), ground_truth.max()], [ground_truth.min(), ground_truth.max()], 'k--', lw=2)
    norm = plt.Normalize(z_scored_age_gap.min(), z_scored_age_gap.max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='z-scored age gap')
    plt.xlabel('chronological age',fontsize=18)
    plt.ylabel('predicted age',fontsize=18)
    plt.title(f'age prediction',fontsize=18)
    plt.text(0.05, 0.95, f'r = {r:.2f}', transform=plt.gca().transAxes, 
            fontsize=18, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))
    plt.savefig(fig_name, dpi=300, bbox_inches='tight') 
    if show_fig:
        plt.show()
    else:
        plt.close()

def cal_cosine_sim(matrixA, matrixB=None, format="pairweise"): # choose from pariweise, general
    # calculate pairweise cosine similarity, e.g. one cell's cell type embedding with its own age embedding
    similarity_matrix = cosine_similarity(matrixA, matrixB)
    if matrixB is None:
        return similarity_matrix[np.tril_indices_from(similarity_matrix, k = -1)]
    elif format == "pairweise":
        assert matrixA.shape == matrixB.shape
        return np.diag(similarity_matrix)
    elif format == "general":
        return similarity_matrix.flatten()

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
    t_stat, p_value = ttest_ind(lst1, lst2, equal_var=False)
    if stars:
        return p_2_sign(p_value)
    else:
        return p_value
    

def cal_sim_between_two(matrix_gt, matrix_ref=None, age_group_list=None, ref_age_group_list = None,#for gene-gene, situations like the shape of two gene embs are diff, thus age lists are diff as well
                        age_order = None, by_age_group = True, format="pairweise", ref_same=False):
    if format == "pairweise":
        assert matrix_gt.shape == matrix_ref.shape
    assert matrix_gt.shape[0] == len(age_group_list)
    if by_age_group:
        assert age_group_list is not None, "age_group_list cannot be None when by_age_group is True"
        sims_by_age = {}
        if ref_same:
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
                    similarities = cal_cosine_sim(matrix_gt[indices], None, format="general") # to cal inner each age group gene embedding variance
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
        similarities = cal_cosine_sim(matrix_gt, matrix_ref, format="")
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
    return cal_sim_between_two(matrix_gene, matrix_ref=None, age_group_list = age_group_list, age_order=age_order, by_age_group = True, format="general")
