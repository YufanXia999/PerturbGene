import scanpy as sc
import anndata as ad

from .read import read_h5ad_file
import os
import matplotlib.pyplot as plt
import numpy as np


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'configs/credentials.json'

filename = "/home/rohola/codes/perturbgene/data/Tabula_Sapiens_ranked.h5ad"
combined_adata = read_h5ad_file(filename)

print(combined_adata)

data = combined_adata.X.data

# print(sum(data<0.1))
# print(len(data))
#
# plt.hist(data, bins='auto')  # You can specify the number of bins or use 'auto' for automatic binning
# plt.title('Histogram of Expression Values')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig("histogram.png", dpi=300)


# Calculate histogram bins and counts
counts, bin_edges = np.histogram(data, bins='auto')
# number of bins
num_bins = len(counts)
print(num_bins)

# To avoid taking the log of zero, replace any zero counts with a small number (e.g., 1) before taking the log
counts[counts == 0] = 1
log_counts = np.log(counts)

# Choose a plot style
plt.style.use('seaborn-darkgrid')  # Using seaborn-darkgrid for a nice background and grid

# Create a more beautiful bar plot
plt.figure(figsize=(10, 6))  # Set the figure size for better readability
plt.bar(bin_edges[:-1], log_counts, width=np.diff(bin_edges), color='skyblue', edgecolor='blue', alpha=0.7)

# Adding additional plot elements for better aesthetics
plt.title('Histogram of Expression Values with Logarithmic Frequencies', fontsize=15)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Log(Frequency)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# plt.grid(True, which="both", ls="--", linewidth=0.5)  # Adding a grid for better readability
plt.tight_layout()  # Adjust the layout to make room for the plot elements

plt.show()
plt.savefig("log_frequency_histogram_beautiful.png", dpi=300)
