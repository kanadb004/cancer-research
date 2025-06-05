import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(
        file_path, 
        sep='\t'
    )

    print(df.shape)
    df.head()

    df.columns

    # Set index and clean
    df = df.set_index("Gene Symbol")
    df = df.apply(pd.to_numeric, errors='coerce')

    # df = df.dropna() # df = df.dropna(how="any") # removes rows with any NaN entry
    df = df.dropna(how="all") # remove rows with with all NaN entries

    df = df.fillna(0)  # assume missing as normal copy number (neutral)

    df.columns

    # Verify all genes appear only once (are unique)
    print(df.index.value_counts())
    print(len(df.index.unique()))

    # Check for Isoforms
    # Check gene names ending with .<digit> (.1, .10, .99 so on)
    genes_with_suffix = df.index[df.index.str.contains(r"\.\d+$")]

    # Convert to list
    genes_with_suffix = genes_with_suffix.tolist()

    # Convert to Series to count occurrences
    gene_suffix_counts = pd.Series(genes_with_suffix).value_counts()

    # Top 50
    print(genes_with_suffix[:50])
    # print(gene_suffix_counts)

    # Round -> nearest whole number (1.8 -> 2, -1.4 -> -1) and 
    # Clip values to -2, -1, 0, 1, 2 (value < -2 -> -2, any value > 2 -> 2)
    df = df.round(0).clip(-2, 2).astype(int)
    df.head()

    return df


def plot_heatmap(data, index_labels, title, ylabel, cmap, norm, figsize=(20, 15), yticks_every=None):
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap=cmap, norm=norm, aspect='auto')

    if yticks_every is None:
        yticks_every = max(len(index_labels)//100, 1)

    plt.yticks(ticks=range(0, len(index_labels), yticks_every),
               labels=[index_labels[i] for i in range(0, len(index_labels), yticks_every)])
    plt.xticks([])

    cbar = plt.colorbar(ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([
        'Deep Deletion (-2)', 'Deletion (-1)', 'Normal (0)',
        'Gain (1)', 'Amplification (2)'
    ])
    cbar.set_label('Copy Number Alteration')

    plt.title(title)
    plt.xlabel("Samples (hidden)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def main():
    # File path to your data file
    file_path = "/Users/kanadb/Work/IIT-KGP Summer/cancer-research/datasets/UCSC-Xena-Copy-Number-Gene-Level-GISTIC2-Thresholded/TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes"

    df = load_and_preprocess_data(file_path)

    data = df.values
    data

    # Get rows (genes) where any sample has an individual value
    # axis=0 -> column wise, axis=1 -> row wise
    genes_with_amp = df[df.eq(-2).any(axis=1)].index.tolist() # check each row (gene) to see if 'any' sample has -2
    genes_with_amp[:5]

    # Create colormap
    cmap = mcolors.ListedColormap([
        '#800000',  # dark red (-2)
        '#ff6666',  # light red (-1)
        '#ffffff',  # white (0)
        '#6699ff',  # light blue (1)
        '#000080'   # dark blue (2)
    ])

    # Set bins to map colors to
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    # value = -2 -> falls in [-2.5, -1.5] -> norm maps it to index 0 -> cmap(0) = dark red
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # PLOT
    plt.figure(figsize=(20, 15))

    # Plot the heatmap
    plt.imshow(data, cmap=cmap, norm=norm, aspect='auto')

    # Show all gene names on Y-axis (clutter)
    # plt.yticks(ticks=range(len(df.index)), labels=df.index)

    # Compute how often to show gene names to avoid clutter
    yticks_every = max(len(df.index)//100, 1) # total number of genes (25000) // 100 = 250 -> every 250th gene
                                              # max -> if number of genes < 100 ( len(df.index) // 100 = 0 ) -> fall back to 1 as the minimum spacing

    # Set Y-axis ticks (gene name at every 'yticks_every' interval)
    plt.yticks(ticks=range(0, len(df.index), yticks_every),
               labels=[df.index[i] for i in range(0, len(df.index), yticks_every)])

    # Hide X-axis ticks (sample IDs)
    plt.xticks([])

    # Set colorbar (legend)
    cbar = plt.colorbar(ticks=[-2, -1, 0, 1, 2])

    # cbar.ax -> Axes object (a matplotlib Axes instance) inside the colorbar -> controls how the colorbar is drawn, including ticks and labels
    cbar.ax.set_yticklabels([
        'Deep Deletion (-2)', 'Deletion (-1)', 'Normal (0)',
        'Gain (1)', 'Amplification (2)'
    ])
    cbar.set_label('Copy Number Alteration')

    plt.title("Copy Number Alterations for All Genes Across Samples (-2 to 2)")
    plt.xlabel("Samples (hidden)")
    plt.ylabel("Genes (subset labeled)")

    plt.tight_layout()
    plt.show()

    # Compute CNA frequency per gene (% of samples altered)
    altered = df != 0  # boolean DataFrame where True means altered
    altered.head()

    cna_freq = altered.sum(axis=1) / df.shape[1] * 100
    cna_freq = cna_freq.sort_values(ascending=False)

    print("Top 20 genes by alteration frequency:")
    print(cna_freq.head(20))

    # Convert Series to Dataframe
    cna_freq_df = cna_freq.to_frame(name='Alteration Frequency')
    cna_freq_df.index.name = 'Gene Symbol'

    print(cna_freq_df.head())

    # Save to csv file
    cna_freq_df.to_csv("UCSC-GISTIC2-Thresholded-TCGA-BRCA-Copy-Number-Gene-Level_Altered-Gene-Frequency_Feature-Matrix.csv")

    # Feature matrix of altered genes
    feature_gene = altered.sum(axis=1) # axis=1 -> row wise
    feature_gene = feature_gene.sort_values(ascending=False)

    print("Feature Matrix of altered genes (top 20):")
    print(feature_gene.head(20))

    # Heatmap for top 100 altered genes
    top100_genes = cna_freq.head(100).index
    df_top100 = df.loc[top100_genes]

    plot_heatmap(
        df_top100.values,
        df_top100.index,
        "Heatmap: Top 100 Altered Genes by CNA Frequency",
        "Genes",
        cmap,
        norm
    )

    # Significantly amplified genes
    # Identify genes amplified in more than 5% of samples (copy number value == 2)
    amp_freq = (df == 2).sum(axis=1) / df.shape[1] * 100
    significant_amp_genes = amp_freq[amp_freq > 5].sort_values(ascending=False)
    print(f"Significantly amplified genes (>5% samples): {len(significant_amp_genes)}")
    print(significant_amp_genes.head(10))

    # Plot heatmap for top 50 significantly amplified genes if available
    top_amp_genes = significant_amp_genes.head(50).index
    df_amp = df.loc[top_amp_genes]

    plt.figure(figsize=(20, 10))
    plt.imshow(df_amp.values, cmap=cmap, norm=norm, aspect='auto')
    plt.yticks(ticks=range(len(df_amp.index)), labels=df_amp.index)
    plt.xticks([])

    cbar = plt.colorbar(ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([
        'Deep Deletion (-2)', 'Deletion (-1)', 'Normal (0)',
        'Gain (1)', 'Amplification (2)'
    ])
    cbar.set_label('Copy Number Alteration')

    plt.title("Heatmap: Top Significantly Amplified Genes (>5% samples)")
    plt.xlabel("Samples (hidden)")
    plt.ylabel("Genes")
    plt.tight_layout()
    plt.show()

    # Significantly deleted genes
    # Identify genes deleted in more than 5% of samples (copy number value == -2 -> deep deletion)
    del_freq = (df == -2).sum(axis=1) / df.shape[1] * 100
    significant_del_genes = del_freq[del_freq > 5].sort_values(ascending=False)
    print(f"Significantly deleted genes (>5% samples): {len(significant_del_genes)}")
    print(significant_del_genes.head(10))

    # Plot heatmap for top 50 significantly deleted genes if available
    top_del_genes = significant_del_genes.head(50).index
    df_del = df.loc[top_del_genes]

    plt.figure(figsize=(20, 10))
    plt.imshow(df_del.values, cmap=cmap, norm=norm, aspect='auto')
    plt.yticks(ticks=range(len(df_del.index)), labels=df_del.index)
    plt.xticks([])

    cbar = plt.colorbar(ticks=[-2, -1, 0, 1, 2])
    cbar.ax.set_yticklabels([
        'Deep Deletion (-2)', 'Deletion (-1)', 'Normal (0)',
        'Gain (1)', 'Amplification (2)'
    ])
    cbar.set_label('Copy Number Alteration')

    plt.title("Heatmap: Top Significantly Deleted Genes (>5% samples deep deletion)")
    plt.xlabel("Samples (hidden)")
    plt.ylabel("Genes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
