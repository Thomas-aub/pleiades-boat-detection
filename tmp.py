import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. The input string
raw_text = """
══════════════════════════════════════════════════════════════════════
  📐 PER-CLUSTER AVERAGES (Total Variation & BRISQUE)
══════════════════════════════════════════════════════════════════════
  Group - Cluster      | Mean TV      | Mean BRISQUE
----------------------------------------------------------------------
  Gen - Cluster 0      | 0.046957     | 115.41      
  Gen - Cluster 1      | 0.498506     | 80.63       
  Gen - Cluster 2      | 0.097387     | 114.99      
  PL - Cluster 0       | 0.030513     | 111.92      
  PL - Cluster 1       | 0.005767     | 116.38      
  PL - Cluster 2       | 0.001925     | 111.49      
  PLN - Cluster 0      | 0.026162     | 115.76      
  PLN - Cluster 1      | 0.003421     | 107.10      
  PLN - Cluster 2      | 0.055259     | 104.02      

══════════════════════════════════════════════════════════════════════════════════════════
  🔬 PAIRWISE QUALITY GRID (Mean W1 & Mean FFT ΔdB)
     * Sampled up to 200 random pairs per combination to ensure speed.
══════════════════════════════════════════════════════════════════════════════════════════
  Source Cluster       | Target Cluster       | Mean W1      | Mean FFT ΔdB
------------------------------------------------------------------------------------------
  Gen - Cluster 0      | Gen - Cluster 0      | 198.79       | 20.9712      
  Gen - Cluster 0      | Gen - Cluster 1      | 798.59       | 35.0450      
  Gen - Cluster 0      | Gen - Cluster 2      | 321.30       | 20.4373      
  Gen - Cluster 0      | PL - Cluster 0       | 134.19       | 17.5809      
  Gen - Cluster 0      | PL - Cluster 1       | 85.03        | 17.3322      
  Gen - Cluster 0      | PL - Cluster 2       | 92.17        | 24.8917      
  Gen - Cluster 0      | PLN - Cluster 0      | 113.95       | 18.0716      
  Gen - Cluster 0      | PLN - Cluster 1      | 81.61        | 14.9339      
  Gen - Cluster 0      | PLN - Cluster 2      | 104.86       | 21.5288      
  Gen - Cluster 1      | Gen - Cluster 1      | 492.46       | 9.0416       
  Gen - Cluster 1      | Gen - Cluster 2      | 806.13       | 20.6041      
  Gen - Cluster 1      | PL - Cluster 0       | 842.17       | 23.5574      
  Gen - Cluster 1      | PL - Cluster 1       | 840.87       | 39.7603      
  Gen - Cluster 1      | PL - Cluster 2       | 795.64       | 50.5506      
  Gen - Cluster 1      | PLN - Cluster 0      | 817.93       | 25.5670      
  Gen - Cluster 1      | PLN - Cluster 1      | 849.09       | 38.4322      
  Gen - Cluster 1      | PLN - Cluster 2      | 832.16       | 19.1488      
  Gen - Cluster 2      | Gen - Cluster 2      | 486.37       | 13.9663      
  Gen - Cluster 2      | PL - Cluster 0       | 269.93       | 11.7236      
  Gen - Cluster 2      | PL - Cluster 1       | 325.02       | 22.0829      
  Gen - Cluster 2      | PL - Cluster 2       | 313.16       | 36.9169      
  Gen - Cluster 2      | PLN - Cluster 0      | 212.84       | 12.3443      
  Gen - Cluster 2      | PLN - Cluster 1      | 278.87       | 20.9459      
  Gen - Cluster 2      | PLN - Cluster 2      | 257.50       | 10.7268      
  PL - Cluster 0       | PL - Cluster 0       | 19.85        | 3.2545       
  PL - Cluster 0       | PL - Cluster 1       | 30.77        | 17.6400      
  PL - Cluster 0       | PL - Cluster 2       | 38.48        | 31.3463      
  PL - Cluster 0       | PLN - Cluster 0      | 37.05        | 6.6559       
  PL - Cluster 0       | PLN - Cluster 1      | 36.49        | 16.6687      
  PL - Cluster 0       | PLN - Cluster 2      | 31.98        | 6.1853       
  PL - Cluster 1       | PL - Cluster 1       | 13.79        | 9.7667       
  PL - Cluster 1       | PL - Cluster 2       | 9.13         | 17.6657      
  PL - Cluster 1       | PLN - Cluster 0      | 46.86        | 13.3855      
  PL - Cluster 1       | PLN - Cluster 1      | 11.74        | 9.1651       
  PL - Cluster 1       | PLN - Cluster 2      | 45.82        | 21.4715      
  PL - Cluster 2       | PL - Cluster 2       | 2.66         | 23.4331      
  PL - Cluster 2       | PLN - Cluster 0      | 47.50        | 29.5510      
  PL - Cluster 2       | PLN - Cluster 1      | 3.56         | 14.1186      
  PL - Cluster 2       | PLN - Cluster 2      | 52.00        | 37.5101      
  PLN - Cluster 0      | PLN - Cluster 0      | 56.60        | 8.9774       
  PLN - Cluster 0      | PLN - Cluster 1      | 44.60        | 14.4852      
  PLN - Cluster 0      | PLN - Cluster 2      | 49.24        | 9.4084       
  PLN - Cluster 1      | PLN - Cluster 1      | 3.45         | 3.8247       
  PLN - Cluster 1      | PLN - Cluster 2      | 55.89        | 21.7729      
  PLN - Cluster 2      | PLN - Cluster 2      | 34.65        | 5.4178       
"""

# 2. Parse the text
data = []
lines = raw_text.strip().split('\n')
in_pairwise_section = False

for line in lines:
    if "Source Cluster" in line and "Target Cluster" in line:
        in_pairwise_section = True
        continue
    
    if in_pairwise_section:
        if "---" in line or "===" in line:
            continue
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 4:
                src, tgt = parts[0], parts[1]
                w1 = float(parts[2])
                fft = float(parts[3])
                
                data.append({"Source": src, "Target": tgt, "Mean W1": w1, "Mean FFT ΔdB": fft})
                if src != tgt:
                    data.append({"Source": tgt, "Target": src, "Mean W1": w1, "Mean FFT ΔdB": fft})

df = pd.DataFrame(data)

# Dictionary mapping the raw abbreviations to their full names
RENAME_MAP = {
    "Gen": "Generated",
    "PL": "Pléiade",
    "PLN": "pléiades néo"
}

# 3. Helper function to group and plot
def plot_grouped_heatmap(df, value_col, title, cmap, filename):
    # Extract the main group prefix, and rename it using the dictionary
    df['Source_Group'] = df['Source'].apply(lambda x: RENAME_MAP.get(x.split(' - ')[0], x.split(' - ')[0]))
    df['Target_Group'] = df['Target'].apply(lambda x: RENAME_MAP.get(x.split(' - ')[0], x.split(' - ')[0]))
    
    # Group by the main Group combinations and find the minimum (best) score
    df_grouped = df.groupby(['Source_Group', 'Target_Group'])[value_col].min().reset_index()
    
    # Pivot the grouped data into a matrix
    pivot_df = df_grouped.pivot(index='Source_Group', columns='Target_Group', values=value_col)
    
    # Ensure the 3x3 grid follows our custom order with the new names
    group_order = ["Generated", "Pléiade", "pléiades néo"]
    pivot_df = pivot_df.reindex(index=group_order, columns=group_order)
    
    # Plotting (reduced figure size to fit a smaller 3x3 grid nicely)
    plt.figure(figsize=(7, 5))
    
    ax = sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".1f" if "W1" in value_col else ".2f", 
        cmap=cmap, 
        cbar=True, 
        square=True, 
        linewidths=0.5, 
        linecolor='lightgray'
    )
    
    plt.title(title, fontsize=14, pad=20, fontweight='bold')
    plt.xlabel("Target Group", fontsize=12)
    plt.ylabel("Source Group", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

# 4. Generate the grouped plots
plot_grouped_heatmap(
    df, 
    value_col="Mean W1", 
    title="Wasserstein-1 Distance\n(Best Cluster-Match between Groups)", 
    cmap="YlGnBu", 
    filename="group_heatmap_w1.png"
)

plot_grouped_heatmap(
    df, 
    value_col="Mean FFT ΔdB", 
    title="FFT Power Spectral Density Error\n(Best Cluster-Match between Groups)", 
    cmap="YlOrRd", 
    filename="group_heatmap_fft.png"
)