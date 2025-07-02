import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import datetime
# === Load the compiled CSV ===
compiled_path = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_latest_iMNTB_TeNT_fits/_last_round/test_mutation/_latest_iMNTB_TeNT_fits/compiled_fit_results_coductance.csv"
df = pd.read_csv(compiled_path)
timestamp = datetime.datetime.now().strftime("%Y%m%d")
# === Prepare for plotting ===
df_plot = df.copy()
df_plot["group"] = pd.Categorical(df_plot["group"], categories=["TeNT", "iMNTB"], ordered=True)
custom_palette = {"TeNT": "#d62728", "iMNTB": "#7f7f7f"}

# === Detect numeric, plottable columns ===
non_plottable_cols = {"source_folder", "group", "datetime", "fit_file", "fit_type", "fit_quality"}
plottable_cols = [
    col for col in df_plot.columns
    if col not in non_plottable_cols and pd.api.types.is_numeric_dtype(df_plot[col])
]

# === Optional: save directory for plots ===
save_dir = os.path.join(os.path.dirname(compiled_path),"..", f"figures_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
# === Initialize results list ===
stats_results = []
# === Generate and save plots ===
for col in plottable_cols:
    plt.figure(figsize=(8, 5))

    # Boxplot and stripplot
    sns.boxplot(data=df_plot, x="group", y=col, hue="group", palette=custom_palette, legend=False)
    sns.stripplot(data=df_plot, x="group", y=col, color='black', alpha=0.6)

    # === Statistical Test ===
    group1 = df_plot[df_plot["group"] == "TeNT"][col].dropna()
    group2 = df_plot[df_plot["group"] == "iMNTB"][col].dropna()

    try:
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    except ValueError:
        stat, p = float('nan'), float('nan')

    # Significance star
    if pd.isna(p):
        star = "n/a"
    elif p < 0.001:
        star = "***"
    elif p < 0.01:
        star = "**"
    elif p < 0.05:
        star = "*"
    else:
        star = "ns"

    p_text = f"p = {p:.3e}" if p >= 0.001 else "p < 0.001"
    plt.text(0.5, max(df_plot[col].dropna()) * 1.05, p_text,
             ha='center', va='bottom', fontsize=12)

    # Plot labels and saving
   # plt.title(f"{col} by Group")
    plt.ylabel(col)
    plt.xlabel("Group")
    plt.tight_layout()

    fig_path = os.path.join(save_dir, f"{col}_by_group.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"📊 Saved: {fig_path} | {p_text}")

    # Add to results
    stats_results.append({
        "parameter": col,
        "statistic": stat,
        "p_value": p,
        "significance": star
    })

# === Save stats as CSV ===
stats_df = pd.DataFrame(stats_results)
stats_csv_path = os.path.join(save_dir, "stats_summary.csv")
stats_df.to_csv(stats_csv_path, index=False)
print(f"\n📁 Saved statistical summary: {stats_csv_path}")