import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load the compiled CSV ===
compiled_path = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/latest_iMNTB_TeNT_fits/compiled_fit_results_coductance.csv"
df = pd.read_csv(compiled_path)

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
save_dir = os.path.join(os.path.dirname(compiled_path), "figures")
os.makedirs(save_dir, exist_ok=True)

# === Generate and save plots ===
for col in plottable_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_plot, x="group", y=col, hue="group", palette=custom_palette, legend=False)

    sns.stripplot(data=df_plot, x="group", y=col, color='black', alpha=0.6)
    plt.title(f"{col} by Group")
    plt.ylabel(col)
    plt.xlabel("Group")
    plt.tight_layout()


    # Save figure
    fig_path = os.path.join(save_dir, f"{col}_by_group.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"📊 Saved: {fig_path}")
