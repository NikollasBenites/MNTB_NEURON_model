import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

rcParams['pdf.fonttype'] = 42   # TrueType
# === Set style ===
sns.set_theme(style="whitegrid")

# === Get the directory where the script is located ===
script_dir = os.path.dirname(os.path.abspath(__file__))

# === Create a subfolder for results ===
output_dir = os.path.join(script_dir, "results_boxplots")
os.makedirs(output_dir, exist_ok=True)

# === Prepare to store all data ===

df = pd.read_csv("combined_simulation_summary.csv")  # or whatever your file is called

# # === Loop through all CSV files in the script directory ===
# for filename in os.listdir(script_dir):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(script_dir, filename)
#         df = pd.read_csv(file_path)
#
#         # Infer group from filename
#         if "iMNTB" in filename:
#             df["group"] = "iMNTB"
#         elif "TeNT" in filename:
#             df["group"] = "TeNT"
#         else:
#             continue  # Skip if group not identifiable
#
#         df["source_file"] = filename
#         all_data.append(df)

# === Combine all data into one DataFrame ===
combined_df = df

# === Save the combined data ===
compiled_path = os.path.join(output_dir, "compiled_data.csv")
combined_df.to_csv(compiled_path, index=False)
print(f"✅ Compiled CSV saved at: {compiled_path}")

# === Define custom colors ===
custom_palette = {
    "iMNTB": "#4d4d4d",  # dark grey
    "TeNT": "#fca4a4"  # light red
}

# === Plot barplots with points and statistical annotations ===
numeric_cols = combined_df.select_dtypes(include="number").columns

# === Define custom y-limits for specific parameters ===
ylim_dict = {
    "gNa": (0, 450),
    "gKHT": (0, 450),
    "gKA": (0, 450),
    "gKLT": (0, 45),
    "gIH": (0, 45),
    "gLeak": (0, 45),
    "ELeak": (-80, 0),
    "kbm": (-0.035, 0),
}

stats_results = []

for col in numeric_cols:
    plt.figure(figsize=(3, 6))

    sns.barplot(
        data=combined_df, x="group", y=col,
        hue="group", palette=custom_palette,
        capsize=0.2, errorbar="se", err_kws={'linewidth': 1.0},
        legend=False
    )

    sns.stripplot(
        data=combined_df, x="group", y=col,
        hue="group", palette=custom_palette,
        dodge=False, alpha=0.6, linewidth=1.0,
        edgecolor="black", size=12, legend=False
    )

    # === Extract values by group ===
    values_imntb = combined_df[combined_df["group"] == "iMNTB"][col].dropna()
    values_tent = combined_df[combined_df["group"] == "TeNT"][col].dropna()

    # === Normality check ===
    stat_imntb, p_imntb = shapiro(values_imntb)
    stat_tent, p_tent = shapiro(values_tent)
    normal = p_imntb > 0.05 and p_tent > 0.05

    # === Select statistical test ===
    if normal:
        stat, pval = ttest_ind(values_imntb, values_tent, equal_var=False)
        test_name = "t-test"
    else:
        stat, pval = mannwhitneyu(values_imntb, values_tent, alternative="two-sided")
        test_name = "M-W U"

    # === Determine asterisk level ===
    if pval < 0.001:
        p_text = "***"
    elif pval < 0.01:
        p_text = "**"
    elif pval < 0.05:
        p_text = "*"
    else:
        p_text = ""

    # === Add asterisk above bar ===
    if p_text:
        ymax = ylim_dict[col][1] if col in ylim_dict else combined_df[col].max()
        plt.text(0.5, ymax * 0.95, p_text, ha="center", va="top", fontsize=14)

    # === Apply y-axis limits ===
    if col in ylim_dict:
        plt.ylim(ylim_dict[col])
    elif combined_df[col].min() >= 0:
        plt.ylim(bottom=0)

    # === Clean black axes with ticks ===
    ax = plt.gca()
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1)
        ax.spines[spine].set_color("black")
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)

    ax.tick_params(
        axis='both', which='both',
        bottom=True, left=True,
        top=False, right=False,
        direction='out',
        width=1, length=5,
        color='black'
    )

    stats_results.append({
        "Variable": col,
        "Test": test_name,
        "p-value": pval,
        "Group 1": "iMNTB",
        "Group 2": "TeNT",
        "Normality iMNTB p": p_imntb,
        "Normality TeNT p": p_tent,
        "Used parametric test": normal
    })

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"barplot_{col}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📄 Saved: {plot_path}")

# === Rename columns for export ===
rename_columns = {
    "gNa": "gna",
    "gKHT": "gkht",
    "gKA": "gka",
    "gKLT": "gklt",
    "gIH": "gh",
    "gLeak": "gleak",
    "ELeak": "erev",  # or keep as 'eleak' if preferred
    # You can include others if needed
}

# Apply to combined_df just before exporting
combined_df.rename(columns=rename_columns, inplace=True)


# === Save stats after all plots ===
stats_df = pd.DataFrame(stats_results)
stats_path = os.path.join(output_dir, "group_comparison_stats.csv")
stats_df.to_csv(stats_path, index=False)
print(f"📄 Statistical summary saved at: {stats_path}")

# === Compute and save transposed averages separately for iMNTB and TeNT ===
avg_imntb = combined_df[combined_df["group"] == "iMNTB"].mean(numeric_only=True)
avg_tent = combined_df[combined_df["group"] == "TeNT"].mean(numeric_only=True)

# Transpose for readability
avg_imntb_df = avg_imntb.to_frame(name="avg_iMNTB").T
avg_tent_df = avg_tent.to_frame(name="avg_TeNT").T

# Save to CSV
avg_imntb_path = os.path.join(output_dir, "avg_iMNTB_transposed.csv")
avg_tent_path = os.path.join(output_dir, "avg_TeNT_transposed.csv")

avg_imntb_df.to_csv(avg_imntb_path, index=False)
avg_tent_df.to_csv(avg_tent_path, index=False)

print(f"📊 Transposed average iMNTB saved at: {avg_imntb_path}")
print(f"📊 Transposed average TeNT saved at: {avg_tent_path}")

