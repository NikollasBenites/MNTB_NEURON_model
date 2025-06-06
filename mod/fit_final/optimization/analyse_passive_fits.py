import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import  re

# === Load both datasets ===
base_dir = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_passive_results"
tent_path = os.path.join(base_dir, "passive_fit_summary_TeNT.csv")
imntb_path = os.path.join(base_dir, "passive_fit_summary_iMNTB.csv")

df_tent = pd.read_csv(tent_path)
df_imntb = pd.read_csv(imntb_path)

# === Add group labels if not already present ===
df_tent["group"] = "TeNT"
df_imntb["group"] = "iMNTB"

# === Combine both into a single DataFrame ===
df = pd.concat([df_tent, df_imntb], ignore_index=True)


# === Parse age and clean up ===
df["age_num"] = df["age"].str.extract(r"P(\d+)").astype(float)
df = df[df["r2_score"] > 0.85]  # Optional: filter low-quality fits

# === Create output directory ===
output_dir = os.path.join(base_dir, "stats_passive")

os.makedirs(output_dir, exist_ok=True)

per_cell_dir = os.path.join(output_dir, "per_cell_barplots")
os.makedirs(per_cell_dir, exist_ok=True)

for source_file in df["source_file"].unique():
    df_cell = df[df["source_file"] == source_file]

    group = df_cell["group"].iloc[0] if not df_cell["group"].isnull().all() else "Unknown"
    cell_id = df_cell["cell_id"].iloc[0] if not df_cell["cell_id"].isnull().all() else "?"

    df_melted = df_cell.melt(
        id_vars=["age", "age_num", "group", "cell_id", "source_file"],
        value_vars=["gleak", "gklt", "gh"],
        var_name="conductance",
        value_name="value"
    )

    if df_melted["value"].isnull().all():
        continue

    plt.figure(figsize=(8, 8))
    sns.barplot(
        data=df_melted,
        x="age_num", y="value",
        hue="conductance",
        errorbar="sd",
        palette="pastel"
    )



    # Clean file name
    clean_name = re.sub(r'^passive_summary_experimental_data_', '', source_file)
    clean_name = re.sub(r'\.json.*$', '', clean_name)
    safe_filename = clean_name.replace("/", "_")

    filename = f"{safe_filename}_{group}.png"
    plt.title(f"{safe_filename} ({group}, {cell_id})")
    plt.xlabel("Age (P)")
    plt.ylabel("Conductance (nS)")
    plt.ylim(0, 50)
    plt.legend(title="Conductance")
    plt.tight_layout()
    plt.savefig(os.path.join(per_cell_dir, filename), dpi=300)
    plt.close()



# === Violin plots ===
for param in ["gleak", "gklt", "gh"]:
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="age_num", y=param, hue="group", inner="box", cut=0)
    plt.title(f"{param} by Age and Phenotype")
    plt.ylabel(f"{param} (nS)")
    plt.ylim(0, 50)
    plt.xlabel("Age (P)")
    plt.legend(title="Group", loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{param}_violinplot.png"), dpi=300)
    plt.close()

# === Run ANOVA for each parameter ===
anova_results = {}
for param in ["gleak", "gklt", "gh"]:
    subdf = df[["group", "age_num", param]].dropna()
    if subdf["group"].nunique() < 2 or subdf["age_num"].nunique() < 2:
        print(f"⚠️ Skipping ANOVA for {param}: not enough group or age variation.")
        continue

    model = ols(f"{param} ~ C(group) + C(age_num)", data=subdf).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    anova_results[param] = anova_table

    print(f"\n📊 ANOVA for {param}")
    print(anova_table)
    anova_table.to_csv(os.path.join(output_dir, f"anova_{param}.csv"))

# === Aggregate summary stats
agg = df.groupby(["group", "age_num"])[["gleak", "gklt", "gh"]].agg(["mean", "std", "count"])
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
agg.reset_index(inplace=True)
agg.to_csv(os.path.join(output_dir, "grouped_summary_stats.csv"), index=False)

print(f"\n✅ Analysis complete. All outputs saved to:\n{output_dir}")
