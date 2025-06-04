import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# === Load data ===
summary_path = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_passive_results/passive_fit_summary_TeNT.csv"  # Adjust if needed
df = pd.read_csv(summary_path)

# === Parse age and clean up ===
df["age_num"] = df["age"].str.extract(r"P(\d+)").astype(float)
df = df[df["r2_score"] > 0.85]  # Optional: filter low-quality fits

# === Create output directory ===
output_dir = os.path.join(os.path.dirname(summary_path), "stats_passive")
os.makedirs(output_dir, exist_ok=True)

# === Violin plots ===
for param in ["gleak", "gklt", "gh"]:
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="age_num", y=param, hue="group", inner="box", cut=0)
    plt.title(f"{param} by Age and Phenotype")
    plt.ylabel(f"{param} (nS)")
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
