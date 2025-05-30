import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# === Load the global summary ===
fenotype = "P9_iMNTB"
script_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(script_dir, "..", "results", "_fit_results", f"APs_{fenotype}",f"summary_all_ap_{fenotype}_fits.csv")
df = pd.read_csv(summary_path)

# === Basic overview ===
print("\nSummary statistics for parameters:")
print(df.describe())

# === Parse metadata from filename ===
df['age'] = df['filename'].str.extract(r'P(\d+)').astype(float)
df['condition'] = df['filename'].apply(lambda x: 'TeNT' if 'TeNT' in x else 'Control')

# === Scatter plot: gNa vs gKHT ===
plt.figure(figsize=(6, 5))
plt.scatter(df['gna'], df['gkht'], c='blue')
plt.xlabel('gNa (nS)')
plt.ylabel('gKHT (nS)')
plt.title('gNa vs gKHT across AP fits')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Boxplot: Distribution of Conductances ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['gna', 'gkht', 'gklt', 'gh', 'gka', 'gleak']])
plt.title("Distribution of Fitted Conductances")
plt.ylabel("nS")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Pairplot of AP features ===
feature_cols = ['amp', 'width', 'latency', 'AHP', 'threshold', 'rest', 'peak']
sns.pairplot(df[feature_cols].dropna())
plt.suptitle("Pairwise Relationships of AP Features", y=1.02)
plt.show()

# # === Plot gNa vs Age grouped by condition ===
# plt.figure(figsize=(8, 5))
# sns.scatterplot(data=df, x='age', y='gna', hue='condition')
# plt.title('gNa vs Age by Condition')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
