import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Set style ===
sns.set_theme(style="whitegrid")

# === Get the directory where the script is located ===
script_dir = os.path.dirname(os.path.abspath(__file__))

# === Create a subfolder for results ===
output_dir = os.path.join(script_dir, "results_boxplots")
os.makedirs(output_dir, exist_ok=True)

# === Prepare to store all data ===
all_data = []

# === Loop through all CSV files in the script directory ===
for filename in os.listdir(script_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(script_dir, filename)
        df = pd.read_csv(file_path)

        # Infer group from filename
        if "iMNTB" in filename:
            df["group"] = "iMNTB"
        elif "TeNT" in filename:
            df["group"] = "TeNT"
        else:
            continue  # Skip if group not identifiable

        df["source_file"] = filename
        all_data.append(df)

# === Combine all data into one DataFrame ===
combined_df = pd.concat(all_data, ignore_index=True)

# === Save the combined data ===
compiled_path = os.path.join(output_dir, "compiled_data.csv")
combined_df.to_csv(compiled_path, index=False)
print(f"✅ Compiled CSV saved at: {compiled_path}")

# === Define custom colors ===
custom_palette = {
    "iMNTB": "#4d4d4d",  # dark grey
    "TeNT": "#fca4a4"  # light red
}

# === Plot boxplots with points for each numeric column ===
numeric_cols = combined_df.select_dtypes(include="number").columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))

    sns.boxplot(
        data=combined_df, x="group", y=col,
        hue="group", palette=custom_palette,  # required fix
        width=0.5, fliersize=0, dodge=False
    )

    sns.stripplot(
        data=combined_df, x="group", y=col,
        hue="group", palette=custom_palette,  # required fix
        dodge=False, alpha=0.6, linewidth=0.5,
        edgecolor="black", size=5, legend=False
    )

    plt.title(f"{col} by Group")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"boxplot_{col}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Saved: {plot_path}")
