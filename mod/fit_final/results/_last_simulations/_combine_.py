import pandas as pd

# Load both files
df_imntb = pd.read_csv("simulation_summary_cleaned_iMNTB.csv")
df_tent = pd.read_csv("simulation_summary_cleaned_TeNT.csv")

# Add group identifiers
df_imntb["group"] = "iMNTB"
df_tent["group"] = "TeNT"

# Combine into one DataFrame
combined_df = pd.concat([df_imntb, df_tent], ignore_index=True)

# Save to CSV if needed
combined_df.to_csv("combined_simulation_summary.csv", index=False)

# Show preview
print(combined_df.head())
