# Metabolic Syndrome Analysis Project
# Authors: Maria Eva Mellor Ortiz & Diana Mihaela Pal Japa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Question 1: How many patients have metabolic syndrome in the dataset?
df = pd.read_csv("metabolic_syndrome.txt")

print(df["MetabolicSyndrome"].value_counts())
print(df["MetabolicSyndrome"].value_counts(normalize=True) * 100)

# Plotting the results
plt.figure(figsize=(8, 6))
df["MetabolicSyndrome"].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Number of Patients with Metabolic Syndrome")
plt.xlabel("Metabolic Syndrome Status")
plt.ylabel("Number of Patients")
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.show()

# Question 2: How are the different variables (age, gender, etc.) distributed?
import os

# create folder to save figures
fig_dir = "figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# handle missing values (brief, we drop for plotting but keep original df)
df_plot = df.copy()

# Identify numeric and categorical columns based on the dataset description
numeric_cols = ["Age", "WaistCirc", "BMI", "UrAlbCr", "UricAcid",
                "BloodGlucose", "HDL", "Triglycerides"]
categorical_cols = ["Sex", "Marital", "Income", "Race", "Albuminuria", "MetabolicSyndrome"]

# ensure columns exist (defensive)
numeric_cols = [c for c in numeric_cols if c in df_plot.columns]
categorical_cols = [c for c in categorical_cols if c in df_plot.columns]

# Basic missing-value report
print("\nMissing values per column:")
print(df_plot.isna().sum())

# Summary statistics: numeric
print("\nSummary statistics (numeric):")
num_summary = df_plot[numeric_cols].describe().transpose()
print(num_summary)

# Save numeric summary to CSV
num_summary.to_csv("summary_statistics_numeric.csv")

# Summary for categorical variables (counts and proportions)
print("\nCategorical counts and proportions:")
cat_summary = {}
for c in categorical_cols:
    counts = df_plot[c].value_counts(dropna=False)
    props = df_plot[c].value_counts(normalize=True, dropna=False) * 100
    print(f"\nColumn: {c}")
    print(counts)
    print((props).round(2))
    cat_summary[c] = pd.concat([counts, props.round(2)], axis=1).rename(columns={c: "count", 0: "proportion(%)"})
# Optionally save categorical summaries
# Convert to one CSV per categorical column
for c in categorical_cols:
    cat_summary[c].to_csv(f"summary_{c}.csv")

# Plot: histograms + KDE for numeric columns
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_plot[col].dropna(), kde=True, bins=25)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"hist_{col}.png"))
    plt.show()

# Plot: boxplots for numeric columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_plot[col].dropna())
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"box_{col}.png"))
    plt.show()

# Plot: countplots for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(x=df_plot[col])
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"count_{col}.png"))
    plt.show()

print("\nFinished Question 2: numeric summary saved as summary_statistics_numeric.csv, figures saved in 'figures/'")
