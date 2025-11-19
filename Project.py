# Metabolic Syndrome Analysis Project
# Authors: Maria Eva Mellor Ortiz & Diana Mihaela Pal Japa
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency

#  Question 1: How many patients have metabolic syndrome in the dataset?
# Load the dataset
df = pd.read_csv("metabolic_syndrome.txt")

print("\n--- Question 1: Metabolic Syndrome Counts ---")
print(df["MetabolicSyndrome"].value_counts())
print("\nProportions (%):")
print(df["MetabolicSyndrome"].value_counts(normalize=True) * 100)

# Plotting the results
plt.figure(figsize=(8, 5))
df["MetabolicSyndrome"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Number of Patients with Metabolic Syndrome")
plt.xlabel("Metabolic Syndrome Status")
plt.ylabel("Number of Patients")
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.tight_layout()
plt.show()

# %% Question 2: How are the different variables (age, gender, etc.) distributed?

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

# Convert to one CSV per categorical column
for c in categorical_cols:
    cat_summary[c].to_csv(f"summary_{c}.csv")

# Plot: histograms + KDE for numeric columns
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        df_plot[col].dropna(),
        color="red",
        linewidth=2
    )
    sns.histplot(
        df_plot[col].dropna(),
        bins=25,
        color="skyblue",
        stat="density",
        alpha=0.6
    )
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"hist_{col}.png"))
    plt.show()

# Plot: boxplots for numeric columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_plot[col].dropna(), color="skyblue")
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"box_{col}.png"))
    plt.show()

# Plot: countplots for categorical columns
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df_plot[col], color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"count_{col}.png"))
    plt.show()

print("\nFinished Question 2: numeric summary saved as summary_statistics_numeric.csv, figures saved in 'figures/'")

# %% Question 3: How do the different variables interact? 
# Are the biological variables similarly distributed in the different gender group?

print("\n--- Question 3: Variable Interactions ---")
bio_vars = ["WaistCirc", "BMI", "UrAlbCr", "UricAcid",
            "BloodGlucose", "HDL", "Triglycerides"
            ]

print("\n--- Interaction: Biological Variables by Sex ---")
print(df.groupby("Sex")[bio_vars].describe().transpose())

# Boxplots
for var in bio_vars:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Sex", y=var, color="skyblue")
    plt.title(f"{var} by Sex")
    plt.xlabel("Sex")
    plt.ylabel(var)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"box_{var}_by_Sex.png"))
    plt.show()

# What correlations exist between the biological variables?
print("\n--- Correlation Matrix of Biological Variables ---")
corr_matrix = df[bio_vars].corr()
print(corr_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Biological Variables")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "correlation_matrix_biological_vars.png"))
plt.show()

print("\nFinished Question 3: interaction plots saved in 'figures/'")

# %% Question 4: What factor is the most linked to metabolic syndrome?
print("\n--- Question 4: Factors Linked to Metabolic Syndrome ---")
# We will analyze both numeric and categorical variables to see which ones are most strongly associated with metabolic syndrome.
# 1) Numeric variables: we calculate mean difference between groups.

numeric_effects = {}
for col in numeric_cols:
    means = df.groupby("MetabolicSyndrome")[col].mean()
    diff = abs(means.iloc[1] - means.iloc[0])
    numeric_effects[col] = diff

# Sort numeric variables by difference
numeric_effects = dict(sorted(numeric_effects.items(), key=lambda item: item[1], reverse=True))
print("\nNumeric variables sorted by difference between MetSyn groups:")
for col, diff in numeric_effects.items():
    print(f"{col}: Mean difference = {diff:.2f}")

# Plot numeric variables
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="MetabolicSyndrome", y=col, color="skyblue")
    plt.title(f"{col} by Metabolic Syndrome Status")
    plt.xlabel("Metabolic Syndrome Status")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"box_{col}_by_MetabolicSyndrome.png"))
    plt.show()

# 2) Categorical variables: chi-square to see strongest association 

cat_effects = {}
for col in categorical_cols:
    contingency = pd.crosstab(df[col], df["MetabolicSyndrome"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    cat_effects[col] = chi2

# Sort categorical variables by chi-square value
cat_effects = dict(sorted(cat_effects.items(), key=lambda item: item[1], reverse=True))
print("\nCategorical variables sorted by chi-square association with MetSyn:")
for col, chi2_val in cat_effects.items():
    print(f"{col}: Chi-square = {chi2_val:.2f}")

# Plot categorical variables
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    counts = pd.crosstab(df[col], df["MetabolicSyndrome"])
    counts.plot(kind="bar", stacked=True, figsize=(8,5), color=["skyblue", "salmon"])
    plt.title(f"{col} by Metabolic Syndrome Status")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"count_{col}_by_MetabolicSyndrome.png"))
    plt.show()

# Summary 
top_numeric = list(numeric_effects.keys())[0]
top_categorical = list(cat_effects.keys())[0]

print(f"\nMost relevant numeric factor: {top_numeric}")
print(f"Most relevant categorical factor: {top_categorical}")

print("\nFinished Question 4: analysis plots saved in 'figures/' and top factors identified.")

# %% Question 5: Predict Metabolic Syndrome
print("\n--- Question 5: Predicting Metabolic Syndrome ---")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Separate features and target
X = df.drop(columns=["MetabolicSyndrome"])
Y = df["MetabolicSyndrome"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Train a simple Random Forest classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Evaluation of the model
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

# %% Plot top 10 features by importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind="barh", figsize=(8, 5), color="skyblue")
plt.title("Top 10 Features by Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "feature_importance.png"))
plt.show()

print("\nFinished Question 5: prediction model evaluated and feature importance plot saved in 'figures/'.")
print("\n--- End of Analysis ---")


