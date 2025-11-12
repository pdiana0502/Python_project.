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
