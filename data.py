from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
  
cardiotocography = fetch_ucirepo(id=193) 
  
X = cardiotocography.data.features 
y = cardiotocography.data.targets 
  
#print(cardiotocography.metadata) 
  
#print(cardiotocography.variables) 

# Inspecting the data
df = pd.concat([X, y], axis=1)

#print("Shape:", df.shape)
#print("\nFirst rows:\n", df.head())
#print("\nTarget distribution:\n", df['NSP'].value_counts())

# Checking for missing values + info
#print(df.isnull().sum())
#print(df.dtypes)

# Descriptive Stats
#print(df.describe)

# Class Distribution
#sns.countplot(x="NSP", data=df, palette="viridis")
#plt.title("Class Distribution (1=Normal, 2=Suspect, 3=Pathologic)")
#plt.show()

# Histogram
#df.hist(figsize=(15, 10), bins=20)
#plt.suptitle("Feature Distributions", fontsize=16)
#plt.show()

# Heatmap
#plt.figure(figsize=(12, 8))
#sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
#plt.title("Feature Correlation Heatmap")
#plt.show()

# Feature Differences
plt.figure(figsize=(10,6))
sns.boxplot(x="NSP", y="CLASS", data=df, palette="Set2")
plt.title("Baseline FHR by Class")
plt.show()