import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('titanic.csv')  

# Ensure images/plot folder exists
os.makedirs('images/plot', exist_ok=True)

# 1️⃣ Plot 1: Age distribution
plt.figure(figsize=(8,6))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('images/plot/plot1.png')
plt.close()

# 2️⃣ Plot 2: Fare distribution
plt.figure(figsize=(8,6))
sns.histplot(df['Fare'], bins=30, kde=True, color='green')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.savefig('images/plot/plot2.png')
plt.close()

# 3️⃣ Plot 3: Survival count
plt.figure(figsize=(8,6))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.savefig('images/plot/plot3.png')
plt.close()

# 4️⃣ Plot 4: Survival by Pclass
plt.figure(figsize=(8,6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.savefig('images/plot/plot4.png')
plt.close()

# 5️⃣ Plot 5: Survival by Sex
plt.figure(figsize=(8,6))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.savefig('images/plot/plot5.png')
plt.close()

# 6️⃣ Plot 6: Age vs Fare scatter
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.title('Age vs Fare by Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.savefig('images/plot/plot6.png')
plt.close()

# 7️⃣ Plot 7: Parch vs SibSp scatter
plt.figure(figsize=(8,6))
sns.scatterplot(x='SibSp', y='Parch', hue='Survived', data=df)
plt.title('SibSp vs Parch by Survival')
plt.xlabel('Siblings/Spouses Aboard')
plt.ylabel('Parents/Children Aboard')
plt.savefig('images/plot/plot7.png')
plt.close()

# 8️⃣ Correlation Heatmap
# Only select numeric columns
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('images/plot/heatmap.png')
plt.close()
