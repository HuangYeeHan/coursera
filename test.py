import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, pearsonr
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Load the dataset
data_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(data_url)

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Task 2: Descriptive Statistics and Visualizations
# Boxplot for the "Median value of owner-occupied homes"
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['medv'], color="skyblue")
plt.title("Boxplot of Median Value of Owner-Occupied Homes (MEDV)")
plt.ylabel("Median Value ($1000's)")
plt.show()

# Bar plot for the Charles River variable
plt.figure(figsize=(8, 6))
df['chas'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Number of Houses Bounded by Charles River (CHAS)")
plt.xlabel("CHAS (1: Yes, 0: No)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# Boxplot for MEDV vs AGE groups
age_groups = pd.cut(df['age'], bins=[0, 35, 70, 100], labels=['<=35', '35-70', '>70'])
df['age_group'] = age_groups
plt.figure(figsize=(8, 6))
sns.boxplot(x='age_group', y='medv', data=df, palette="Pastel1")
plt.title("Boxplot of MEDV by Age Groups")
plt.xlabel("Age Groups")
plt.ylabel("Median Value ($1000's)")
plt.show()

# Scatter plot for NOX and INDUS
plt.figure(figsize=(8, 6))
sns.scatterplot(x='nox', y='indus', data=df, color="purple")
plt.title("Scatter Plot of Nitric Oxides Concentration (NOX) vs Non-Retail Business Acres (INDUS)")
plt.xlabel("NOX (parts per 10 million)")
plt.ylabel("INDUS (% non-retail business acres)")
plt.show()

# Histogram for the pupil-teacher ratio (PTRATIO)
plt.figure(figsize=(8, 6))
df['ptratio'].plot(kind='hist', bins=15, color="lightgreen", edgecolor='black')
plt.title("Histogram of Pupil-Teacher Ratio (PTRATIO)")
plt.xlabel("PTRATIO")
plt.ylabel("Frequency")
plt.show()

# Task 3: Statistical Analysis
# 1. T-test for significant difference in MEDV based on CHAS
group1 = df[df['chas'] == 1]['medv']
group2 = df[df['chas'] == 0]['medv']
stat, p_value = ttest_ind(group1, group2)
print("T-test results:")
print(f"Statistic: {stat}, P-value: {p_value}")
if p_value < 0.05:
    print("There is a significant difference in median value of houses bounded by the Charles River.")
else:
    print("No significant difference in median value of houses bounded by the Charles River.")

# 2. ANOVA for difference in MEDV by AGE groups
anova_stat, anova_p_value = f_oneway(
    df[df['age_group'] == '<=35']['medv'],
    df[df['age_group'] == '35-70']['medv'],
    df[df['age_group'] == '>70']['medv']
)
print("ANOVA results:")
print(f"Statistic: {anova_stat}, P-value: {anova_p_value}")
if anova_p_value < 0.05:
    print("There is a significant difference in MEDV based on age groups.")
else:
    print("No significant difference in MEDV based on age groups.")

# 3. Pearson Correlation for NOX and INDUS
corr, corr_p_value = pearsonr(df['nox'], df['indus'])
print("Pearson Correlation results:")
print(f"Correlation Coefficient: {corr}, P-value: {corr_p_value}")
if corr_p_value < 0.05:
    print("There is a significant relationship between NOX and INDUS.")
else:
    print("No significant relationship between NOX and INDUS.")

# 4. Regression analysis for DIS and MEDV
X = df[['dis']]
X = sm.add_constant(X)
Y = df['medv']
model = sm.OLS(Y, X).fit()
print(model.summary())

# Interpretation of regression results
if model.pvalues['dis'] < 0.05:
    print("Weighted distance to employment centers (DIS) significantly impacts MEDV.")
else:
    print("No significant impact of DIS on MEDV.")
