# Python/Unsupervised Machine Learning Customer Segmentation Project

## Project Overview


To understand the Target Customers for the Marketing Team to plan a strategy, I segmented customer data utilizing the following techniques:

1. **Bivariate Analysis/Bivariate Clustering**
2. **K-Means Algorithm and the Elbow Methodology**
3. **Summary Statistics**

With these techniques, I identified the most important shopping groups based on income, age and the mall shopping score and created labels for each of the groups.

## Objectives


1. **Perform EDA**
2. **Use KMEANS Clustering Algorithm to Create Segments**
3. **Use Summary Statistics on the Clusters**
4. **Visualize the Results**

## Project Structure


### 1. Importing Required Libraries

The project begins by importing the required libraries for data analysis, visualization, and clustering and then importing the data.



```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings 
warnings.filterwarnings('ignore')
```
```python
df = pd.read_csv("Mall_Customers.csv")
```



### 2. Data Analysis and Findings

- **Univariate Analysis**: Understand the data by using Histograms, Probability Density Plots and KDE Plots.
  

```python
sns.distplot(df['Annual Income (k$)']);
```
```python
sns.kdeplot(data=df, x='Annual Income (k$)', hue='Gender', fill=True)
```
```python
columns = [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns: 
    plt.figure()
    sns.boxplot(data=df, x='Gender', y= df[i], hue='Gender')
```



- **Bivariate Analysis**: Understand the data by using Scatter Plots, Pair Plots and KDE Plots.


```python
sns.scatterplot(data=df, x= 'Annual Income (k$)', y = 'Spending Score (1-100)')
```
```python
sns.pairplot(df)
```
```python
sns.pairplot(df, hue='Gender')
```

Grouping the Age, Annual Income and Spending score by gender shows us that, while men tend to have higher average annual incomes, women exhibit higher average spending scores.
```python
df.groupby('Gender')[[ 'Age', 'Annual Income (k$)','Spending Score (1-100)']].mean()
```
<img width="853" height="264" alt="image" src="https://github.com/user-attachments/assets/f8aeca8c-c9aa-43e7-b395-9238a74ce7a0" />

A correlation analysis between Age, Annual Income, and Spending Score reveals that income and spending score are largely independent, suggesting that high-spending customers exist across income levels. Age shows a slight negative correlation with spending score, indicating that younger customers tend to spend more, while income and age are essentially uncorrelated.
```python
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
```
<img width="989" height="237" alt="image" src="https://github.com/user-attachments/assets/259a560f-a809-4e7e-9cda-164e98f58bde" />


### 3. Clustering - Univariate, Bivariate and Multivariate

-**Univariate Clustering** : Using a K-Means model, 

```python
clustering1 = KMeans(n_clusters=3)
```
```python
clustering1.fit(df[['Annual Income (k$)']])	
```
```python
clustering1.labels_
```
```python
df['Income Cluster'] = clustering1.labels_
df.head()
```
```python
df['Income Cluster'].value_counts()
```
```python
clustering1.inertia_
```
```python
inertia_scores=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)
```
```python
plt.plot(range(1,11),inertia_scores)
```
```python
df.groupby('Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
```

-**Bivariate Clustering** : Cluster 2 are our customers with the highest Income and Spending Score. Women are the dominate demographic within this cluster, representing 59% of the group. 
<img width="865" height="677" alt="image" src="https://github.com/user-attachments/assets/83f2a16a-7a97-447b-9485-1335327f55f7" />

```python
clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
df.head()
```
```python
centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['X','Y']
```
```python
plt.figure(figsize=(10,8))
plt.scatter(x=centers['X'], y=centers['Y'], s=100, c='black',marker='*')
sns.scatterplot(data=df, x = 'Annual Income (k$)', y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
```
```python
pd.crosstab(df['Spending and Income Cluster'], df['Gender'])
```
```python
pd.crosstab(df['Spending and Income Cluster'], df['Gender'], normalize='index')
```
<img width="334" height="224" alt="image" src="https://github.com/user-attachments/assets/67e2b517-b699-4aa5-bb0e-b78cc9b8bf44" />

```python
clustering1.inertia_
```
```python
inertia_scores=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)
```
```python
inertia_scores2=[]
for i in range (1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),inertia_scores2)
```
```python
df.groupby('Spending and Income Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
```

-**Multivariate Clustering** : Using a K-Means model, 
```python
from sklearn.preprocessing import StandardScaler
```
```python
scale = StandardScaler()
```
```python
dff = pd.get_dummies(df,drop_first=True)
dff.head()
```
```python
dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
dff.head()
```
```python
dff = scale.fit_transform(dff)
```
```python
dff = pd.DataFrame(scale.fit_transform(dff))
```
```python
inertia_scores3=[]
for i in range (1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    inertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),inertia_scores3)
```
```python
dff = scale.fit_transform(dff)
```
```python
df.to_csv('Clustering.csv')
```

## Findings

- **Cluster**: Target Marketing Group would be Cluster 3, which has a high Spending Score and high income.
- **Demographics**: 59% of Cluster 3 shoppers are women. We develop marketing campaigns that target popular items in this cluster to attract these customers.
- **Customer Insights**: Cluster 2 shoppers (low Annual Income but high Shopping Score) present an interesting opportunity to market to customers for sales events on popular items. 

## Conclusions

Based on the clustering analysis, the customer base can be segmented into groups representing various spending behaviors and income levels, enabling more targeted and data-driven marketing strategies. Using bivariate analysis, summary statistics and K-Means algorithms guided by the elbow method, clear patterns emerged across income, age and spending score.

Cluster 3 represents the primary target marketing group, which are customers who exhibit both high annual income and high spending scores. This indicates strong purchasing power and a high liklihood of engagement with premium or high value products. With 54% of this cluster consisting of women, marketing campaigns can be strategically tailored toward products and promotions that resonate with this demographic to maximise conversion and retention. 

Cluster 2, characterized by lower annual income but high spending scores, reveals a high-engagement segment that is price-sensitive yet active. This group presents a valuable opportunity for promotional campaigns, discounds and sales events on popular items, as they are likely to respond positively to value-driven marketing strategies. 

Overall, the segmentation approach allowed for the identification of key customer groups and their behavioral patterns, providing actionable insights for the marketing team. By leveraging K-Means clustering alongside exploratory and bivariate analysis, the project successfully transformed raw customer data into meaningful segments that support strategic decision-making, personalized marketing, and improved customer targeting.



