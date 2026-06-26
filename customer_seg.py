# ============================================
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# Complete Data Analyst Project
# ============================================


# ==============================
# 1. IMPORT REQUIRED LIBRARIES
# ==============================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



# ==============================
# 2. LOAD DATASET
# ==============================

df = pd.read_csv("Mall_Customers.csv")


print("First 5 Records")
print(df.head())



# ==============================
# 3. BASIC DATA INFORMATION
# ==============================


print("\nDataset Shape:")
print(df.shape)


print("\nDataset Information:")
print(df.info())


print("\nStatistical Summary:")
print(df.describe())



# ==============================
# 4. DATA CLEANING
# ==============================


# Check missing values

print("\nMissing Values:")
print(df.isnull().sum())


# Check duplicate values

print("\nDuplicate Values:")
print(df.duplicated().sum())


# Remove duplicates

df.drop_duplicates(inplace=True)



# ==============================
# 5. EXPLORATORY DATA ANALYSIS
# ==============================



# Gender Distribution

plt.figure(figsize=(6,4))

sns.countplot(
    x="Gender",
    data=df
)

plt.title("Customer Gender Distribution")

plt.show()



# Age Distribution


plt.figure(figsize=(8,5))


sns.histplot(
    df["Age"],
    bins=20,
    kde=True
)


plt.title("Age Distribution")

plt.show()



# Income Distribution


plt.figure(figsize=(8,5))


sns.histplot(
    df["Annual Income (k$)"],
    kde=True
)


plt.title("Annual Income Distribution")

plt.show()



# Spending Score Distribution


plt.figure(figsize=(8,5))


sns.histplot(
    df["Spending Score (1-100)"],
    kde=True
)


plt.title("Spending Score Distribution")


plt.show()




# ==============================
# 6. FEATURE SELECTION
# ==============================


# Selecting important features

X = df[
[
"Annual Income (k$)",
"Spending Score (1-100)"
]
]


print(X.head())



# ==============================
# 7. FEATURE SCALING
# ==============================


scaler = StandardScaler()


X_scaled = scaler.fit_transform(X)



print(X_scaled[:5])



# ==============================
# 8. FIND OPTIMAL NUMBER OF CLUSTERS
# ELBOW METHOD
# ==============================


wcss = []


for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42
    )


    model.fit(X_scaled)


    wcss.append(model.inertia_)



plt.figure(figsize=(8,5))


plt.plot(
    range(1,11),
    wcss,
    marker="o"
)


plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")


plt.title("Elbow Method")


plt.show()




# ==============================
# 9. SILHOUETTE SCORE
# ==============================


for i in range(2,11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42
    )


    labels = kmeans.fit_predict(X_scaled)


    score = silhouette_score(
        X_scaled,
        labels
    )


    print(
        "Cluster:",
        i,
        "Score:",
        score
    )





# ==============================
# 10. APPLY K-MEANS MODEL
# ==============================


# Choose optimal clusters
# Usually elbow gives 5


kmeans = KMeans(
    n_clusters=5,
    random_state=42
)



cluster_labels = kmeans.fit_predict(
    X_scaled
)




# ==============================
# 11. ADD CLUSTER COLUMN
# ==============================


df["Cluster"] = cluster_labels



print(df.head())




# ==============================
# 12. VISUALIZE CUSTOMER SEGMENTS
# ==============================


plt.figure(figsize=(10,6))


sns.scatterplot(

    x="Annual Income (k$)",

    y="Spending Score (1-100)",

    hue="Cluster",

    data=df,

    s=100

)



plt.title(
"Customer Segmentation"
)


plt.show()





# ==============================
# 13. CLUSTER CENTROIDS
# ==============================


centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)



centers_df = pd.DataFrame(

centers,

columns=[
"Income",
"Spending Score"
]

)


print("\nCluster Centers")

print(centers_df)




# ==============================
# 14. CUSTOMER SEGMENT ANALYSIS
# ==============================



cluster_analysis = df.groupby(
"Cluster"
).mean(numeric_only=True)



print("\nCluster Analysis")

print(cluster_analysis)





# ==============================
# 15. CUSTOMER COUNT PER CLUSTER
# ==============================


cluster_count = df["Cluster"].value_counts()


print("\nCustomers per Segment")

print(cluster_count)



plt.figure(figsize=(7,5))


sns.countplot(
x="Cluster",
data=df
)


plt.title(
"Customers in Each Segment"
)


plt.show()




# ==============================
# 16. ADD SEGMENT NAMES
# ==============================


segment_names = {

0:"Budget Customers",

1:"Premium Customers",

2:"Average Customers",

3:"Careful Customers",

4:"Potential Customers"

}



df["Customer Segment"] = df["Cluster"].map(
segment_names
)



print(df.head())





# ==============================
# 17. FINAL CUSTOMER REPORT
# ==============================


final_report = df[
[
"CustomerID",
"Gender",
"Age",
"Annual Income (k$)",
"Spending Score (1-100)",
"Customer Segment"
]

]



print(final_report.head(20))




# ==============================
# 18. SAVE FINAL OUTPUT FILE
# ==============================


final_report.to_csv(
"Customer_Segmentation_Result.csv",
index=False
)



print(
"Project Completed Successfully"
)
