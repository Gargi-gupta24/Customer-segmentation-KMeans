# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading dataset
data = pd.read_csv("dataset/Mall_Customers.csv")

# selecting columns
X = data[['Annual Income (k$)',
          'Spending Score (1-100)']]

# empty list
wcss = []

# checking clusters from 1 to 10
for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42
    )

    kmeans.fit(X)

    # storing WCSS values
    wcss.append(kmeans.inertia_)

# plotting graph
plt.plot(range(1, 11), wcss)

# labels
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

# title
plt.title("Elbow Method")
plt.savefig("elbow_output.png")
# show graph
plt.show()