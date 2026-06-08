#import pandas as pd

#data = pd.read_csv("dataset/Mall_Customers.csv")

#print(data.head())

# importing libraries
#import pandas as pd
#import matplotlib.pyplot as plt

# loading dataset
#data = pd.read_csv("dataset/Mall_Customers.csv")

# creating graph
plt.scatter(
    #data['Annual Income (k$)'],
    #data['Spending Score (1-100)']
#)

# labels
#plt.xlabel("Annual Income")

#plt.ylabel("Spending Score")

# title
#plt.title("Customer Distribution")

# showing graph
#plt.show()

# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading dataset
data = pd.read_csv("dataset/Mall_Customers.csv")

# selecting important columns
X = data[['Annual Income (k$)',
          'Spending Score (1-100)']]

# applying K-Means
kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

# predicting clusters
y_kmeans = kmeans.fit_predict(X)

# plotting clusters
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y_kmeans,
    cmap='rainbow'
)

# labels and title
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.savefig("output.png")
# show graph
plt.show()