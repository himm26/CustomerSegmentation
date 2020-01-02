import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import k_means

DATA_PATH = '/home/himanshu/Downloads/DATA/'

data = pd.read_csv(DATA_PATH + 'customer-segmentation-dataset/Mall_Customers.csv')


#Categorical features

# dummies = pd.get_dummies(data['Gender'], prefix='Gender')
# data = pd.concat([data, dummies], axis=1)
# data.drop('Gender', axis=1, inplace=True)


X = data[data.columns.difference(['Gender', 'CustomerID'])]

data_transform = minmax_scale(X)

sum_of_squared_distance = []

# TO FIND TO OPTIMAL NUMBER OF CLUSTER FOR GIVEN DATA
K = range(1,15)

for k in K:
    kmean = k_means(X, k)
    sum_of_squared_distance.append(kmean[-1])

plt.plot(K, sum_of_squared_distance, 'bx-')
plt.show()

kmean = k_means(X, 5)

print(kmean) #SHOWS CENTER OF CLUSTERED DATA, WHICH CLUSTER DATA BLONGS TO
