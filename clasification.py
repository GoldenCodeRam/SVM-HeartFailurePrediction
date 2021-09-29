from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


data = pd.read_csv ('Data.csv', sep = ',')

X = data.drop('HeartDisease', axis=1) # caracteristicas
Y = data['HeartDisease'] # clase 

# estandarizar
scaler = StandardScaler()

# estandarizar las variables
scaler.fit(X)
X_scaled = scaler.transform(X)

# transformar los datos
pca = PCA(n_components = 11, random_state =457).fit(X_scaled)

# calcular varianza

import numpy as np

print("varianza", sum(pca.explained_variance_ratio_ * 100))

np.cumsum(pca.explained_variance_ratio_ * 100)

plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel('Number of variables')
plt.ylabel('Explained variance')
plt.show()

print("Variance for one variables" , np.cumsum(pca.explained_variance_ratio_ * 100)[0])
print("Variance for two variables" , np.cumsum(pca.explained_variance_ratio_ * 100)[1])
print("Variance for three variables" , np.cumsum(pca.explained_variance_ratio_ * 100)[2])







# Transformar datos a  2D
pca_2D = PCA(n_components = 2, random_state =457).fit(X_scaled)
X_pca_2D = pca_2D.transform(X_scaled)

plt.figure(figsize =(10,7))
sns.scatterplot(x = X_pca_2D[:,0], y=X_pca_2D[:,1], s = 70, hue=Y)
plt.show()


# Transformar 3D

from mpl_toolkits import mplot3d

pca_3D = PCA(n_components = 3, random_state =457).fit(X_scaled)
X_pca_3D = pca_3D.transform(X_scaled)

fig = plt.figure(figsize = (12, 8))
ax = plt.axes(projection="3d")

sctt = ax.scatter3D(X_pca_3D[:, 0], X_pca_3D[:,1], X_pca_3D[:,2], c = Y, s=50, alpha=0.6)

plt.title("3D Scatterplot: 72.64% of the variability captured", pad=15)
plt.show()

