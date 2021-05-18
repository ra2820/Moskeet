import pandas as pd 
train_df = pd.read_csv('./7_AMCA_Cleaned.csv')
#print(train_df.head())
train_df = train_df.dropna()
train_df_2 = train_df.drop('Genre',1)
train_df_3 = train_df_2.drop('ActivityTime',1)
#print(train_df_2)
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

#print(train_df_2)
scaler=StandardScaler()
scaler.fit(train_df_3)
scaled_data=scaler.transform(train_df_3)
scaled_dataframe = pd.DataFrame(scaled_data,columns=train_df_3.columns)


print(scaled_data)
print(scaled_dataframe)

# PCA

from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(scaled_dataframe)
x_pca=pca.transform(scaled_dataframe)

principalDf = pd.DataFrame(data = x_pca
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

finalDf = pd.concat([principalDf, train_df[['Genre']]], axis = 1)

print(finalDf)

#print(train_df['Genre'].unique())




fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax = plt.axes(projection='3d')
#ax = fig.add_subplot(1,1,1) 

ax.set_title('3 component PCA', fontsize = 20)
targets = train_df['Genre'].unique()
colors = ['red', 'green', 'blue','yellow','black','pink','grey']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Genre'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , alpha=0.2
               , c = color
               , s = 50)

ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.legend(targets)
ax.grid()

plt.show()

# individual PCA

for item in train_df['Genre'].unique():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    target = item
    indicesToKeep = finalDf['Genre'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , finalDf.loc[indicesToKeep, 'principal component 3']
                , alpha=0.2
                , c = 'red'
                , s = 50)
    ax.set_title(item, fontsize = 20)

    ax.grid()
    plt.show()



# TSNE 

from sklearn import manifold
tsne = manifold.TSNE(n_components=3, init='pca',random_state=0, perplexity= 30, n_iter=5000)
trans_data = tsne.fit_transform(scaled_dataframe)
print(trans_data)

principalDf = pd.DataFrame(data = trans_data
             , columns = ['component 1', 'component 2','component 3'])

finalDf = pd.concat([principalDf, train_df[['Genre']]], axis = 1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax = plt.axes(projection='3d')
#ax = fig.add_subplot(1,1,1) 

ax.set_title('TSNE', fontsize = 20)
targets = train_df['Genre'].unique()
colors = ['red', 'green', 'blue','yellow','black','pink','grey']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Genre'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'component 1']
               , finalDf.loc[indicesToKeep, 'component 2']
               , finalDf.loc[indicesToKeep, 'component 3']
               , alpha=0.2
               , c = color
               , s = 50)

ax.set_xlabel(fontsize = 15)
ax.set_ylabel(fontsize = 15)
ax.set_zlabel(fontsize = 15)
ax.legend(targets)
ax.grid()

plt.show()

