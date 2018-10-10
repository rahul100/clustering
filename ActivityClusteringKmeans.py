
# coding: utf-8

# In[229]:

import pandas as pd
import numpy as np
#import cPickle as pickle
import pickle
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic(u'matplotlib inline')
import pdb 
import collections



# In[137]:

dataDir = '/data/analytics/Nikunj/'
modDtBasedSegment_1to3month_train = pd.read_csv(dataDir+'modDtBasedSegment_1to3month_train.csv')
modDtBasedSegment_1to3month_train.head()


# # Plot PCA and T-NSE representation to visualize clustering quality

# In[138]:

def plotPCA(data_train_cluster):

    pca = PCA(n_components=2).fit(data_train_cluster.drop('cluster' , axis=1))
    pca_2d = pca.transform(data_train_cluster.drop('cluster' , axis=1))
    
    pca_df = pd.DataFrame(pca_2d)
    pca_df.columns = ['PCAdim1' , 'PCAdim2']
    pca_df['cluster']= data_train_cluster['cluster']
    print pca_df.head()
                           
    ### Plotting the PCA dimesions and the cluster

    fig = plt.figure()

    ax = fig.add_subplot(111)
    scatter = ax.scatter(pca_df['PCAdim1'],pca_df['PCAdim2'],
                     c=pca_df['cluster'],s=50)
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    plt.colorbar(scatter) 


# In[139]:

def plotTNSE(data_train_cluster):
    sample = data_train_cluster.sample(n=5000).reset_index(drop=True)
    tnse = TSNE(n_components=2).fit_transform(sample.drop('cluster', axis=1))
    print tnse.shape
    tnse_df = pd.DataFrame(tnse)
    tnse_df.columns = ['TNSEdim1' , 'TNSEdim2']
    tnse_df['cluster']= sample['cluster']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(tnse_df['TNSEdim1'],tnse_df['TNSEdim2'],
                     c=tnse_df['cluster'],s=50)
    ax.set_title('K-Means Clustering tnse rep')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    plt.colorbar(scatter)


# In[140]:

def getStats(model_file,data_file):
    km_pkl = open(dataDir+model_file, 'rb') 
    km = pickle.load(km_pkl)
    data_train = pd.read_csv(dataDir+data_file)
    cluster_values = km.predict(data_train.as_matrix())
    data_train_cluster = data_train
    data_train_cluster['cluster'] = cluster_values
    print collections.Counter(list(cluster_values)) 
#    print "Cluster Centers: "  , km_1to3month.cluster_centers_
    print "Mean of the cluster\n" , data_train_cluster.groupby('cluster').mean()
    print "Count of the clusters\n", data_train_cluster.groupby('cluster').count()
    print "PCA"
    plotPCA(data_train_cluster)
    
    
    print "TNSE"
    plotTNSE(data_train_cluster)
    
    print "********************************************************************************************"
    
    


# In[141]:

getStats('km_1to3month.p','modDtBasedSegment_1to3month_train.csv')


# In[142]:

getStats('km_3to6month.p','modDtBasedSegment_3to6month_train.csv')


# In[143]:

getStats('km_post6month.p','modDtBasedSegment_post6month_train.csv')


# # Distribution of applies across mod-date buckets 

# In[94]:

modDtBasedSegment_upto1month_train = pd.read_csv(dataDir+'modDtBasedSegment_upto1month_train.csv')
modDtBasedSegment_1to3month_train = pd.read_csv(dataDir+'modDtBasedSegment_1to3month_train.csv')
modDtBasedSegment_3to6month_train = pd.read_csv(dataDir+'modDtBasedSegment_3to6month_train.csv')
modDtBasedSegment_post6month_train = pd.read_csv(dataDir+'modDtBasedSegment_post6month_train.csv')


# In[132]:

apply_viewupto1month = np.sum(np.sum(modDtBasedSegment_upto1month_train , axis =1),axis = 0)
apply_view1to3month = np.sum(np.sum(modDtBasedSegment_1to3month_train , axis =1),axis = 0)
apply_view3to6month = np.sum(np.sum(modDtBasedSegment_3to6month_train , axis =1),axis = 0)
apply_viewpost6month = np.sum(np.sum(modDtBasedSegment_post6month_train , axis =1),axis = 0)
stats = pd.DataFrame.from_dict({"apply_viewupto1month":[888523838.0] , "apply_view1to3month":[248856736.0] , "apply_view3to6month": [143731644.0] , "apply_viewpost6month": [223172305.0]} )
stats['total']  = np.sum(stats , axis=1)
stats_perc = {}
for column in ['apply_view1to3month','apply_view3to6month','apply_viewpost6month','apply_viewupto1month']:
    stats_perc[column] = round(stats[column]/stats['total']*100,2)
stats_perc


# # Calculating BIC for all the clusters & the model

# In[145]:

from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
    BIC_clusters =[n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]
    return(BIC , BIC_clusters)


# In[147]:

def getBIC(model_file,data_file):
    km_pkl = open(dataDir+model_file, 'rb') 
    km = pickle.load(km_pkl)
    data_train = pd.read_csv(dataDir+data_file).as_matrix()
    print model_file
    print compute_bic(km,data_train)


# In[149]:

getBIC('km_upto1month.p','modDtBasedSegment_upto1month_train.csv')
getBIC('km_1to3month.p','modDtBasedSegment_1to3month_train.csv')
getBIC('km_3to6month.p','modDtBasedSegment_3to6month_train.csv')
getBIC('km_post6month.p','modDtBasedSegment_post6month_train.csv')


# # Subdividing the low activity clusters further

# In[151]:

km_pkl = open(dataDir+'km_1to3month.p', 'rb') 
km = pickle.load(km_pkl)
data_train = pd.read_csv(dataDir+'modDtBasedSegment_upto1month_train.csv')
cluster_values = km.predict(data_train.as_matrix())
data_train_cluster = data_train
data_train_cluster['cluster'] = cluster_values


# In[154]:

data_train_cluster.groupby('cluster').count()
upto1month_high = data_train_cluster[data_train_cluster['cluster']==0]
upto1month_high = upto1month_high.drop('cluster',axis=1)
upto1month_high.to_csv(dataDir + 'modDtBasedSegment_upto1month_low.csv',index = False)


# In[162]:

getStats('km_upto1month_low.p','modDtBasedSegment_upto1month_low.csv')


# In[164]:

km_pkl = open(dataDir+'km_upto1month_low.p', 'rb') 
km = pickle.load(km_pkl)
data_train = pd.read_csv(dataDir+'modDtBasedSegment_upto1month_low.csv')
cluster_values = km.predict(data_train.as_matrix())
data_train_cluster = data_train
data_train_cluster['cluster'] = cluster_values


# In[181]:

data_train_cluster[(data_train_cluster['cluster']==0) & (data_train_cluster['0to15days']>100.0)]


# # Merging all data irrespective of mod date and kmeans clustering for cluster size = [3,4...10]

# In[180]:

getStats('Model_Rahul/km_3_clusters.p','Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')


# In[178]:

getStats('Model_Rahul/km_4_clusters.p','Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')


# In[179]:

getStats('Model_Rahul/km_5_clusters.p','Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')


# In[182]:

getStats('Model_Rahul/km_6_clusters.p','Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')


# In[202]:

getStats('Model_Rahul/km_10_clusters.p','Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')


# In[183]:

from sklearn.mixture import GaussianMixture
viewApplyActivity_since_20180101_20180815_train = pd.read_csv('viewApplyActivity_since_20180101_20180815_train.csv')
gmm = GaussianMixture(n_components=6, covariance_type='full')
gmm.fit(viewApplyActivity_since_20180101_20180815_train.as_matrix())


# In[184]:

pickle.dump(gmm, open("gmm_6clusters.p", 'wb'))


# In[185]:

getStats('Model_Rahul/gmm_6clusters.p','Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')


# In[186]:

gmm.bic(viewApplyActivity_since_20180101_20180815_train.as_matrix())


# In[201]:

gmm.predict(x)


# In[200]:

x = np.array([90.0,180.0,0.0,0.0,0.0]).reshape((1,5))


# In[218]:

getStats('Model_Rahul/gmm_4_clusters.p','Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')


# In[204]:

getStats('Model_Rahul/gmm_9_clusters.p','Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')


# In[214]:

km_pkl = open(dataDir+'Model_Rahul/gmm_6_clusters.p', 'rb') 
km = pickle.load(km_pkl)
data_train = pd.read_csv(dataDir+'Model_Rahul/viewApplyActivity_since_20180101_20180815_train.csv')
cluster_values = km.predict(data_train.as_matrix())
data_train_cluster = data_train
data_train_cluster['cluster'] = cluster_values
data_train_cluster = pd.DataFrame(data_train_cluster)


# # Clustering on 3 days data

# In[238]:

'Hi'

import sys
sys.path.append('/data/analytics/Nikunj/Model/')
from DataConnections.MySQLConnect.MySQLConnect import MySQLConnect
import MySQLdb

host = '172.10.112.94'
username="user_analytics"
user = username
password="anaKm7Iv80l"
database="mynaukri"
# unix_socket="/tmp/mysql.sock"
# unix_socket="/tmp/mysql3306.sock"
#port = port
port = 3308

import pandas as pd
from pandas import DataFrame




#mysql_conn = MySQLdb(database, host, user, password, port) 
mysql_conn = MySQLdb.connect(db = database,host =  host,user =  user,passwd =  password,port =  port)
cmd_for3days = '''select USERNAME, count(*) as Apply_count_3days from APPLY_LOG_FULL where applogDate >= '2017-07-02' and applogDate < '2017-07-05' group by USERNAME'''
df_for3days = pd.read_sql(cmd_for3days, mysql_conn)
print len(df_for3days)


# In[239]:

mat_for3days = df_for3days.drop("USERNAME" , axis=1).as_matrix()
km_df_for3days = sklearn.cluster.KMeans(n_clusters=3).fit(mat_for3days)


# In[240]:

km_df_for3days.cluster_centers_


# In[241]:

df_for3days['cluster'] = km_df_for3days.labels_


# In[242]:

df_for3days.head()


# In[243]:

print "Mean of the cluster\n" , df_for3days.groupby('cluster').mean()
print "Count of the clusters\n", df_for3days.groupby('cluster').count()


# In[244]:

print "Min of the cluster\n" , df_for3days.groupby('cluster').min()
print "Max of the clusters\n", df_for3days.groupby('cluster').max()


# In[245]:

print "Mean of the cluster\n" , df_for3days.groupby('cluster').sum()


# In[1]:

a = {1:'One',2:'Two'}
a.update({2:'Three'})


# In[3]:

a.update({3:'Three'})


# In[4]:

a


# In[ ]:



