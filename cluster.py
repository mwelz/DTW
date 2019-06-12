import numpy as np 
import dtw_funs as dtw
import matplotlib.pyplot as plt
import pathlib as pl
import pandas as pd 

np.random.seed(252311)

## 0) Import data & User Input
core_path = "/Users/mwelz/Documents/work/ra_franses/2019/wk17/code/clusters"
data_df   = pd.read_csv('/Users/mwelz/Documents/work/ra_franses/2019/wk17/data/africa_gdp_index.csv', header=0)
K_set     = [3,4,5]  # number of clusters
n_iters   = 5

## 1) Prepare data
data_df.drop(data_df.tail(4).index,inplace=True) # drop last 4 rows (constant growth everywhere)
del data_df['DATE'] # drop dates from data frame
countries = list(data_df)
X         = data_df.values
# optional: remove Equatorial Guinea and Botswana (much larger scale, they will form their own cluster)
idx       = [i for i,j in enumerate(countries) if(j == "EQGUINEA" or j == "BOTSWANA")]
countries.remove("EQGUINEA")
countries.remove("BOTSWANA")
X         = np.delete(X, idx, axis=1)
X         = X.T # transpose such that the time series are contained in the rows, whereas the T periods are the colums. This is in line with tidy data as we want to cluster the series, not periods! Hence, the series are the observations!
N,P       = np.shape(X)



for K in K_set:

    clusters_dir = core_path + '/' + str(K) + 'clusters'
    pl.Path(clusters_dir).mkdir(parents=True, exist_ok=True) # create a directory to store the plots in

    ## 2) Apply K-Means clustering
    output     = dtw.kMeansClustering(X, K, n_iters=n_iters, max_iters=30)
    S          = output['clusters']
    centroids  = output['centroids']
    dispersion = output['totalDispersons']
    x          = 1960 + np.arange(P) # x-axis

    print("For K=" + str(K) + ", we have " + str(output['iterations']) + " iterations and empty was " + str( output['empty']))


    ## 3) Plots:
    clusters_arr  = np.empty((N,K)) * np.nan  # will keep the elements of each cluster in the columns
    centroids_arr = np.empty((P,K)) * np.nan  # columns hold each cluster's centroid

    for k in np.arange(K):

        # access k-th cluster:
        S_k = S[k]
        N_k = len(S_k)
        X_k = X[S_k,:]
        C_k = centroids[k]

        countries_k = [countries[i] for i in S_k]
        countries_k.append('DBA')
        data = X_k.copy()
        data = np.append(data, np.reshape(C_k,(1,P)), axis=0)

        # store results:
        S_k.extend(np.arange(N-len(S_k)) * np.nan)
        clusters_arr[:,k] = S_k
        centroids_arr[:,k] = C_k


        # create the plot of the k-th cluster:
        plots = []
        f = plt.figure()
        for i in np.arange(N_k+1):
            country = countries_k[i]
            if (country == 'BOTSWANA') or (country == 'EQGUINEA'): # Omit Botswana and Eqguinea (too large)
                continue
            if i != N_k:
                plots.append(plt.plot(x, data[i,:], label=country))
            else:
                plots.append(plt.plot(x, data[i,:], '--ro', label=country))
        plt.legend(loc = 'upper left', prop={'size': 4})
        # plt.show()
        tosave = clusters_dir + "/cluster" + str(k+1) + ".pdf"
        f.savefig(tosave, bbox_inches='tight')
        plt.close(f)

    np.savetxt(clusters_dir + "/clusters.csv", clusters_arr, delimiter=",") # save the clusters, their centroids and their total dispersion!!
    np.savetxt(clusters_dir + "/centroids.csv", centroids_arr, delimiter=",")
    np.savetxt(clusters_dir + "/dispersion.csv", dispersion, delimiter=",")
