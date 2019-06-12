import numpy as np
import pandas as pd

def get_accumulated_cost_matrix(X,Y):
    ''' Obtain the accumulated cost matrix as in Müller (2007) plus the local cost'''

    N = len(X)
    M = len(Y)

    ## Initialize the accumulated cost matrix Gamma (called "D" in Müller (2007))
    Gamma = np.empty((N,M))

    ## Initialize matrices holding the local cost values at each increment
    f_local_X_mat = np.zeros((N,2))
    f_local_Y_mat = np.zeros((M,2))

    ## Get accumulated cost matrix (Müller (2007), p.73)
    for i in np.arange(N):
        f_local_X, f_global_X = dist_measures(X,i)
        f_local_X_mat[i,:] = f_local_X

        for j in np.arange(M):
            f_local_Y, f_global_Y = dist_measures(Y,j)
            distance = np.abs(f_local_X[0] - f_local_Y[0]) + np.abs(f_local_X[1] - f_local_Y[1]) + np.abs(f_global_X[0] - f_global_Y[0]) + np.abs(f_global_X[1] - f_global_Y[1])
            f_local_Y_mat[j,:] = f_local_Y

            # the following is equivalent to the initialization suggested in Müller (2007), p.73:
            if i == 0 and j == 0:
                gamma = distance
            elif i == 0 and j != 0:
                gamma = Gamma[i,j-1] + distance
            elif i != 0 and j == 0:
                gamma = Gamma[i-1,j] + distance
            else:
                gamma = np.min(( Gamma[i-1,j-1], Gamma[i-1,j], Gamma[i,j-1] )) + distance
            Gamma[i,j] = gamma
    # total warping cost:        
    cost = Gamma[N-1, M-1] / (N+M)  # (scaled) upper right element of D
    return Gamma, cost, f_local_X_mat, f_local_Y_mat 



def dist_measures(series, idx):
    '''computes f_global and f_local by means of growth'''

    f_local     = [0,0]
    f_global    = [0,0]
    N           = len(series)

    if idx == 0:
        mean_before_idx = np.mean(series[:(idx+1)])
        mean_after_idx  = np.mean(series[(idx+2):])
        f_local[0]      = (series[idx+1] - series[idx]) / series[idx]
        f_local[1]      = (series[idx+1] - series[idx+2]) / series[idx+2]
        f_global[0]     = (series[idx+1] - mean_before_idx) / mean_before_idx
        f_global[1]     = (series[idx+1] - mean_after_idx) / mean_after_idx

    elif idx == N-1:
        mean_before_idx = np.mean(series[:(idx-1)]) 
        mean_after_idx  = np.mean(series[idx:])
        f_local[0]      = (series[idx-1] - series[idx-2]) / series[idx-2]
        f_local[1]      = (series[idx-1] - series[idx]) / series[idx]
        f_global[0]     = (series[idx-1] - mean_before_idx) / mean_before_idx
        f_global[1]     = (series[idx-1] - mean_after_idx) / mean_after_idx 

    else:
        mean_before_idx = np.mean(series[:idx]) 
        mean_after_idx  = np.mean(series[(idx+1):]) 
        f_local[0]      = (series[idx] - series[idx-1]) / series[idx-1]
        f_local[1]      = (series[idx+1] - series[idx]) / series[idx]
        f_global[0]     = (series[idx] - mean_before_idx) / mean_before_idx
        f_global[1]     = (series[idx] - mean_after_idx) / mean_after_idx
    
    return f_local, f_global


def get_optimal_path(D):
    ''' D is the accumulated cost Matrix; implements the algorithm in Müller (2007), p.73'''

    N, M = np.shape(D)
    ct = N * M
    pstar = np.zeros((ct,2))
    ct = ct-1 # reduce confusion

    n = N - 1
    m = M - 1
    pstar[ct,0] = n
    pstar[ct,1] = m
    while n > 0 or m > 0:
        if n == 0:
            m = m - 1
        elif m == 0:
            n = n - 1
        else:
            if D[n-1,m] == np.min((D[n-1,m-1], D[n-1,m], D[n,m-1])):
                n = n - 1
            elif D[n,m-1] == np.min((D[n-1,m-1], D[n-1,m], D[n,m-1])):
                m = m - 1
            else:
                n = n - 1
                m = m - 1
        ct = ct-1
        pstar[ct, 0] = n
        pstar[ct, 1] = m
    pstar = pstar[ct:]
    pstar = pstar.astype(int)
    return pstar
    

def initialize_kMeans(X, K):
    ''' 
    Initializes the k-Means clustering algorithm as described in Franses & Wiemann (2018).
    Requires: data matrix X [NxP], number of clusters K.
    Returns an initial list of cluster assignmentments S. S is of length K and each element is a list itself, i.e. a cluster, that contains all observations that have been assigned to that particular cluster.
    '''
    ## Initialization of KMeans:
    # Requires: data matrix X [NxP], number of clusters K
    N = np.shape(X)[0]
    S = [[] for _ in range(K)] # initialize clusters. Each sublist (i.e. cluster) k will contain the observations of that particular cluster.
    C = [[] for _ in range(K)] # inizialize centroids of each cluster. Each sublist (i.e. cluster) k will contain its [Px1] centroid.
    assigned_obs = []          # indices of observations that have been assigned to be an initial centroid already. Will be of length K in the end.

    ### 0) For each centroid, find a centroid
    for k in np.arange(K): 
        if k == 0:
            i = np.random.randint(0, N+1) # draw random observation as a start
        else: 
            cost_list = [None]*N # holds warping cost for each observation
            for i in np.arange(N):
                if i in assigned_obs:
                    cost_list[i] = 0 # if i has already been assigned, make it have 0 influence (will never be picked as cost is always > 0)
                    continue
                cost_per_cluster = [None]*len(assigned_obs)
                for c in np.arange(len(assigned_obs)):
                    _, cost, _, _       = get_accumulated_cost_matrix(X[i,:], C[c])
                    cost_per_cluster[c] = cost
                cost_list[i] = np.sum(cost_per_cluster) ** 2
            
            ## Now, obtain the probabilities and pick the largest
            summed_SqCost = np.sum(cost_list)
            omega         = cost_list / summed_SqCost # equation (5) in Franses & Wiemann (2018) for all i
            i             = np.argmax(omega)

        C[k] = list(X[i,:]) # TODO: maybe remove list() if there is a problem with DTW
        S[k].append(i)
        assigned_obs.append(i)

    ### 1) With the centroids, assign the remaining observations

    for i in np.arange(N):
        if i in assigned_obs:
            continue
        
        cost_list = [None]*K
        for k in np.arange(K):
            _, cost, _, _ = get_accumulated_cost_matrix(X[i,:], C[k])
            cost_list[k]  = cost
        k = np.argmin(cost_list) # the cluster to assign observation i to
        S[k].append(i)
        assigned_obs.append(i)

    return S


def DBA(X, n_iters = 10):
    '''
    Implements algorithm 5 in Petitjean et al. (2011). X [NxP] is a data matrix. For time series data, the time series are contained in the rows (such that P=T for the columns).
    '''

    N,P = np.shape(X)
    currentAverageSequence = np.ones((P,1))*(-1e20) # ensure that the initial point-mapping is the diagonal
    assocTab = [[] for _ in range(P)] # assocTab maps each element in 'currentAverageSequence' with one or more elements in the other series. These mappings are contained in each sub-list of 'assocTab'.

    for _ in np.arange(n_iters):
        totalDispersion = 0
        for i in np.arange(N): 
            seq = X[i,:]
            acc_cost_mat,cost,_,_ = get_accumulated_cost_matrix(currentAverageSequence, seq)
            totalDispersion += cost
            path = get_optimal_path(acc_cost_mat) # counts forward
            [assocTab[p].append(seq[path[p,1]]) for p in range(P)] # map each value of current Sequence with values from the other sequences (since it doesn't matter where the mapping comes from, I do not report from which series the mapping was taken.)
        newAverageSequence = [np.mean(assocTab[p]) for p in range(P)] # the Barycenter
        currentAverageSequence = newAverageSequence.copy()
    return {'DBA':currentAverageSequence, 'totalDispersion':totalDispersion}


def kMeansClustering(X, K, n_iters=5, max_iters = 50):
    '''
    Performs K-Means clustering. X is data matrix, K is number of clusters and n_iters is the number of iterations to be performed for the DBA. Returns the clusters and how many iterations were required. Since we need a random series as initialization, the usage of a seed is recommended.
    '''
    # Let X be an NxP data matrix.
    # we have i=1,...,N observations (time series here) 
    # we have j=1,...,P variables (number of time series elements here (P=T))
    # we have k=1,...,K clusters (pre-defined)
    # Let S be a list that contains K sub-lists. Each sublist k contains the indices of the observations belonging to cluster k.
    N,P = np.shape(X)
    S   = initialize_kMeans(X, K)
    ct  = 1

    while True:
        S_new     = [[] for _ in range(K)] # initialize list of K empty sublists
        for i in np.arange(N):
            inner_products = [] 
            for k in range(K):
                X_cluster_k = X[S[k],:] # observations contained in k-th cluster
                centroid_k  = DBA(X_cluster_k, n_iters=n_iters)
                centroid_k  = np.reshape(centroid_k['DBA'], (P,1)) 
                deviation   = np.reshape(X[i,:], (P,1)) - centroid_k
                inner_products.append(float(np.dot(deviation.T, deviation)))
            S_new[np.argmin(inner_products)].append(i)

        emptyness = 0 # takes 1 if there were empty clusters
        ### Dealing with empty clusters: Pop largest deviation(s) from biggest cluster to empty cluster(s)
        clusterSizes = [len(S_new[l]) for l in range(K)]
        if 0 in clusterSizes:  # Check for empty clusters
            emptyness                = 1
            emptyClusters            = [i0 for i0,x0 in enumerate(clusterSizes) if x0 == 0] # find empty cluster(s)
            X_largest_cluster        = X[S_new[np.argmax(clusterSizes)],:]
            N_largest_cluster        = np.shape(X_largest_cluster)[0]
            centroid_largest_cluster = DBA(X_largest_cluster, n_iters=n_iters)
            centroid_largest_cluster = np.reshape(centroid_largest_cluster['DBA'], (P,1))     
            deviations = []
            for i0 in range(N_largest_cluster):
                deviation = np.reshape(X_largest_cluster[i0,:], (P,1)) - centroid_largest_cluster
                deviations.append(float(np.dot(deviation.T, deviation)))

            largest_deviations = sorted(range(len(deviations)), key=lambda i0: np.abs(deviations)[i0])
            largest_deviations = largest_deviations[::-1] 
            largest_deviations =  [largest_deviations[i0] for i0 in range(len(emptyClusters))] # selects the K0 largest deviations' indices (K0 = number of empty clusters)
            for i0 in np.arange(len(emptyClusters)):
                toBePopped = S_new[np.argmax(clusterSizes)][(largest_deviations[i0])] #S_new[np.argmax(clusterSizes)].pop(largest_deviations[i0])
                S_new[emptyClusters[i0]].append(toBePopped)
            toBeDeleted = [S_new[np.argmax(clusterSizes)][i0] for i0 in largest_deviations]
            print(toBeDeleted)
            S_new[np.argmax(clusterSizes)] = [i0 for i0 in S_new[np.argmax(clusterSizes)] if i0 not in toBeDeleted]
        ### Now, there are no empty clusters anymore.

        ## Check if changes occured. (Use sets as the order of clusters and their observations doesn't matter).
        S_setOfSets     = set(frozenset(k) for k in S) # use frozensets to make the inner sets hashable
        S_new_setOfSets = set(frozenset(k) for k in S_new)
        if (S_new_setOfSets == S_setOfSets) or (ct == max_iters):
            break
        else:
            S   = S_new.copy()
            ct += 1
            print(ct)

    ## Centroids of each cluster
    centroids = [[] for _ in range(K)]
    totalDispersions = [] # total dispersion per cluster
    for k in np.arange(K):
        X_cluster_k  = X[S[k],:] # observations contained in k-th cluster
        centroid_k = DBA(X_cluster_k, n_iters)
        centroids[k] = list(centroid_k['DBA'])
        totalDispersions.append(centroid_k['totalDispersion'])

    ## Organize output in a dictionary
    output = {'clusters':S, 'centroids':centroids, 'totalDispersons':totalDispersions, 'iterations':ct, 'empty':emptyness}
    return output

