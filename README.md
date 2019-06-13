# DTW
Dynamic Time Warping (DTW) is a widely-applied machine learning algorithm in fields such as speech, pattern, and movement recognition and data mining. In a novel approach, [Franses and Wiemann (2018)](https://ideas.repec.org/p/ems/eureir/109916.html) apply DTW to examine similarities of business cycles by using a feature-based distance function to account for the special properties of business cycles. They furthermore apply _k_-Means clustering to identify DTW-based clusters using DTW Barycenter Averaging (DBA) for the centroids.

This repository provides the code of a project building on the work of [Franses and Wiemann (2018)](https://ideas.repec.org/p/ems/eureir/109916.html), where we use real GDP growth data of 52 African countries from 1961 to 2016 as our time series. The data can be found in [Franses and Vasilev (2019)](https://ideas.repec.org/p/ems/eureir/116541.html).

# Organization of the Repository
This repository contains the following codes I wrote, all of them written in the Python programming language:
- `dtw_main.ipynb` Jupyter notebook to give the reader an overview of the project with some explanations
- `cluster.py` Main file for DTW clustering
- `dtw_funs.py` Contains functions that supply all of the above files
