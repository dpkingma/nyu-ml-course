--[[
K-Means clustering algorithm implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The k-means algorithm should be presented here. You can implement it in any way
you want. For your convenience, a clustering object is provided at mcluster.lua

Here is how I implemented it:

kmeans(n,k) is a constructor to return an object km which will perform k-means
algorithm on data of dimension n with k clusters. The km object stores the i-th
cluster at km[i], which is an mcluster object. The km object has the following
methods:

km:g(x): the decision function to decide which cluster the vector x belongs to.
Return a scalar representing cluster index.

km:f(x): the output function to output a prototype that could replace vector x.

km:learn(x): learn the clusters using x, which is a m*n matrix representing m
data samples.
]]

dofile("mcluster.lua")

-- Create a k-means learner
-- n: dimension of data
-- k: number of clusters
function kmeans(n,k)
-- Remove the following line and add your stuff
print("You have to define this function by yourself!");
end
