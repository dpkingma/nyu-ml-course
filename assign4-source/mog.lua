--[[
Mixture of Gaussians Implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

The mixture of gaussians algorithm should be presented here. You can implement
it in anyway you want. For your convenience, a multivariate gaussian object is
provided at gaussian.lua.

Here is how I implemented it:

mog(n,k) is a constructor to return an object m which will perform MoG
algorithm on data of dimension n with k gaussians. The m object stores the i-th
gaussian at m[i], which is a gaussian object. The m object has the following
methods:

m:g(x): The decision function which returns a vector of k elements indicating
each gaussian's likelihood

m:f(x): The output function to output a prototype that could replace vector x.

m:learn(x,p,eps): learn the gaussians using x, which is an m*n matrix
representing m data samples. p is regularization to keep each gaussian's
covariance matrices non-singular. eps is a stop criterion.
]]

dofile("gaussian.lua")
dofile("kmeans.lua")

-- Create a MoG learner
-- n: dimension of data
-- k: number of gaussians
function mog(n,k)
-- Remove the following line and add your stuff
print("You have to define this function by yourself!");
end
