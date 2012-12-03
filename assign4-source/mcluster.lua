--[[
K-means cluster implementation
Multivariate gaussian implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

Here contains the implementation of a cluster for k-means algorithm. You can
create a cluster object using the constructor 'mcluster' by passing it the
dimension of data, as follows:
t7> c = mcluster(20)

To evaluate the euclidean distance of a vector in this cluster, you can use
c:eval(x) where x is a vector:
t7> = c:eval(torch.rand(20))
2.8470922909493

To learn from a set of vectors with some (unnormalized) weights, you can use
c:learn(x,r) where x is a matrix of size m*n, indicating m vectors. r is a
vector of size m indicating the weights.
t7> x = torch.rand(1000,20)
t7> r = torch.rand(1000)
t7> c:learn(x,r)
t7> = c:eval(torch.rand(20))
1.35269220869

The center vector is stored in c.m. You can use the method set_m to update it
t7> c:set_m(torch.rand(20)) 
]]

-- Return a cluster object
-- n: the dimension of data
function mcluster(n)
   -- Create the object
   local c = {}
   -- Allocate the center
   c.m = torch.zeros(n)
   -- Set the mean
   function c:set_m(m)
      c.m = m:clone()
   end
   -- Evaluate to get the euclidean distance
   function c:eval(x)
      -- Return the distance
      return torch.norm(x-c.m)
   end
   -- Learn the mean from set of examples
   function c:learn(x,r)
      -- The new mean
      local m = torch.zeros(n)
      -- The normalization factor of weights
      local w = torch.sum(r)
      -- Weighted mean
      for i = 1,x:size()[1] do
	 m = m + x[i]*r[i]/w
      end
      -- Set the mean
      c.m = m
   end
   -- Return the object
   return c
end