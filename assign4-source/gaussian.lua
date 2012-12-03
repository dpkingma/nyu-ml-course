--[[
Multivariate gaussian implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

Here contains the implementation of a multivariate gaussian. You can create a
gaussian object using the constructor 'gaussian' by passing it the dimension of
data, as follows:
t7> g = gaussian(20)

To evaluate the probability density of a vector in this gaussian, you can use
g:eval(x) where x is a vector:
t7> = g:eval(torch.rand(20))
5.4393975890361e-10

To learn from a set of vectors with some (unnormalized) weights, you can use
g:learn(x,r,p) where x is a matrix of size m*n, indicating m samples. r is a
vector of size m indicating the weights. p is a regularization parameter to
ensure the covariance matrix is non-singular. Something like 1e-4 is good.
t7> x = torch.rand(1000,20)
t7> r = torch.rand(1000)
t7> g:learn(x,r,1e-4)
t7> = g:eval(torch.rand(20))
0.014640555053801

The center of gaussian is stored at g.m. You can use the set_m method to modify
it
t7> g:set_m(torch.rand(20))

The covariance matrix, its inverse and a normalization factor is stored at g.c,
g.ci and g.normalizer. You can modify them using a new covariance matrix with
the set_c method, i.e.,
t7> g:set_c(torch.eye(20))
]]

-- Return a gaussian object
-- n: the dimension
function gaussian(n)
   -- A constant pi
   local pi = 3.1415926535897932385
   -- Create the object
   local g = {}
   -- Allocate the mean
   g.m = torch.zeros(n)
   -- Allocate the covariance matrix
   g.c = torch.eye(n)
   -- Stores the inverse of the covariance matrix
   g.ci = torch.inverse(g.c)
   -- Initialize the normalizer (2*pi)^{n/2} * |c|^{1/2}
   g.normalizer = math.sqrt(math.pow(2*pi,n)*torch.prod(torch.symeig(g.c),1)[1])
   -- Set the mean
   function g:set_m(m)
      -- Copy data
      g.m = m:clone()
   end
   -- Set the covariance matrix
   function g:set_c(c)
      -- Copy data
      g.c = c:clone()
      -- Get the inverse
      g.ci = torch.inverse(g.c)
      -- Compute the normalizer (2*pi)^{n/2} * |c|^{1/2}
      g.normalizer = math.sqrt(math.pow(2*pi,n)*torch.prod(torch.symeig(g.c),1)[1])
   end
   -- Return the likelihood of vector x under the gaussian
   function g:eval(x)
      -- Return the probability
      return math.exp(-1/2*torch.dot(x-g.m,torch.mv(g.ci,x-g.m)))/g.normalizer
   end
   -- Learn the mean and variance from a set of vectors
   -- x: a matrix of size m*n, indicating m samples
   -- r: a vector of size m indicating the weight of each sample
   -- p: a regularization parameter to ensure g.c is non-singular. Make something like 1e-4 will be good.
   function g:learn(x,r,p)
      -- The new mean
      local m = torch.zeros(n)
      -- The new covariance
      local c = torch.eye(n)*p
      -- The normalization factor of weights
      local w = torch.sum(r)
      -- Weighted mean
      for i = 1,x:size()[1] do
	 m = m + x[i]*r[i]/w
      end
      -- Weighted covariance
      for i = 1,x:size()[1] do
	 c = c + torch.ger(x[i]-m,x[i]-m)*r[i]/w
      end
      -- Set the mean and covariance
      g:set_m(m)
      g:set_c(c)
   end
   -- Return the object
   return g
end
