--[[
Kernels implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 10/08/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you should implement kernels that will be used combining with
the xsvm library.

A kernel object is a callable k(x,y) such that it computes a positive definite
symmetric kernel function based on two equally sized torch tensors x and y. It
should return a double number.

As an example, the linear kernel (inner-product between x and y) is given.

For additional information regarding xsvm, please refer to xsvm.lua.
--]]

-- The linear kernel
function kernLin()
   -- Create a callable which returns innner product
   local kfunc = function (x,y)
      return torch.dot(x,y)
   end
   -- Return this callable
   return kfunc
end

-- The polynomial kernel kernel(x,y) = (x^T y + c)^d
-- c: the constant to be added to the inner product
-- d: the degree of the polynomial
function kernPoly(c, d)
   -- Remove the following line and add your stuff
   print("You have to define this function by yourself!");
end

-- A kind of PDS kernels from norms, which are NDS.
-- kfunc(x,y) = exp(-\gamma ||x - y||_p^p)
-- The gaussian radial basis is a special case of this kernel kind where p = 2.
function kernNorm(gamma, p)
   -- Create a callable which returns this kernel
   local kfunc = function (x,y)
      return math.exp(-gamma*torch.sum(torch.pow(torch.abs(x-y), p)))
   end
   -- Return this callable
   return kfunc
end
