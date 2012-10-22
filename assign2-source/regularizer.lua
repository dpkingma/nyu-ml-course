--[[
Models implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 09/24/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you should implement regularizers satisfying the following
convention, so that they can be used by models you will implement in model.lua.

A regularizer object r consists of the following fields:

r:l(w): the loss function. The return value should be a scalar number.

r:dw(w): the gradient function. The return value should be a tensor of the
same dimension with w.

The way I would recommend you to program the regularizer above is to write a
function which returns a table containing the fields above. As an example, a
l2 regularizer is provided.

For additional information regarding models, please refer to model.lua.

]]

-- No regularization
function regNone()
   local regularizer = {}
   -- The loss is 0
   function regularizer:l(w) return 0 end
   -- The gradient is 0
   function regularizer:dw(w) return torch.zeros(w:size()) end
   -- Return this regularizer
   return regularizer
end

-- l2 regularization module
-- lambda: the regularization parameter
function regL2(lambda)
   local regularizer = {}
   -- The loss value of this regularizer
   function regularizer:l(w) return torch.sum(torch.pow(w,2))*lambda end
   -- The gradient of l with respect to w
   function regularizer:dw(w) return w*2*lambda end
   -- Return this regularizer
   return regularizer
end

-- l1 regularization module
-- lambda: the regularization parameter
function regL1(lambda)
   -- Remove the following line and add your stuff
   print("You have to define this function by yourself!");
end
