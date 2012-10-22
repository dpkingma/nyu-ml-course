--[[
Trainers implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 09/24/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file consists of two kinds of object: step objects and trainer objects.

Step objects are used for giving a step of a gradient update. It should have
the field step:step(t) which returns a step at step t.

Trainer objects are used to train and test a model towards a dataset using
generic learning algorithms such as gradient descent (but not closed-form
solutions). We recommend the following convention for implementing trainers:

You should write a function for each kind of trainer to initialize them. The
initializer function should accept a model object and a step object. Additional
parameters are at your choice.

A trainer object consists of the following fields:

trainer:train(dataset, ...): train the model with dataset (using model:dw(x,y)
and step:step(t)). Additional parameters are at your choice. The average loss
and error rate on the training dataset should be returned.

trainer:test(dataset, ...): test the model with dataset. Additional parameters
are at your choice. The average loss and error rate on the testing dataset
should be returned.

As an example, we provide a batch trainer (trainerBatch).

For more information regarding models, please refer to model.lua.

]]


-- A constant stepsize object
function stepCons(stepsize)
   -- A step object
   local step = {}
   -- The function of a step
   function step:eta(t)
      return stepsize
   end
   -- return this step object
   return step
end

-- A harmonic stepsize object: eta(t) = alpha / (beta+t)
function stepHarm(alpha, beta)
   -- A step object
   local step = {}
   -- The function of a step
   function step:eta(t)
      return alpha/(beta+t)
   end
   -- return this step object
   return step
end

-- A batch trainer using a module and a single number step object
-- model: some model object; step: some step object
function trainerBatch(model, step)
   local trainer = {}
   -- Train a module using batch method with max_step size
   function trainer:train(dataset, max_step)
      -- Do this many steps of training
      for i = 1,max_step do
	 -- Compute the batch gradients
	 local dw = torch.zeros(model.w:size())
	 -- Iterative average
	 for j = 1,dataset:size() do
	    dw = dw*(j-1)/j + model:dw(dataset[j][1], dataset[j][2])/j
	 end
	 -- Take batch gradient step
	 model.w = model.w - dw*step:eta(i)
      end
      -- return the training loss and error
      return trainer:test(dataset)
   end
   -- Test a module, returning with average loss and error rate
   function trainer:test(dataset)
      -- Average loss
      local loss = 0
      -- Counter for wrong classifications
      local error = 0
      -- Iterate over all the datasets
      for i = 1,dataset:size() do
	 -- Iterative loss averaging
	 loss = loss*(i-1)/i + model:l(dataset[i][1], dataset[i][2])/i
	 -- Iterative error rate computation
	if torch.sum(torch.ne(model:g(dataset[i][1]), dataset[i][2])) == 0 then
	    error = error*(i-1)/i
	 else
	    error = (error*i-error + 1)/i
	 end
      end
      -- Return the loss and error ratio
      return loss, error
   end
   -- Return this trainer
   return trainer
end

-- A stochastic gradient descent trainer using a module and a single number step object
-- model: some model object; step: some step object
function trainerSGD(model, step)
   -- Remove the following line and add your stuff
   print("You have to define this function by yourself!");
end
