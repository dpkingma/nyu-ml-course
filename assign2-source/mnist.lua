--[[
MNIST dataset implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.2, 10/04/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In general, all you need to do is to load this file with mnist_train_32x32.t7
and mnist_test_32x32.t7 presented in you current directory as follows:
t7> dofille "mnist.lua"
then, you can split out shuffled and normalized training and testing data by
calling spambase:getDatasets(train_size,test_size), for example:
t7> train, test = mnist:getDatasets(6000,1000)

The sets train and tes follow the datasets convention defined in torch tutorial
http://www.torch.ch/manual/tutorial/index , and I quote it here:
"A dataset is an object which implements the operator dataset[index] and
implements the method dataset:size(). The size() methods returns the number of
examples and dataset[i] has to return the i-th example. An example has to be
an object which implements the operator example[field], where field often
takes the value 1 (for input features) or 2 (for corresponding labels), i.e
an example is a pair of input and output objects."

For example, using train[3][1], you get the inputs of the third training
example which is a 1025-dim vector, where the last dimension is constantly 1
so you do not need to worry about bias in a linear model. Using train[3][2],
you get the output of the third training example which is a 1-dim vector
whose sole element can be any integer between 1 and 10. Notice that class i
means digit i-1.
]]


-- The mnist dataset
mnist = {};

-- The dataset has 60000 training + 10000 testing samples
function mnist:size() return 70000 end

-- Each row (observation) has 32x32 = 1024 dimensions
function mnist:features() return 1024 end

-- We have 10 classes, where the digit i is class (i+1).
function mnist:classes() return 10 end

-- Read data from mnist_train_32x32.t7 and mnist_test_32x32.t7. Splitting
-- is automatically done.
function mnist:readFile()
   -- Reading raw files
   mnist.orig = {}
   mnist.orig.train = torch.load('mnist_train_32x32.t7','ascii')
   mnist.orig.test = torch.load('mnist_test_32x32.t7','ascii')
end

-- Split the mnist dataset to training and testing dataset
-- Note: mnist:readFile() must have been executed
function mnist:split(train_size, test_size)
   local train = {}
   local test = {}
   function train:size() return train_size end
   function test:size() return test_size end
   function train:features() return mnist:features() end
   function test:features() return mnist:features() end
   function train:classes() return mnist:classes() end
   function test:classes() return mnist:classes() end
   -- Randomize, vectorize and insert training data all into train
   local rorder = torch.randperm(train:size())
   for i = 1,train:size() do
      local input = mnist.orig.train.data[i]:double():reshape(mnist:features())
      local output = torch.ones(1)*mnist.orig.train.labels[i]
      train[rorder[i]] = {input, output}
   end
   -- Randomize, vectorize and insert testing data into test
   local rorder = torch.randperm(test:size())
   for i = 1,test:size() do
      local input = mnist.orig.test.data[i]:double():reshape(mnist:features())
      local output = torch.ones(1)*mnist.orig.test.labels[i]
      test[rorder[i]] = {input, output}
   end
   -- Return the datasets
   return train, test
end

-- Globally normalize. Note: features-wise local normalization is error-prone
-- since some of the pixel is constantly 0 across the dataset.
function mnist:normalize(train, test)
   -- Allocate mean and variance vectors
   local mean = 0
   local var = 0
   -- Iterative mean computation
   for i = 1,train:size() do
      mean = mean*(i-1)/i + train[i][1]:mean()/i
   end
   -- Iterative variance computation
   for i = 1,train:size() do
      var = var*(i-1)/i + torch.sum(torch.pow(train[i][1] - mean,2))/train:features()/i
   end
   -- Get the standard deviation
   local std = math.sqrt(var)
   -- If any std is 0, make it 1
   if std == 0 then std = 1 end
   -- Normalize the training dataset
   for i = 1,train:size() do
      train[i][1] = (train[i][1]-mean)/std
   end
   -- Normalize the testing dataset
   for i = 1,test:size() do
      test[i][1] = (test[i][1]-mean)/std
   end

   return train, test
end

-- Add a dimension to the inputs which are constantly 1
-- This is useful to make simple linear modules without thinking about the bias
function mnist:appendOne(train, test)
   -- Sanity check. If dimensions do not match, do nothing.
   if train:features() ~= mnist:features() or test:features() ~= mnist:features() then
      return train, test
   end
   -- Redefine the features() functions
   function train:features() return mnist:features() + 1 end
   function test:features() return mnist:features() + 1 end
   -- Add dimensions
   for i = 1,train:size() do
      train[i][1] = torch.cat(train[i][1], torch.ones(1))
   end
   for i = 1, test:size() do
      test[i][1] = torch.cat(test[i][1], torch.ones(1))
   end
   -- Return them back
   return train, test
end

-- Get the train and test datasets
function mnist:getDatasets(train_size, test_size)
   -- If file not read, read the files
   if mnist.orig == nil then mnist:readFile() end
   -- Split the datasets
   local train, test = mnist:split(train_size, test_size)
   -- Normalize the dataset
   train, test = mnist:normalize(train, test)
   -- Append one to each input
   train, test = mnist:appendOne(train, test)
   -- Return the train and test datasets
   return train, test
end
