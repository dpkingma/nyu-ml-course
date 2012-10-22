--[[
Spambase dataset implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.2, 10/04/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In general, all you need to do is to load this file with spambase.data
presented in you current directory as follows:
t7> dofille "spambase.lua"
then, you can split out shuffled and normalized training and testing data by
calling spambase:getDatasets(train_size,test_size), for example:
t7> train, test = spambase:getDatasets(3000,1000)

The sets train and test (and even spambase itself) follow the datasets
convention defined in torch tutorial http://www.torch.ch/manual/tutorial/index
, and I quote it here:
"A dataset is an object which implements the operator dataset[index] and
implements the method dataset:size(). The size() methods returns the number of
examples and dataset[i] has to return the i-th example. An example has to be
an object which implements the operator example[field], where field often
takes the value 1 (for input features) or 2 (for corresponding labels), i.e
an example is a pair of input and output objects."

For example, using train[3][1], you get the inputs of the third training
example which is a 58-dim vector, where the last dimension is constantly 1 so
you do not need to worry about the bias in a linear model. Using train[3][2],
you get the output of the  third training example which is a 1-dim vector
whose sole element can only be +1 or -1.
]]

-- the spambase dataset
spambase = {};

-- The dataset has 4601 rows (observations) 
function spambase:size() return 4601 end

-- Each row (observaton) has 57 features
function spambase:features() return 57 end

-- Read csv files from the spambase.data
function spambase:readFile()
   -- CSV reading using simple regular expression :)
   local file = 'spambase.data'
   local fp = assert(io.open (file))
   local csvtable = {}
   for line in fp:lines() do
      local row = {}
      for value in line:gmatch("[^,]+") do
	 -- note: doesn\'t work with strings that contain , values
	 row[#row+1] = value
      end
      csvtable[#csvtable+1] = row
   end
   -- Generating random order
   local rorder = torch.randperm(spambase:size())
   -- iterate over rows
   for i = 1, spambase:size() do	
      -- iterate over columns (1 .. num_features)
      local input = torch.Tensor(spambase:features())
      local output = torch.Tensor(1)
      for j = 1, spambase:features() do
	 -- set entry in feature matrix
	 input[j] = csvtable[i][j]
      end
      -- get class label from last column (num_features+1)
      output[1] = csvtable[i][spambase:features()+1]
      -- it should be class -1 if output is 0
      if output[1] == 0 then output[1] = -1 end
      -- Shuffled dataset
      spambase[rorder[i]] = {input, output}
   end
end

-- Split the dataset into two sets train and test
-- spambase:readFile() must have been executed
function spambase:split(train_size, test_size)
   local train = {}
   local test = {}
   function train:size() return train_size end
   function test:size() return test_size end
   function train:features() return spambase:features() end
   function test:features() return spambase:features() end
   -- iterate over rows
   for i = 1,train:size() do
      -- Cloning data instead of referencing, so that the datset can be split multiple times
      train[i] = {spambase[i][1]:clone(), spambase[i][2]:clone()}
   end
   -- iterate over rows
   for i = 1,test:size() do
      -- Cloning data instead of referencing
      test[i] = {spambase[i+train:size()][1]:clone(), spambase[i+train:size()][2]:clone()}
   end

   return train, test
end

-- Normalize the dataset using training set's mean and std
-- train and test must be returned from spambase:split
function spambase:normalize(train, test)
   -- Allocate mean and variance vectors
   local mean = torch.zeros(train:features())
   local var = torch.zeros(train:features())
   -- Iterative mean computation
   for i = 1,train:size() do
      mean = mean*(i-1)/i + train[i][1]/i
   end
   -- Iterative variance computation
   for i = 1,train:size() do
      var = var*(i-1)/i + torch.pow(train[i][1] - mean,2)/i
   end
   -- Get the standard deviation
   local std = torch.sqrt(var)
   -- If any std is 0, make it 1
   std:apply(function (x) if x == 0 then return 1 end end)
   -- Normalize the training dataset
   for i = 1,train:size() do
      train[i][1] = torch.cdiv(train[i][1]-mean, std)
   end
   -- Normalize the testing dataset
   for i = 1,test:size() do
      test[i][1] = torch.cdiv(test[i][1]-mean, std)
   end

   return train, test
end

-- Add a dimension to the inputs which are constantly 1
-- This is useful to make simple linear modules without thinking about the bias
function spambase:appendOne(train, test)
   -- Sanity check. If dimensions do not match, do nothing.
   if train:features() ~= spambase:features() or test:features() ~= spambase:features() then
      return train, test
   end
   -- Redefine the features() functions
   function train:features() return spambase:features() + 1 end
   function test:features() return spambase:features() + 1 end
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
function spambase:getDatasets(train_size, test_size)
   -- If file not read, read the files
   if spambase[1] == nil then spambase:readFile() end
   -- Split the dataset
   local train, test = spambase:split(train_size, test_size)
   -- Normalize the dataset
   train, test = spambase:normalize(train, test)
   -- Append one to each input
   train, test = spambase:appendOne(train, test)
   -- return train and test datasets
   return train, test
end
