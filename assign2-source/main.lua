--[[
Main file
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com>) @ New York University
Version 0.1, 10/10/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file contains sample of experiments.
--]]

-- Load required libraries and files
dofile("spambase.lua")
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")
dofile("kernel.lua")
dofile("crossvalid.lua")
dofile("xsvm.lua")
dofile("mult.lua")
dofile("mnist.lua")

-- An example of using xsvm
function main()
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(3000,1000)

   -- 2. Initialize a dual SVM with linear kernel, and C = 0.05.
   print("Initializing a linear kernel SVM with C = 0.05...")
   local model = xsvm.vectorized{kernel = kernLin(), C = 0.05}
   
   -- 3. Train the kernel SVM
   print("Training the kernel SVM...")
   local error_train = model:train(data_train)
   
   -- 4. Testing using the kernel SVM
   print("Testing the kernel SVM...")
   local error_test = model:test(data_test)

   -- 6. Print the result
   print("Train error = "..error_train.."; Testing error = "..error_test)
end

main()