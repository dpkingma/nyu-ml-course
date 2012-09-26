Linear Models Assignment Description
Written by Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com)
Version 0.1, 09/24/2012

This file is written for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Your assignment should be sent by email to the TA
- Send all the source code files (.lua) plus a description in one zip or
tar.gz. Please excluding datasets (.t7, .data), or your email may not be
received.
- The description can be a simple text file, with any mathematical notion
(if there is any) written in LaTeX convention. But please do NOT include any
msword or html files. If you do not like writing formulas in LaTeX convention,
we can accept pdf if it is small. Write all your answers to the QUESTIONS
section (below) in this description file.
- You must implement the code by yourself, but you may discuss your results
with other students.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following learning algorithms must be implemented with ridge (l2) and
LASSO (l1) regularization:
- The perceptron algorithm
- Direct solution of linear regression
- On-line (stochastic) linear regression
- On-line (stochastic) logistic regression
- On-line (stochastic) multinomial logistic regression

In order to implement the programs, you must download the datasets for this
assignment (available form the course website), and put all files in the same
path as your source code. Notice that for the mnist dataset we are using a
different file (native to torch) than that provided on its website.

The dataset with which to test the binary classification models is called
uci-spambase. It is a two-class classificiation problem from UC Irvine
machine learning dataset repository
(http://www.ics.uci.edu/~mlearn/MLRepository.html):
The dataset contains 4601 samples with 57 variables. The task is to predict
whether an email message is spam or not.

The datset with which to test the multinomial model is called MNIST
(http://yann.lecun.com/exdb/mnist/). It has a training set of 60,000 examples,
and a test set of 10,000 examples. The task is to classify each 32x32 image
into 10 categories (corresponding to digits 0-9).

The torch code to read the datasets and format them is provided. Please refer
to spambase.lua and mnist.lua. Upon calling the getDatasets() methods, all the
data will be appended with an extra dimension which will constantly be 1, so
that you do not need to worry about the bias when implementing a linear model.

You simply need to edit the files main.lua, model.lua, regularizer.lua,
trainer.lua and implement the algorithms. The detailed design paradigm is
illustrated in the heading comments of each file. Please read them. Examples
are given for each kind of object you will have to implement.

The places where you should implement something is identified by the follwoing
two lines:
-- Remove the following line and add your stuff
print("You have to define this function by yourself!");

Your training code should
- Starting by computing and displaying loss and the error rate on the training
set and test set.
- Make a training pass over the training set
- Compute and display the loss and error rate on the training set and test set.
- Iterate to 2 until convergence (pick a criterion).

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

QUESTIONS:

1 - Implement:
  a. - The perceptron learning algorithm
  b. - Linear regression with square loss trained with direction solution and 
       stochastic gradient descent
  c. - The Logistic regression algorithm trained with stochastic gradient
       descent
  d. - The multinomial Logistic regression algorithm trained with stochastic
       gradient descent

  ==> Include your code as attachment of an email to the TA.

2 - Experiments with the Spambase dataset
     - Set the training set size to 1000 and the test set to 1000.
     - Use the (ridge) l2 regularization
     - The stochastic gradient method for linear regression and logistic
       regression requires you to find good values for the learning rate
       (the step size eta using stepCons)
  a. - What learning rate will cause linear regression and logistic regression
       to diverge?
  b. - What learnin rate for linear regression and logistic regression that
       produces the fastest convergence?
  c. - Implement a stopping criterion that detects convergence.
  d. - Train logistic regression with 10, 30, 100, 500, 1000 and 3000 trianing
       samples, and 1000 test samples. For each size of the training set,
       provide:
       - The final value of the average loss, and the classification errors on
         the training set and the test set
       - The value of the learning rate used
       - The number of iterations performed
  e. - What is the asymptotic value of the training/test error for very large
       training sets?

3. - L2 and L1 regularization
     When the training set is small, it is often helpful to add a
     regularization term to the loss function. The most popular ones are:
     L2 Norm: lambda*||W||^2 (aka "Ridge")
     L1 Norm: lambda*[\sum_{i} |W_i|] (aka "LASSO")
  a. - How is the linear regression with direct solution modified by the
       addition of an L2 regularizer?
  b. - Implement the L1 regularizer. Experiment with your logistic regression
       code with the L2 and L1 regularizers. Can you improve the performance on
       the test set for training set sizes of 10, 30 and 100? What value of
       lambda gives the best results?

4. - Multinomial logistic regression
     Implement the multinomial logistic regression as a model. Your can
     experiment with larger datasets if your machine has enough memory. In this
     part we experiment on the MNIST dataset.
  a. - Testing your model with training sets of size 100, 500, 1000, and 6000,
       with a testing set of size 1000. What are the error rates on the
       training and testing datasets?
  b. - Use both L2 and L1 regularization. Do they make some difference?
