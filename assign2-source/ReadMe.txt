Support Vector Machines Assignment Description
Written by Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com)
Version 0.1, 10/08/2012

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Note: Please start this assignment as early as possible, since QUESTIONS part 2
may take several tens of hours to finish. We recommend you do finish the
programming of part 2, run a small-scale test (with smaller polynomial degree,
smaller range of C and smaller cross-validation folds) to know whether your
programs are correct. If they are correct, you can then do the tens of hours
of work on some machine that is available to you automatically, and concentrate
yourself to other parts of the assignment.

Your assignment should be sent by email to the TA
- Send all the source code files (.lua) plus a description in one zip or
tar.gz. Please EXCLUDE datasets (.t7, .data), or your email may not be
received.
- The description can be a simple text file, with any mathematical notion
(if there is any) written in LaTeX convention. Please do NOT include any
msword, rtf or html files. If you do not like writing formulas in LaTeX
convention, we can accept pdf if it is small. Write all your answers to the
QUESTIONS section (below) in this description file.
- If you choose to write a pdf file, please embed all the plots into it. Or
you can save your plot as eps, ps, or pdf files and send them altogether with
your description file, and give indicator in the description which plot
is for which problem.
- You must implement the code by yourself, but you may discuss your results
with other students.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In this assignment, you will be asked to implement the following:
- Primal SGD SVM using the Hinge loss.
- Experiments with an experimental kernel SVM package
- Cross-validation
- Multiclass classification using one versus one and one versus all.

In order to implement the programs, you must download the datasets for this
assignment (available form the course website), and put all files in the same
path as your source code.

The dataset with which to experiment the binary classification is called
uci-spambase, same as the assignment before. It is a two-class classificiation
problem from UC Irvine machine learning dataset repository
(http://www.ics.uci.edu/~mlearn/MLRepository.html):
The dataset contains 4601 samples with 57 variables. The task is to predict
whether an email message is spam or not.

The dataset with which to test multi-class classification is called MNIST
(http://yann.lecun.com/exdb/mnist/, also same as last time). It has a training
set of 60,000 examples, and a test set of 10,000 examples. The task is to
classify each 32x32 image into 10 categories (corresponding to digits 0-9).

The torch code to read the datasets and format them is provided. Please refer
to spambase.lua and mnist.lua. Upon calling the getDatasets() methods, all the
data will be appended with an extra dimension which will constantly be 1.

Part of this homework is largely dependent on your previous homework. You will
need your trainerSGD implementation for training of the primal SVM model. As a
result, you need to edit the files main.lua, model.lua, trainer.lua, mult.lua,
kernel.lua and crossvalid.lua to implement the algorithms. For your convenience
, we have implemented an experimental kernel support vector machines library
called 'XSVM'. You will only be asked to experiment on it, but you will have to
implement the kernels by yourself. Please send your complaints to the TA if you
find bugs in this library.

The places where you should implement something is identified by the follwoing
two lines:
-- Remove the following line and add your stuff
print("You have to define this function by yourself!");

Your code should run the experiments automatically when we execute
t7> dofile("main.lua")
The plots should be automatically shown whenever necessary.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

QUESTIONS:

Dagobert is an M.S. student who has a great interest in machine learning. One
day he decided to join the lab of a pointy-haired professor who also likes
machine learning. The professor wants him to complete the following tasks
using the machine learning environment Torch 7, on the spambase and MNIST
datasets to prepare Dagobert for a research project.

1 - Implement
  Dagobert's pointy-haired professor wants him to implement the following
  algorithms as a start.
  a. - the primal support vector machines algorithm using Hinge loss with L2
       regularization, based on the SGD trainer implementation that Dagobert
       already has.
       The hinge loss: l(xi,yi,w) = max(0, 1-yi*(w^T xi))
  b. - Polynomial kernels for dual support vector machines, to be used for an
       experimental kernel SVM library called 'XSVM' implemented in the
       pointy-haired professor's lab.
  c. - Cross-validation for spambase datasets. This is a common technique that
       everybody should be familiar with.
  d. - Since Dagobert has done well in the experiments on binary classification
       models using spambase dataset, the pointy-haired professor challenged
       him to further implement the two techniques of multi-class
       classification using binary classifiers: one v.s. one and one v.s. all,
       so that Dagobert could use the code he already gets to experiment on the
       MNIST dataset

2 - Cross-validation with kernel SVM on spambase
  One problem facing Dagobert is that even though he has many nice algorithms,
  there are still hyper-parameters that he must determine before they can be
  put into practice. These may include regularization parameters, step sizes
  and choices of kernels. His pointy-haired professor told him that one
  good technique to determine them is to use cross-validation. In all the
  following experiments, Dagobert finds the fastest kernel SVM to use is
  xsvm.vectorized().
  a. - Dagobert splits the spambase dataset into 2000 training samples and 1000
     testing samples. Then, he splitted the training data into 10 equal-sized
     disjoint sets. For each value of the polynomial kernel with degrees from 1
     to 4, he plotted the average cross-validation error as a function of C,
     varying C in powers of 2 (that is for some value of k, C is 2^-k to 2^k).
     Dagobert finds that the training time may be significantly longer if C is
     very large. What the best average cross-validation error Dagobert should
     have obtained?
  b. - Dagobert selected a good C he found in the experiment above. This time,
     he chose a large range of polynomial degree d, and plotted the average
     trainig errors, average cross-validation errors and average test errors
     he obtained for each model. Should Dagobert find that the cross-validation
     errors more close to training errors or the test errors? Can you give him
     an explanation?
  c. - Check with Dagobert the best pair of (C,d) you have found during the
     training.

3. Multi-classification using binary classifiers on MNIST
   Being happy about his experimental results on the SVM codes, Dagoberts
   went on to try multi-class classification using binary classifiers, with
   primal SVM. For all the experiments, he used at least 6000 training samples
   from the MNIST dataset. But you can use more if your machine has enough
   memory.
  a. - The primal SVM uses the hinge loss regularized by a L2 regularizer.
     The pointy-haird professor told Dagobert that there is a relationship
     between the l2 regularization parameter \lambda and the slack variable
     parameter C in the original SVM formulation. What it might be?
  b. - Dagobert's pointy-haired professor had taught him the "one versus all"
     technique in class. But Dagobert got sleepy on the lecture and he did
     not get the details. Can you tell him how many SVMs should be trained,
     how should each SVM be trained, and how should the trained SVMs be
     combined to produce a classification? Implement this and check with
     Dagobert the testing error you get.
  c. - The pointy-haired professor went on to teach Dagobert a second technique
     as a bonus for joining his lab. It is called "pairwise SVM". Dagobert has
     learnt that it consists in training 45 SVMs for MNIST datasets, one for
     each pair of character classes. Do you know how each SVM should be trained
     and on what part of the training set, and how the outputs of the SVMs
     should be combined?
  d. - Dagobert tried the one-vs-one technique. Do the same, and check with him
     on the testing error you got. Is it better or worse than the one-vs-all
     technique?

4. Bonus, Challenging and Optional Questions
   Being agitated for the first several experiments in machine learning,
   Dagobert wanted to figure out more by himself after finishing all that the
   pointy-haired professor has asked him to do.
  a. - Dagobert discovered that for each model in XSVM, model.a stores the
     trained dual variables and model.x stores the support vectors. He found a
     way to determine whether model.x[i] is a marginal support vector (that is,
     a support vector on the marginal hyperplanes w^T x + b = 1 and w^T x + b =
     -1) just by looking at model.a[i]. How did he do it?
  b. - Dagobert plotted the average total number of support vectors and average
     number of marginal support vectors as functions of d in the same
     experiment as 2b. He asked the pointy-haired professor to explain the
     trend. What might the professor say? (Hint: high-dimensional space is
     more likely to be separable)
  c. - Dagobert finds some interesting properties of the hinge loss. For the
     primal SVM, he replaced the number '1' in hinge loss by a parameter p,
     i.e., l(xi,yi,w) = max(0, p-yi*(w^T xi)). He then did the experiments
     for p = 1,2,3,4,5, with l2 regularization parameter \lambda/p for each
     case. In the experiments he used appropriate choices of step sizes such
     that each case produces almost the same training errors and testing errors
     on a spambase dataset of 3000 training samples and 1000 testing samples.
     He showed you the plot of the training loss as a function of p. What can
     you conclude from his plots? What role does the parameter p play in the
     generalized hinge loss?
  d. - Dagobert told you that for the degenerated case such that p = 0, he
     discovered that it is equivalent to an algorithm from a lecture on
     linear models, once taught by the pointy-haired professor. Do you
     remember what this algorithm was?

Super! After completing the required tasks with a happy extra self-study time,
Dagobert feels more experienced in applying support vector machines. He is also
ready to accept further challenges from the pointy-haired professor. Are you
ready too? (Dagobert's story continues on Assignment 4)
