K-Means and Mixture of Gaussians Assignment Description
Written by Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com)
Version 0.1, 11/26/2012

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Your assignment should be sent by email to the TA
- Send all the source code files (.lua) plus a description in one zip or
tar.gz. Please EXCLUDE datasets (.t7, .data), or your email may not be
received.
- The description can be a simple text file, with any mathematical notion
(if there is any) written in LaTeX convention. Please do NOT include any
msword, rtf or html files. If you do not like writing formulas in LaTeX
convention, we can accept pdf if it is small. Write all your answers to the
QUESTIONS section (below) in this description file.
- If you choose to write a pdf file, please embed all the result images in it.
Or you can save your image as png files and send them altogether with your
description file, and give indicator in the description which image is for
which problem.
- You must implement the code by yourself, but you may discuss your results
with other students.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In this assignment, you will be asked to implement the following:
- The K-Means algorithm
- The Mixture of Gaussians algorithm using Expectation-Maximization

In order to implement the programs, you must download the datasets for this
assignment (available form the course website), and put all files in the same
path as your source code. The dataset is a set of grayscale images.

This assignment will need the 'image' package. You can install it using the
following command:
$ torch-pkg install image
If you want to install it only to yourself, please use -local option.

The places where you should implement something is identified by the follwoing
two lines:
-- Remove the following line and add your stuff
print("You have to define this function by yourself!");

Your code should run the experiments automatically when we execute
t7> dofile("main.lua")

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

QUESTIONS:

Recently Dagobert turns his interests into the area of unsupervised learning.
From the lectures he knew two techniques that were usedly extensively - K-Means
and Mixture of Gaussians. These methods are able to do clutering, that is,
assigning a set of vectors into groups without labels on them.

1 - Implement
  a. - Dagobert implements the K-Means algorithm in the following way
     - Initialize K prototype vectors P1,...,Pk by setting each of them to a
     different randomly chosen training sample
     - Repeat untill convergence
         - For all i in [1...p], for all j in [1...k], compute the
         responsibilities Rij. Rij = 1 if Xi is closest to Pj than to any other
         prototype; Rij = 0 otherwise.
	 - Recompute the prototypes so that each prototype is equal to the mean
         of all the samples assigned to it Pi = 1/[sum_i Rij] sum_i Rij*Xi
   b. - Dagobert used EM to estimate the parameters of a Mixture of Gaussians.
      From the lectures he knows that given a set of P training vectors X1,...,
      Xp, the likelihood of the data modeled by MoG with K component Gaussians
      is: P(X1...Xp | W1...Wk,M1...Mk,A1...AK) = \prod_i \sum_j Wj*G(Xi,Mj,Aj),
      where i ranges over [1,p] and j over [1,K] (K is the number of components
      in the mixture). G(Xi,Mj,Aj) is a multivariate Gaussian density with mean
      Mj and covariance matrix Aj evaluated at point Xi. To ensure
      normalization, the Wj must be between 0 and 1, and must verify \sum_j Wj
      = 1. The EM algorithm for MoG Dagobert implemented is as follows:
      - Initialize K prototype vectors P1,...,Pk by running the K-Means
      algorithm.
      - Repeat until convergence:
          - for all i in [1...p], for all j in [1...k], compute the
          responsibilities Rij = Wj*G(Xi,Mj,Aj)/sum_h Wh*G(Xi,Mh,Ah)
          - Set Mj to the mean of all the sample *weighted* by their
          responsibilities Rij
          - Set Aj to the covariance of all the samples *weighted* by their
          responsibilities Rij
          - Set Wj to [sum_i Rij] / [sum_j sum_i Rij]

2. - Image Compression using K-Means and Mixture of Gaussians
   a. - A simple technique for image compression, called vector quantization,
      consists in (1) cutting up an image into small non-overlapping tiles (e.g
      8 by 8 pixels), (2) viewing this set of tiles as a data set of 64-
      dimensional vectors (each vector component being a pixel) (3) running the
      K-means algorithm on this dataset (4) coding each tile by a code that
      represents the index of the nearest prototype to the tile. The image in
      the PNG files in the directory (800x600 pixels, grayscale) were cut up
      into 7500 non-overlapping, 8x8-pixel tiles. Those tiles are turned into
      64-dimensional vectors (by lining up the pixels in row-major order from
      left to right and top to bottom). The file "tile.lua" has protocols to
      transform an image into tiles and transform tiles back to an image. Run
      the K-means algorithm with K = 2,4,8,64,256 for each of the image. Report
      the mean-squared error inccured by replacing tile by its closest protypes
      i.e., the average of ||Xi-Tk(i)||^2 over i, where Xi is the i-th tile,
      and Tk(i) is the prototype (among the K prototypes) that is closest to Xi
   b. - Dagobert reconstructed the "compressed' test image obtained with K =
      256 by by replacing each tile by its closest prototype. For your
      convenience, Dagobert wrote a function tile.tileim() to transform tiles 
      back to an image, to make it easier for you to repeat his results.
   c. - Dagobert tested the EM algorithm for Mixture of Gaussians. Starting
      from the result obtained from 2.a (result of K-means on the tile dataset)
      Dagobert trained a mixture of Gaussians with 8 components using EM on the
      dataset. Dagobert figured that EM finds the parameteres that minimize
      the negative log-likelihood of the dataset given the model:
      L = -sum_i log(ModelProb(xi)). Compare with Dagobert the initial value
      of negative log-likelihood (after running k-means and one M step to
      compute the covariance matrices, using the binary responsibilities
      produced by K-means), and the value at the end of the convergence of EM.

3. Bonus and Open Questions
   a. - In experiment 2.a, for k = 8, Dagobert computed the binary entropy of
      the sequence of codes for the whole image: make an integer vector Zi
      with as many elements as samples in the dataset, where each component
      contains the index of the prototype closest to the corresponding tile
      [k(i)]. Then compute the normalized histogram of the values in that
      vector (i.e., a vector Hk where element k is the proportion of elements
      in Zi that are equal to k (k in [1,8]). Compute the binary entropy of
      the histogram:
      Histogram_Entropy = -sum_k Hk*log2[Hk] where log2[x] is the base 2
      logarithm of x. This is the average number of bits we would need to
      specify the index of the prototype of each tile in the image, if we had a
      perfect entropy coder to turn those integers into short bit strings.
      (e.g. an arithmetic coder, such as the Z-coder used in DjVu). The total
      number of bits we would need to compress the whole image would be
      Number_of_bits = Number_of_tiles * Histogram_Entropy. Report this value.
   b. (Open question) - Dagober has read a paper saying the histogram Hk we
      proposed above could actually be used as a feature for images regardless
      of the sizes of them. Can you propose a type of classification task where
      this idea may be good, so that Dagobert could give it a try?
