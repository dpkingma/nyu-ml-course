--[[
Tile implementation for images
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 11/28/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

This file provides the tile functions to transform an image into tiles of data,
and transform tiles of data into an image. The usages are as follows:

im = tile.imread(file): Read an image. If it is an RGB image, transform it into
grayscale.

t = tile.imtile(im,tsize): Transform an image to tile vectors. tsize indicates
the size of the tile patches, as {height,width}. t is a matrix whose rows are
stacked tile patches in row-major order.

im = tile.tileim(t,tsize,imsize): Transform tile vectors into an image. tsize
indicates the size of the tile patches, as {height,width}. imsize is the size
of the image

tile.imwrite(im,file): Write image data to file. The file type is determined
by the postfix of file. The supported ones are pgm, ppm, jpg and png.

For example:
-- Read the boat.png image
im = tile.imread('boat.png')
-- Transform it to a 8x8 tiles. t is a 7500*64 tensor.
t = tile.imtile(im,{8,8})
-- Transform the tile back to an image
im1 = tile.tileim(t,{8,8},im:size())
-- Save the image to boat1.png
tile.imwrite(im1,'boat1.png')
]]

-- requires image package
require("image")

-- the namespace
tile = {}

-- Read the image file
-- file: a string indicating an image file
function tile.imread(file)
   -- Load the image
   local im = image.load(file)
   if im:size()[1] > 1 then
      -- Assuming RGB image
      im = image.rgb2y(im)
   end
   -- Return the image
   return im[1]
end

-- Save an image
-- im: image data
-- file: the file string
function tile.imwrite(im,file)
   return image.save(file,im)
end

-- transform an image into tiles
-- im: image data. must be a 2-dim array of double type
-- tsize: a table indicating size of tiles {height,width}. e.g., {8,8}
function tile.imtile(im,tsize)
   -- Determine the size of final dataset
   local ncol = math.floor(im:size()[1]/tsize[1])
   local nrow = math.floor(im:size()[2]/tsize[2])
   -- Create the tile as a matrix
   local t = torch.zeros(ncol*nrow,tsize[1]*tsize[2])
   -- Loop in row-major order
   for i = 1,ncol do
      for j = 1,nrow do
	 t[(i-1)*nrow + j] = im[{{(i-1)*tsize[1]+1,i*tsize[1]},{(j-1)*tsize[2]+1,j*tsize[2]}}]:clone():resize(tsize[1]*tsize[2])
      end
   end
   -- Return the tile
   return t
end

-- Transform tiles to an image
-- t: tile data
-- tsize: size of tile patches as {height,width}. e.g., {8,8}
-- imsize: size of the image as {height,width}. e.g., {600,800}
function tile.tileim(t,tsize,imsize)
   -- Determine number of tile rows and tile columns
   local ncol = math.floor(imsize[1]/tsize[1]);
   local nrow = math.floor(imsize[2]/tsize[2]);
   -- Allocate the image
   local im = torch.zeros(imsize[1],imsize[2]);
   -- Make the image corresponding size
   for i = 1,ncol do
      for j = 1,nrow do
	 -- Check size
	 if (i-1)*nrow+j > t:size()[1] then
	    return im
	 end
	 -- Set the image
	 im[{{(i-1)*tsize[1]+1,i*tsize[1]},{(j-1)*tsize[2]+1,j*tsize[2]}}] = t[(i-1)*nrow+j]:clone():resize(tsize[1],tsize[2])
      end
   end
   -- Return the image
   return im
end
