import h5py
import numpy as np
import os

# FROM HDF5 BLOG: "HDF5 FILE FOR LARGE IMAGE DATASETS" BY SOUMYA TRIPATHY
# https://blade6570.github.io/soumyatripathy/hdf5_blog.html
# ENABLES IMAGES TO BE STORED IN A HDF5 FILE WHILE MAINTAINING FOLDERS AND SUBFOLDERS

basePath = './data/by_class'
savePath = './data/test.hdf5'

# open the file in append mode
hf = h5py.File(savePath, 'a')

# read all the folders
for i in os.listdir(basePath):
    name = os.path.join(basePath, i)
    group = hf.create_group(name)

    # read all the folders inside the folders
    for j in os.listdir(name):
        track = os.path.join(name, j)
        subgroup = group.create_group(j)
    
        # find all images in the subfolders
        for k in os.listdir(track):
            imagePath = os.path.join(track, k)

            # open images as python binary
            with open(imagePath, 'rb') as image:
                binaryData = image.read()
            
            # create numpy array storing python binary
            binaryDataNp = np.asarray(binaryData)

            # save all images in the current subfolder
            dset = subgroup.create_dataset(k, data = binaryDataNp)