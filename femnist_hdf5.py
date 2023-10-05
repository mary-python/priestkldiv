import h5py
import numpy as np
import os

# FROM HDF5 BLOG: "HDF5 FILE FOR LARGE IMAGE DATASETS" BY SOUMYA TRIPATHY
# https://blade6570.github.io/soumyatripathy/hdf5_blog.html
# ENABLES IMAGES TO BE STORED IN A HDF5 FILE WHILE MAINTAINING FOLDERS AND SUBFOLDERS

basePath = '.\\Sayan Biswas\\data\\by_class'
# savePath = '.\\Sayan Biswas\\data\\femnist_digits.hdf5'
savePath = '.\\Sayan Biswas\\data\\test.hdf5'

# open the file in append mode
hf = h5py.File(savePath, 'a')
numberFiles = 747260
print(f"\nConverting directory to hdf5 format...\n")

from alive_progress import alive_bar
with alive_bar(numberFiles) as bar:

    # read all the folders
    for i in os.listdir(basePath):
        name = os.path.join(basePath, i)

        # read all the folders inside the folders
        for j in os.listdir(name):
            track = os.path.join(name, j)
    
            # find all images in the subfolders
            for k in os.listdir(track):
                imagePath = os.path.join(track, k)

                # open images as python binary
                with open(imagePath, 'rb') as image:
                    binaryData = image.read()
            
                # create numpy array storing python binary
                binaryDataNp = np.asarray(binaryData)

                # TESTING: save all images without folders or subfolders
                # AVOID DUPLICATE NAME: dset
                dset = hf.create_dataset(k, data = binaryDataNp)
                bar()

hf.close()
print(f"\nDone.\n")