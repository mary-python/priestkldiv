"""Modules are to load directories, display a progress bar, provide both a high- and low-level
interface to the HDF5 library, and work with arrays in Python."""
import os
from alive_progress import alive_bar
import h5py
import numpy as np

# FROM HDF5 BLOG: "HDF5 FILE FOR LARGE IMAGE DATASETS" BY SOUMYA TRIPATHY
# https://blade6570.github.io/soumyatripathy/hdf5_blog.html
# ENABLES IMAGES TO BE STORED IN A HDF5 FILE WHILE MAINTAINING FOLDERS AND SUBFOLDERS

BASE_PATH = '.\\Sayan Biswas\\data\\by_class'
# basePath = '.\\Sayan Biswas\\data\\by_class_mit'
# savePath = '.\\Sayan Biswas\\data\\femnist_digits.hdf5'
SAVE_PATH = '.\\Sayan Biswas\\data\\femnist_digits_mit.hdf5'
# savePath = '.\\Sayan Biswas\\data\\test.hdf5'

# open the file in append mode
hf = h5py.File(SAVE_PATH, 'a')
NUMBER_FILES = 747260
print("\nConverting directory to hdf5 format...\n")

with alive_bar(NUMBER_FILES) as bar:

    # read all the folders
    for i in os.listdir(BASE_PATH):
        name = os.path.join(BASE_PATH, i)
        group = hf.create_group(name)

        # read all the folders inside the folders
        for j in os.listdir(name):
            track = os.path.join(name, j)
            subgroup = group.create_group(j)

            # find all images in the subfolders
            for k in os.listdir(track):
                if os.path.isdir():
                    imagePath = os.path.join(track, k)
                else:
                    # open MIT files as python binary
                    with open(track, 'rb') as mit:
                        binaryDataMit = mit.read()

                    # create numpy array storing python binary
                    binaryDataMitNp = np.asarray(binaryDataMit)

                    # save all MIT files in the current folder
                    dset = group.create_dataset(k, data = binaryDataMitNp)

                # open images as python binary
                with open(imagePath, 'rb') as image:
                    binaryData = image.read()

                # create numpy array storing python binary
                binaryDataNp = np.asarray(binaryData)

                # save all images in the current subfolder
                dset = subgroup.create_dataset(k, data = binaryDataNp)
                bar()

hf.close()
print("\nDone.\n")
