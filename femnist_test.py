import h5py
import numpy as np
# from PIL import Image
# import io
import torch
import matplotlib.pyplot as plt

# FROM HDF5 BLOG: "HDF5 FILE FOR LARGE IMAGE DATASETS" BY SOUMYA TRIPATHY
# https://blade6570.github.io/soumyatripathy/hdf5_blog.html
# ENABLES IMAGES IN FOLDERS AND SUBFOLDERS TO BE EXTRACTED FROM THEIR HDF5 FILE 

# data = [] # keys to access image data
# group = [] # list all groups and subgroups in hdf5 file
# print(f"\nExtracting keys, groups and subgroups from hdf5 file...")

# def func(name, obj):
    # if isinstance(obj, h5py.Dataset):
        # data.append((name, obj))
    # elif isinstance(obj, h5py.Group):
        # group.append((name, obj))

# fetch hdf5 file from current directory
# path = './Sayan Biswas/data/femnist_digits.hdf5'
# path = './Sayan Biswas/data/femnist_digits_mit.hdf5'
# path = './Sayan Biswas/data/test.hdf5'
# path = './Sayan Biswas/data/write_all.hdf5'
path = './data/write_all.hdf5'
file = h5py.File(path, 'r')
# file.visititems(func)

# numberFiles = 747260
# print(f"Extracting images from hdf5 file...\n")

# from alive_progress import alive_bar
# with alive_bar(numberFiles) as bar:

    # read the image files in their proper format
    # for j in data:
        # kk = np.array(file[j])
        # img = Image.open(io.BytesIO(kk)) # our image file
        # bar()

# FROM HDF5-FEMNIST BY XIAO-CHENGUANG
# https://github.com/Xiao-Chenguang/HDF5-FEMNIST
# ENABLES EASY ACCESS AND FAST LOADING TO THE FEMNIST DATASET FROM LEAF WITH THE HELP OF HDF5

# create a list where each index stores the images and labels associated with a particular writer
writers = sorted(file.keys())
# writers = sorted(data.keys())
print(f'\nThe dataset contains images from {len(writers)} writers.') # should be 3580

# store writer 0's images and labels in numpy arrays
w0_images = file[writers[0]]['images'][:]
w0_labels = file[writers[0]]['labels'][:]

# writer 0's numpy arrays ---> torch tensors
w0_images_tensor = torch.from_numpy(w0_images)
w0_labels_tensor = torch.from_numpy(w0_labels)

# sort writer 0's images into classes (digits 0 to 9)
classes, counts = np.unique(w0_labels, return_counts = True)
print(f'\nWriter 0 has {len(w0_images)} images') # should be 104
print(f'Writer 0 has {len(classes)} classes') # should be 10
print(f'Writer 0 has {counts} images per class\n') # should be [12 11 11 12 9 8 10 11 9 11]

# dataset 0 stores all of writer 0's data
d0 = file[writers[0]]
fig, axs = plt.subplots(10, 10, figsize = (15, 15))

for i in range(10):
    for j in range(10):

        # randomly choose a 10 x 10 grid of labels to compare to images
        id = np.random.randint(len[d0['labels']])
        axs[i, j].imshow(d0['images'][id])
        axs[i, j].axis('off')
        axs[i, j].set_title(d0['labels'][id] - 30)

plt.show()