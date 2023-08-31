import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

# FROM HDF5-FEMNIST BY XIAO-CHENGUANG
# https://github.com/Xiao-Chenguang/HDF5-FEMNIST
# ENABLES EASY ACCESS AND FAST LOADING TO THE FEMNIST DATASET FROM LEAF WITH THE HELP OF HDF5

# is this path genuine or do I need to change this to my path?
path = '.\Sayan Biswas\data'
file = h5py.File(path, 'r')

# create an list where each index stores the images and labels associated with a particular writer
writers = sorted(file.keys)
print(f'the dataset contains images from {len(writers)} writers') # should be 3580

# store writer 0's images and labels in numpy arrays
w0_images = file[writers[0]]['images'][:]
w0_labels = file[writers[0]]['labels'][:]

# writer 0's numpy arrays ---> torch tensors
w0_images_tensor = torch.from_numpy(w0_images)
w0_labels_tensor = torch.from_numpy(w0_labels)

# sort writer 0's images into classes (digits 0 to 9)
classes, counts = np.unique(w0_labels, return_counts = True)
print(f'writer 0 has {len(w0_images)} images') # should be 104
print(f'writer 0 has {len(classes)} classes') # should be 10
print(f'writer 0 has {counts} images per class') # should be [12 11 11 12 9 8 10 11 9 11]

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