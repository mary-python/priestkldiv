# Distributed, communication-efficient, and differentially private estimation of KL divergence: Experimental evaluation

This repository contains all the supporting files for the **experimental section** of the paper **_Distributed, communication-efficient, and differentially private estimation of KL divergence_**, including all the Python files necessary for anyone to repeat all of the experiments given as evidence for the results in the paper.

## Environment

- Install the [**latest version of Python 3**](https://www.python.org/downloads/), and install the additional packages **h5py, matplotlib, numpy and tensorflow** using **PIP**.
- Download the Python files **group_by_writer.py**, **converter.py** and **femnist_test.py**, for preprocessing purposes.
- Download the Python files **femnist_eps.py** and/or **femnist_eps_low_Ct.py** and/or **femnist_eps_mid_Ct.py** and/or **femnist_eps_super_Ct.py**, if interested in the experiments involving the privacy parameter $\varepsilon$ and/or the client sample size $|C_t|$.
- Download the ZIP folder **by_class.zip** (~1GB) from the [**National Institute of Standards and Technology**](https://s3.amazonaws.com/nist-srd/SD19/by_class.zip).
- Unzip this folder using unzipping software for large folders, e.g. [**7-Zip File Manager**](https://7-zip.org/download.html).
- **Create a new folder named "data"** in the same folder as the Python files, and **place the unzipped by_class folder in this new "data" folder**.

## Preprocessing

- After setting up the Python environment and downloading all the required files as outlined above, open and run **group_by_writer.py**. A new folder named *write_all* will appear, and it will contain folders of the form *fxxx_xx*, each of which contains the images linked to one writer.
- Within each of the *fxxx_xx* folders, **only the folders "30" to "39" inclusive need to be kept** (they contain the images of digits 0-9). **If space is limited, then delete all other folders** (they contain the images of uppercase letters A-Z and lowercase letters a-z).
- Now open and run **converter.py**, which creates an eponymous HDF5 file from the "write_all" folder structure.
- Now open and run **femnist_test.py**, which loads the HDF5 dataset, and checks its data and properties. The desired outputs are stated on the relevant lines in the Python file.

## Experiments

- If all outputs of **femnist_test.py** are as expected, open and run **femnist_eps.py** or **femnist_eps_low_Ct.py** or **femnist_eps_mid_Ct.py** or **femnist_eps_super_Ct.py**. There will be various text updates appearing in the terminal, indicating which experiment is being run.
- When the final text update indicating the runtime appears, the experiments have finished and the final plots have been saved in the same folder as the Python files. This should happen after approximately **6-12 hours**, depending on the computer or laptop used. 
- These final plots should be **exactly the graphs found in the experimental section** of **_Distributed, communication-efficient, and differentially private estimation of KL divergence_** featuring the privacy parameter $\varepsilon$ and/or the client sample size $|C_t|$ respectively.
- Repeat all steps in the **Experiments** section with the another Python file, if interested in the experiments featuring other parameters.

## Authors

- **[Mary Scott](https://mary-python.github.io/)**, Department of Computer Science, University of Warwick
- **[Sayan Biswas](https://people.epfl.ch/sayan.biswas?lang=en)**, EPFL, Switzerland
- **[Graham Cormode](http://dimacs.rutgers.edu/~graham/)**, Department of Computer Science, University of Warwick
- **[Carsten Maple](https://warwick.ac.uk/fac/sci/wmg/people/profile/?wmgid=1102)**, WMG, University of Warwick
