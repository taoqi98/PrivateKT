# PrivateKT
- Codes of our PrivateKT method


# Code Files
- preprocessing.py: load raw data, partition local training data
- utils.py: containing some util functions
- model.py: implementation of the basic model
- evaluation.py: performance evaluation function
- PrivateKT.py: implementation of the PrivateKT method, including the importance sampling mechanism, knowledge buffer mechanism, self-training mechanism, randomized response mechanism, and unbiased knowledge aggregation, it also includes the fedetarated traning framework of PrivateKT 
- main.ipynb: organizing the whole knowledge transfer framework


# Running Instructions
- We provide demo data in the path "../demo_data". The codes can be executed by running the "main.ipynb" file. The expected outputs of the codes are stored in the log of the 7-th cell of the "main.ipynb" file. For a normal server (with an Nvidia GPU), it usually takes about 2 minutes to download this repository and takes about 24 hours to execute the codes.
