import numpy as np
import time, os, sys
from utils import store_data, getIdxList, read_data


def getDataFromSingleRun(folderPath):
    data_pickle, data_json = read_data(folderPath)
    data = data_pickle["data"]
    tArray = data_pickle["tArray"]
    dt = data_json["s1"]["dt"]

    X, y = list(), list()

    for tCur in tArray:
        # List in indices in the sequence
        idxList, _ = getIdxList(
            tCur=tCur, nSeq=nSeq, dtSeq=dtSeq, dtBase=dt, tArray=tArray
        )

        if idxList is None:
            # print(tCur)
            break

        # gather input and output parts of the pattern
        seq_x, seq_y = data[idxList[:-1], :], data[idxList[-1], :]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    return X, y, data_pickle, data_json


#############################################
############## CODE START HERE ##############
#############################################

# Parameters
nSeq = 5
dtSeq = 2 / 8

# Data folders
cwd = sys.path[0]
outputFolderPath = os.path.join(cwd, "o1_pfc_train")
folderNames = os.listdir(outputFolderPath)


X = list()
y = list()

for folderName in folderNames:
    folderPath = os.path.join(outputFolderPath, folderName)
    _X, _y, data_dict, data_json = getDataFromSingleRun(folderPath)
    X.append(_X)
    y.append(_y)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# Writing data to file
dJson = dict(
    folderNames=folderNames, nSeq=nSeq, dtSeq=dtSeq, X_shape=X.shape, y_shape=y.shape
)
data_store_pickle = dict(X=X, y=y)
data_store_json = {**dict(s2=dJson), **data_json}

# Make output folder
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
folderPath = os.path.join(cwd, "o2_train_data", timestamp)
store_data(
    folderPath=folderPath,
    data_store_pickle=data_store_pickle,
    data_store_json=data_store_json,
)
