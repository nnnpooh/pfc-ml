import numpy as np
import time, os, sys
from utils import store_data, getIdxList, read_data


def getDataFromSingleRun(nSeq, dtSeq, dtBase, tArray, data):
    X, y = list(), list()

    for tCur in tArray:
        # List in indices in the sequence
        idxList, _ = getIdxList(
            tCur=tCur, nSeq=nSeq, dtSeq=dtSeq, dtBase=dtBase, tArray=tArray
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

    return X, y


#############################################
############## CODE START HERE ##############
#############################################

models = {
    "m2": {
        "dtMul": 2,
    },
    "m4": {
        "dtMul": 4,
    },
    "m8": {
        "dtMul": 8,
    },
    "m16": {
        "dtMul": 16,
    },
    "m32": {
        "dtMul": 32,
    },
}

# Parameters
modelName = "m16"
nSeq = 1

# Data folders
cwd = sys.path[0]
outputFolderPath = os.path.join(cwd, "o2_base_train")
folderNames = os.listdir(outputFolderPath)

X = list()
y = list()

for folderName in folderNames:
    folderPath = os.path.join(outputFolderPath, folderName)

    data_pickle, data_json = read_data(folderPath)
    data = data_pickle["data"]
    tArray = data_pickle["tArray"]
    dtMul = models[modelName]["dtMul"]
    dtBase = data_json["s1"]["dt"]
    dtSeq = dtBase * dtMul

    _X, _y = getDataFromSingleRun(
        nSeq=nSeq, dtSeq=dtSeq, dtBase=dtBase, tArray=tArray, data=data
    )

    X.append(_X)
    y.append(_y)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# Writing data to file
dJson = dict(
    folderNames=folderNames,
    modelName=modelName,
    nSeq=nSeq,
    dtSeq=dtSeq,
    dtBase=dtBase,
    X_shape=X.shape,
    y_shape=y.shape,
)
data_store_pickle = dict(X=X, y=y)
data_store_json = {**dict(s4=dJson), **data_json}

# Make output folder
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
folderPath = os.path.join(cwd, "o4_ml_train_data", modelName, timestamp)

if not (os.path.exists(folderPath)):
    os.makedirs(folderPath)

store_data(
    folderPath=folderPath,
    data_store_pickle=data_store_pickle,
    data_store_json=data_store_json,
)
