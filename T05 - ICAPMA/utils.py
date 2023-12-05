import os, pickle
import pickle, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Flatten,
    ConvLSTM2D,
)
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import json


def dKeys(myDict, keys):
    return dict((k, myDict[k]) for k in keys if k in myDict)


# Store data in pickle file and json file
def store_data(folderPath, data_store_pickle, data_store_json):
    if not (os.path.exists(folderPath)):
        os.mkdir(folderPath)

    if data_store_pickle:
        filePath = os.path.join(folderPath, "data_store.pickle")
        with open(filePath, "wb") as handle:
            pickle.dump(data_store_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if data_store_json:
        filePath = os.path.join(folderPath, "parameters.json")
        with open(filePath, "w", encoding="utf-8") as f:
            json.dump(data_store_json, f, ensure_ascii=False, indent=6)


def read_data(folderPath):
    filePath = os.path.join(folderPath, "data_store.pickle")

    if os.path.exists(filePath):
        with open(filePath, "rb") as handle:
            data_pickle = pickle.load(handle)
    else:
        data_pickle = None

    filePath = os.path.join(folderPath, "parameters.json")
    if os.path.exists(filePath):
        with open(filePath, "rb") as f:
            data_json = json.load(f)
    else:
        data_json = None

    return data_pickle, data_json


def createModel(type, nSeq, nRow, nCol, nChannel):
    inputLayer = Input(shape=(nSeq, nRow, nCol, nChannel))

    if type == "ConvLSTM":
        layer = ConvLSTM2D(filters=64, kernel_size=(1, 3), activation="relu")(
            inputLayer
        )
        layer = Flatten()(layer)

    outputLayer = Dense(nCol)(layer)
    model = Model(inputs=inputLayer, outputs=outputLayer, name=type)
    return model


def get_latest_checkpoint_folder(path):
    a = os.listdir(path)
    if a:
        a.sort()
        return os.path.join(path, a[-1])
    else:
        return None


def getIdxList(tCur, dtSeq, dtBase, tArray, nSeq=None, tEnd=None):
    if (nSeq is None) and (tEnd is None):
        raise Exception("nSeq and tEnd cannot be both None.")

    if (nSeq is not None) and (tEnd is not None):
        raise Exception("nSeq and tEnd cannot be both specified.")

    idxSkip = dtSeq / dtBase
    # Make sure idxSkip is an integer
    if np.floor(idxSkip) != idxSkip:
        raise Exception("Invalid dtSeq")

    s = np.where(tArray == tCur)[0]  # Array containing indices
    if s.shape[0] != 1:
        raise Exception("Invalid t value")
    iCur = s[0]

    if tEnd is None:
        # Get data for training mode
        # Time at the end of sequence (note that the end of this sequence is the response value)
        tEnd = tCur + nSeq * dtSeq

        # Find corresponding index
        s = np.where(tArray == tEnd)[0]
        if s.shape[0] != 1:
            return None, None
    else:
        # Get data for forecasting mode
        # Find corresponding index
        s = np.where(tArray == tEnd)[0]
        if s.shape[0] != 1:
            raise Exception("Invalid tEnd value")

    iEnd = s[0]
    idxList = np.arange(iCur, iEnd + 1, dtSeq / dtBase).astype(int)
    tList = tArray[idxList]
    return idxList, tList
