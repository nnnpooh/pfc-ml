import tensorflow as tf
import os, sys, time
from utils import (
    get_latest_folder,
    getIdxList,
    store_data,
    read_data,
)
import numpy as np
from utilsML import createModel


def load_model_weights(modelName, model):
    cwd = sys.path[0]
    cpPath = os.path.join(cwd, "o5_checkpoints", modelName)
    latestDir = get_latest_folder(cpPath)

    if latestDir is None:
        raise Exception("No folder")

    print(f"Load model from {latestDir}")
    latestFile = tf.train.latest_checkpoint(latestDir)

    model.load_weights(latestFile)
    return model


def construct_model(modelName):
    cwd = sys.path[0]
    folderPath = os.path.join(cwd, "o5_model_params", modelName)

    _, data_json = read_data(folderPath)

    nSeq = data_json["s5"]["nSeq"]
    nRow = data_json["s5"]["nRow"]
    nCol = data_json["s5"]["nCol"]
    nChannel = data_json["s5"]["nChannel"]

    model = createModel(
        type="ConvLSTM", nSeq=nSeq, nRow=nRow, nCol=nCol, nChannel=nChannel
    )
    return model, data_json


def get_pfc_data(folderName, mode):
    cwd = sys.path[0]
    if mode == "TRAIN_RUN":
        outputFolderPath = os.path.join(cwd, "o2_base_train")
    if mode == "TEST_RUN":
        outputFolderPath = os.path.join(cwd, "o2_base_test")
    else:
        raise Exception("Mode not defined.")

    folderPath = os.path.join(outputFolderPath, folderName)

    if not (os.path.exists(folderPath)):
        raise Exception("PFC data folder does not exist.")

    data_pickle, data_json = read_data(folderPath)

    data = data_pickle["data"]
    tArray = data_pickle["tArray"]
    return data, tArray, data_json


def predictValue(model, X_, nSeq, nRow, nCol, nChannel):
    X = X_.reshape((1, nSeq, nRow, nCol, nChannel))
    yhat_ = model.predict(X, verbose=0)
    yhat = yhat_.flatten()
    return yhat


def calculateForecast(
    model, tCur, tEnd, data, nSeq, dtSeq, tArray, dtBase, nRow, nCol, nChannel
):
    idxList, tArrayPred = getIdxList(
        tCur=tCur, tEnd=tEnd, dtSeq=dtSeq, dtBase=dtBase, tArray=tArray
    )
    X_true = data[idxList]

    X_pred = data[idxList[:nSeq]]

    for _ in idxList[nSeq:]:
        X = X_pred[-nSeq:, :]
        y = predictValue(model, X, nSeq=nSeq, nRow=nRow, nCol=nCol, nChannel=nChannel)
        y = y.reshape(1, -1)
        X_pred = np.concatenate((X_pred, y), axis=0)

    return X_pred, X_true, tArrayPred


#############################################
############## CODE START HERE ##############
#############################################

# modelName = "m1"
# modelName = "m2"
# modelName = "m4"
# modelName = "m8"
modelName = "m16"
# modelName = "m32"

mode = "TRAIN_RUN"
# mode = "TEST_RUN"

cwd = sys.path[0]
if mode == "TRAIN_RUN":
    sourceFolder = "o2_base_train"
    outputFolder = os.path.join("o6_ml_train", modelName)
elif mode == "TEST_RUN":
    sourceFolder = "o2_base_test"
    outputFolder = os.path.join("o6_ml_test", modelName)
else:
    raise ValueError("mode must be either TRAIN_RUN or TEST_RUN")
outputFolderPath = os.path.join(cwd, sourceFolder)
folderNames = os.listdir(outputFolderPath)
for folderName in folderNames:
    folderPath = os.path.join(outputFolderPath, folderName)
    data_pickle_pfc, data_json_pfc = read_data(folderPath)
    data = data_pickle_pfc["data"]
    tArray = data_pickle_pfc["tArray"]

    model, data_json_ml = construct_model(modelName)
    model = load_model_weights(modelName, model)

    tCur = data_json_pfc["s2"]["tStart"]
    tEnd = data_json_pfc["s2"]["tEnd"]
    X_pred, X_true, tArrayPred = calculateForecast(
        model=model,
        tCur=tCur,
        tEnd=tEnd,
        data=data,
        tArray=tArray,
        nSeq=data_json_ml["s5"]["nSeq"],
        nRow=data_json_ml["s5"]["nRow"],
        nCol=data_json_ml["s5"]["nCol"],
        nChannel=data_json_ml["s5"]["nChannel"],
        dtSeq=data_json_ml["s4"]["dtSeq"],
        dtBase=data_json_ml["s2"]["dt"],
    )

    dJson = dict(
        tCur=tCur,
        tEnd=tEnd,
        mode=mode,
        folderNameBase=folderName,
        modelName=modelName,
    )
    data_store_json = {**dict(s6=dJson), **data_json_ml}
    data_store_pickle = dict(X_pred=X_pred, X_true=X_true, tArrayPred=tArrayPred)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    folderPath = os.path.join(sys.path[0], outputFolder, timestamp)

    if not (os.path.exists(folderPath)):
        os.makedirs(folderPath)

    store_data(
        folderPath=folderPath,
        data_store_pickle=data_store_pickle,
        data_store_json=data_store_json,
    )
