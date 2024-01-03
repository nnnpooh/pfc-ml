import os, sys, time
from utils import (
    store_data,
    read_data,
    calculate_err,
)
from utilsPFC import calcFreeEnergy, calcPhiAve
import numpy as np


#############################################
############## CODE START HERE ##############
#############################################


mode = "ML_TEST"
# modelName = "m1"
# modelName = "m2"
# modelName = "m4"
# modelName = "m8"
modelName = "m16"
# modelName = "m32"

# mode = "PFC_TEST"
# modelName = "p16"

if mode == "ML_TEST":
    sourceFolder = "o6_ml_train"
    outputFolder = os.path.join("o7_analyze_train", modelName)
elif mode == "ML_TEST":
    sourceFolder = "o6_ml_test"
    outputFolder = os.path.join("o7_analyze_test", modelName)
elif mode == "PFC_TRAIN":
    sourceFolder = "o3_pfc_train"
    outputFolder = os.path.join("o7_analyze_train", modelName)
elif mode == "PFC_TEST":
    sourceFolder = "o3_pfc_test"
    outputFolder = os.path.join("o7_analyze_test", modelName)
else:
    raise ValueError("Unknown mode")

sourceFolder = os.path.join(sourceFolder, modelName)

cwd = sys.path[0]
outputFolderPath = os.path.join(cwd, sourceFolder)
folderNames = os.listdir(outputFolderPath)

for folderName in folderNames:
    folderPath = os.path.join(outputFolderPath, folderName)
    print(folderPath)
    data_pickle, data_json = read_data(folderPath)

    L = data_json["s2"]["L"]
    n = data_json["s2"]["n"]
    eps = data_json["s2"]["eps"]

    X_true = data_pickle["X_true"]
    X_pred = data_pickle["X_pred"]
    tArrayPred = data_pickle["tArrayPred"]

    mse, mape, mseArray, mapeArray = calculate_err(X_true=X_true, X_pred=X_pred)
    feArrayPred = calcFreeEnergy(X_pred, n, L, eps)
    feArrayTrue = calcFreeEnergy(X_true, n, L, eps)
    phiAveArrayPred = calcPhiAve(data=X_pred, n=n)
    phiAveArrayTrue = calcPhiAve(data=X_true, n=n)

    dJson = dict(
        mse=mse,
        mape=mape,
        mode=mode,
        modelName=modelName,
        folderName=folderName,
        sourceFolder=sourceFolder,
    )

    data_store_json = {**dict(s7=dJson), **data_json}
    data_store_pickle = dict(
        X_pred=X_pred,
        X_true=X_true,
        tArrayPred=tArrayPred,
        mseArray=mseArray,
        mapeArray=mapeArray,
        feArrayPred=feArrayPred,
        feArrayTrue=feArrayTrue,
        phiAveArrayPred=phiAveArrayPred,
        phiAveArrayTrue=phiAveArrayTrue,
    )

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    folderPath = os.path.join(sys.path[0], outputFolder, timestamp)

    if not (os.path.exists(folderPath)):
        os.makedirs(folderPath)

    store_data(
        folderPath=folderPath,
        data_store_pickle=data_store_pickle,
        data_store_json=data_store_json,
    )
    time.sleep(1)
