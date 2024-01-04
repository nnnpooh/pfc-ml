import os, time, sys
from utils import dKeys, read_data, getIdxList
from utilsPFC import calcTime, runPFC, export_data


# Return true data with correct time step for comparison
def getTrueData(data, tCur, tEnd, dtSeq, dtBase, tArray):
    idxList, tArrayPred = getIdxList(
        tCur=tCur, tEnd=tEnd, dtSeq=dtSeq, dtBase=dtBase, tArray=tArray
    )

    X_true = data[idxList]

    return X_true, tArrayPred


#############################################
############## CODE START HERE ##############
#############################################

models = {
    "p2": {
        "dtMul": 2,
    },
    "p4": {
        "dtMul": 4,
    },
    "p8": {
        "dtMul": 8,
    },
    "p16": {
        "dtMul": 16,
    },
    "p32": {
        "dtMul": 32,
    },
}

# modelName = "p2"
# modelName = "p4"
# modelName = "p8"
modelName = "p16"
# p32 results is NaN.

mode = "TRAIN_RUN"
# mode = "TEST_RUN"

dtMul = models[modelName]["dtMul"]
cwd = sys.path[0]
if mode == "TRAIN_RUN":
    sourceFolder = "o2_base_train"
elif mode == "TEST_RUN":
    sourceFolder = "o2_base_test"
else:
    raise ValueError("mode must be either TRAIN_RUN or TEST_RUN")

outputFolderPath = os.path.join(cwd, sourceFolder)
folderNames = os.listdir(outputFolderPath)
for folderName in folderNames:
    folderPath = os.path.join(cwd, sourceFolder, folderName)
    data_pickle, data_json = read_data(folderPath)

    # Base data
    phiBase = data_pickle["data"]
    tArrayBase = data_pickle["tArray"]
    dtBase = data_json["s2"]["dt"]

    # Parameters
    n = data_json["s2"]["n"]
    L = data_json["s2"]["L"]
    eps = data_json["s2"]["eps"]
    dx = data_json["s2"]["dx"]
    tStart = data_json["s2"]["tStart"]
    tEnd = data_json["s2"]["tEnd"]
    dtSeq = dtBase * dtMul

    mTotal, mArray, tArray = calcTime(tStart, tEnd, dtSeq)

    dStore = dict(
        n=n,
        L=L,
        eps=eps,
        dx=dx,
        dtSeq=dtSeq,
        dt=dtSeq,
        tStart=tStart,
        tEnd=tEnd,
        mTotal=mTotal,
        tArray=tArray,
        mArray=mArray,
        modelName=modelName,
        dtMul=dtMul,
        folderName=folderName,
        sourceFolder=sourceFolder,
        X_true=None,
        X_pred=None,
        tArrayPred=None,
        runErr=None,
    )

    prRunScalar = ["n", "L", "eps", "dt", "mTotal"]
    prRunArray = ["mArray"]
    prRun = [*prRunScalar, *prRunArray]
    prStoreScalar = [
        *prRunScalar,
        "dtSeq",
        "dtMul",
        "modelName",
        "dx",
        "tStart",
        "tEnd",
        "folderName",
        "sourceFolder",
        "mode",
        "runErr",
    ]
    prStoreArray = ["X_true", "X_pred", "tArrayPred"]

    # Run
    phiInit = phiBase[0, :]
    paramsDict = dKeys(dStore, prRun)
    X_pred, dStore["runErr"] = runPFC(
        **paramsDict, phiInit=phiInit, printTag=folderName
    )

    # Get true data
    X_true, tArrayPred = getTrueData(
        data=phiBase,
        tCur=tStart,
        tEnd=tEnd,
        dtSeq=dtSeq,
        dtBase=dtBase,
        tArray=tArrayBase,
    )

    dStore["X_true"] = X_true
    dStore["X_pred"] = X_pred
    dStore["tArrayPred"] = tArrayPred

    # Prepare data
    data_store_json = {**dict(s3=dKeys(dStore, prStoreScalar), **data_json)}
    data_store_pickle = dKeys(dStore, prStoreArray)

    export_data(data_store_pickle, data_store_json, mode=mode, modelName=modelName)
    time.sleep(1)
