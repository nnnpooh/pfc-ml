import os, time, sys
from utils import dKeys, read_data
from utilsPFC import calcTime, runPFC, export_data


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

# mode = "TRAIN_RUN"
mode = "TEST_RUN"

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

    # Parameters
    n = data_json["s2"]["n"]
    L = data_json["s2"]["L"]
    eps = data_json["s2"]["eps"]
    dx = data_json["s2"]["dx"]
    tStart = data_json["s2"]["tStart"]
    tEnd = data_json["s2"]["tEnd"]
    dt = data_json["s2"]["dt"] * dtMul

    mTotal, mArray, tArray = calcTime(tStart, tEnd, dt)

    dStore = dict(
        n=n,
        L=L,
        eps=eps,
        dx=dx,
        dt=dt,
        tStart=tStart,
        tEnd=tEnd,
        mTotal=mTotal,
        tArray=tArray,
        mArray=mArray,
        modelName=modelName,
        dtMul=dtMul,
        data=None,
        folderName=folderName,
        sourceFolder=sourceFolder,
    )

    prRunScalar = ["n", "L", "eps", "dt", "mTotal"]
    prRunArray = ["mArray"]
    prRun = [*prRunScalar, *prRunArray]
    prStoreScalar = [
        *prRunScalar,
        "dtMul",
        "modelName",
        "dx",
        "tStart",
        "tEnd",
        "folderName",
        "sourceFolder",
        "mode",
    ]
    prStoreArray = ["tArray", "data"]

    # Run
    phiInit = data_pickle["data"][0, :]
    paramsDict = dKeys(dStore, prRun)
    dStore["data"] = runPFC(**paramsDict, phiInit=phiInit)

    # Prepare data
    data_store_json = {**dict(s3=dKeys(dStore, prStoreScalar), **data_json)}
    data_store_pickle = dKeys(dStore, prStoreArray)

    export_data(data_store_pickle, data_store_json, mode=mode, modelName=modelName)
    time.sleep(1)
