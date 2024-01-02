import os, time, sys
from utils import dKeys, read_data
from utilsPFC import calcTime, runPFC, export_data


#############################################
############## CODE START HERE ##############
#############################################

# mode = "TRAIN_RUN_BASE"
mode = "TEST_RUN_BASE"

cwd = sys.path[0]
if mode == "TRAIN_RUN_BASE":
    sourceFolder = "o1_init_train"
elif mode == "TEST_RUN_BASE":
    sourceFolder = "o1_init_test"
else:
    raise ValueError("mode must be either TRAIN_RUN_BASE or TEST_RUN_BASE")

outputFolderPath = os.path.join(cwd, sourceFolder)
folderNames = os.listdir(outputFolderPath)

for folderName in folderNames:
    folderPath = os.path.join(cwd, sourceFolder, folderName)
    data_pickle, data_json = read_data(folderPath)

    # Setting
    tStart = 0  # Time start
    tEnd = 800  # Time end
    dt = 1 / 4  # Time step

    # Parameters
    n = data_json["s1"]["n"]
    L = data_json["s1"]["L"]
    eps = data_json["s1"]["eps"]
    dx = data_json["s1"]["dx"]
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
        mode=mode,
        data=None,
        folderName=folderName,
        sourceFolder=sourceFolder,
    )

    prRunScalar = ["n", "L", "eps", "dt", "mTotal"]
    prRunArray = ["mArray"]
    prRun = [*prRunScalar, *prRunArray]
    prStoreScalar = [
        *prRunScalar,
        "dx",
        "tStart",
        "tEnd",
        "folderName",
        "sourceFolder",
        "mode",
    ]
    prStoreArray = ["tArray", "data"]

    # Run
    phiInit = data_pickle["data"][-1, :]
    paramsDict = dKeys(dStore, prRun)
    dStore["data"] = runPFC(**paramsDict, phiInit=phiInit)

    # Prepare data
    data_store_json = {**dict(s2=dKeys(dStore, prStoreScalar), **data_json)}
    data_store_pickle = dKeys(dStore, prStoreArray)

    export_data(data_store_pickle, data_store_json, mode=mode)
    time.sleep(1)
