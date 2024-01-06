import os, sys
import numpy as np
from utils import read_data
import pandas as pd


def getBoxPlot(row):
    des = row.describe()
    upper_quartile = des["75%"]
    lower_quartile = des["25%"]
    iqr = upper_quartile - lower_quartile

    upper_whisker = row[row <= (upper_quartile + 1.5 * iqr)].max()
    lower_whisker = row[row >= (lower_quartile - 1.5 * iqr)].min()
    return lower_whisker, upper_whisker


def getFolderNames(modelName, sourceMode):
    if sourceMode == "TRAIN":
        sourceFolder = "o7_analyze_train"
    elif sourceMode == "TEST":
        sourceFolder = "o7_analyze_test"
    else:
        raise ValueError("Unknown mode")
    sourceFolder = os.path.join(sourceFolder, modelName)

    cwd = sys.path[0]
    sourceFolderPath = os.path.join(cwd, sourceFolder)
    folderNames = os.listdir(sourceFolderPath)

    return folderNames, sourceFolderPath


def getModelDataNum(modelName, sourceMode, visualizeMode, cutoffRow=None):
    folderNames, sourceFolderPath = getFolderNames(modelName, sourceMode)

    data = {}
    for folderName in folderNames:
        folderPath = os.path.join(sourceFolderPath, folderName)
        data_pickle, data_json = read_data(folderPath)

        if visualizeMode == "PHI_MAPE":
            valueArray = data_pickle["mapeArray"]
        elif visualizeMode == "PHI_MSE":
            valueArray = data_pickle["mseArray"]
        elif visualizeMode == "FE_DIFFERENCE":
            feArrayTrue = np.array(data_pickle["feArrayTrue"])
            feArrayPred = np.array(data_pickle["feArrayPred"])
            valueArray = np.abs(feArrayTrue - feArrayPred)
        elif visualizeMode == "PHI_AVE":
            phiAveArrayTrue = np.array(data_pickle["phiAveArrayTrue"])
            phiAveArrayPred = np.array(data_pickle["phiAveArrayPred"])
            diffArray = np.abs(phiAveArrayTrue - phiAveArrayPred)

            # Cutoff to zero
            diffArray = np.where(diffArray < 1e-12, 0, diffArray)

            # Avoid division by zero
            # https://stackoverflow.com/a/37977222
            valueArray = np.divide(
                diffArray,
                phiAveArrayTrue,
                out=np.zeros_like(diffArray),
                where=phiAveArrayTrue != 0,
            )

            valueArray = diffArray
        else:
            raise ValueError("Unknown mode")
        data[folderName] = valueArray

    tArrayPred = data_pickle["tArrayPred"]
    df = pd.DataFrame(data=data, index=tArrayPred)
    colsFolders = df.columns.values

    if cutoffRow is not None:
        df = df.iloc[:cutoffRow, :]

    # Create wide form data
    dfWide = pd.DataFrame(index=df.index)
    dfWide["mean"] = df.apply(lambda row: row.describe()["mean"], axis=1)
    dfWide["std"] = df.apply(lambda row: row.describe()["std"], axis=1)
    dfWide["25%"] = df.apply(lambda row: row.describe()["25%"], axis=1)
    dfWide["75%"] = df.apply(lambda row: row.describe()["75%"], axis=1)
    dfWide["lower_whisker"] = df.apply(lambda row: getBoxPlot(row)[0], axis=1)
    dfWide["upper_whisker"] = df.apply(lambda row: getBoxPlot(row)[1], axis=1)
    dfWide["median"] = df.apply(lambda row: row.median(), axis=1)

    # Create long form data
    dft = df.copy()
    dft["time"] = dft.index
    dfLong = pd.melt(
        dft,
        id_vars=["time"],
        value_vars=colsFolders,
        var_name="folder",
        value_name="value",
    )
    dfLong["model"] = modelName

    return df, dfLong, dfWide


def getModelDataReduce(modelName, sourceMode, visualizeMode, cutoffRow=None):
    folderNames, sourceFolderPath = getFolderNames(modelName, sourceMode)

    data = {}
    for folderName in folderNames:
        folderPath = os.path.join(sourceFolderPath, folderName)
        data_pickle, data_json = read_data(folderPath)

        if visualizeMode == "FE_REDUCE":
            valueArray = np.array(data_pickle["feArrayPred"])
            valueArray = np.concatenate(([0], np.diff(valueArray)))
            valueArray = np.where(valueArray > 0, 1, 0)
        else:
            raise ValueError("Unknown mode")
        data[folderName] = valueArray

    tArrayPred = data_pickle["tArrayPred"]
    df = pd.DataFrame(data=data, index=tArrayPred)

    if cutoffRow is not None:
        df = df.iloc[:cutoffRow, :]

    # Create wide form data
    dfWide = pd.DataFrame(index=df.index)
    dfWide["sum"] = df.apply(lambda row: row.sum(), axis=1)
    dfWide["tot"] = df.apply(lambda row: row.shape[0], axis=1)
    dfWide["frac"] = df.apply(lambda row: row.sum() / row.shape[0], axis=1)
    value_vars = ["frac"]

    # Create long form data from dfWide
    dft = dfWide.copy()
    dft["time"] = dft.index
    dfLong = pd.melt(
        dft,
        id_vars=["time"],
        value_vars=value_vars,
        var_name="type",
        value_name="value",
    )
    dfLong["model"] = modelName

    return df, dfLong, dfWide
