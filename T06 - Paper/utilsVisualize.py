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


def getModelData(modelName, sourceMode, visualizeMode, cutoffRow=None):
    if sourceMode == "TRAIN":
        sourceFolder = "o7_analyze_train"
    elif sourceMode == "TEST":
        sourceFolder = "o7_analyze_test"
    else:
        raise ValueError("Unknown mode")
    sourceFolder = os.path.join(sourceFolder, modelName)

    cwd = sys.path[0]
    outputFolderPath = os.path.join(cwd, sourceFolder)
    folderNames = os.listdir(outputFolderPath)

    data = {}
    for folderName in folderNames:
        folderPath = os.path.join(outputFolderPath, folderName)
        data_pickle, data_json = read_data(folderPath)

        if visualizeMode == "PHI_MAPE":
            valueArray = data_pickle["mapeArray"]
        elif visualizeMode == "FE_DIFF":
            valueArray = valueArray = np.array(data_pickle["feArrayTrue"]) - np.array(
                data_pickle["feArrayPred"]
            )
        elif visualizeMode == "FE_REDUCE":
            valueArray = np.array(data_pickle["feArrayPred"])
            valueArray = np.concatenate(([0], np.diff(valueArray)))
            valueArray = np.where(valueArray < 0, 1, 0)
        else:
            raise ValueError("Unknown mode")
        data[folderName] = valueArray

    tArrayPred = data_pickle["tArrayPred"]
    df = pd.DataFrame(data=data, index=tArrayPred)

    if cutoffRow is not None:
        df = df.iloc[:cutoffRow, :]

    # Create long form data
    dft = df.copy()
    dft["time"] = dft.index
    value_vars = [col for col in dft.columns.values if col != "time"]
    dfLong = pd.melt(
        dft,
        id_vars=["time"],
        value_vars=value_vars,
        var_name="folder",
        value_name="value",
    )
    dfLong["model"] = modelName

    # Create wide form data
    dfWide = df.copy()
    if visualizeMode == "PHI_MAPE":
        dfWide["mean"] = df.apply(lambda row: row.describe()["mean"], axis=1)
        dfWide["std"] = df.apply(lambda row: row.describe()["std"], axis=1)
        dfWide["25%"] = df.apply(lambda row: row.describe()["25%"], axis=1)
        dfWide["75%"] = df.apply(lambda row: row.describe()["75%"], axis=1)
        dfWide["lower_whisker"] = df.apply(lambda row: getBoxPlot(row)[0], axis=1)
        dfWide["upper_whisker"] = df.apply(lambda row: getBoxPlot(row)[1], axis=1)
        dfWide["median"] = df.apply(lambda row: row.median(), axis=1)
    if visualizeMode == "FE_REDUCE":
        dfWide["sum"] = df.apply(lambda row: row.sum(), axis=1)
        dfWide["tot"] = df.apply(lambda row: row.shape[0], axis=1)
        dfWide["frac"] = df.apply(lambda row: row.sum() / row.shape[0], axis=1)
    return df, dfLong, dfWide
