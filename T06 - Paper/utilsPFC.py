import os, time, sys
import numpy as np
from utils import store_data


def calcTime(tStart, tEnd, dt):
    # Number of time step
    mTotal = int(np.ceil((tEnd - tStart) / dt)) + 1

    # Recalculate tEnd in case the specified value is not consistent with mTotal
    tEnd = tStart + (mTotal - 1) * dt

    # print(f"Total time steps: {mTotal}")
    # print(f"Time: {tStart} -> {tEnd}")

    # Calculate array of time steps and time
    mArray = np.arange(0, mTotal)
    tArray = np.linspace(tStart, tEnd, mTotal)

    return mTotal, mArray, tArray


def runPFC(n, L, eps, dt, mTotal, mArray, phiInit):
    # Parameter for skipping plotting (animation)
    mPrintSkip = 20

    phiOld = phiInit
    k = (
        2 * np.pi / L * np.concatenate((np.arange(0, n / 2), np.arange(-n / 2, 0)))
    )  # or np.fft.fftfreq(100, d=dx/L)
    k2 = np.power(k, 2)
    k4 = np.power(k, 4)
    k6 = np.power(k, 6)

    # Storing data
    data = np.zeros((mTotal, n))
    data[0, :] = phiOld  # Initial data
    for m in mArray[1:]:  # Use [1] to skip overwriting inital data
        phiOldHat = np.fft.fft(phiOld)
        phiCubeOldHat = np.fft.fft(np.power(phiOld, 3))
        numerator = -dt * k2 * phiCubeOldHat + phiOldHat
        denominator = 1 + dt * (1 - eps) * k2 - 2 * dt * k4 + dt * k6
        phiNewHat = np.divide(numerator, denominator)
        phiNew = np.fft.ifft(phiNewHat)
        phiOld = np.real(phiNew)

        if np.isnan(phiOld).sum() > 0:
            print("Nan detected")
            return data

        data[m, :] = phiOld

        if np.mod(m, mPrintSkip) == 0:
            print(m)

    return data


def export_data(data_store_pickle, data_store_json, mode, modelName=""):
    # Make output folder
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    # cwd = os.getcwd() # This will not work in py file.
    cwd = sys.path[0]  # https://stackoverflow.com/a/5475224

    if mode == "TRAIN_INIT":
        folderPath = os.path.join(cwd, "o1_init_train", timestamp)
    elif mode == "TEST_INIT":
        folderPath = os.path.join(cwd, "o1_init_test", timestamp)
    elif mode == "TRAIN_RUN_BASE":
        folderPath = os.path.join(cwd, "o2_base_train", timestamp)
    elif mode == "TEST_RUN_BASE":
        folderPath = os.path.join(cwd, "o2_base_test", timestamp)
    elif mode == "TRAIN_RUN":
        folderPath = os.path.join(cwd, "o3_pfc_train", modelName, timestamp)
    elif mode == "TEST_RUN":
        folderPath = os.path.join(cwd, "o3_pfc_test", modelName, timestamp)
    else:
        raise Exception("Unknown mode.")

    if not (os.path.exists(folderPath)):
        os.makedirs(folderPath)

    store_data(
        folderPath=folderPath,
        data_store_pickle=data_store_pickle,
        data_store_json=data_store_json,
    )
