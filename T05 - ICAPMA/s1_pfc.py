import os, time, pickle, sys
import matplotlib.pyplot as plt
import numpy as np
from utils import store_data, dKeys


def runPFC(
    n,
    L,
    eps,
    phiBar,
    phiAmp,
    dt,
    mTotal,
    mArray,
):
    # Parameter for skipping plotting (animation)
    mPrintSkip = 20

    # Init value
    phi = np.random.random(n)
    # Make sure that the average is phi_bar
    phiAve = np.sum(phi) / n
    phi = (phi - phiAve) * phiAmp + phiBar

    print(np.sum(phi) / n)

    phiOld = phi
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

        data[m, :] = phiOld

        if np.mod(m, mPrintSkip) == 0:
            print(m)

    return data


def export_data(data_store_pickle, data_store_json, mode):
    # Make output folder
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    # cwd = os.getcwd() # This will not work in py file.
    cwd = sys.path[0]  # https://stackoverflow.com/a/5475224

    if mode == "TRAIN":
        folderPath = os.path.join(cwd, "o1_pfc_train", timestamp)
    else:
        folderPath = os.path.join(cwd, "o1_pfc_test", timestamp)
    store_data(
        folderPath=folderPath,
        data_store_pickle=data_store_pickle,
        data_store_json=data_store_json,
    )


#############################################
############## CODE START HERE ##############
#############################################
# mode = "TRAIN"
mode = "TEST"
repeat = 10

# Parameters
n = 32  # Number of grid
L = 8.4 * np.pi  # Length of the domain
eps = 0.2  # Undercooling degree
phiBar = 0.2  # Average density
phiAmp = phiBar * 2  # Amplitude of the initial perturbation
dx = L / n  # Grid spacing
dt = 1 / 8  # Time step
tStart = 0  # Time start
tEnd = 50  # Time end

# Number of time step
mTotal = int(np.ceil((tEnd - tStart) / dt)) + 1

# Recalculate tEnd in case the specified value is not consistent with mTotal
tEnd = tStart + (mTotal - 1) * dt

print(f"Total time steps: {mTotal}")
print(f"Time: {tStart} -> {tEnd}")

# Calculate array of time steps and time
mArray = np.arange(0, mTotal)
tArray = np.linspace(tStart, tEnd, mTotal)

dStore = dict(
    n=n,
    L=L,
    eps=eps,
    phiBar=phiBar,
    phiAmp=phiAmp,
    dx=dx,
    dt=dt,
    tStart=tStart,
    tEnd=tEnd,
    mTotal=mTotal,
    tArray=tArray,
    mArray=mArray,
    mode=mode,
    data=None,
)

# Run

paramsRun1 = ["n", "L", "eps", "phiBar", "phiAmp", "dt", "mTotal"]
paramsRun2 = ["mArray"]

for i in range(0, repeat):
    paramsDict = dKeys(dStore, [*paramsRun1, *paramsRun2])
    dStore["data"] = runPFC(**paramsDict)

    # Prepare data
    kJson = [*paramsRun1, "dx", "tStart", "tEnd", "mode"]
    data_store_pickle = dKeys(dStore, ["tArray", "data"])
    data_store_json = dict(s1=dKeys(dStore, kJson))

    export_data(data_store_pickle, data_store_json, mode)
    time.sleep(1)
