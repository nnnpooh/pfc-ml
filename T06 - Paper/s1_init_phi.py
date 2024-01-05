import time
import numpy as np
from utils import dKeys
from utilsPFC import calcTime, runPFC, export_data

#############################################
############## CODE START HERE ##############
#############################################
# mode = "TRAIN_INIT"
mode = "TEST_INIT"
repeat = 50

scale = 4
# Parameters
n = 32 * scale  # Number of grid
L = 8.5 * np.pi * scale  # Length of the domain
eps = 0.2  # Undercooling degree
phiBar = 0.2  # Average density
phiAmp = phiBar * 2  # Amplitude of the initial perturbation
dx = L / n  # Grid spacing
dt = 1 / 4  # Time step
tStart = 0  # Time start
tEnd = tStart + dt  # Time end
mTotal, mArray, tArray = calcTime(tStart, tEnd, dt)

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
    runErr=None,
)

prRunScalar = ["n", "L", "eps", "dt", "mTotal"]
prRunArray = ["mArray"]
prRun = [*prRunScalar, *prRunArray]
prStoreScalar = [
    *prRunScalar,
    "dx",
    "tStart",
    "tEnd",
    "mode",
    "phiBar",
    "phiAmp",
    "runErr",
]
prStoreArray = ["tArray", "data"]

for i in range(0, repeat):
    phi = np.random.random(n)
    phiAve = np.sum(phi) / n  # Make sure that the average is phi_bar
    phi = (phi - phiAve) * phiAmp + phiBar

    paramsDict = dKeys(dStore, prRun)
    dStore["data"], dStore["runErr"] = runPFC(**paramsDict, phiInit=phi)

    # Prepare data
    data_store_json = dict(s1=dKeys(dStore, prStoreScalar))
    data_store_pickle = dKeys(dStore, prStoreArray)

    export_data(data_store_pickle, data_store_json, mode)
    time.sleep(1)
