import tensorflow as tf
import os, sys, time
from tensorflow.keras.callbacks import EarlyStopping
from utils import (
    get_latest_checkpoint_folder,
    store_data,
    createModel,
    read_data,
    dKeys,
)


def runModel(
    model,
    X,
    y,
    cpPath="./tmp",
    cpFilename="cp",
    learning_rate=0.001,
    patience=2,
    epochs=200,
):
    # Check if model exist
    if not (os.path.exists(cpPath)):
        os.mkdir(cpPath)
        latestDir = None
    else:
        latestDir = get_latest_checkpoint_folder(cpPath)

    if latestDir:
        print(f"Load model from {latestDir}")
        latestFile = tf.train.latest_checkpoint(latestDir)
        model.load_weights(latestFile)
    else:
        print("Load new model")

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    earlyStoppingCallback = EarlyStopping(
        monitor="loss", patience=patience, min_delta=1e-4
    )

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cpPath, timestamp, cpFilename),
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )

    history = model.fit(
        X,
        y,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpointCallback, earlyStoppingCallback],
    )

    hist = history.history
    print(hist["loss"])


#############################################
############## CODE START HERE ##############
#############################################

dm = {
    "m1": "2023-12-06-03-06-36",
    "m2": "2023-12-06-03-06-55",
    "m4": "2023-12-06-03-07-00",
    "m8": "2023-12-06-03-07-07",
    "m16": "2023-12-06-03-07-13",
    "m32": "2023-12-06-03-07-17",
    "m64": "2023-12-06-03-07-22",
}

# modelName = "m1"
# modelName = "m2"
# modelName = "m4"
# modelName = "m8"
modelName = "m16"
# modelName = "m32"
# modelName = "m64"

folderName = dm[modelName]

cwd = sys.path[0]
folderPath = os.path.join(cwd, "o2_train_data", folderName)
data_pickle, data_json = read_data(folderPath)

X = data_pickle["X"]
y = data_pickle["y"]
nCol = data_json["s1"]["n"]
nSeq = data_json["s2"]["nSeq"]
nRow = 1
nChannel = 1

# Reshape to match the input of the model
X = X.reshape(X.shape[0], nSeq, nRow, nCol, nChannel)

# Model building
tf.keras.backend.clear_session()
tf.random.set_seed(1)

ConvLSTM = createModel(
    type="ConvLSTM", nSeq=nSeq, nRow=nRow, nCol=nCol, nChannel=nChannel
)
ConvLSTM.summary()

modelParam = {
    "model": ConvLSTM,
    "cpPath": os.path.join(cwd, "o3_checkpoints", modelName),
    "cpFilename": "cp-{epoch:04d}.ckpt",
    "learning_rate": 0.001,
    "epochs": 2000,
    "patience": 3,
    "X": X,
    "y": y,
}


runModel(**modelParam)

# Export data
dJson = {
    **dict(
        modelName=modelName,
        nSeq=nSeq,
        nRow=nRow,
        nCol=nCol,
        nChannel=nChannel,
    ),
    **dKeys(modelParam, ["cpPath", "learning_rate", "epochs", "patience"]),
}
dJson = {**dict(s3=dJson), **data_json}
folderPath = os.path.join(cwd, "o3_model_params", modelName)
store_data(folderPath=folderPath, data_store_pickle=None, data_store_json=dJson)
