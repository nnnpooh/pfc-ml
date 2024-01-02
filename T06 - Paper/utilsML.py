from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Flatten,
    ConvLSTM2D,
)

def createModel(type, nSeq, nRow, nCol, nChannel):
    inputLayer = Input(shape=(nSeq, nRow, nCol, nChannel))

    if type == "ConvLSTM":
        layer = ConvLSTM2D(filters=64, kernel_size=(1, 3), activation="relu")(
            inputLayer
        )
        layer = Flatten()(layer)

    outputLayer = Dense(nCol)(layer)
    model = Model(inputs=inputLayer, outputs=outputLayer, name=type)
    return model

