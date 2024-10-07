from tensorflow import keras

def conv_block(inputs, out_ch, rate=1):
    x = keras.layers.Conv2D(out_ch, 3, padding="same", dilation_rate=rate)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x