from conv_block import conv_block
from tensorflow import keras

def RSU_Middle_Block(x, out_channels, intermediate_channels):
    x0 = conv_block(x, intermediate_channels, rate=1)
    
    x1 = conv_block(x0, intermediate_channels, rate=1)
    x2 = conv_block(x1, intermediate_channels, rate=2)
    x3 = conv_block(x2, intermediate_channels, rate=4)
    
    x4 = conv_block(x3, intermediate_channels, rate=8)
    
    x = keras.layers.Concatenate()([x4, x3])
    x = conv_block(x, intermediate_channels, rate=4)
    
    x = keras.layers.Concatenate()([x, x2])
    x = conv_block(x, intermediate_channels, rate=2)
    
    x = keras.layers.Concatenate()([x, x1])
    x = conv_block(x, intermediate_channels, rate=1)
    
    y = keras.layers.Add()([x, x0])
    return y