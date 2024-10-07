from conv_block import conv_block
from tensorflow import keras

def RSU_Block(x, out_channels, intermediate_channels, n_layers, rate=2):
    skip=[]
    
    x = conv_block(x, out_channels)
    original_features = x
    
    #Encoder
    x = conv_block(x, intermediate_channels)
    skip.append(x)
    
    for i in range(n_layers-2):
        x = keras.layers.MaxPool2D((2, 2))(x)
        x = conv_block(x, intermediate_channels)
        skip.append(x)
        
    #Bridge
    x = conv_block(x, intermediate_channels, rate=rate)
    
    #Decoder
    skip.reverse()
    x = keras.layers.Concatenate()([x, skip[0]])
    x = conv_block(x, intermediate_channels)
    
    for i in range(n_layers-3):
        x = keras.layers.UpSampling2D(size = (2,2), interpolation='bilinear')(x)
        x = keras.layers.Concatenate()([x, skip[i+1]])
        x = conv_block(x, intermediate_channels)
        
    x = keras.layers.UpSampling2D(size = (2,2), interpolation='bilinear')(x)
    x = keras.layers.Concatenate()([x, skip[-1]])
    x = conv_block(x, out_channels)
    
    y = keras.layers.Add()([x, original_features])
    return y