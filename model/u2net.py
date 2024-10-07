from tensorflow import keras
from keras.models import Model
from model.rsu_block import RSU_Block
from model.rsu_bridge_block import RSU_Middle_Block

def u2net(input_shape):
    inputs = keras.layers.Input(input_shape)
    s0 = inputs
    
    #Encoder Path
    s1 = RSU_Block(x=s0, out_channels=64, intermediate_channels=32, n_layers=7)
    p1 = keras.layers.MaxPooling2D((2, 2))(s1)
    
    s2 = RSU_Block(x=p1, out_channels=128, intermediate_channels=32, n_layers=6)
    p2 = keras.layers.MaxPooling2D((2, 2))(s2)
    
    s3 = RSU_Block(x=p2, out_channels=256, intermediate_channels=64, n_layers=5)
    p3 = keras.layers.MaxPooling2D((2, 2))(s3)
    
    s4 = RSU_Block(x=p3, out_channels=512, intermediate_channels=128, n_layers=4)
    p4 = keras.layers.MaxPooling2D((2, 2))(s4)

    s5 = RSU_Middle_Block(p4, out_channels=512, intermediate_channels=256)
    p5 = keras.layers.MaxPooling2D((2, 2))(s5)
    print(f'last encoder rsu block layer shape: {s5.shape}')
    
    #Bridge
    b1 = RSU_Middle_Block(p5, out_channels=512, intermediate_channels=256)
    b2 = keras.layers.UpSampling2D(size=2, interpolation='bilinear')(b1)
    print(f'bridge shape: {b2.shape}')
    
    #Decoder
    d1 = keras.layers.Concatenate()([b2, s5])
    d1 = RSU_Middle_Block(d1, out_channels=512, intermediate_channels=256)
    up1 = keras.layers.UpSampling2D(size=2, interpolation='bilinear')(d1)
    
    d2 = keras.layers.Concatenate()([up1, s4])
    d2 = RSU_Block(d2, out_channels=256, intermediate_channels=128, n_layers=4)
    up2 = keras.layers.UpSampling2D(size=2, interpolation='bilinear')(d2)
    
    d3 = keras.layers.Concatenate()([up2, s3])
    d3 = RSU_Block(d3, out_channels=128, intermediate_channels=64, n_layers=5)
    up3 = keras.layers.UpSampling2D(size=2, interpolation='bilinear')(d3)
    
    d4 = keras.layers.Concatenate()([up3, s2])
    d4 = RSU_Block(d4, out_channels=64, intermediate_channels=32, n_layers=6)
    up4 = keras.layers.UpSampling2D(size=2, interpolation='bilinear')(d4)
    
    d5 = keras.layers.Concatenate()([up4, s1])
    d5 = RSU_Block(d5, out_channels=64, intermediate_channels=16, n_layers=7)
    print(f'last decoder rsu block layer shape: {d5.shape}')
    
    #Side Output
    side_1 = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", strides=1)(d5)
    
    side_2 = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", strides=1)(d4)
    side_2 = keras.layers.UpSampling2D(size=2, interpolation='bilinear')(side_2)
    
    side_3 = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", strides=1)(d3)
    side_3 = keras.layers.UpSampling2D(size=4, interpolation='bilinear')(side_3)
    
    side_4 = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", strides=1)(d2)
    side_4 = keras.layers.UpSampling2D(size=8, interpolation='bilinear')(side_4)
    
    side_5 = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", strides=1)(d1)
    side_5 = keras.layers.UpSampling2D(size=16, interpolation='bilinear')(side_5)
    
    side_6 = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", strides=1)(b1)
    side_6 = keras.layers.UpSampling2D(size=32, interpolation='bilinear')(side_6)
    
    print(f'side_1 shape:{side_1.shape}')
    print(f'side_2 shape:{side_2.shape}')
    print(f'side_3 shape:{side_3.shape}')
    print(f'side_4 shape:{side_4.shape}')
    print(f'side_5 shape:{side_5.shape}')
    print(f'side_6 shape:{side_6.shape}')
    
    fusion = keras.layers.Concatenate()([side_1, side_2, side_3, side_4, side_5, side_6])
    fusion_conv = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same")(fusion)
    
    #Activation Functions
    side_1 = keras.layers.Activation("sigmoid")(side_1)
    side_2 = keras.layers.Activation("sigmoid")(side_2)
    side_3 = keras.layers.Activation("sigmoid")(side_3)
    side_4 = keras.layers.Activation("sigmoid")(side_4)
    side_5 = keras.layers.Activation("sigmoid")(side_5)
    side_6 = keras.layers.Activation("sigmoid")(side_6)
    fusion_conv = keras.layers.Activation("sigmoid")(fusion_conv)
    
    model = Model(inputs=inputs, outputs=[fusion_conv, side_1, side_2, side_3, side_4, side_5, side_6])
    return model
    