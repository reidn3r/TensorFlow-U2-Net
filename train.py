import os
import tensorflow as tf
from data.split_data import split_data
from data.load_dataset import load_dataset
from model.u2net import u2net
from model.callbacks.cb import model_callbacks
from dotenv import load_dotenv

load_dotenv() 
def train(dataset_dir:str):
    #Param. Definitions
    img_size = os.getenv("img_size")
    bsize = os.getenv("bsize")
    input_shape = (img_size, img_size, 3)

    #Data Split and creating tf dataset 
    train_img_path, val_img_path, train_mask_path, val_mask_path = split_data(dataset_dir)
    train_ds = load_dataset(train_img_path, train_mask_path, batch_size=bsize, img_size=img_size, shuffle=True)
    val_ds = load_dataset(val_img_path, val_mask_path, batch_size=bsize, img_size=img_size, shuffle=False)

    #Model Building
    model = u2net(input_shape)
    
    #Compile Model
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    )

    # model.summary()

    #Fit Callbacks
    cb = model_callbacks()

    #Fit method
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=cb,
        verbose=1
    )

train('/')