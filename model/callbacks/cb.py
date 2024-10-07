from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from dotenv import load_dotenv

load_dotenv() 
def model_callbacks():
    #Base Path
    base_model_path = os.getenv(base_model_path)
    model_p = base_model_path + 'model.keras'
    weights_p = base_model_path + 'weights.weights.h5'


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)

    early = EarlyStopping(
        monitor="val_loss",
        patience=15, 
        verbose=1,
        mode="min", 
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(model_p, 
        monitor = "val_loss",
        mode = "min",
        verbose=1,
        save_best_only=True, 
        save_weights_only=False
    )

    weights_checkpoint = ModelCheckpoint(weights_p, 
        monitor = "val_loss",
        mode = "min",
        save_best_only=True, 
        save_weights_only=True
    )

    cb = [early, model_checkpoint, weights_checkpoint, reduce_lr]
    return cb