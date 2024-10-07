import os
import tensorflow as tf
from dotenv import load_dotenv
from inference.single_shot import single_shot

load_dotenv() 
def main(model_path:str):
    model = tf.keras.models.load_model(model_path)
    single_shot('./io/test.jpg', './io/result.png', model)


if __name__ == "__main__":
    path = os.getenv('model_path')
    main(path)