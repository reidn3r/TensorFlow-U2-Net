import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv() 
def single_shot(image_path:str, output_path:str, model):
    img_size = 512

    def preprocess_image(image_path,img_size):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)

        #Original Dimensions
        dim = img.shape

        img = tf.image.resize(img, [img_size, img_size])
        img = tf.math.scalar_mul(1.0/255, img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.expand_dims(img, axis=0)
        return img, dim

    def save_result_image(img, mask, dim):
        np_image = np.array(img[0])

        np_mask = mask[0][0].numpy()
        np_mask = np.concatenate([np_mask, np_mask, np_mask], axis=-1)

        result = np_image * np_mask
        result = np.clip(result, 0, 1)

        w, h = dim[0], dim[1]
        result = tf.image.resize(result, [w, h]).numpy()

        alpha_channel = mask[0][0].numpy()
        alpha_channel = tf.image.resize(alpha_channel, [w, h]).numpy()

        result_with_alpha = np.dstack([result, alpha_channel])
        plt.imsave(output_path, result_with_alpha, format='png')
        # return result_with_alpha


    img, dim = preprocess_image(image_path, img_size)
    output = model(img)
    save_result_image(img, output, dim)
