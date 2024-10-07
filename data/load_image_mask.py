import tensorflow as tf

def load_image_and_mask(image_path, mask_path, img_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.math.scalar_mul(1.0/255, img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [img_size, img_size])
    mask = tf.math.scalar_mul(1.0/255, mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    return img, mask