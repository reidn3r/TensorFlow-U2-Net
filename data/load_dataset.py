import tensorflow as tf
from load_image_mask import load_image_and_mask

def load_dataset(img_paths, mask_paths, batch_size, img_size=128, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths), reshuffle_each_iteration=True)
    
    ds = ds.map(lambda img, mask: load_image_and_mask(img, mask, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds