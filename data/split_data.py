from sklearn.model_selection import train_test_split
import os

def split_data(base_dir:str):
    mask_path, img_path = [], []
    for d in os.listdir(base_dir):
        class_dir = os.path.join(base_dir + '/' + d + '/im')
        images_name = os.listdir(class_dir)
        for name in images_name:
            image_path = class_dir + '/' + name
            img_path.append(image_path)
            
            mask_p = image_path.replace('.jpg', '.png').replace('/im', '/gt')
            mask_path.append(mask_p)
            
    train_img_path, val_img_path, train_mask_path, val_mask_path = train_test_split(
        img_path, mask_path, test_size=0.2, random_state=42, shuffle=True)
    
    return train_img_path, val_img_path, train_mask_path, val_mask_path