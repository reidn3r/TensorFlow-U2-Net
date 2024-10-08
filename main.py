import os
from tensorflow import keras
from huggingface_hub import hf_hub_download
from inference.single_shot import single_shot
from model.u2net import u2net

os.environ["KERAS_BACKEND"] = "tensorflow"
def main():
    model = u2net(input_shape=(512,512,3))
    
    weights_path = hf_hub_download(repo_id="reidn3r/u2net-image-rembg", filename="model.weights.h5")    
    model = model.load_weights(weights_path)    
    single_shot('./io/test.jpg', './io/result.png', model)


if __name__ == "__main__":
    main()