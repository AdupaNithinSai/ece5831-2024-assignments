import sys
from PIL import Image
import numpy as np
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp
import matplotlib.pyplot as plt

def load_image(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)  # resize for MNIST compatibility
    img_array = np.asarray(img).astype(np.float32)
    img_array = img_array.flatten()
    img_array = (255.0 - img_array) / 255.0  # normalize
    return img_array

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python module6.py <image_path> <true_label>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    true_label = int(sys.argv[2])

    image_array = load_image(image_path)
    
    # Load and test model
    network = TwoLayerNetWithBackProp(input_size=784, hidden_size=50, output_size=10)
    network.load_params("adupa_mnist_model.pkl")

    prediction = np.argmax(network.predict(image_array.reshape(1, -1)))
    
    if prediction == true_label:
        print(f"Success: Image {image_path} is for digit {true_label} is recognized as {true_label}.")
    else:
        print(f"Fail: Image {image_path} is for digit {true_label} but the inference result is {prediction}.")
    
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.show()