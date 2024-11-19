import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from le_net import LeNet

def load_image(image_path):
    """Load and preprocess image for prediction."""
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)  
    img_array = np.asarray(img).astype(np.float32)
    img_array = (255.0 - img_array) / 255.0 
    return img_array

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python module8.py <image_path> <true_label>")
        sys.exit(1)

    image_path = sys.argv[1]
    true_label = int(sys.argv[2])

    image_array = load_image(image_path)
    
    lenet = LeNet()
    lenet.load('adupa_cnn_model') 

    prediction = lenet.predict([image_array])
    predicted_label = prediction[0] if prediction is not None else None

    # Display the image
    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.title(f"True Label: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
    
    # Output result
    if predicted_label == true_label:
        print(f"Success: Image {image_path} is for digit {true_label} recognized as {true_label}.")
    else:
        print(f"Fail: Image {image_path} is for digit {true_label} but the inference result is {predicted_label}.")