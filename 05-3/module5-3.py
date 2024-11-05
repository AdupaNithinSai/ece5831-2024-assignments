from mnist import Mnist
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_custom_image(image_path):
    #Load and preprocess the image to match the input format of the trained model.
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img = np.asarray(img, dtype=np.float32).reshape(1, 784)
    img = img / 255.0
    return img

def main():
    parser = argparse.ArgumentParser(description='MNIST Digit Prediction')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('true_digit', type=int, help='True digit of the image.')
    args = parser.parse_args()

    mnist = Mnist()
    mnist.init_network()
    
    img = load_custom_image(args.image_path)
    
    if img is not None:
        prediction = mnist.predict(img)
        predicted_digit = np.argmax(prediction)
        
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'True: {args.true_digit}, Predicted: {predicted_digit}')
        plt.show()
        
        if predicted_digit == args.true_digit:
            print(f"Success: Image {args.image_path} is for digit {args.true_digit} and is recognized as {predicted_digit}.")
        else:
            print(f"Fail: Image {args.image_path} is for digit {args.true_digit} but the inference result is {predicted_digit}.")
    else:
        print('Image could not be loaded.')

if __name__ == "__main__":
    main()