import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_labels(label_file):
    with open(label_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main(image_path):
    model = tf.keras.models.load_model('keras_model.h5')
    class_names = load_labels('labels.txt')
    
    img = preprocess_image(image_path)
    
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence_score = predictions[0][class_idx]
    
    print(f"Class: {class_names[class_idx]}")
    print(f"Confidence Score: {confidence_score:.4f}")
    
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {class_names[class_idx]} ({confidence_score:.4f})")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify an image of rock, paper, or scissors.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()
    main(args.image)
