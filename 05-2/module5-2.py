import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mnist_data import MnistData

def main():
    parser = argparse.ArgumentParser(description="Test MnistData class.")
    parser.add_argument('dataset', choices=['train', 'test'], help='Dataset type to load (train/test)')
    parser.add_argument('index', type=int, help='Index of the image to display')

    args = parser.parse_args()

    mnist_data = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load()

    dataset_choices = {
        'train': (train_images, train_labels),
        'test': (test_images, test_labels)
    }

    images, labels = dataset_choices[args.dataset]
    idx = args.index

    plt.imshow(images[idx].reshape(mnist_data.image_dim))
    plt.title(f'Label: {np.argmax(labels[idx])}')
    plt.show()

    print(f'Label (one-hot): {labels[idx]}')
    print(f'Label: {np.argmax(labels[idx])}')

if __name__ == "__main__":
    main()