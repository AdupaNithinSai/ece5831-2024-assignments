import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import io

# Create a revised CNN model
def create_model():
    input_layer = tf.keras.Input(shape=(28, 28, 1))
    
    # Convolution layer 1 with 16 filters
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='Conv2D_1')(input_layer)
    
    # Downsampling layer 1 (Max Pooling)
    x = layers.MaxPooling2D((2, 2), name='MaxPooling2D_1')(x)
    
    # Convolution layer 2 with 32 filters
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv2D_2')(x)
    
    # Downsampling layer 2 (Max Pooling)
    x = layers.MaxPooling2D((2, 2), name='MaxPooling2D_2')(x)
    
    # Flatten and Fully-connected (Dense) layers
    x = layers.Flatten()(x)
    
    # Fully-connected layer 1 with 64 nodes
    x = layers.Dense(64, activation='relu', name='Dense_1')(x)
    
    # Fully-connected layer 2 with 32 nodes
    x = layers.Dense(32, activation='relu', name='Dense_2')(x)
    
    # Output layer with 10 nodes (for 10 classes)
    output_layer = layers.Dense(10, activation='softmax', name='Output')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

# Create and initialize the model
model = create_model()
model.summary()

# Create a model that will return the intermediate layer outputs
layer_outputs = [layer.output for layer in model.layers]
intermediate_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Function to preprocess the drawing image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    img_array = img_array / 255.0  # Normalize
    return img_array

# Function to add color to the feature maps
def add_color_to_feature_map(feature_map):
    colored_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
    return colored_map

# Function to plot the processed nodes in each layer
def plot_feature_maps(layer_indices, feature_maps, architecture_img):
    max_display_size = 150
    margin = 10
    width_needed = (max_display_size + margin) * len(layer_indices) + margin
    height_needed = architecture_img.shape[0] + max_display_size * 2 + 40  # Adjust height for dense layer

    output_img = np.ones((height_needed, width_needed, 3), dtype=np.uint8) * 255
    output_img[:architecture_img.shape[0], :architecture_img.shape[1]] = architecture_img

    y_offset = architecture_img.shape[0] + 10

    for idx, layer_index in enumerate(layer_indices):
        features = feature_maps[layer_index]
        layer_name = model.layers[layer_index].name

        if len(features.shape) == 4:  # Conv2D or MaxPooling2D layer
            num_filters = features.shape[-1]
            size = features.shape[1]
            grid_cols = min(max_display_size // size, num_filters)
            grid_rows = (num_filters + grid_cols - 1) // grid_cols
            display_grid = np.zeros((size * grid_rows, size * grid_cols))

            for i in range(num_filters):
                x = features[0, :, :, i]
                x -= x.mean()
                x /= (x.std() + 1e-5)
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')

                row = i // grid_cols
                col = i % grid_cols
                display_grid[row * size:(row + 1) * size, col * size:(col + 1) * size] = x

            display_grid = np.clip(display_grid, 0, 255).astype('uint8')
            display_grid = cv2.resize(display_grid, (max_display_size, max_display_size), interpolation=cv2.INTER_NEAREST)
            display_grid = add_color_to_feature_map(display_grid)

            x_offset = margin + (idx * (max_display_size + margin))
            output_img[y_offset:y_offset + max_display_size, x_offset:x_offset + max_display_size] = display_grid

            # Add layer name below the feature map
            cv2.putText(output_img, layer_name, (x_offset, y_offset + max_display_size + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        elif len(features.shape) == 2:  # Dense layer
            fig = plt.figure(figsize=(1.5, 1.5))
            plt.bar(range(features.shape[1]), features[0])
            plt.ylim(0, 1)
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            bar_img = np.array(Image.open(buf))
            bar_img = cv2.resize(bar_img, (max_display_size, max_display_size))

            x_offset = margin + (idx * (max_display_size + margin))
            bar_img = cv2.cvtColor(bar_img, cv2.COLOR_RGBA2BGR)
            h, w = bar_img.shape[:2]
            output_img[y_offset:y_offset + h, x_offset:x_offset + w] = bar_img

            # Add layer name below the bar plot
            cv2.putText(output_img, layer_name, (x_offset, y_offset + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Output', output_img)

# Function to create the CNN architecture visualization
def draw_cnn_architecture():
    fig, ax = plt.subplots(figsize=(num_layers * 2, 5))
    layer_types = [layer.__class__.__name__ for layer in model.layers]

    for i, layer_type in enumerate(layer_types):
        rect = patches.FancyBboxPatch((i * 2, 1), 1.8, 0.5, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightgray')
        ax.add_patch(rect)
        plt.text(i * 2 + 0.9, 1.2, layer_type, ha='center', va='bottom')

    plt.xlim(0, num_layers * 2)
    plt.ylim(0, 2)
    plt.axis('off')

    buf = io.BytesIO()
    fig.patch.set_alpha(1.0)  # Remove transparency
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    architecture_img = np.array(Image.open(buf))

    architecture_img = cv2.cvtColor(architecture_img, cv2.COLOR_RGBA2BGR)
    return architecture_img

# Drawing settings
canvas_width, canvas_height = 200, 200

# Create an OpenCV canvas
img = np.ones((canvas_width, canvas_height, 3), np.uint8) * 255

drawing = False
ix, iy = -1, -1

# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (0, 0, 0), 3)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 0, 0), 3)
        update_feature_maps()

def update_feature_maps():
    global img
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    preprocessed_img = preprocess_image(pil_img)
    feature_maps = intermediate_model.predict(preprocessed_img)
    plot_feature_maps(range(len(model.layers)), feature_maps, architecture_img)

# Create a named window
cv2.namedWindow('Draw')

# Set mouse callback for the drawing
cv2.setMouseCallback('Draw', draw_circle)

# Generate the CNN architecture visualization
num_layers = len(model.layers)
architecture_img = draw_cnn_architecture()
cv2.imshow('Output', architecture_img)

# Main loop
while True:
    cv2.imshow('Draw', img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cv2.destroyAllWindows()