from multilayer_percetron import MultiLayerPerceptron
import numpy as np

def test_mlp():
    # Initialize and configure the MLP
    mlp = MultiLayerPerceptron(activation='sigmoid')
    mlp.init_network()
    
    # Test data
    input_data = np.array([[0.5, 0.1], [1.0, -1.0], [-0.5, 0.7]])
    
    # Forward pass
    outputs = []
    for data in input_data:
        output = mlp.forward(data)
        outputs.append(output)
        print(f"Input: {data}, Output: {output}")

if __name__ == "__main__":
    print("Testing MultiLayerPerceptron...")
    test_mlp()
