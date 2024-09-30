import numpy as np

class LogicGate:
    def __init__(self):
        pass

    def and_gate(self, x1, x2):
        return int(np.logical_and(x1, x2))

    def nand_gate(self, x1, x2):
        return int(np.logical_not(np.logical_and(x1, x2)))

    def or_gate(self, x1, x2):
        return int(np.logical_or(x1, x2))

    def nor_gate(self, x1, x2):
        return int(np.logical_not(np.logical_or(x1, x2)))

    def xor_gate(self, x1, x2):
        return int(np.logical_xor(x1, x2))

if __name__ == "__main__":
    print("LogicGate class")
    print("Available methods:")
    print("1. and_gate(x1, x2)")
    print("2. nand_gate(x1, x2)")
    print("3. or_gate(x1, x2)")
    print("4. nor_gate(x1, x2)")
    print("5. xor_gate(x1, x2)")
