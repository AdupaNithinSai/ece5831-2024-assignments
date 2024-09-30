from logic_gate import LogicGate

def test_logic_gates():
    lg = LogicGate()
    
    print("AND Gate:")
    print("0 AND 0 =", lg.and_gate(0, 0))
    print("0 AND 1 =", lg.and_gate(0, 1))
    print("1 AND 0 =", lg.and_gate(1, 0))
    print("1 AND 1 =", lg.and_gate(1, 1))

    print("\nNAND Gate:")
    print("0 NAND 0 =", lg.nand_gate(0, 0))
    print("0 NAND 1 =", lg.nand_gate(0, 1))
    print("1 NAND 0 =", lg.nand_gate(1, 0))
    print("1 NAND 1 =", lg.nand_gate(1, 1))

    print("\nOR Gate:")
    print("0 OR 0 =", lg.or_gate(0, 0))
    print("0 OR 1 =", lg.or_gate(0, 1))
    print("1 OR 0 =", lg.or_gate(1, 0))
    print("1 OR 1 =", lg.or_gate(1, 1))

    print("\nNOR Gate:")
    print("0 NOR 0 =", lg.nor_gate(0, 0))
    print("0 NOR 1 =", lg.nor_gate(0, 1))
    print("1 NOR 0 =", lg.nor_gate(1, 0))
    print("1 NOR 1 =", lg.nor_gate(1, 1))

    print("\nXOR Gate:")
    print("0 XOR 0 =", lg.xor_gate(0, 0))
    print("0 XOR 1 =", lg.xor_gate(0, 1))
    print("1 XOR 0 =", lg.xor_gate(1, 0))
    print("1 XOR 1 =", lg.xor_gate(1, 1))

if __name__ == "__main__":
    test_logic_gates()
