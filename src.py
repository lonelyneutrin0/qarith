from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from numpy import pi
#############################################################
#                 Quantum Fourier Transform                 #
#############################################################

def qft_(circuit: QuantumCircuit, n: int): 
    """
    Recursive function for applying Quantum Fourier Transform (reverse qubit order) 
    """
    if n == 0: 
        return circuit
    # Zero indexing
    n -= 1 
    circuit.h(n)
    for qubit in range(n): 
        circuit.cp(pi/(2**(n-qubit)), qubit, n)
    qft_(circuit, n)

def swap_(circuit: QuantumCircuit, n: int): 
    """
    Swapping function to ensure proper qubit order 
    """
    for i in range(n//2): 
        circuit.swap(i, n-i-1)
    return circuit

def QFT(circuit: QuantumCircuit): 
    """
    Quantum Fourier Transform 
    """
    n = circuit.num_qubits 
    qft_(circuit, n)
    swap_(circuit, n)
    return circuit

def iQFT(circuit, n):
    """
    Inverse Quantum Fourier Transform 
    """
    qft_circ = QFT(circuit=QuantumCircuit(n))
    invqft_circ = qft_circ.inverse()
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose()

#############################################################
#                 Quantum Arithmetic Gates                  #
#############################################################

def NQubitAdder(A: QuantumRegister, B: QuantumRegister, carry_in: QuantumRegister=None)->QuantumCircuit: 
    """
    N-Qubit Adder Circuit using Toffoli and CNOT gates. 
    """
    if A.size != B.size: raise ValueError("The given quantum registers are not of the same size")
    if carry_in is None: 
        carry_in = QuantumRegister(1)
    N = A.size
    
    S = QuantumRegister(N, 'sum')
    carry_out = QuantumRegister(N, 'c_out')
    c_S = ClassicalRegister(N, 'c_sum')
    c_carry_out = ClassicalRegister(N, 'c_c_out')

    circuit = QuantumCircuit(A,B,S,carry_in, carry_out, c_S, c_carry_out)
    
    for i in range(N): 

        # Compute the partial sum
        circuit.cx(A[i], S[i])
        circuit.cx(B[i], S[i])
        if i == 0: 
            circuit.cx(carry_in[0], S[i])
        else: 
            circuit.cx(carry_out[i-1], S[i])
        
        # Carry Out Calculations
        if i == 0: 
            circuit.ccx(A[i], B[i], carry_out[i])
            circuit.ccx(A[i], carry_in[0], carry_out[0])
            circuit.ccx(B[i], carry_in[0], carry_out[0])
        else: 
            circuit.ccx(A[i], B[i], carry_out[i])
            circuit.ccx(A[i], carry_out[i-1], carry_out[i])
            circuit.ccx(B[i], carry_out[i-1], carry_out[i])

    # Measurements 
    for i in range(N): 
        circuit.measure(S[i], c_S[i])
        circuit.measure(carry_out[i], c_carry_out[i])
    
    return circuit
