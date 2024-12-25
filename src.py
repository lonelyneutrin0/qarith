from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import qiskit
import numpy as np 
from qiskit_aer import Aer
from typing import Tuple 
#############################################################
#                 Quantum Fourier Transform                 #
#############################################################

def QFT(self: QuantumCircuit, 
        target: QuantumRegister=None, 
        target_range: Tuple[int, int]=None, 
        )->None: 
    """
    Quantum Fourier Transform.
    
    Parameters
    -----------
    
    self : QuantumCircuit 
        The circuit to which the QFT is applied. 
    
    target : QuantumCircuit, optional
        The register to which the QFT is applied.
        If not specified, QFT is applied to the entire circuit.
    
    range : Tuple(min, max), optional. max exclusive
        The range of qubits over which the QFT is applied. 
        If not specific, QFT is applied to the selected 
        target or the entire circuit. 
    
    Return Object
    --------------
    Does not return an object. Purely a mutator method
    """
    circuit = self

    if target is None and target_range is None:
        n = circuit.num_qubits
        # Quantum Fourier Transform 
        for i in range(n): 
            circuit.h(n-i-1)
            for j in range(n-i-1):
                circuit.cp(np.pi/2**(n-i-1-j), j, n-i-1)
        
        # Swapping
        for i in range(n//2): 
            circuit.swap(i, n-1-i)
    
    elif target_range is None: 
        for i in range(target.size): 
            circuit.h(target[target.size-i-1])
            for j in range(target.size-i-1): 
                circuit.cp(np.pi/2**(target.size-i-1-j), target[j], target[target.size-i-1])
        
        for i in range(target.size//2): 
            circuit.swap(target[i], target[target.size-i-1])
    
    else: 
        n = target_range[1] - target_range[0] 
        for i in range(n): 
            circuit.h(target_range[0] + n-i-1)
            for j in range(n-i-1):
                circuit.cp(np.pi/2**(n-i-1-j), target_range[0] + j, target_range[0] + n-i-1)
        # Swapping
        for i in range(n//2): 
            circuit.swap(target_range[0] + i, target_range[0] + n-1-i)

def QFTCircuit(n: int)->QuantumCircuit: 
    """ 
    Implementation Specifications
    ------------------------------

    Returns a quantum Fourier transform circuit.

    Parameters
    -----------

    n : int
        Number of qubits to create the circuit with.

    Return Object
    --------------
    
    QuantumCircuit
    """

    circuit = QuantumCircuit(n)
    for i in range(n): 
        circuit.h(n-i-1)
        for j in range(n-i-1):
            circuit.cp(np.pi/2**(n-i-1-j), j, n-i-1)
    
    # Swapping
    for i in range(n//2): 
        circuit.swap(i, n-1-i)
    
    return circuit

def iQFT(self: QuantumCircuit, 
        target: QuantumRegister=None, 
        target_range: Tuple[int, int]=None)->None:
    """
    Inverse Quantum Fourier Transform.
    
    Parameters
    -----------
    
    self : QuantumCircuit 
        The circuit to which the iQFT is applied. 
    
    target : QuantumCircuit, optional
        The register to which the iQFT is applied.
        If not specified, QFT is applied to the entire circuit.
    
    range : Tuple(min, max), optional. max exclusive
        The range of qubits over which the QFT is applied. 
        If not specific, QFT is applied to the selected 
        target or the entire circuit. 
    
    Return Object
    --------------
    Does not return an object. Purely a mutator method
    """
    if target is None and target_range is None: 
        n = self.num_qubits
        qft_circ = QFTCircuit(n)
        invqft_circ = qft_circ.inverse()
        invqft_circ.decompose() 
        self.append(invqft_circ, self.qubits[:n])
    elif target_range is None: 
        n = target.size
        qft_circ = QFTCircuit(n)
        invqft_circ = qft_circ.inverse()
        invqft_circ.decompose() 
        self.append(invqft_circ, target)
    
    else: 
        n = target_range[1]-target_range[0]
        qft_circ = QFTCircuit(n)
        invqft_circ = qft_circ.inverse()
        invqft_circ.decompose() 
        self.append(invqft_circ, qargs=np.arange(start=target_range[0], stop=target_range[1]).tolist())        

QuantumCircuit.QFT = QFT
QuantumCircuit.iQFT = iQFT

#############################################################
#                 Quantum Arithmetic Gates                  #
#############################################################

def FullCarryAdder(
        A: QuantumRegister, 
        B: QuantumRegister, 
        carry_in: QuantumRegister=None, 
        carry_out: QuantumRegister=None, 
        S: QuantumRegister=None,
        classical_S: ClassicalRegister=None, 
        classical_carry_out: ClassicalRegister=None
        )->qiskit.QuantumCircuit:

    """
    N-Qubit Adder Circuit using Toffoli and CNOT gates. (2N+1, 2N) Implementation. Use .compose() to attach to other circuits
    :param A: Input Register 1 (N Qubits)
    :param B: Input Register 2 (N Qubits)
    :param carry_in: carry-in register (1 qubit only)
    :param carry_out: carry-out register (N Qubits)
    :param S: Partial Sum Register (N Qubits)
    :param classical_S: Partial Sum Storage (N Bits)
    :param classical_carry_out: Carry-out Register (N bits only)
    :returns Instruction: If all registers are specified correctly 
    :returns Circuit: If one or more required Registers are not specified
    """
    if(all(x is not None for x in [carry_in, carry_out, S, classical_carry_out, classical_S])): 
        N = A.size
        for i in [B, carry_out, S, classical_carry_out, classical_S]: 
            if i.size != N: raise ValueError("Improper Register Qubit Count!")
        if(carry_in.size != 1): raise ValueError("Improper Carry-In Qubit")
        
        circuit = QuantumCircuit(A,B,S,carry_in, carry_out, classical_S, classical_carry_out)
        
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
            circuit.measure(S[i], classical_S[i])
            circuit.measure(carry_out[i], classical_carry_out[i])
        
        instr = circuit.to_instruction()
        instr.name = "Adder"
        return instr
    else: 
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

def Adder(self: QuantumCircuit, A: QuantumRegister, B: QuantumRegister, sum: QuantumRegister=None)->None: 
    """ 
    Implementation Specifications
    ------------------------------
    Quantum Fourier Transform based adder gate. |A⟩|B⟩ ↦ |A⟩|A+B⟩ OR |A⟩|B⟩|0⟩ ↦ |A⟩|B⟩|A+B⟩
    
    Qubit Requirement - (2N+2) for 2 N-Qubit Integers
    
    Strictly speaking, only 2N Qubits are needed for the gate.
    However, QFT-based addition is intrinsically modular. 
    Hence, the function actually operates on a qudit as |A⟩|B⟩ ↦ |A⟩|A+B mod 2^d⟩.
    This is resolved by represented the qudits in d+1 qubits (e.g. 011 -> 0011)
    
    Parameters
    ----------- 

    self : QuantumCircuit 
        The circuit to which the gate is to be added. 
    
    A : QuantumRegister, size d [see qubit requirements above]
        The first input register for the sum 
    
    B : QuantumRegister, size d [see qubit requirements above]
        The second input register for the sum 
    
    sum : QuantumRegister, size d [see qubit requirements above], optional 
        The output register for the sum. 
        Defaults to None, output is mapped to B

    Return Object
    ---------------

    None : This is purely a mutator method 
    """
    
    # Check gate sizes 
    if A.size != B.size: raise ValueError("Improper register size!")
    
    # If a designated sum register is not specified
    if(sum is None): 

        # QFT on B 
        for i in range(B.size): 
            self.h(B[B.size-i-1])
            for j in range(B.size-i-1): 
                self.cp(np.pi/2**(B.size-i-1-j), B[j], B[B.size-i-1])
        
        # QFT from A to B
        for i in range(B.size): 
            for j in range(B.size-i-1): 
                self.cp(np.pi/2**(B.size-i-1-j), A[j], B[B.size-i-1])   
            
        # Phase Shifts from A 
        for i in range(B.size): 
            self.cp(np.pi, A[i], B[i])
        for i in range(B.size//2): 
            self.swap(B[i], B[B.size-i-1])          
        
        self.iQFT(B)
    
    else: 
        if sum.size != A.size: raise ValueError("Improper registry size!")
        for i in range(sum.size): 
            self.h(sum[sum.size-i-1])
            for j in range(sum.size-i-1): 
                self.cp(np.pi/2**(sum.size-i-1-j), A[j], sum[sum.size-i-1])
        for i in range(sum.size): 
            for j in range(sum.size-i-1): 
                self.cp(np.pi/2**(sum.size-i-1-j), B[j], sum[sum.size-i-1])   
        for i in range(sum.size): 
            self.cp(np.pi, A[i], sum[i])
            self.cp(np.pi, B[i], sum[i])
        for i in range(sum.size//2): 
            self.swap(sum[i], sum[sum.size-i-1])   
        
        self.iQFT(sum)
        
QuantumCircuit.Adder = Adder