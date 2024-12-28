from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import qiskit
import numpy as np 
import qiskit.circuit
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
        invqft_circ.name='iQFT'
        self.append(invqft_circ, self.qubits[:n])
    elif target_range is None: 
        n = target.size
        qft_circ = QFTCircuit(n)
        invqft_circ = qft_circ.inverse()
        invqft_circ.decompose() 
        invqft_circ.name='iQFT'
        self.append(invqft_circ, target)
    
    else: 
        n = target_range[1]-target_range[0]
        qft_circ = QFTCircuit(n)
        invqft_circ = qft_circ.inverse()
        invqft_circ.decompose() 
        invqft_circ.name='iQFT'
        self.append(invqft_circ, qargs=np.arange(start=target_range[0], stop=target_range[1]).tolist())        

QuantumCircuit.QFT = QFT
QuantumCircuit.iQFT = iQFT

#############################################################
#                  Custom Controlled Gates                  #
#############################################################

def ccp(
    self, 
    theta: float, 
    control_qubit1, 
    control_qubit2, 
    target_qubit
    )->None: 
    
    """
    Implementation Specifications
    ------------------------------
    Controlled controlled phase gate (CCRZ/CCP)
    
    Qubit Requirement - 3
    
    Introduces a phase change in target qubit if both control qubits are true
    
    Parameters
    ----------- 
    
    self : QuantumCircuit 
        The circuit to which the gate is to be added. 
    
    theta : float 0 <= θ <= 2π
        The phase change to introduce
    
    control_qubit1 : QubitSpecifier
        The first control qubit 
    
    control_qubit2 : QubitSpecifier
        The second control qubit 
    
    target_qubit : QubitSpecifier
        The target qubit
    
    Return Object
    ---------------
    
    None : This is purely a mutator method 
    """
    rz_matrix = np.array([[1, 0],
                        [0, np.exp(1j * theta)]])
    
    rz_matrix = np.block(
        [
        [np.eye(2), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))], 
        [np.zeros((2,2)), np.eye(2), np.zeros((2,2)), np.zeros((2,2))], 
        [np.zeros((2,2)), np.zeros((2,2)), np.eye(2), np.zeros((2,2))],
        [np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), rz_matrix]
        ]
        )
    
    ccp = qiskit.circuit.library.UnitaryGate(rz_matrix, label='ccp')
    if control_qubit1 == control_qubit2: 
        self.cp(theta, control_qubit1, target_qubit)
    else:
        self.append(ccp, [control_qubit1, control_qubit2, target_qubit])

def cccp(self, theta, control_qubit1, control_qubit2, control_qubit3, target_qubit):    
    """
    Implementation Specifications
    ------------------------------
    Controlled controlled controlled phase gate (CCCRZ/CCCP)
    
    Qubit Requirement - 3
    
    Introduces a phase change in target qubit if all control qubits are true
    
    Parameters
    ----------- 
    
    self : QuantumCircuit 
        The circuit to which the gate is to be added. 
    
    theta : float 0 <= θ <= 2π
        The phase change to introduce
    
    control_qubit1 : QubitSpecifier
        The first control qubit 
    
    control_qubit2 : QubitSpecifier
        The second control qubit 
    
    control_qubit3 : QubitSpecifier
        The third control qubit 

    target_qubit : QubitSpecifier
        The target qubit
    
    Return Object
    ---------------
    
    None : This is purely a mutator method 
    """
    rz_matrix = np.array([
                [1, 0],
                [0, np.exp(1j * theta)]
                ])
    
    rz_matrix = np.block([
        [np.eye(2), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))], 
        [np.zeros((2,2)), np.eye(2), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))],
        [np.zeros((2,2)), np.zeros((2,2)), np.eye(2), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))],
        [np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.eye(2), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))], 
        [np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.eye(2), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))], 
        [np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.eye(2), np.zeros((2,2)), np.zeros((2,2))],
        [np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.eye(2), np.zeros((2,2))],
        [np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), rz_matrix] 
    ])

    cccp = qiskit.circuit.library.UnitaryGate(rz_matrix, label='ccp')
    if control_qubit1 == control_qubit2 == control_qubit3: 
        self.cp(theta, control_qubit1, target_qubit)
    elif control_qubit1 == control_qubit2: 
        self.ccp(theta, control_qubit1, control_qubit3, target_qubit)
    elif control_qubit2 == control_qubit3: 
        self.ccp(theta, control_qubit1, control_qubit2, target_qubit)
    elif control_qubit1 == control_qubit3: 
        self.ccp(theta, control_qubit2, control_qubit3, target_qubit)
    else:
        self.append(cccp, [control_qubit1, control_qubit2, control_qubit3, target_qubit])


QuantumCircuit.ccp = ccp
QuantumCircuit.cccp = cccp 

#############################################################
#                 Quantum Arithmetic Gates                  #
#############################################################

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
    
    This implementation uses an AQFT with a k-cutoff of log2(n), which is more 
    accurate than QFT in the presence of decoherence [Draper].
    
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
        
        # CP from A to B
        for i in range(B.size): 
            for j in range(B.size-i-1): 
                if B.size-i-j-1 < np.log2(B.size):
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

def Subtractor(self, A: QuantumRegister, B: QuantumRegister, difference: QuantumRegister=None)->None:
    """ 
    Implementation Specifications
    ------------------------------
    Quantum Fourier Transform based subtractor gate. |A⟩|B⟩ ↦ |A⟩|A-B⟩ OR |A⟩|B⟩|0⟩ ↦ |A⟩|B⟩|A-B⟩
    Negative numbers wrap around 
    
    Qubit Requirement - (2N+2) for 2 N-Qubit Integers
    
    Strictly speaking, only 2N Qubits are needed for the gate.
    However, QFT-based subtraction is intrinsically modular. 
    Hence, the function actually operates on a qudit as |A⟩|B⟩ ↦ |A⟩|A-B mod 2^d⟩.
    This is resolved by represented the qudits in d+1 qubits (e.g. 011 -> 0011)
    
    Parameters
    ----------- 
    
    self : QuantumCircuit 
        The circuit to which the gate is to be added. 
    
    A : QuantumRegister, size d [see qubit requirements above]
        The first input register for the difference 
    
    B : QuantumRegister, size d [see qubit requirements above]
        The second input register for the difference
    
    difference : QuantumRegister, size d [see qubit requirements above], optional 
        The output register for the difference. 
        Defaults to None, output is mapped to B
    
    Return Object
    ---------------
    
    None : This is purely a mutator method 
    """
    
    # Check gate sizes 
    if A.size != B.size: raise ValueError("Improper register size!")
    
    # If a designated difference register is not specified
    if(difference is None): 
        
        # QFT on B 
        for i in range(B.size): 
            self.h(B[B.size-i-1])
            for j in range(B.size-i-1): 
                self.cp(-np.pi/2**(B.size-i-1-j), B[j], B[B.size-i-1])
        
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
        if difference.size != A.size: raise ValueError("Improper registry size!")
        for i in range(difference.size): 
            self.h(difference[difference.size-i-1])
            for j in range(difference.size-i-1): 
                self.cp(np.pi/2**(difference.size-i-1-j), A[j], difference[difference.size-i-1])
        for i in range(difference.size): 
            for j in range(difference.size-i-1): 
                self.cp(-np.pi/2**(difference.size-i-1-j), B[j], difference[difference.size-i-1])   
        for i in range(difference.size): 
            self.cp(np.pi, A[i], difference[i])
            self.cp(-np.pi, B[i], difference[i])
        for i in range(difference.size//2): 
            self.swap(difference[i], difference[difference.size-i-1])   
        
        self.iQFT(difference)

def Multiplier(self, A: QuantumRegister, B: QuantumRegister, prod: QuantumRegister)->None: 
    """ 
    Implementation Specifications
    ------------------------------
    Quantum Fourier Transform based multiplier gate. |A⟩|B⟩|0⟩ ↦ |A⟩|B⟩|A*B⟩
    
    Qubit Requirement - (6N) for 2 N-Qubit Integers (worst case)
    
    Strictly speaking, only 3N Qubits are needed for the gate.
    However, QFT-based multiplication is intrinsically modular. 
    Hence, the function actually operates on a qudit as |A⟩|B⟩|0⟩ ↦ |A⟩|B⟩|A*B mod 2^d⟩.
    This is resolved by represented the qudits in 2d qubits (e.g. 011 -> 000011)
    
    Parameters
    ----------- 
    
    self : QuantumCircuit 
        The circuit to which the gate is to be added. 
    
    A : QuantumRegister, size d [see qubit requirements above]
        The first input register for the difference 
    
    B : QuantumRegister, size d [see qubit requirements above]
        The second input register for the difference
    
    prod : QuantumRegister, size d [see qubit requirements above], optional 
        The output register for the difference. 
        Defaults to None, output is mapped to B
    
    Return Object
    ---------------
    
    None : This is purely a mutator method 
    """
    n = A.size 
    if(B.size != n or prod.size != n): raise ValueError("Improper register size!")
    
    for i in range(prod.size): 
        self.h(prod[i])
        
    for k in range(prod.size): 
        for i in range(prod.size): 
            for j in range(B.size-i-1): 
                self.ccp((2**k)*np.pi/2**(B.size-i-1-j), A[j], B[k], prod[B.size-i-1])  
    
    for k in range(prod.size): 
        for i in range(prod.size): 
            self.ccp((2**k)*np.pi, A[i], B[k], prod[i])
    
    for i in range(prod.size//2): 
        self.swap(prod[i], prod[B.size-i-1])
    
    self.iQFT(prod)

QuantumCircuit.Multiplier = Multiplier
QuantumCircuit.Subtractor = Subtractor
QuantumCircuit.Adder = Adder 

#############################################################
#                     Utility Functions                     #
#############################################################

def encode(self, reg: QuantumRegister, n: int ): 
    for i in range(reg.size): 
        if (n >> i) & 1: 
            self.x(reg[i])

QuantumCircuit.encode = encode