This project builds quantum arithmetic subcircuits in Qiskit which implement addition, subtraction and multiplication. This is done through a quantum Fourier transform
(QFT)-based approach which reduces qubit requirements. An approximate quantum Fourier transform (AQFTs) is implemented in addition which can improve accuracy in 
decoherent systems as shown by Draper. 
The advantage of QFT-based arithmetic is the benefit of non-modular options. For operations on 2 n-qubit integers, 2N qubits are required for the operation $A+B MOD 2^N$
and 2N+2 qubits are required for complete addition. 

