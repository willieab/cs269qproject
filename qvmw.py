import numpy as np
from pyquil import Program
from pyquil.quilatom import substitute_array

class qvmw:

    #Initialize class variables
    def __init__(self, seed=None):
        self.seed = seed

        self.n_qubits = 0
        self.n_to_binary = []
        self.N = 0
        self.amplitudes = []
        self.idx_map = dict()

        self.gates = dict()
        self.defgates = dict()

        self.cmemory = dict()
        self.memory_types = {'BIT': bool, 'REAL': float, 'INT': int}

        #Pauli gates
        self.gates['I'] = np.identity(2)
        self.gates['X'] = np.array([[0, 1], [1, 0]], dtype='bool')
        self.gates['Y'] = np.array([[0, -1.j], [1.j, 0]])
        self.gates['Z'] = np.array([[1, 0], [0, -1]], dtype='int')

        #Cartesian rotation gates
        self.gates['RX'] = lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
        self.gates['RY'] = lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
        self.gates['RZ'] = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(-1j*theta/2)]])

        #Hadamard gate
        self.gates['H'] = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

        #Phase gates
        self.gates['PHASE'] = lambda phi: np.diag([1, np.exp(1j*phi)])
        self.gates['S'] = self.gates['PHASE'](np.pi/2)
        self.gates['T'] = self.gates['PHASE'](np.pi/4)
        self.gates['CPHASE'] = lambda phi: np.diag([1, 1, 1, np.exp(1j*phi)])
        self.gates['CPHASE00'] = lambda phi: np.diag([np.exp(1j*phi), 1, 1, 1])
        self.gates['CPHASE01'] = lambda phi: np.diag([1, np.exp(1j*phi), 1, 1])
        self.gates['CPHASE10'] = lambda phi: np.diag([1, 1, np.exp(1j*phi), 1])

        #Controlled gates
        self.gates['CNOT'] = self.gates['X']
        self.gates['CZ'] = self.gates['Z']

        #Swap gates
        self.gates['PSWAP'] = lambda theta: np.array([[1, 0, 0, 0], [0, 0, np.exp(1j*theta), 0], [0, np.exp(1j*theta), 0, 0], [0, 0, 0, 1]])
        self.gates['SWAP'] = self.gates['PSWAP'](0)
        self.gates['ISWAP'] = self.gates['PSWAP'](np.pi/2)

        #Measurement gates
        self.gates['MEASURE0'] = np.array([[1, 0], [0, 0]])
        self.gates['MEASURE1'] = np.array([[0, 0], [0, 1]])

    #Initialize state according to number of qubits in p
    def _setup_state(self, p):
        self.n_qubits = len(p.get_qubits())
        self.N = 2 ** self.n_qubits
        self.n_to_binary = np.zeros([2 ** self.n_qubits, self.n_qubits], dtype='bool')
        for n in range(2 ** self.n_qubits):
            self.n_to_binary[n] = np.array(list(np.binary_repr(n, width=self.n_qubits))).astype(bool)
        self.n_to_binary = np.fliplr(self.n_to_binary)
        ctr = 0
        for idx in sorted(p.get_qubits()):
            self.idx_map[idx] = ctr
            ctr += 1
        self.amplitudes = np.zeros(2 ** self.n_qubits, dtype='complex')

        #Clear classical registers and user-defined gates
        self.cmemory = dict()
        self.defgates = dict()

    #Produce the wavefunction resulting from program
    def wavefunction(self, p: Program, ground_state=0):
        self._setup_state(p)
        self.amplitudes[ground_state] = 1.

        #Create user-defined gates
        for gate in p.defined_gates:
            self.defgates[gate.name] = [gate.parameters, gate.matrix]

        #Loop through instructions in program
        for instruction in p:
            #Declare a classical memory region
            if 'Declare' in str(type(instruction)):
                mem_type = self.memory_types.get(instruction.memory_type)
                if mem_type is None:
                    raise ValueError('Memory type not understood')
                if self.cmemory.get(instruction.name) is None:
                    self.cmemory[instruction.name] = np.zeros(instruction.memory_size, dtype=mem_type)
                else:
                    raise ValueError('Classical register has already been declared')
            #Apply the gate defined by program instruction
            elif 'Gate' in str(type(instruction)) and instruction.name in self.gates:
                self._applyGate(instruction.name, instruction.params, instruction.qubits)
            elif 'Gate' in str(type(instruction)) and instruction.name in self.defgates:
                name = instruction.name
                vals = {}
                if not self.defgates[name][0] is None:
                    for i, param in enumerate(self.defgates[name][0]):
                        vals[param] = instruction.params[i]
                self.gates[name] = substitute_array(self.defgates[name][1], vals)
                self._applyGate(name, None, instruction.qubits)
                del self.gates[name]
            #Measure a qubit and record the result in a classical register
            elif 'Measure' in str(type(instruction)):
                target = instruction.get_qubits(indices=False).pop()
                idx = self.idx_map[target.index]
                p1 = sum(np.absolute(self.amplitudes[self.n_to_binary[:,idx]]) ** 2)
                result = np.random.choice([0,1], size=1, p=[1-p1, p1])[0]
                self.cmemory[instruction.classical_reg.name][instruction.classical_reg.offset] = result
                self._applyGate('MEASURE'+str(result), None, [target])
                self._normalizeState()
            else:
                raise NotImplementedError('Instuction ', instruction, ' is not yet implemented')

    #Apply a gate defined by a matrix
    def _applyGate(self, gate_name, params, qubits):
        matrix = self.gates[gate_name]
        if params: matrix = matrix(*params)

        #Record qubit indices
        first, second = sorted([self.idx_map[qubits[0].index], self.idx_map[qubits[-1].index]])

        #Identify target and control qubits for a controlled gate
        if gate_name in ['CNOT', 'CZ']:
            control_idx = self.idx_map[qubits[0].index]
            first = self.idx_map[qubits[-1].index]
            second = first

        #Apply permutation matrices in order to lift 2-qubit gate
        for i in reversed(range(first, second-1)):
            self._swapAdjQubits(first+second-i-2)

        #Apply a 1- or 2-qubit gate
        jump = 2 ** max(first, second - 1)
        block_size = 2 ** min(2, second - first + 1) * jump
        for block in np.arange(0, self.N, block_size):
            for j in range(jump):
                if gate_name in ['CNOT', 'CZ'] and not self.n_to_binary[block+jump+j, control_idx]:
                    continue
                affected = np.arange(block+j, block+j+block_size, jump)
                self.amplitudes[affected] = matrix @ self.amplitudes[affected]

        #Reverse permutations
        for i in range(first, second-1):
            self._swapAdjQubits(first+second-i-2)

    #Swap qubits idx, idx+1
    def _swapAdjQubits(self, idx):
        matrix = self.gates['SWAP']
        jump = 2 ** idx
        block_size = 4 * jump
        for block in np.arange(0, self.N, block_size):
            for j in range(jump):
                affected = np.arange(block+j, block+j+block_size, jump)
                self.amplitudes[affected] = matrix @ self.amplitudes[affected]

    def _normalizeState(self):
        self.amplitudes /= np.linalg.norm(self.amplitudes)

    #Run a program and measure its qubits trials number of times
    def run_and_measure(self, p: Program, trials=10):
        self.wavefunction(p)
        np.random.seed(self.seed)
        p = np.absolute(self.amplitudes) ** 2
        results = np.random.choice(2 ** self.n_qubits, size=trials, p=p)
        return self.n_to_binary[results].astype(int)

    #Return a unitary matrix corresponding to the Program
    def program_unitary(self, p: Program):
        self.wavefunction(p)
        u = np.zeros([self.N, self.N], dtype='complex')
        u[0] = self.amplitudes
        for n in range(1, self.N):
            self.wavefunction(p, ground_state=n)
            u[n] = self.amplitudes
        return u.T
