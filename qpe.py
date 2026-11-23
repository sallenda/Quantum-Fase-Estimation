import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.primitives import StatevectorSampler


def _qft_inversa(t: int) -> QuantumCircuit:
    qc = QuantumCircuit(t, name="QFT†")
    # Deshacer los swaps de la QFT
    for j in range(t // 2):
        qc.swap(j, t - 1 - j)
    # Inversa de las rotaciones controladas + Hadamards, en orden inverso
    for j in range(t):
        for k in range(j):
            qc.cp(-np.pi / 2 ** (j - k), k, j)
        qc.h(j)
    return qc


def circuito_qpe(U: np.ndarray, w: np.ndarray, t: int) -> QuantumCircuit:
    U = np.asarray(U, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    dim = U.shape[0]
    n = int(round(np.log2(dim)))

    fase = QuantumRegister(t, "fase")
    objetivo = QuantumRegister(n, "obj")
    c = ClassicalRegister(t, "c")
    qc = QuantumCircuit(fase, objetivo, c)

    # 1. Preparación de |0^t> ⊗ |w>  (StatePreparation no usa reset)
    w = w / np.linalg.norm(w)
    qc.append(StatePreparation(w, label="|w>"), objetivo)

    # 2. Superposición uniforme en el registro de fase
    qc.h(fase)

    # 3. Potencias controladas U^(2^j), controladas por el qubit de fase j
    potencia = U.copy()
    for j in range(t):
        gate = UnitaryGate(potencia, label=f"U^{2 ** j}").control(1)
        qc.append(gate, [fase[j], *objetivo])
        potencia = potencia @ potencia  # U^(2^(j+1))

    # 4. QFT inversa sobre el registro de fase
    qc.append(_qft_inversa(t).to_gate(), fase)

    # 5. Medición únicamente del registro de fase
    qc.measure(fase, c)
    return qc


def ejecutar_qpe(U: np.ndarray, w: np.ndarray, t: int,
                 num_iter: int = 10000) -> dict[int, int]:
    qc = circuito_qpe(U, w, t)
    sampler = StatevectorSampler()
    resultado = sampler.run([qc], shots=num_iter).result()
    conteos = resultado[0].data.c.get_int_counts()
    return {int(l): int(c) for l, c in conteos.items()}
