import numpy as np

from qpe import ejecutar_qpe

# ---------------------------------------------------------------------------
# Construcción de U y |w>
# ---------------------------------------------------------------------------
_c = np.cos(np.pi / np.sqrt(2))
_s = np.sin(np.pi / np.sqrt(2))
_r1 = np.sqrt(2 + np.sqrt(2))
_r2 = np.sqrt(2 - np.sqrt(2))

a = (-4 - _r1 + 2 * _c + 1j * (_r2 + 2 * _s)) / 16
b = (-_r1 - 2 * _c + 1j * (_r2 - 2 * _s)) / 16
d = (12 - _r1 + 2 * _c + 1j * (_r2 + 2 * _s)) / 16

U = np.array([
    [d, b, a, b, a, b, a, b],
    [b, d, b, a, b, a, b, a],
    [a, b, d, b, a, b, a, b],
    [b, a, b, d, b, a, b, a],
    [a, b, a, b, d, b, a, b],
    [b, a, b, a, b, d, b, a],
    [a, b, a, b, a, b, d, b],
    [b, a, b, a, b, a, b, d],
], dtype=np.complex128)

w = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.complex128) / np.sqrt(8)

THETA = np.sqrt(2) / 4

# Combinaciones (m, epsilon) del experimento, con epsilon = 1/inv_eps
COMBINACIONES = [(3, 8), (3, 32), (7, 8), (7, 32)]


def calcular_t(m: int, epsilon: float) -> int:
    return m + int(np.ceil(np.log2(2 + 1 / (2 * epsilon))))


def frecuencia_exito(conteos: dict[int, int], t: int, m: int,
                     theta: float = THETA) -> float:
    Q = 2 ** t
    total = sum(conteos.values())
    exitos = sum(c for l, c in conteos.items()
                 if abs(l / Q - theta) <= 2.0 ** (-m))
    return exitos / total


def mejor_estimacion(conteos: dict[int, int]) -> int:
    return max(conteos, key=conteos.get)


def imprimir_linea(m: int, inv_eps: int, p: float, l_star: int, t: int):
    print(f"Para m = {m} y epsilon = 1/{inv_eps}: "
          f"frecuencia de éxito = {p:.4f}; fase estimada = {l_star}/{2 ** t}")


if __name__ == "__main__":
    for m, inv_eps in COMBINACIONES:
        t = calcular_t(m, 1 / inv_eps)
        conteos = ejecutar_qpe(U, w, t, num_iter=10000)
        p_sim = frecuencia_exito(conteos, t, m)
        l_star = mejor_estimacion(conteos)
        imprimir_linea(m, inv_eps, p_sim, l_star, t)
