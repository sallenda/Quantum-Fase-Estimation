import json
import os
import sys

from qiskit.transpiler import generate_preset_pass_manager

from pregunta2 import (COMBINACIONES, U, calcular_t, frecuencia_exito,
                       imprimir_linea, mejor_estimacion, w)
from qpe import circuito_qpe

SHOTS = 1000
ARCHIVO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "resultados3.json")


def construir_circuitos(backend):
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    circuitos = []
    for m, inv_eps in COMBINACIONES:
        t = calcular_t(m, 1 / inv_eps)
        circuitos.append((m, inv_eps, t, pm.run(circuito_qpe(U, w, t))))
    return circuitos


def uso_qpu(job, intentos: int = 12, espera: float = 5.0) -> float:
    import time
    for _ in range(intentos):
        uso = job.usage()
        if uso:
            return uso
        time.sleep(espera)
    return job.usage()


def actualizar_usos(datos):
    pendientes = [e for e in datos["experimentos"] if not e["uso_qpu_segundos"]]
    if not pendientes:
        return False
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    for exp in pendientes:
        exp["uso_qpu_segundos"] = uso_qpu(service.job(exp["job_id"]))
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2)
    return True


def ejecutar_en_ibm():
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    nombre = os.environ.get("IBM_BACKEND")
    backend = (service.backend(nombre) if nombre
               else service.least_busy(operational=True, simulator=False))
    print(f"Backend: {backend.name}", file=sys.stderr)

    sampler = SamplerV2(mode=backend)
    datos = {"backend": backend.name, "experimentos": []}

    # Se envían los 4 trabajos (uno por circuito) y luego se esperan.
    trabajos = []
    for m, inv_eps, t, qc in construir_circuitos(backend):
        job = sampler.run([qc], shots=SHOTS)
        print(f"m={m}, eps=1/{inv_eps}, t={t}: job {job.job_id()} "
              f"(profundidad {qc.depth()})", file=sys.stderr)
        trabajos.append((m, inv_eps, t, job))

    for m, inv_eps, t, job in trabajos:
        resultado = job.result()
        conteos = resultado[0].data.c.get_int_counts()
        datos["experimentos"].append({
            "m": m,
            "inv_eps": inv_eps,
            "t": t,
            "job_id": job.job_id(),
            "uso_qpu_segundos": uso_qpu(job),
            "conteos": {str(l): int(c) for l, c in conteos.items()},
        })
        # Se guarda tras cada trabajo para no perder resultados
        with open(ARCHIVO, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2)
    return datos


def probar_transpilacion():
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    backend = FakeMarrakesh()
    for m, inv_eps, t, qc in construir_circuitos(backend):
        ops = qc.count_ops()
        dos_q = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
        print(f"m={m}, eps=1/{inv_eps}, t={t}: profundidad={qc.depth()}, "
              f"compuertas de 2 qubits={dos_q}")


def main():
    if "--probar" in sys.argv:
        probar_transpilacion()
        return

    if "--enviar" in sys.argv or not os.path.exists(ARCHIVO):
        datos = ejecutar_en_ibm()
    else:
        with open(ARCHIVO, encoding="utf-8") as f:
            datos = json.load(f)
        actualizar_usos(datos)

    for exp in datos["experimentos"]:
        m, inv_eps, t = exp["m"], exp["inv_eps"], exp["t"]
        conteos = {int(l): c for l, c in exp["conteos"].items()}
        assert sum(conteos.values()) == SHOTS
        p_ibm = frecuencia_exito(conteos, t, m)
        l_star = mejor_estimacion(conteos)
        imprimir_linea(m, inv_eps, p_ibm, l_star, t)

    print(f"\nBackend: {datos['backend']}", file=sys.stderr)
    for exp in datos["experimentos"]:
        print(f"m = {exp['m']}, epsilon = 1/{exp['inv_eps']}: "
              f"uso de QPU = {exp['uso_qpu_segundos']} s "
              f"(job {exp['job_id']})", file=sys.stderr)


if __name__ == "__main__":
    main()
