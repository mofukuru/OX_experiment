from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# IBM量子使うための操作
# register account
from qiskit_ibm_runtime import QiskitRuntimeService

"""
class emurate_qc_noise:
    def __init__(self, service, backend: str):
        self.emulate_backend = service.get_backend(backend)

    def noise(self):
        noise_model = NoiseModel.from_backend(self.emulate_backend)
        coupling_map = self.emulate_backend.configuration().coupling_map
        basis_gates = noise_model.basis_gates
        sim_backend = AerSimulator(noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates, method="statevector", device="GPU", cuStateVec_enable=True)
        properties = self.emulate_backend.properties()
        max_memory_mb = getattr(properties, "max_memory_mb", "Attribute not found")
        print(f"Max memory (MB): {max_memory_mb}")
        return sim_backend
"""

noise_model = NoiseModel()
error_rate = 0.05
# error = depolarizing_error(0.05, 1)
error = depolarizing_error(error_rate, 1)
noise_model.add_all_qubit_quantum_error(error, ["id"])

# please save the account if you use first
# QiskitRuntimeService.save_account(
#     channel="ibm_cloud",
#     token="<API_TOKEN>",
#     overwrite=True,
#     set_as_default=True,
# )
# using account
service = QiskitRuntimeService(
    channel="ibm_cloud", token="<API_TOKEN>"
)
### The method of working in real device is in progress.
REAL_DEVICE = True

# set backend
if REAL_DEVICE:
    # if you use REAL_DEVICE, occuring errors
    # backend = service.get_backend("ibm_kawasaki")
    backend = service.get_backend("ibm_torino")
    # backend = service.get_backend("ibm_osaka")
    pass
else:
    # backend = service.get_backend("simulator_statevector")
    backend = AerSimulator(
        method="statevector", device="GPU", cuStateVec_enable=True
    )
    backend_noise = AerSimulator(
        method="statevector",
        noise_model=noise_model,
        device="GPU",
        cuStateVec_enable=True,
    )
    # backend = emurate_qc_noise(service, "ibm_torino").noise()
if REAL_DEVICE:
    print(backend.name, backend.status().pending_jobs)
else:
    print(backend.name, backend.options.device)
# print(service.backends())
"""noise_model=emurate_qc_noise("ibm_torino"), """
