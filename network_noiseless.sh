# noiseless
# python3 train.py cqc_network \
# --noised_quantum_channel=False \
# --nn_network=3 \
# --total_episodes=5000 \
# --n_qubits=8 \
# --embedding_type=ZFeatureMap \
# --ansatz_type=RealAmplitudes \
# --network_model=1 \
# --early_stopping=True

python3 train.py cqc_network \
--noised_quantum_channel=False \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--embedding_type=ZFeatureMap \
--ansatz_type=RealAmplitudes \
--network_model=2 \
--early_stopping=True

python3 train.py cqc_network \
--noised_quantum_channel=False \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--embedding_type=ZZFeatureMap \
--ansatz_type=RealAmplitudes \
--network_model=1 \
--early_stopping=True

python3 train.py cqc_network \
--noised_quantum_channel=False \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--embedding_type=ZZFeatureMap \
--ansatz_type=RealAmplitudes \
--network_model=2 \
--early_stopping=True

python3 train.py cqc_network \
--noised_quantum_channel=False \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--embedding_type=TPE \
--ansatz_type=RealAmplitudes \
--network_model=1 \
--early_stopping=True

python3 train.py cqc_network \
--noised_quantum_channel=False \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--embedding_type=TPE \
--ansatz_type=RealAmplitudes \
--network_model=2 \
--early_stopping=True

python3 train.py cqc_network \
--noised_quantum_channel=False \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--embedding_type=HEE \
--ansatz_type=RealAmplitudes \
--network_model=1 \
--early_stopping=True

python3 train.py cqc_network \
--noised_quantum_channel=False \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--embedding_type=HEE \
--ansatz_type=RealAmplitudes \
--network_model=2 \
--early_stopping=True
