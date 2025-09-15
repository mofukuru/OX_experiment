# with noisy quantum channel
# python3 train.py cqc_network \
# --nn_network=3 \
# --total_episodes=5000 \
# --n_qubits=8 \
# --distance=100 \
# --embedding_type=ZFeatureMap \
# --ansatz_type=RealAmplitudes \
# --network_model=1 \
# --early_stopping=True

python3 train.py cqc_network \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--distance=100 \
--embedding_type=ZFeatureMap \
--ansatz_type=RealAmplitudes \
--network_model=2 \
--early_stopping=True

python3 train.py cqc_network \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--distance=100 \
--embedding_type=ZZFeatureMap \
--ansatz_type=RealAmplitudes \
--network_model=1 \
--early_stopping=True

python3 train.py cqc_network \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--distance=100 \
--embedding_type=ZZFeatureMap \
--ansatz_type=RealAmplitudes \
--network_model=2 \
--early_stopping=True

python3 train.py cqc_network \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--distance=100 \
--embedding_type=TPE \
--ansatz_type=RealAmplitudes \
--network_model=1 \
--early_stopping=True

python3 train.py cqc_network \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--distance=100 \
--embedding_type=TPE \
--ansatz_type=RealAmplitudes \
--network_model=2 \
--early_stopping=True

python3 train.py cqc_network \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--distance=100 \
--embedding_type=HEE \
--ansatz_type=RealAmplitudes \
--network_model=1 \
--early_stopping=True

python3 train.py cqc_network \
--nn_network=3 \
--total_episodes=5000 \
--n_qubits=8 \
--distance=100 \
--embedding_type=HEE \
--ansatz_type=RealAmplitudes \
--network_model=2 \
--early_stopping=True
