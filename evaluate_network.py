def evaluate_noised(type: int, n_qubit: int, train_noise: bool):
    network_list = ["zr", "zzr", "tr", "hr", "ze", "zze", "te", "he"]
    for iter in range(len(network_list)):
        from src.agent_re import RandomPolicy, CQCAgent_network
        from env_for_network import Env
        import torch

        rate_per_NN = []
        randagent = RandomPolicy()

        elorate = 1500
        load_path = f"models/trainwithnetwork/noised/train_cqc_agent_cqcnn_sampler_{network_list[iter]}_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth" if train_noise \
                    else f"models/trainwithnetwork/noiseless/train_cqc_agent_cqcnn_sampler_{network_list[iter]}_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent_network(type=type, network=f"CQCNN_sampler_{network_list[iter].upper()}_network", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
        with open(f"result_re/trainwithnetwork/noised/train_cqc_agent_cqcnn_sampler_{network_list[iter]}_type{type}_times{iter*10+10}_n_qubits{n_qubit}_train_noise_is_{train_noise}_evaluate_noise_is_True.txt", "a") as f:
            print(rate_per_NN, file=f)
            f.write("\n")

def evaluate_noiseless(type: int, n_qubit: int, train_noise: bool):
    network_list = ["zr", "zzr", "tr", "hr", "ze", "zze", "te", "he"]
    for iter in range(len(network_list)):
        from src.agent_re import RandomPolicy, CQCAgent_network
        from src.env import Env
        import torch

        rate_per_NN = []
        randagent = RandomPolicy()

        elorate = 1500
        load_path = f"models/trainwithnetwork/noised/train_cqc_agent_cqcnn_sampler_{network_list[iter]}_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth" if train_noise \
                    else f"models/trainwithnetwork/noiseless/train_cqc_agent_cqcnn_sampler_{network_list[iter]}_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent_network(type=type, network=f"CQCNN_sampler_{network_list[iter].upper()}_network", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
        with open(f"result_re/trainwithnetwork/noiseless/train_cqc_agent_cqcnn_sampler_{network_list[iter]}_type{type}_times{iter*10+10}_n_qubits{n_qubit}_train_noise_is_{train_noise}_evaluate_noise_is_False.txt", "a") as f:
            print(rate_per_NN, file=f)
            f.write("\n")

