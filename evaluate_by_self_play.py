#%%
def evaluate_cqc_agent_qcnn(n_qubit: int):
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_qcnn_z_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCCNN_Z", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_qcnn_z_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_qcnn_zz_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCCNN_ZZ", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_qcnn_zz_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_qcnn_t_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCCNN_T", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_qcnn_t_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_qcnn_h_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCCNN_H", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_qcnn_h_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

#%%
def evaluate_cqc_agent_sampler_former(n_qubit: int):
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_qcnn_h_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_sampler_ZR", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_qcnn_h_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_zzr_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_sampler_ZZR", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_sampler_zzr_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_tr_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_sampler_TR", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_sampler_tr_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_hr_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_sampler_HR", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_sampler_hr_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

#%%
def evaluate_cqc_agent_sampler_latter(n_qubit: int):
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_ze_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_sampler_ZE", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_sampler_ze_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_zze_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_sampler_ZZE", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_sampler_zze_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_te_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_sampler_TE", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_sampler_te_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_he_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_sampler_HE", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_sampler_he_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

#%%
def evaluate_cqc_agent_estimator_former(n_qubit: int):
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zr_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_estimator_ZR", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zr_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zzr_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_estimator_ZZR", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zzr_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_tr_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_estimator_TR", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_estimator_tr_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_hr_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_estimator_HR", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_estimator_hr_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

#%%
def evaluate_cqc_agent_estimator_latter(n_qubit: int):
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_ze_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_estimator_ZE", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_estimator_ze_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zze_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_estimator_ZZE", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zze_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_te_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_estimator_TE", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_estimator_te_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_he_times{iter*10+10}_n_qubits{n_qubit}.pth'
        load_weights = torch.load(load_path)
        cqc_agent = CQCAgent(network="CQCNN_estimator_HE", num_of_qubit=n_qubit)
        cqc_agent.HNN.load_state_dict(load_weights)

        cqc_agent.eval()

        environment = Env(cqc_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_cqc_agent_cqcnn_estimator_he_times{iter*10+10}_n_qubits{n_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

#%%
def evaluate_qcnn_agent():
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_qcnn_agent_z_times{iter*10+10}.pth'
        load_weights = torch.load(load_path)
        qcnn_agent = CQCAgent(network="QCNN_Z")
        qcnn_agent.QCNN.load_state_dict(load_weights)

        qcnn_agent.eval()

        environment = Env(qcnn_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_qcnn_agent_z_times{iter*10+10}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_qcnn_agent_zz_times{iter*10+10}.pth'
        load_weights = torch.load(load_path)
        qcnn_agent = CQCAgent(network="QCNN_ZZ")
        qcnn_agent.QCNN.load_state_dict(load_weights)

        qcnn_agent.eval()

        environment = Env(qcnn_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_qcnn_agent_zz_times{iter*10+10}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_qcnn_agent_t_times{iter*10+10}.pth'
        load_weights = torch.load(load_path)
        qcnn_agent = CQCAgent(network="QCNN_T")
        qcnn_agent.QCNN.load_state_dict(load_weights)

        qcnn_agent.eval()

        environment = Env(qcnn_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_qcnn_agent_t_times{iter*10+10}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
    from src.agent import RandomPolicy, CNNAgent, CQCAgent, QCNNAgent
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        load_path = f'./models/nonrandomtrain/train_qcnn_agent_h_times{iter*10+10}.pth'
        load_weights = torch.load(load_path)
        qcnn_agent = CQCAgent(network="QCNN_H")
        qcnn_agent.QCNN.load_state_dict(load_weights)

        qcnn_agent.eval()

        environment = Env(qcnn_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"./result/nonrandomtrain/train_qcnn_agent_h_times{iter*10+10}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
def evaluate_qcnn_p_agent(circuit_name=None):
    from src.agent_re import QCNNAgent_weight_change, RandomPolicy
    from src.env import Env
    import torch
    
    rate_per_NN = []
    randagent = RandomPolicy()
    
    for iter in range(24, 25):
        elorate = 1500
        load_path = f"models/train/train_qcnn_p_{circuit_name}_times{iter*10+10}.pth"
        load_weights = torch.load(load_path)
        qcnn_agent = QCNNAgent_weight_change(network=circuit_name)
        qcnn_agent.QCNN.load_state_dict(load_weights)
        
        qcnn_agent.eval()
        
        environment = Env(qcnn_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"result/eval/train_qcnn_p_{circuit_name}_times{iter*10+10}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
        
def evaluate_qnn_p_agent(num_of_qubit=9, circuit_name=None):
    from src.agent_re import QAgent, RandomPolicy
    from src.env import Env
    import torch

    rate_per_NN = []
    randagent = RandomPolicy()
    
    for iter in range(24, 25):
        elorate = 1500
        load_path = f"models/train/train_qnn_p_{circuit_name}_times{iter*10+10}_n_qubits_{num_of_qubit}.pth"
        load_weights = torch.load(load_path)
        qnn_agent = QAgent(network=circuit_name)
        qnn_agent.QNN.load_state_dict(load_weights)
        
        qnn_agent.eval()
        
        environment = Env(qnn_agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
    with open(f"result/train/train_qnn_p_{circuit_name}_times{iter*10+10}_n_qubits_{num_of_qubit}.txt", "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
    