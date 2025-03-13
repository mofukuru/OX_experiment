#%%
def train_cqc_agent_qcnn(type: int, n_qubit: int):
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch

    cqc_agent = CQCAgent_network(type=type, network="CQCCNN_Z_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"models/trainwithnetwork/train_cqc_agent_qcnn_z_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCCNN_ZZ_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"models/trainwithnetwork/train_cqc_agent_qcnn_zz_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCCNN_T_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_qcnn_t_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCCNN_H_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_qcnn_h_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_cqc_agent_sampler_former(type: int, n_qubit: int):
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_sampler_ZR_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_sampler_zr_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_sampler_ZZR_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_sampler_zzr_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_sampler_TR_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_sampler_tr_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_sampler_HR_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_sampler_hr_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_cqc_agent_sampler_latter(type: int, n_qubit: int):
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_sampler_ZE_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_sampler_ze_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_sampler_ZZE_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_sampler_zze_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_sampler_TE_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_sampler_te_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_sampler_HE_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_sampler_he_type{type}_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_cqc_agent_estimator_former(type: int, n_qubit: int):
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_estimator_ZR_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_estimator_zr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_estimator_ZZR_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_estimator_zzr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_estimator_TR_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_estimator_tr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_estimator_HR_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_estimator_hr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_cqc_agent_estimator_latter(type: int, n_qubit: int):
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_estimator_ZE_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_estimator_ze_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_estimator_ZZE_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_estimator_zze_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_estimator_TE_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_estimator_te_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent_network(type=type, network="CQCNN_estimator_HE_network", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_cqc_agent_cqcnn_estimator_he_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_qcnn_agent(type: int):
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    qcnn_agent = QCNNAgent_network(type=type, network="QCNN_Z_network")
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"models/trainwithnetwork/train_qcnn_agent_z_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)
        
    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    qcnn_agent = QCNNAgent_network(type=type, network="QCNN_ZZ_network")
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"models/trainwithnetwork/train_qcnn_agent_zz_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    qcnn_agent = QCNNAgent_network(type=type, network="QCNN_T_network")
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_qcnn_agent_t_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)

    from src.agent_re import CQCAgent_network, QCNNAgent_network
    from src.env import Environment
    import torch
        
    qcnn_agent = QCNNAgent_network(type=type, network="QCNN_H_network")
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"./models/trainwithnetwork/train_qcnn_agent_h_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)
        