#%%
def train_cqc_agent_qcnn(n_qubit: int):
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch

    cqc_agent = CQCAgent(network="CQCCNN_Z", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_qcnn_z_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCCNN_ZZ", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_qcnn_zz_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCCNN_T", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_qcnn_t_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCCNN_H", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_qcnn_h_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_cqc_agent_sampler_former(n_qubit: int):
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_sampler_ZR", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_zr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_sampler_ZZR", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_zzr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_sampler_TR", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_tr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_sampler_HR", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_hr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_cqc_agent_sampler_latter(n_qubit: int):
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_sampler_ZE", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_ze_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_sampler_ZZE", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_zze_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_sampler_TE", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_te_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_sampler_HE", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_sampler_he_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_cqc_agent_estimator_former(n_qubit: int):
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_estimator_ZR", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_estimator_ZZR", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zzr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_estimator_TR", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_tr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_estimator_HR", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_hr_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_cqc_agent_estimator_latter(n_qubit):
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_estimator_ZE", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_ze_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_estimator_ZZE", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_zze_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_estimator_TE", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_te_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    cqc_agent = CQCAgent(network="CQCNN_estimator_HE", num_of_qubit=n_qubit)
    print(cqc_agent.HNN)
    
    for iter in range(25):
        cqc_agent.train()
        environment = Environment(cqc_agent, cqc_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_cqc_agent_cqcnn_estimator_he_times{iter*10+10}_n_qubits{n_qubit}.pth"
        torch.save(cqc_agent.HNN.state_dict(), model_path)

#%%
def train_qcnn_agent():
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    qcnn_agent = QCNNAgent(network="QCNN_Z")
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_qcnn_agent_z_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)
        
    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    qcnn_agent = QCNNAgent(network="QCNN_ZZ")
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_qcnn_agent_zz_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    qcnn_agent = QCNNAgent(network="QCNN_T")
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_qcnn_agent_t_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)

    from src.agent import CNNAgent, CQCAgent, QCNNAgent
    from src.env import Environment
    import torch
        
    qcnn_agent = QCNNAgent(network="QCNN_H")
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"./models/nonrandomtrain/train_qcnn_agent_h_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)
        
def train_qcnn_p_agent(circuit_name=None):
    from src.agent_re import QCNNAgent_weight_change
    from src.env import Environment
    import torch

    qcnn_agent = QCNNAgent_weight_change(network=circuit_name)
    print(qcnn_agent.QCNN)
    
    for iter in range(25):
        qcnn_agent.train()
        environment = Environment(qcnn_agent, qcnn_agent)
        environment.train(10)
        
        model_path = f"models/train/train_qcnn_p_{circuit_name}_times{iter*10+10}.pth"
        torch.save(qcnn_agent.QCNN.state_dict(), model_path)
        
def train_qnn_p_agent(num_of_qubit=9, circuit_name=None):
    from src.agent_re import QAgent
    from src.env import Environment
    import torch

    qnn_agent = QAgent(network=circuit_name)
    print(qnn_agent.QNN)
    
    for iter in range(25):
        qnn_agent.train()
        environment = Environment(qnn_agent, qnn_agent)
        environment.train(10)
        
        model_path = f"models/train/train_qnn_p_{circuit_name}_times{iter*10+10}_n_qubits_{num_of_qubit}.pth"
