import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

from train_by_self_play import train_qcnn_p_agent

train_qcnn_p_agent(circuit_name="TPE")