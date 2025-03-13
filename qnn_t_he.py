import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

from train_by_self_play import train_qnn_p_agent

train_qnn_p_agent(circuit_name="HE")
