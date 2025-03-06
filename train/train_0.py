import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from train_by_self_play import train_cqc_agent_qcnn, train_cqc_agent_sampler_former, train_cqc_agent_sampler_latter, train_cqc_agent_estimator_former, train_cqc_agent_estimator_latter, train_qcnn_agent

train_cqc_agent_qcnn(10)

