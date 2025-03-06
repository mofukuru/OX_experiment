import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from evaluate_by_self_play import evaluate_cqc_agent_qcnn, evaluate_cqc_agent_sampler_former, evaluate_cqc_agent_sampler_latter, evaluate_cqc_agent_estimator_former, evaluate_cqc_agent_estimator_latter, evaluate_qcnn_agent

evaluate_cqc_agent_qcnn(18)
