"""Federated-learning pipeline (Step 7).

Implemented: datasets (MNIST/Fashion-MNIST via torchvision, config-controlled
download) + partitioning (IID + Dirichlet non-IID).
Next: models, client/server, aggregation (FedAvg + AirComp distortion from
Step 6), and the federated training loop that integrates SCOUT-FL selection.
DeepSense 6G / WiMANS come only after A1-Full works on synthetic + MNIST/Fashion.
"""
