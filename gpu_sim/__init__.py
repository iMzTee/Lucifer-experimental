"""gpu_sim — GPU-accelerated Rocket League physics simulator.

All physics, rewards, observations, and collection run as batched PyTorch
tensor operations on GPU. Zero CPU↔GPU transfers in the hot loop.
"""
