from src.adapter import Adapter

# if (torch.gpu.is_available()):
#   device = torch.device('cuda')
# else: 
device = torch.device('cpu')

# N is batch size, K_j is input size
# K_j is hidden layer size, n_i is output projection
N, k_j, n_i = 64, 1000, 100


