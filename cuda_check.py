import torch

cuda = torch.cuda.is_available()

if cuda:
    print("cuda is available")
else:
    print("cuda is not availble")