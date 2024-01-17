
import torch.nn



l  = torch.nn.Linear(2, 2)



print(l(torch.FloatTensor([1, 2])))