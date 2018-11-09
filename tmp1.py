import torch
import pdb
import numpy as np

rnn1 = torch.nn.LSTM(10, 20, 2, bidirectional=True) # 40, 64

rnn2 = torch.nn.LSTM(20*2, 20, 2, bidirectional=True) # 64*2, 

input = torch.randn(5, 3, 10)

h0 = torch.randn(4, 3, 20)
c0 = torch.randn(4, 3, 20)

output1, (hn, cn) = rnn1(input, (h0, c0))

pdb.set_trace()
output2, (hn, cn) = rnn2(output1, (h0, c0))
mask = torch.from_numpy(np.random.binomial(1,0.1,(4,3,20)))

pdb.set_trace()