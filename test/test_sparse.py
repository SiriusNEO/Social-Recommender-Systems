import torch
import time

print(torch.__version__)        # 返回pytorch的版本
print(torch.version.cuda)

device = "cuda"
print(device)

big_sparse = torch.sparse.FloatTensor(10000000, 10000000)
print(big_sparse)

index = torch.LongTensor([[i*100, j*100] for j in range(1, 1001) for i in range(1, 1001)])

index = index.t()

value = torch.FloatTensor([v for v in range(1000*1000)])

print(index.size())

adj = torch.sparse.FloatTensor(index, value).to(device)
print(adj.size())
print(adj + adj)
print(adj * adj)

multi = torch.FloatTensor([[j for j in range(1, 6)] for i in range(100001)]).to(device)

print(multi.size())

print(adj.matmul(multi).size())


