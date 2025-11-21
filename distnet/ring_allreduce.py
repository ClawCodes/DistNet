import os
from torch import nn
from torch import Tensor
from torch import distributed as dist
import torch
from torch.multiprocessing import Process


def ring_reduce(tensor: torch.Tensor):
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    #Split tensor into world_size chunks
    chunks=list(torch.chunk(tensor,world_size))
    tensor=torch.cat(chunks)
    print('Process {} has tensor {}'.format(rank, tensor))

    send_idx=rank%world_size
    recv_idx=((rank-1)+world_size)%world_size
    
    #Empty buffer for receiving tensors
    recv_buff = torch.zeros_like(chunks[recv_idx])

    #Reduce-scatter
    for i in range(world_size):
        #send to next neighbor
        dist.send(chunks[send_idx], (rank+1)%world_size)
        #receive from prev neighbor
        dist.recv(recv_buff,recv_idx)
        #update local tensor with received value
        chunks[recv_idx] += recv_buff[:]
        tensor=torch.cat(chunks)
        #decrement send/receive index + wrap around if needed
        send_idx=((send_idx-1)+world_size)%world_size
        recv_idx=((recv_idx-1)+world_size)%world_size
    print('Reduce-scatter done, process {} has tensor {}'.format(rank, tensor))
    #update indices, might not need
    send_idx = (recv_idx+1)%world_size
    recv_idx = ((send_idx - 1)+world_size)%world_size

    #All Gather
    for i in range(world_size):
        #send to next neighbor
        dist.send(chunks[send_idx], (rank+1)%world_size)
        #receive from prev neighbor
        dist.recv(recv_buff,recv_idx)
        #update local tensor with received value
        chunks[recv_idx] += recv_buff[:]
        tensor=torch.cat(chunks)
        #decrement send/receive index + wrap around if needed
        send_idx=((send_idx-1)+world_size)%world_size
        recv_idx=((recv_idx-1)+world_size)%world_size
    print('Gathered, process {} has tensor {}'.format(rank, tensor))