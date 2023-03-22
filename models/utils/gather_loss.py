import time
import torch
import torch.distributed as dist


class GatherLoss(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
        )

class VariedShapeGatherLoss(torch.autograd.Function):
    """An autograd function that performs allgather on varied length tensor."""

    @staticmethod
    def forward(ctx, q, rank, ws): 
        """
        Gathers tensor arrays of different lengths across multiple gpus
        
        Parameters
        ----------
            q : tensor array
            ws : world size
            device : current gpu device
            
        Returns
        -------
            all_q : list of gathered tensor arrays from all the gpus

        """
            
        ctx.rank = rank
        ctx.batch_size = q.shape[0]
        device = q.device
        # q 的 维度是 m * d , 只会在第一维上不同， 所以只比较第一维
        local_size = torch.tensor(q.size(0), device=device)
        
        all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
        dist.all_gather(all_sizes, local_size)
        if max(all_sizes).item() > 1024 or (min(all_sizes).item() < 0.001 and min(all_sizes).item()>0): # batch size maximum
            print('SSL gather wrong, all size :{} on rank {}'.format(all_sizes, rank))
            time.sleep(60)
            raise Exception("SSL gather wrong")
        max_size = max(all_sizes)
        ctx.all_sizes = torch.tensor(all_sizes).cumsum_(dim=0).tolist()
        size_diff = max_size.item() - local_size.item()
        if size_diff:
            # try:
            
            padding = torch.zeros(((size_diff,) + q.size()[1:]), device=device, dtype=q.dtype)
            q = torch.cat((q, padding))
            # except Exception as e:
            #     print("all-gather-padding error info:",repr(e))
            #     print(size_diff)
            #     print(max_size.item())
            #     print(local_size.item())
            #     print(q.size())
            #     print("all-gather-padding error info ended")
        # try:
        all_qs_padded = [torch.zeros_like(q, dtype=q.dtype) for _ in range(ws)]
        # except Exception as e:
        #     print("all-gather-all_qs error info:",repr(e))
        #     print(ws)
        #     print(q.size())
        #     print("all-gather-all_qs error info ended") 
        
        dist.all_gather(all_qs_padded, q)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return torch.cat(all_qs, dim=0)

    #@staticmethod
    #def forward(ctx, q, rank, ws): 
    #    """
    #    Gathers tensor arrays of different lengths across multiple gpus
    #    
    #    Parameters
    #    ----------
    #        q : tensor array
    #        ws : world size
    #        device : current gpu device
    #        
    #    Returns
    #    -------
    #        all_q : list of gathered tensor arrays from all the gpus
#
    #    """
    #    ctx.rank = rank
    #    ctx.batch_size = q.shape[0]
    #    device = q.device
    #    # q 的 维度是 m * d , 只会在第一维上不同， 所以只比较第一维
    #    local_size = torch.tensor(q.size(0), device=device)
    #    if local_size > 1024:
    #        print('Big local_size on rank ', rank)
    #        print('local_size = ', local_size)
    #    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    #    i = 0
    #    while i<100:
    #        i += 1
    #        dist.all_gather(all_sizes, local_size)
    #        max_size = max(all_sizes)
    #        ctx.all_sizes = torch.tensor(all_sizes).cumsum_(dim=0).tolist()
    #        size_diff = max_size.item() - local_size.item()
    #        if size_diff:
    #            # try:
    #            if size_diff > 1024: # batch size maximum
    #                print('Big size diff on rank ', rank)
    #                print('Big size = ', max_size)
    #                continue
    #                raise Exception("Big size diff")
    #            padding = torch.zeros(((size_diff,) + q.size()[1:]), device=device, dtype=q.dtype)
    #            q = torch.cat((q, padding))
    #        break
    #        # except Exception as e:
    #        #     print("all-gather-padding error info:",repr(e))
    #        #     print(size_diff)
    #        #     print(max_size.item())
    #        #     print(local_size.item())
    #        #     print(q.size())
    #        #     print("all-gather-padding error info ended")
    #        # try:
    #    if i==100:
    #        raise Exception("repeat too many time")
    #    all_qs_padded = [torch.zeros_like(q, dtype=q.dtype) for _ in range(ws)]
    #    # except Exception as e:
    #    #     print("all-gather-all_qs error info:",repr(e))
    #    #     print(ws)
    #    #     print(q.size())
    #    #     print("all-gather-all_qs error info ended") 
    #    
    #    dist.all_gather(all_qs_padded, q)
    #    all_qs = []
    #    for q, size in zip(all_qs_padded, all_sizes):
    #        all_qs.append(q[:size])
    #    return torch.cat(all_qs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.all_sizes[ctx.rank - 1] if ctx.rank > 0 else 0
        end = ctx.all_sizes[ctx.rank]
        return (
            grad_output[start : end],
            None,
            None,
        )

    #@staticmethod
        
    #@staticmethod
    #def forward(ctx, q, rank, ws): 
    #    """
    #    Gathers tensor arrays of different lengths across multiple gpus
    #    
    #    Parameters
    #    ----------
    #        q : tensor array
    #        ws : world size
    #        device : current gpu device
    #        
    #    Returns
    #    -------
    #        all_q : list of gathered tensor arrays from all the gpus
#
    #    """
    #    ctx.rank = rank
    #    ctx.batch_size = q.shape[0]
    #    device = q.device
    #    # q 的 维度是 m * d , 只会在第一维上不同， 所以只比较第一维
    #    local_size = torch.tensor(q.size(0), device=device)
    #    if local_size > 1024:
    #        print('Big local_size on rank ', rank)
    #        print('local_size = ', local_size)
    #    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    #    dist.all_gather(all_sizes, local_size)
    #    max_size = max(all_sizes)
    #    ctx.all_sizes = torch.tensor(all_sizes).cumsum_(dim=0).tolist()
    #    size_diff = max_size.item() - local_size.item()
    #    if size_diff:
    #        # try:
    #        if size_diff > 1024: # batch size maximum
    #            print('Big size diff on rank ', rank)
    #            print('Big size = ', max_size)
    #            raise Exception("Big size diff")
    #        padding = torch.zeros(((size_diff,) + q.size()[1:]), device=device, dtype=q.dtype)
    #        q = torch.cat((q, padding))
    #        # except Exception as e:
    #        #     print("all-gather-padding error info:",repr(e))
    #        #     print(size_diff)
    #        #     print(max_size.item())
    #        #     print(local_size.item())
    #        #     print(q.size())
    #        #     print("all-gather-padding error info ended")
    #    # try:
    #    all_qs_padded = [torch.zeros_like(q, dtype=q.dtype) for _ in range(ws)]
    #    # except Exception as e:
    #    #     print("all-gather-all_qs error info:",repr(e))
    #    #     print(ws)
    #    #     print(q.size())
    #    #     print("all-gather-all_qs error info ended") 
    #    
    #    dist.all_gather(all_qs_padded, q)
    #    all_qs = []
    #    for q, size in zip(all_qs_padded, all_sizes):
    #        all_qs.append(q[:size])
    #    return torch.cat(all_qs, dim=0)
    #    
    #def forward(ctx, q, rank, ws): 
    #    """
    #    Gathers tensor arrays of different lengths across multiple gpus
    #    
    #    Parameters
    #    ----------
    #        q : tensor array
    #        ws : world size
    #        device : current gpu device
    #        
    #    Returns
    #    -------
    #        all_q : list of gathered tensor arrays from all the gpus
#
    #    """
    #    ctx.rank = rank
    #    ctx.batch_size = q.shape[0]
    #    device = q.device
    #    # q 的 维度是 m * d , 只会在第一维上不同， 所以只比较第一维
    #    local_size = torch.tensor(q.size(0), device=device)
    #    if local_size > 1024:
    #        print('Big local_size on rank ', rank)
    #        print('local_size = ', local_size)
    #    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    #    while True:
    #        dist.all_gather(all_sizes, local_size)
    #        max_size = max(all_sizes)
    #        ctx.all_sizes = torch.tensor(all_sizes).cumsum_(dim=0).tolist()
    #        size_diff = max_size.item() - local_size.item()
    #        if size_diff and size_diff > 1024:
#
    #            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
    #            print('Big size diff on rank ', rank)
    #            print('Big size = ', max_size)
    #            time.sleep(4)
    #        else:
    #            break
    #    if size_diff:
    #        # try:
    #        #if size_diff > 1024: # batch size maximum
    #        #    print('Big size diff on rank ', rank)
    #        #    print('Big size = ', max_size)
    #        #    raise Exception("Big size diff")
    #        padding = torch.zeros(((size_diff,) + q.size()[1:]), device=device, dtype=q.dtype)
    #        q = torch.cat((q, padding))
    #        # except Exception as e:
    #        #     print("all-gather-padding error info:",repr(e))
    #        #     print(size_diff)
    #        #     print(max_size.item())
    #        #     print(local_size.item())
    #        #     print(q.size())
    #        #     print("all-gather-padding error info ended")
    #    # try:
    #    all_qs_padded = [torch.zeros_like(q, dtype=q.dtype) for _ in range(ws)]
    #    # except Exception as e:
    #    #     print("all-gather-all_qs error info:",repr(e))
    #    #     print(ws)
    #    #     print(q.size())
    #    #     print("all-gather-all_qs error info ended") 
    #    
    #    dist.all_gather(all_qs_padded, q)
    #    all_qs = []
    #    for q, size in zip(all_qs_padded, all_sizes):
    #        all_qs.append(q[:size])
    #    return torch.cat(all_qs, dim=0)