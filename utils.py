import random
import string
import torch


def int2bit(x, bits=8):
    mask = 2**torch.arange(bits, device=x.device)
    out = x.unsqueeze(-1).bitwise_and(mask).ne(0)

    return out

def bit2int(x, bits=8):
    mask = 2 ** torch.arange(bits, device=x.device)
    out = torch.sum(mask * x, -1)

    return out

def float2bit(x, bits=8):
    out = x.mul(2**bits - 1).int()
    out = int2bit(out, bits)

    return out

def bit2float(x, bits=8):
    out = bit2int(x, bits).float()
    out /= 2**bits - 1

    return out

def binarize(x):
    return (x > 0.5).float()

def generate_id(n):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))
