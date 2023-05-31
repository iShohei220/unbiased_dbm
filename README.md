# End-to-end Training of Deep Boltzmann Machines by Unbiased Contrastive Divergence with Local Mode Initialization
Minimum training code of unbiased Deep Boltzmann Machines for MNIST and Fashion MNIST

# Requirements

- torch
- tensorboard

# Training

### MNIST (binary)
```bash
python train.py --dataset MNIST
```

### Fashion MNIST (binary)
```bash
python train.py --dataset FashionMNIST
```

### MNIST (8-bit)
```bash
python train.py --dataset MNIST --bits 8
```

### Fashion MNIST (8-bit)
```bash
python train.py --dataset FashionMNIST --bits 8
```
