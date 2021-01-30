import torch

class SquashTransform:
    def __call__(self, inputs):
        return 2 * inputs - 1

def encodeOneHot(labels):
    ret = torch.FloatTensor(labels.shape[0], num_classes)
    ret.zero_()
    ret.scatter_(dim=1, index=labels.view(-1, 1), value=1)
    return ret

def generate_latent_points(latent_dim, n_samples, device):
    return torch.randn(n_samples, latent_dim, 1, 1, device=device)