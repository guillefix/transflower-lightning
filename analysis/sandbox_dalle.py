import torch
from models.cdvae import ConditionalDiscreteVAE

vae = ConditionalDiscreteVAE(
    input_shape = (7,7),
    num_layers = 3,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_tokens = 8192,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
    codebook_dim = 512,       # codebook dimension
    cond_dim = 100,
    hidden_dim = 64,          # hidden dimension
    num_resnet_blocks = 1,    # number of resnet blocks
    temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
    straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
)

images = torch.randn(4, 3, *vae.input_shape)
cond = torch.randn(4, 100, *vae.codebook_layer_shape)

logits = vae(images, cond=cond, return_logits = True)

logits.shape

import numpy as np

torch.randint(0,10,(1,))
image_seq = torch.randint(0,8192, (4,np.prod(vae.codebook_layer_shape)))
image = vae.decode(image_seq, cond=cond)

image.shape

# loss = vae(images, return_loss = True)
# loss.backward()
# loss
# train with a lot of data to learn a good codebook
