from model import Generator, Discriminator
import torch
import random 
import torch.autograd as autograd
device='cuda'
generator = Generator(
        256, 512, 8, channel_multiplier=2
    ).to(device)
# def make_noise(batch, latent_dim, n_noise, device):
#   if n_noise == 1:
#       return torch.randn(batch, latent_dim, device=device)

#   noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

#   return noises


# def mixing_noise(batch, latent_dim, prob, device):
#     if prob > 0 and random.random() < prob:
#         return make_noise(batch, latent_dim, 2, device)

#     else:
#         return [make_noise(batch, latent_dim, 1, device)]

# def estimate_fisher(sample_size=1,ckpt_path=None,batch_size=32):
#   discriminator = Discriminator(
#   size=256, channel_multiplier=2
#   ).to(device)
#   if(ckpt_path):

#     ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
#     discriminator.load_state_dict(ckpt["d"])

  
#   # noise=torch.randn(sample_size,512,device='cuda')
#   noise = mixing_noise(16, 512, 0.9, 'cuda')
  


#   fake_img, _= generator(noise)
#   loglikelihoods=discriminator(fake_img)
#   reqd_param=[]
#   for itr in generator.convs:
#     reqd_param.append(itr.conv.weight)
#   loglikelihood_grads=[]
#   for l in loglikelihoods:
#     loglikelihood_grads.append((autograd.grad(l,reqd_param,retain_graph=True),l))

#   # loglikelihood_grads = zip(*[autograd.grad(
#   #     l, self.parameters(),
#   #     retain_graph=(i < len(loglikelihoods))
#   # ) for i, l in enumerate(loglikelihoods, 1)])

#   print(loglikelihood_grads)

      
generator.estimate_fisher(sample_size=64)

# print(generator.convs)

