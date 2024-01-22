import torch
from torch import Tensor
import nerfacc
from data_loaders.tiny_data_loader import DataLoader
from models.full_NeRF_model import NerfModel
import torch.nn.functional as F
from setup_utils import set_random_seeds, get_tensor_device

num_positional_encoding_functions = 9
num_directional_encoding_functions = 9

set_random_seeds()
device = get_tensor_device()
data_manager = DataLoader(device)

radiance_field = NerfModel(num_positional_encoding_functions, num_directional_encoding_functions).to(device)  # network: a NeRF model
rays_o: Tensor = ...  # ray origins. (n_rays, 3)
rays_d: Tensor = ...  # ray normalized directions. (n_rays, 3)
optimizer = torch.optim.Adam(radiance_field.parameters(), lr=lr)  # optimizer

estimator = nerfacc.OccGridEstimator(...)


def sigma_fn(t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tensor:
    """ Define how to query density for the estimator."""
    t_origins = rays_o[ray_indices]  # (n_samples, 3)
    t_dirs = rays_d[ray_indices]  # (n_samples, 3)
    positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
    sigmas = radiance_field.query_density(positions)
    return sigmas  # (n_samples,)


def rgb_sigma_fn(t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tuple[Tensor, Tensor]:
    """ Query rgb and density values from a user-defined radiance field. """
    t_origins = rays_o[ray_indices]  # (n_samples, 3)
    t_dirs = rays_d[ray_indices]  # (n_samples, 3)
    positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
    rgbs, sigmas = radiance_field(positions, condition=t_dirs)
    return rgbs, sigmas  # (n_samples, 3), (n_samples,)


# Efficient Raymarching:
# ray_indices: (n_samples,). t_starts: (n_samples,). t_ends: (n_samples,).
ray_indices, t_starts, t_ends = estimator.sampling(
    rays_o, rays_d, sigma_fn=sigma_fn, near_plane=0.2, far_plane=1.0,
    early_stop_eps=1e-4, alpha_thre=1e-2,
)

# Differentiable Volumetric Rendering.
# colors: (n_rays, 3). opacity: (n_rays, 1). depth: (n_rays, 1).
color, opacity, depth, extras = nerfacc.rendering(
    t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
)

# Optimize: Both the network and rays will receive gradients
optimizer.zero_grad()
loss = F.mse_loss(color, color_gt)
loss.backward()
optimizer.step()
