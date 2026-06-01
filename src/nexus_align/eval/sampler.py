"""MeanFlow sampler: single- and multi-step generation from mean velocity."""

import torch


@torch.no_grad()
def meanflow_sampler(model, latents, y=None, cfg_scale=1.0, num_steps=1, **kwargs):
    """Generate samples via z_r = z_t - (t - r) u(z_t, r, t).

    Single-step uses (r=0, t=1); multi-step walks t from 1 to 0 over num_steps.
    CFG is applied when y is given and cfg_scale > 1.0.
    """
    batch_size = latents.shape[0]
    device = latents.device

    do_cfg = y is not None and cfg_scale > 1.0
    if do_cfg:
        num_classes = model.module.num_classes if hasattr(model, "module") else model.num_classes
        null_y = torch.full_like(y, num_classes)

    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        if do_cfg:
            z_in = torch.cat([latents, latents], dim=0)
            u = model(z_in, torch.cat([r, r]), torch.cat([t, t]), y=torch.cat([y, null_y]))
            u_cond, u_uncond = u.chunk(2, dim=0)
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = model(latents, r, t, y=y)
        return latents - u

    z = latents
    time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_cur, t_next = time_steps[i], time_steps[i + 1]
        t = torch.full((batch_size,), t_cur, device=device)
        r = torch.full((batch_size,), t_next, device=device)
        if do_cfg:
            z_in = torch.cat([z, z], dim=0)
            u = model(z_in, torch.cat([r, r]), torch.cat([t, t]), y=torch.cat([y, null_y]))
            u_cond, u_uncond = u.chunk(2, dim=0)
            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = model(z, r, t, y=y)
        z = z - (t_cur - t_next) * u
    return z
