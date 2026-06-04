"""
MeanFlow algorithm: mean-velocity flow-matching loss with JVP bootstrap.

References:
    - MeanFlow:
        - Paper: Mean Flows for One-step Generative Modeling
        - Unofficial code: https://github.com/zhuyu-cs/MeanFlow
"""

import torch
import numpy as np
import torch.autograd.forward_ad as fwAD


def jvp(fn, primals, tangents):
    """
    Jacobian-vector product via forward-mode AD.

    torch.func.jvp rejects FSDP's in-place unshard ops, so use the eager
    forward_ad API instead. Runs under no_grad: the JVP output is only used
    as a detached target, and an unused autograd graph through an FSDP
    module corrupts its backward-hook state.
    """
    with torch.no_grad(), fwAD.dual_level():
        duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
        out, dudt = fwAD.unpack_dual(fn(*duals))
    return out, dudt


class SILoss:
    """MeanFlow training loss.

    Samples a time pair (r, t), builds the interpolant z_t, and regresses the
    model's mean velocity u onto a JVP-bootstrapped target; supports CFG mixing
    and adaptive loss weighting.
    """

    def __init__(self, cfg):
        loss_cfg = cfg.algorithm.loss

        self.path_type = loss_cfg.get("path_type", "linear")  # interpolant: "linear" or "cosine"
        self.weighting = loss_cfg.get("weighting", "adaptive")  # loss weighting: "uniform" or "adaptive"

        # Time sampling config
        self.time_sampler = loss_cfg.get("time_sampler", "logit_normal")  # "uniform" or "logit_normal"
        self.time_mu = loss_cfg.get("time_mu", -0.4)  # logit_normal mean
        self.time_sigma = loss_cfg.get("time_sigma", 1.0)  # logit_normal std
        self.ratio_r_not_equal_t = loss_cfg.get("ratio_r_not_equal_t", 0.25)  # ratio of samples where r≠t
        self.label_dropout_prob = cfg.model.get("cfg_prob", 0.1)  # classifier-free guidance
        # Adaptive weight config
        self.adaptive_p = loss_cfg.get("adaptive_p", 1.0)  # power for adaptive weighting

        # CFG config
        self.cfg_omega = loss_cfg.get("cfg_omega", 0.2)  # CFG omega, 1.0 means no CFG
        self.cfg_kappa = loss_cfg.get("cfg_kappa", 0.92)  # CFG kappa for mixing class-cond and uncond u
        self.cfg_min_t = loss_cfg.get("cfg_min_t", 0.0)  # minimum CFG trigger time
        self.cfg_max_t = loss_cfg.get("cfg_max_t", 0.8)  # maximum CFG trigger time


    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]

        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)

        return r, t

    def __call__(self, model, images, model_kwargs=None):
        """
        Compute MeanFlow loss function with bootstrap mechanism
        """
        if model_kwargs == None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        batch_size = images.shape[0]
        device = images.device

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()
            batch_size = y.shape[0]
            num_classes = model.module.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob

            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask  # Used for unconditional velocity computation

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device)

        noises = torch.randn_like(images)

        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises #(1-t) * images + t * noise

        # Calculate instantaneous velocity v_t
        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)

        u = model(z_t, r, t, **model_kwargs)

        # JVP tangent: CFG-mixed velocity for CFG samples, plain v_t otherwise.
        # NOTE: keep the number of model forwards identical on all ranks; FSDP
        # unshard collectives deadlock on data-dependent branching.
        v_hat = v_t
        if model_kwargs.get('y') is not None:
            # Apply CFG within the time window, excluding unconditional samples
            cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)
            num_classes = model.module.num_classes

            # Compute instantaneous cond/uncond velocities u(z_t, t, t) in one batch
            combined_kwargs = {}
            for k, v in model_kwargs.items():
                if torch.is_tensor(v) and v.shape[0] == batch_size:
                    combined_kwargs[k] = torch.cat([v, v], dim=0)
                else:
                    combined_kwargs[k] = v
            y = model_kwargs['y']
            combined_kwargs['y'] = torch.cat([y, torch.full_like(y, num_classes)], dim=0)

            with torch.no_grad():
                combined_u_at_t = model(
                    torch.cat([z_t, z_t], dim=0),
                    torch.cat([t, t], dim=0),
                    torch.cat([t, t], dim=0),
                    **combined_kwargs,
                )
            u_cond_at_t, u_uncond_at_t = torch.chunk(combined_u_at_t, 2, dim=0)
            v_tilde = (self.cfg_omega * v_t +
                    self.cfg_kappa * u_cond_at_t +
                    (1 - self.cfg_omega - self.cfg_kappa) * u_uncond_at_t)
            v_hat = torch.where(cfg_time_mask.view(-1, 1, 1, 1), v_tilde, v_t)

        def fn_current(z, cur_r, cur_t):
            return model(z, cur_r, cur_t, **model_kwargs)

        primals = (z_t, r, t)
        tangents = (v_hat, torch.zeros_like(r), torch.ones_like(t))
        _, dudt = jvp(fn_current, primals, tangents)

        u_target = v_hat - time_diff * dudt

        # Detach the target to prevent gradient flow
        error = u - u_target.detach()
        loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
        # Apply adaptive weighting based on configuration
        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid
        else:
            loss = loss_mid
        loss_mean_ref = torch.mean((error**2))
        return loss, loss_mean_ref
