import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lightning import LightningModule
from contextlib import contextmanager
from functools import partial
from typing import List, Optional

from .components.ldm.denoiser import LitEma


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2)
    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


class LatentDiffusion(LightningModule):
    def __init__(self,
        denoiser,
        autoencoder,
        context_encoder=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        use_ema=True,
        lr=1e-4,
        lr_warmup=0,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",  # assuming fixed variance schedules
        ae_load_state_file: str = None,
        trainable_parts: Optional[List[str]] = None,
        pde_lambda: float = 0.0,       # For wind PDE loss (uv mode)
        pde_mode: Optional[str] = None,  # "uv" or "temp"
        temp_pde_coef: float = 0.0,     # Coefficient for the temperature PDE loss term
        temp_energy_coef: float = 0.0   # Coefficient for the energy consistency loss term
    ):
        super().__init__()
        self.loss_type = loss_type
        self.pde_lambda = pde_lambda
        self.pde_mode = pde_mode
        self.temp_pde_coef = temp_pde_coef
        self.temp_energy_coef = temp_energy_coef

        self.denoiser = denoiser
        self.autoencoder = autoencoder.requires_grad_(False)
        if ae_load_state_file is not None:
            self.autoencoder.load_state_dict(torch.load(ae_load_state_file)["state_dict"])
        self.conditional = (context_encoder is not None)
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup

        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps", "x0" and "v"'
        self.parameterization = parameterization

        self.use_ema = use_ema
        if self.use_ema:
            self.denoiser_ema = LitEma(self.denoiser)

        self.register_schedule(
            beta_schedule=beta_schedule, timesteps=timesteps,
            linear_start=linear_start, linear_end=linear_end,
            cosine_s=cosine_s
        )

        if trainable_parts is not None and len(trainable_parts) > 0:
            self.set_trainable_layers(trainable_parts)
            if self.use_ema:
                self.denoiser_ema = LitEma(self.denoiser)

    def set_trainable_layers(self, trainable_parts: List[str]):
        for name, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if any(tp in name for tp in trainable_parts):
                param.requires_grad = True
                print(f"Unfreezing parameter: {name}")

    def mass_conservation_loss(self, wind_field: torch.Tensor) -> torch.Tensor:
        wind_field = torch.clamp(wind_field, min=-10.0, max=10.0)
        u = wind_field[:, 0:1, :, :]
        v = wind_field[:, 1:2, :, :]
        du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
        dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / 2.0
        du_dx = du_dx[:, :, 1:-1, :]
        dv_dy = dv_dy[:, :, :, 1:-1]
        divergence = du_dx + dv_dy
        return torch.mean(divergence ** 2)

    def temperature_pde_loss(self, T_f: torch.Tensor, T_c: torch.Tensor) -> torch.Tensor:
        """
        Computes the PDE loss for 2m temperature.
        T_f: full-resolution temperature predicted by decoding (B,1,H_f,W_f)
        T_c: coarse temperature field (B,1,H_c,W_c)
        We upsample T_c to the size of T_f and then compute central differences.
        """
        # Upsample coarse field to fine resolution.
        T_c_up = F.interpolate(T_c, size=T_f.shape[-2:], mode='bilinear', align_corners=False)
        # Compute Laplacian for T_c_up (for effective parameter fitting)
        d2T_c_dx2 = T_c_up[:, :, 1:-1, 2:] - 2 * T_c_up[:, :, 1:-1, 1:-1] + T_c_up[:, :, 1:-1, :-2]
        d2T_c_dy2 = T_c_up[:, :, 2:, 1:-1] - 2 * T_c_up[:, :, 1:-1, 1:-1] + T_c_up[:, :, :-2, 1:-1]
        laplacian_T_c = d2T_c_dx2 + d2T_c_dy2

        dT_c_dx = (T_c_up[:, :, 1:-1, 2:] - T_c_up[:, :, 1:-1, :-2]) / 2.0
        dT_c_dy = (T_c_up[:, :, 2:, 1:-1] - T_c_up[:, :, :-2, 1:-1]) / 2.0
        grad_T_c = torch.sqrt(dT_c_dx**2 + dT_c_dy**2 + 1e-8)

        # Fit effective ratio (kappa/alpha) from coarse field.
        kappa = torch.mean(torch.abs(laplacian_T_c), dim=[1,2,3], keepdim=True)
        alpha = torch.mean(grad_T_c, dim=[1,2,3], keepdim=True) + 1e-8
        ratio = kappa / alpha  # effective parameter (B,1,1,1)

        # Now compute Laplacian and gradient for T_f (crop to common interior).
        d2T_f_dx2 = T_f[:, :, 1:-1, 2:] - 2 * T_f[:, :, 1:-1, 1:-1] + T_f[:, :, 1:-1, :-2]
        d2T_f_dy2 = T_f[:, :, 2:, 1:-1] - 2 * T_f[:, :, 1:-1, 1:-1] + T_f[:, :, :-2, 1:-1]
        laplacian_T_f = d2T_f_dx2 + d2T_f_dy2

        dT_f_dx = (T_f[:, :, 1:-1, 2:] - T_f[:, :, 1:-1, :-2]) / 2.0
        dT_f_dy = (T_f[:, :, 2:, 1:-1] - T_f[:, :, :-2, 1:-1]) / 2.0
        # Compute the coarse gradient on the fine grid
        grad_dot = dT_f_dx * dT_f_dx + dT_f_dy * dT_f_dy  # For simplicity, we use the fine gradients as a proxy

        # PDE residual: we want ratio * Laplacian(T_f) to match the magnitude of the gradient (this is a placeholder formulation)
        pde_residual = ratio * laplacian_T_f - grad_dot
        return torch.mean(pde_residual ** 2)

    def temperature_energy_loss(self, T_f: torch.Tensor, T_c: torch.Tensor) -> torch.Tensor:
        """
        Computes an energy consistency loss.
        T_f: full-resolution temperature (B,1,H_f,W_f)
        T_c: coarse temperature (B,1,H_c,W_c)
        We downsample T_f via adaptive average pooling to T_c's resolution and compare.
        """
        T_f_down = F.adaptive_avg_pool2d(T_f, T_c.shape[-2:])
        return torch.mean((T_f_down - T_c) ** 2)

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps,
                                   linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas must be defined for each timestep'
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.denoiser_ema.store(self.denoiser.parameters())
            self.denoiser_ema.copy_to(self.denoiser)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.denoiser_ema.restore(self.denoiser.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def apply_denoiser(self, x_noisy, t, cond=None, return_ids=False):
        if self.conditional:
            if cond is not None:
                # If cond is already a list (of tuples), assume it's in the proper format
                # and pass it directly to the context encoder.
                if isinstance(cond, list):
                    cond = self.context_encoder(cond)
                # Otherwise, if cond isn’t a dict, wrap it as a single-item list.
                elif not isinstance(cond, dict):
                    cond = self.context_encoder([(cond, [0])])
            else:
                cond = None
        with self.ema_scope():
            return self.denoiser(x_noisy, t, context=cond)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='mean' if mean else 'none')
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")
        return loss

    def p_losses(self, x_start, t, noise=None, context=None):
        # In inference mode (evaluation) simply decode and return predictions.
      #  if not self.training:
      #      with torch.no_grad():
      #          if noise is None:
      #              noise = torch.randn_like(x_start)
      #          x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
      #          denoiser_out = self.denoiser(x_noisy, t, context=context)
      #          return self.autoencoder.decode(denoiser_out)
        # Otherwise (training mode), compute the diffusion loss ...
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        denoiser_out = self.denoiser(x_noisy, t, context=context)
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
        diffusion_loss = self.get_loss(denoiser_out, target, mean=False).mean()
        total_loss = diffusion_loss

        # Add PDE/energy terms if requested.
        if self.pde_lambda > 0 or (self.pde_mode == "temp" and (self.temp_pde_coef > 0 or self.temp_energy_coef > 0)):
            if self.pde_mode == "uv":
                wind_field = denoiser_out[:, :2, :, :]
                pde_loss_val = self.mass_conservation_loss(wind_field)
                total_loss = total_loss + self.pde_lambda * pde_loss_val
            elif self.pde_mode == "temp":
                T_f = self.autoencoder.decode(denoiser_out)
                if context is not None and isinstance(context, dict) and "T_c" in context:
                    T_c = context["T_c"]
                else:
                    raise ValueError("For temperature PDE loss, context must contain the key 'T_c'")
                pde_loss_val = self.temperature_pde_loss(T_f, T_c)
                energy_loss_val = self.temperature_energy_loss(T_f, T_c)
                total_loss = total_loss + self.temp_pde_coef * pde_loss_val + self.temp_energy_coef * energy_loss_val
        return total_loss

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def shared_step(self, batch):
        (x, y, z, ts) = batch      # x: coarse input, y: high-res target, z: static, ts: time
        assert not torch.any(torch.isnan(x)).item(), 'coarse input has NaNs'
        assert not torch.any(torch.isnan(y)).item(), 'high-res target has NaNs'
        assert not torch.any(torch.isnan(z)).item(), 'static has NaNs'
        if self.autoencoder.ae_flag == 'residual':
            residual, _ = self.autoencoder.preprocess_batch([x, y, z])
            latent_target = self.autoencoder.encode(residual)[0]
        else:
            latent_target = self.autoencoder.encode(y)[0]   # returns mean ONLY!!!
        # Build a context dictionary that always contains the coarse field as T_c.
        context_dict = {"T_c": x}
        if self.conditional:
            encoder_context = self.context_encoder([(z, [0]), (x, [0])])
            # If encoder_context is already a dict, merge it; otherwise, add it as an extra entry.
            if isinstance(encoder_context, dict):
                context_dict.update(encoder_context)
            else:
                context_dict["encoder_context"] = encoder_context
        return self(latent_target, context=context_dict)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        with self.ema_scope():
            loss_ema = self.shared_step(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val/loss", loss, **log_params, sync_dist=True)
        self.log("val/loss_ema", loss_ema, **log_params, sync_dist=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        with self.ema_scope():
            loss_ema = self.shared_step(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("test/loss", loss, **log_params, sync_dist=True)
        self.log("test/loss_ema", loss_ema, **log_params, sync_dist=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.denoiser_ema(self.denoiser)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr,
                                      betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val/loss_ema",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        if self.trainer.global_step < self.lr_warmup:
            lr_scale = (self.trainer.global_step + 1) / self.lr_warmup
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)
