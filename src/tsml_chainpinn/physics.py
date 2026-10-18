"""Conservative mixture momentum and ZGB transport, equations (2)-(9).

All inputs are pointwise functions of normalized x,y,z,t. Gradients are converted
to SI units before products, stress divergence and residual scaling.
"""
import torch
from torch.nn import functional as F


def derivative(value, coordinates, coord_scale):
    if not value.requires_grad:
        return coordinates * 0
    g = torch.autograd.grad(value, coordinates, torch.ones_like(value),
                            create_graph=True, retain_graph=True, allow_unused=True)[0]
    if g is None:
        return coordinates * 0
    return g / coord_scale


def zgb_sources(pressure_absolute, alpha_liquid, cfg, smooth=True):
    delta = cfg["vapor_pressure_pa"] - pressure_absolute
    if smooth:
        eps = cfg["positive_part_width_pa"]
        # Smooth positive part in pressure units; eps is disclosed, not from paper.
        positive = lambda x: eps * F.softplus(x / eps)
        tiny = torch.finfo(delta.dtype).tiny
    else:
        positive = torch.relu
        tiny = 0.0
    common = 3 * cfg["rho_vapor"] / cfg["bubble_radius_m"]
    evap = (cfg["f_vap"] * cfg["alpha_nuc"] * alpha_liquid * common
            * (2 * positive(delta) / (3 * cfg["rho_liquid"]) + tiny).sqrt())
    cond = (cfg["f_cond"] * (1 - alpha_liquid) * common
            * (2 * positive(-delta) / (3 * cfg["rho_liquid"]) + tiny).sqrt())
    return evap, cond


def residuals(coords, prediction, normalization, cfg, alpha_reference=None,
              alpha_reference_gradient=None, include_vapor=True, smooth=True):
    scale = coords.new_tensor(normalization["coord_scale"])
    flow = (prediction["flow"] * coords.new_tensor(normalization["flow_scale"])
            + coords.new_tensor(normalization["flow_mean"]))
    p, velocity = flow[:, :1], flow[:, 1:4]
    grad = lambda v: derivative(v, coords, scale)
    if alpha_reference is None:
        alpha = prediction["alpha"]
    else:
        if alpha_reference_gradient is None:
            raise ValueError("Stage 1 conservative momentum requires CFD d(alpha_L)/d(x,y,z,t).")
        # A local first-order lift supplies the CFD coefficient derivatives. It
        # leaves alpha values exactly unchanged at each sampled coordinate.
        alpha = alpha_reference + ((coords - coords.detach()) * scale
                                  * alpha_reference_gradient).sum(dim=1, keepdim=True)
    rho = cfg["rho_vapor"] + alpha * (cfg["rho_liquid"] - cfg["rho_vapor"])
    mu = cfg["mu_vapor"] + alpha * (cfg["mu_liquid"] - cfg["mu_vapor"])
    vg = [grad(velocity[:, i:i+1]) for i in range(3)]
    divergence = sum(vg[i][:, i:i+1] for i in range(3))
    pg = grad(p)
    components = []
    for i in range(3):
        ui = velocity[:, i:i+1]
        inertia = grad(rho * ui)[:, 3:4]
        for j in range(3):
            inertia = inertia + grad(rho * ui * velocity[:, j:j+1])[:, j:j+1]
        stress_div = coords[:, :1] * 0
        for j in range(3):
            stress = mu * (vg[i][:, j:j+1] + vg[j][:, i:i+1])
            stress_div = stress_div + grad(stress)[:, j:j+1]
        components.append(inertia + pg[:, i:i+1] - stress_div)
    momentum = torch.cat(components, dim=1)
    uref, length = cfg["reference_velocity_m_s"], cfg["reference_length_m"]
    result = {"divergence": divergence * length / uref,
              "momentum": momentum * length / (cfg["rho_liquid"] * uref ** 2)}
    if include_vapor:
        av = 1 - prediction["alpha"]
        ag = grad(av)
        evaporation, condensation = zgb_sources(p, prediction["alpha"], cfg, smooth)
        vapor = cfg["rho_vapor"] * (ag[:, 3:4] + (velocity * ag[:, :3]).sum(1, keepdim=True)
                                            + av * divergence) - evaporation + condensation
        result["vapor"] = vapor * length / (cfg["rho_vapor"] * uref)
    return result


def physics_losses(*args, **kwargs):
    r = residuals(*args, **kwargs)
    out = {"divergence": r["divergence"].square().mean(),
           "momentum": r["momentum"].square().sum(dim=1).mean()}
    if "vapor" in r:
        cfg = args[3] if len(args) > 3 else kwargs["cfg"]
        out["vapor"] = F.huber_loss(r["vapor"], torch.zeros_like(r["vapor"]),
                                   delta=cfg["huber_delta"])
    return out
