import numpy as np
import torch
from torch import nn
from torch.optim import LBFGS

# Suppose:
# - STLDataClass is your data wrapper class (e.g. STL2DKernel)
# - st_op is an operator such that st_op.apply(DC).to_flatten()
#   returns a 1D tensor of scattering coefficients.
# - target is your reference map of shape (1,1,128,128)
#   and r is the corresponding scattering vector.


class ScatteringMatchModel(nn.Module):
    def __init__(self, st_op, STLDataClass, init_shape, device=None, dtype=None):
        super().__init__()
        self.st_op = st_op
        self.STLDataClass = STLDataClass

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32

        # Learnable field u
        self.u = STLDataClass(torch.randn(init_shape, device=device, dtype=dtype))
        self.u = nn.Parameter(self.u.array)

    #        self.u = nn.Parameter(
    #            st_op.wavelet_op.apply_smooth(self.u, inplace=False).array
    #        )

    def forward(self):
        DC_u = self.STLDataClass(self.u)
        st_u = self.st_op.apply(DC_u, norm="load_ref")
        s_flat_u = st_u.to_flatten(mean_along_batch=True, keepnans=True)
        return s_flat_u


def optimize_scattering_LBFGS(
    target,
    STLDataClass,
    st_op_target,
    st_op_running,
    max_iter=100,
    nbatch=1,
    lr=1.0,
    history_size=50,
    print_iter=10,
    verbose=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if np.any(np.isnan(target)):
        print("NaN detected in the target, the synthesis takes it into account")

    target = torch.as_tensor(target, device=device, dtype=torch.float32)

    # Reference scattering
    DC_target = STLDataClass(target)
    with torch.no_grad():
        r = st_op_target.apply(DC_target, norm="store_ref").to_flatten(keepnans=True)
    r = r.detach()
    target_coeffs_mask = ~torch.isnan(r)
    print("synthesis on {:} ST coefficients".format(target_coeffs_mask.sum().item()))
    st_op_running.S2_ref = st_op_target.S2_ref

    # Model with learnable u
    model = ScatteringMatchModel(
        st_op=st_op_running,
        STLDataClass=STLDataClass,
        init_shape=(nbatch, 1, *target.shape),
        device=device,
        dtype=target.dtype,
    )

    optimizer = LBFGS(
        [model.u],
        lr=lr,
        max_iter=max_iter,  # <-- le nombre d'itérations internes LBFGS
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )

    loss_history = []

    def closure():
        optimizer.zero_grad()
        s_flat_u = model()
        loss = ((s_flat_u[target_coeffs_mask] - r[target_coeffs_mask]).abs() ** 2).sum()
        loss.backward()

        # Log à chaque appel interne
        loss_val = loss.item()
        loss_history.append(loss_val)
        if verbose:
            if len(loss_history) % print_iter == 0:
                print(f"[LBFGS] inner iter {len(loss_history)}, loss = {loss_val:.6e}")

        return loss

    # Un seul appel : toutes les itérations LBFGS internes sont faites ici
    optimizer.step(closure)

    u_opt = model.u.detach()

    if nbatch == 1:
        u_opt = u_opt[0, 0]
    else:
        u_opt = u_opt[:, 0]

    return u_opt, loss_history
