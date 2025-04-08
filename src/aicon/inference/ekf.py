import torch

from aicon.inference.outlier_rejection import reject_outlier
from aicon.inference.util import make_safe_for_inversion


def update_ekf(c_func, mu, Sigma, other_quantity, Sigma_other, R_additive, outlier_rejection_treshold = None, prevent_uncertainty_state_grad=True, preserve_rejection_grad=True):
    innovation = c_func(mu, other_quantity)

    H_this, H_other = torch.func.jacfwd(c_func, argnums=(0, 1), randomness="same")(mu, other_quantity)
    S = torch.mm(torch.mm(H_this, Sigma), H_this.transpose(0, 1)) + torch.mm(torch.mm(H_other, Sigma_other), H_other.transpose(0, 1)) + R_additive
    K = torch.mm(torch.mm(Sigma, H_this.transpose(0, 1)), torch.inverse(make_safe_for_inversion(S, is_batch=False)))

    if prevent_uncertainty_state_grad:
        # We detach to prevent a idiosyncratic outcome of gradient descent on inference
        # Namely, that if the measurement is closer/farther of the goal a logical conclusion
        # is to be more/less certain about it relative to current predicted Sigma
        change = torch.mv(K.detach(), innovation)
    else:
        change = torch.mv(K, innovation)
    mu_new = mu - change

    Sigma_new = Sigma - torch.mm(K, torch.mm(H_this, Sigma))

    if outlier_rejection_treshold is not None:
        mu_new, Sigma_new = reject_outlier(mu_new, Sigma_new, mu, Sigma, torch.mm(torch.mm(H_this, Sigma), H_this.transpose(0, 1)), innovation, outlier_rejection_treshold,
                                           preserve_gradient=preserve_rejection_grad)

    return mu_new, Sigma_new


def update_switching_ekf(c_func, mu, Sigma, other_quantity, Sigma_other, R_additive, on_likelihood, outlier_rejection_treshold = None, prevent_uncertainty_state_grad=True, preserve_rejection_grad=True):
    innovation = c_func(mu, other_quantity)

    H_this, H_other = torch.func.jacfwd(c_func, argnums=(0, 1), randomness="same")(mu, other_quantity)
    S = torch.mm(torch.mm(H_this, Sigma), H_this.transpose(0, 1)) + torch.mm(torch.mm(H_other, Sigma_other), H_other.transpose(0, 1)) + R_additive
    K = on_likelihood * torch.mm(torch.mm(Sigma, H_this.transpose(0, 1)),
                                 torch.inverse(make_safe_for_inversion(S, is_batch=False)))

    if prevent_uncertainty_state_grad:
        # We detach to prevent a idiosyncratic outcome of gradient descent on inference
        # Namely, that if the measurement is closer/farther of the goal a logical conclusion
        # is to be more/less certain about it relative to current predicted Sigma
        change = torch.mv(K.detach(), innovation)
    else:
        if on_likelihood < 0.25:
            # on_likelihood so small that instead of obtaining gradient for directly constraining qunatities,
            # we only obtain them for likelihood
            change = torch.mv(K, innovation.detach())
        else:
            change = torch.mv(K, innovation)

    mu_new = mu - change

    Sigma_new = Sigma - torch.mm(K, torch.mm(H_this, Sigma))

    if outlier_rejection_treshold is not None:
        mu_new, Sigma_new = reject_outlier(mu_new, Sigma_new, mu, Sigma, torch.mm(torch.mm(H_this, Sigma), H_this.transpose(0, 1)), innovation, outlier_rejection_treshold * on_likelihood,
                                           preserve_gradient=preserve_rejection_grad)

    return mu_new, Sigma_new


def update_shifting_ekf(c_func, mu, Sigma, other_quantity, Sigma_other, R_additive, shift_diagonal_matrix, outlier_rejection_treshold = None, prevent_uncertainty_state_grad=True, preserve_rejection_grad=True):
    innovation = c_func(mu, other_quantity)

    H_this, H_other = torch.func.jacfwd(c_func, argnums=(0, 1), randomness="same")(mu, other_quantity)
    S = torch.mm(torch.mm(H_this, Sigma), H_this.transpose(0, 1)) + torch.mm(torch.mm(H_other, Sigma_other), H_other.transpose(0, 1)) + R_additive
    K = torch.mm(torch.mm(Sigma, torch.mm(shift_diagonal_matrix, H_this.transpose(0, 1))),
                                 torch.inverse(make_safe_for_inversion(S, is_batch=False)))

    if prevent_uncertainty_state_grad:
        # We detach to prevent a idiosyncratic outcome of gradient descent on inference
        # Namely, that if the measurement is closer/farther of the goal a logical conclusion
        # is to be more/less certain about it relative to current predicted Sigma
        change = torch.mv(K.detach(), innovation)
    else:
        change = torch.mv(K, innovation)

    mu_new = mu - change

    Sigma_new = Sigma - torch.mm(K, torch.mm(H_this, Sigma))

    if outlier_rejection_treshold is not None:
        mu_new, Sigma_new = reject_outlier(mu_new, Sigma_new, mu, Sigma, torch.mm(torch.mm(H_this, Sigma), H_this.transpose(0, 1)), innovation, outlier_rejection_treshold,
                                           preserve_gradient=preserve_rejection_grad)

    return mu_new, Sigma_new

def update_ekf_triple_connection(c_func, mu, Sigma, quantity_other1, Sigma_other1, quantity_other2, Sigma_other2, R_additive, outlier_rejection_treshold = None, prevent_uncertainty_state_grad=True, preserve_rejection_grad=True):
    innovation = c_func(mu, quantity_other1, quantity_other2)

    H_this, H_other1, H_other2 = torch.func.jacfwd(c_func, argnums=(0, 1, 2), randomness="same")(mu, quantity_other1, quantity_other2)
    S = torch.mm(torch.mm(H_this, Sigma), H_this.transpose(0, 1)) + torch.mm(torch.mm(H_other1, Sigma_other1), H_other1.transpose(0, 1)) + torch.mm(torch.mm(H_other2, Sigma_other2), H_other2.transpose(0, 1)) + R_additive
    K = torch.mm(torch.mm(Sigma, H_this.transpose(0, 1)), torch.inverse(make_safe_for_inversion(S, is_batch=False)))

    change = torch.mv(K, innovation)

    if prevent_uncertainty_state_grad:
        # We detach to prevent a idiosyncratic outcome of gradient descent on inference
        # Namely, that if the measurement is closer/farther of the goal a logical conclusion
        # is to be more/less certain about it relative to current predicted Sigma
        mu_new = mu - change.detach()
    else:
        mu_new = mu - change

    Sigma_new = Sigma - torch.mm(K, torch.mm(H_this, Sigma))

    if outlier_rejection_treshold is not None:
        mu_new, Sigma_new = reject_outlier(mu_new, Sigma_new, mu, Sigma, Sigma, innovation, outlier_rejection_treshold,
                                           preserve_gradient=preserve_rejection_grad)

    return mu_new, Sigma_new


def update_switching_ekf_triple_connection(c_func, mu, Sigma, quantity_other1, Sigma_other1, quantity_other2, Sigma_other2, R_additive, on_likelihood, outlier_rejection_treshold = None, prevent_uncertainty_state_grad=True, preserve_rejection_grad=True):
    innovation = c_func(mu, quantity_other1, quantity_other2)

    H_this, H_other1, H_other2 = torch.func.jacfwd(c_func, argnums=(0, 1, 2), randomness="same")(mu, quantity_other1, quantity_other2)
    S = torch.mm(torch.mm(H_this, Sigma), H_this.transpose(0, 1)) + torch.mm(torch.mm(H_other1, Sigma_other1), H_other1.transpose(0, 1)) + torch.mm(torch.mm(H_other2, Sigma_other2), H_other2.transpose(0, 1)) + R_additive
    K = on_likelihood * torch.mm(torch.mm(Sigma, H_this.transpose(0, 1)),
                                 torch.inverse(make_safe_for_inversion(S, is_batch=False)))

    if prevent_uncertainty_state_grad:
        # We detach to prevent a idiosyncratic outcome of gradient descent on inference
        # Namely, that if the measurement is closer/farther of the goal a logical conclusion
        # is to be more/less certain about it relative to current predicted Sigma
        change = torch.mv(K, innovation)
        mu_new = mu - change.detach()
    else:
        if on_likelihood < 0.25:
            # on_likelihood so small that instead of obtaining gradient for directly constraining qunatities,
            # we only obtain them for likelihood
            change = torch.mv(K, innovation.detach())
        else:
            change = torch.mv(K, innovation)
        mu_new = mu - change

    Sigma_new = Sigma - torch.mm(K, torch.mm(H_this, Sigma))

    if outlier_rejection_treshold is not None:
        mu_new, Sigma_new = reject_outlier(mu_new, Sigma_new, mu, Sigma, Sigma, innovation, outlier_rejection_treshold * on_likelihood,
                                           preserve_gradient=preserve_rejection_grad)

    return mu_new, Sigma_new


def predict_ekf(forward_func, mu, Sigma, Q_additive):
    mu_new = forward_func(mu)

    F_this = torch.func.jacfwd(forward_func, argnums=(0), randomness="same")(mu)

    Sigma_new = F_this.mm(Sigma).mm(F_this.t()) + Q_additive

    return mu_new, Sigma_new

def predict_ekf_other_quantity(c_func, mu, Sigma, other_quantity, Sigma_other, Q_additive):
    mu_new = c_func(mu, other_quantity)

    F_this, F_other = torch.func.jacfwd(c_func, argnums=(0, 1), randomness="same")(mu, other_quantity)

    Sigma_new = F_this.mm(Sigma).mm(F_this.t()) + F_other.mm(Sigma_other).mm(F_other.t()) + Q_additive

    return mu_new, Sigma_new