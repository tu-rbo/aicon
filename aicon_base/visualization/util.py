import torch
from matplotlib.patches import Ellipse


def get_mpl_covariance_ellipse(mu, Sigma, scale=1.0, **kwargs):
    x, y = mu[0], mu[1]
    eigval, eigvec = torch.linalg.eigh(Sigma)  # in ascending order
    angle = (torch.atan2(eigvec[1, 1], eigvec[1, 0]) / 3.141 * 180).item()
    length_minor = 2 * ((4.605**0.5) * eigval[0].abs().item())  # 90% confidence interval
    length_major = 2 * ((4.605**0.5) * eigval[1].abs().item())  # 90% confidence interval
    angle = angle - 90.0  # Matplotlib angle conventions
    return Ellipse(
        xy=(x.item(), y.item()), width=length_minor * scale, height=length_major * scale, angle=angle, **kwargs
    )
