"""
3D Math (se3 and SE3)

Author: Vito Mengers, based in part on Code from Manuel Baum
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import random

import numpy as np
import torch


def homogeneous_transform_inverse(H, is_batch=False):
    if is_batch:
        n_batch = H.shape[0]
        H_inverse = torch.zeros(n_batch, 4, 4, dtype=H.dtype, device=H.device)
        for i in range(n_batch):
            H_inverse[i] = homogeneous_transform_inverse(H[i], is_batch=False)
    else:
        R = H[:3, :3]
        d = H[:3, 3]

        R_inverse = R.inverse()

        H_inverse = construct_h_transform(R_inverse, -R_inverse.mv(d))
    return H_inverse


def compose_pose_and_point_quat_with_uncertainty(poses, Sigma_poses, points, Sigma_points):
    new_points, J_point, J_pose = compose_pose_and_point_quat(poses, points, return_jacs=True)
    # Sigma = Jpose * Spose * Jpose^T + Jpoint * Spoint * Jpoint^T
    new_Sigma = torch.einsum(
        "pbij,pbjl->pbil", J_pose, torch.einsum("bjk,pblk->pbjl", Sigma_poses, J_pose)
    ) + torch.einsum("bij,pbjk->pbik", J_point, torch.einsum("pbij,blj->pbil", Sigma_points, J_point))
    return new_points, new_Sigma


def compose_pose_and_point_quat(poses, points, return_jacs=False):
    point_jac, pose_jac = get_jacobians_pose_point_composition(poses, points)
    new_points = poses[:, :3].unsqueeze(0) + torch.einsum("bij,pbj->pbi", point_jac, points)
    if return_jacs:
        return new_points, point_jac, pose_jac
    return new_points


def compose_pose_and_pose_quat(poses1, poses2, return_jacs=False):
    translation, _, _ = compose_pose_and_point_quat(poses1, poses2[:, :, :3], return_jacs=True)
    new_poses = torch.zeros_like(poses2)
    new_poses[:, :, :3] = translation
    new_poses[:, :, 3] = poses1[3] * poses2[3] - poses1[4] * poses2[4] - poses1[5] * poses2[5] - poses1[6] * poses2[6]
    new_poses[:, :, 4] = poses1[3] * poses2[4] + poses2[3] * poses1[4] + poses1[5] * poses2[6] - poses1[6] * poses2[5]
    new_poses[:, :, 5] = poses1[3] * poses2[5] + poses2[3] * poses1[5] + poses1[6] * poses2[4] - poses1[4] * poses2[6]
    new_poses[:, :, 6] = poses1[3] * poses2[6] + poses2[3] * poses1[6] + poses1[4] * poses2[5] - poses1[5] * poses2[4]

    if return_jacs:
        raise NotImplementedError
    else:
        return new_poses


def invert_pose_quat_with_uncertainty(poses, Sigma_poses):
    new_poses, jac = invert_pose_quat(poses, return_jacs=True)
    new_Sigma = torch.einsum("bij,bjk->bik", jac, torch.einsum("bij,bkj->bik", Sigma_poses, jac))
    return new_poses, new_Sigma


def invert_pose_quat(poses, return_jacs=False):
    jac = get_inversion_jacobian(poses)
    inverted_pose = torch.einsum("bij,bj->bi", jac, poses)
    if return_jacs:
        return inverted_pose, jac
    else:
        return inverted_pose


def construct_h_transform(R, t):
    H = torch.cat(
        [
            torch.cat([R, t.unsqueeze(1)], dim=1),
            torch.cat(
                [torch.zeros(3, dtype=R.dtype, device=R.device), torch.ones(1, dtype=R.dtype, device=R.device)]
            ).unsqueeze(0),
        ]
    )
    return H


def get_inversion_jacobian(poses):
    # TODO check this
    jac = torch.zeros(poses.shape[0], 7, 7, dtype=poses.dtype, device=poses.device)
    # only used times 2, hence directly here as such
    qx_2 = 2 * torch.pow(poses[:, 4], 2)
    qy_2 = 2 * torch.pow(poses[:, 5], 2)
    qz_2 = 2 * torch.pow(poses[:, 6], 2)
    qxqy = 2 * poses[:, 4] * poses[:, 5]
    qyqz = 2 * poses[:, 5] * poses[:, 6]
    qxqz = 2 * poses[:, 4] * poses[:, 6]
    qrqx = 2 * poses[:, 3] * poses[:, 4]
    qrqy = 2 * poses[:, 3] * poses[:, 5]
    qrqz = 2 * poses[:, 3] * poses[:, 6]
    # usually used with a minus sign, hence used as such
    qrtx = -poses[:, 3] * poses[:, 0]
    qrty = -poses[:, 3] * poses[:, 1]
    qrtz = -poses[:, 3] * poses[:, 2]
    qxtx = -poses[:, 4] * poses[:, 0]
    qxty = -poses[:, 4] * poses[:, 1]
    qxtz = -poses[:, 4] * poses[:, 2]
    qytx = -poses[:, 5] * poses[:, 0]
    qyty = -poses[:, 5] * poses[:, 1]
    qytz = -poses[:, 5] * poses[:, 2]
    qztx = -poses[:, 6] * poses[:, 0]
    qzty = -poses[:, 6] * poses[:, 1]
    qztz = -poses[:, 6] * poses[:, 2]
    jac[:, 0, 0] = qy_2 + qz_2 - 1.0
    jac[:, 1, 0] = qrqz - qxqy
    jac[:, 2, 0] = -qrqy - qxqz
    jac[:, 0, 1] = -qrqz - qxqy
    jac[:, 1, 1] = qx_2 + qz_2 - 1.0
    jac[:, 2, 1] = qrqx - qyqz
    jac[:, 0, 2] = qrqy - qxqz
    jac[:, 1, 2] = -qrqx - qyqz
    jac[:, 2, 2] = qx_2 + qy_2 - 1.0
    jac[:, 0, 3] = -qytz + qzty
    jac[:, 1, 3] = qxtz - qztx
    jac[:, 2, 3] = qytx - qxty
    jac[:, 0, 4] = qyty + qztz
    jac[:, 1, 4] = qytx - 2 * qxty + qrtz
    jac[:, 2, 4] = qztx - qrty - 2 * qxtz
    jac[:, 0, 5] = qxty - 2 * qytx - qrtz
    jac[:, 1, 5] = qxtx + qztz
    jac[:, 2, 5] = qzty + qrtx - 2 * qytz
    jac[:, 0, 6] = qxtz + qrty - 2 * qztx
    jac[:, 1, 6] = -qrtx - 2 * qzty + qytz
    jac[:, 2, 6] = qytx + qyty
    jac[:, 3, 3] = 1.0
    jac[:, 4, 4] = -1.0
    jac[:, 5, 5] = -1.0
    jac[:, 6, 6] = -1.0
    return jac


def get_jacobians_pose_point_composition(poses, points):
    # TODO check this
    point_jac = torch.zeros(poses.shape[0], 3, 3, dtype=poses.dtype, device=poses.device)
    qx_2 = torch.pow(poses[:, 4], 2)
    qy_2 = torch.pow(poses[:, 5], 2)
    qz_2 = torch.pow(poses[:, 6], 2)
    qxqy = poses[:, 4] * poses[:, 5]
    qyqz = poses[:, 5] * poses[:, 6]
    qxqz = poses[:, 4] * poses[:, 6]
    qrqx = poses[:, 3] * poses[:, 4]
    qrqy = poses[:, 3] * poses[:, 5]
    qrqz = poses[:, 3] * poses[:, 6]
    point_jac[:, 0, 0] = 0.5 - qy_2 - qz_2
    point_jac[:, 0, 1] = qxqy + qrqz
    point_jac[:, 0, 2] = -qrqy + qxqz
    point_jac[:, 1, 0] = -qrqz + qxqy
    point_jac[:, 1, 1] = 0.5 - qx_2 - qz_2
    point_jac[:, 1, 2] = qyqz + qrqx
    point_jac[:, 2, 0] = qxqz + qrqy
    point_jac[:, 2, 1] = -qrqx + qyqz
    point_jac[:, 2, 2] = 0.5 - qx_2 - qy_2
    point_jac = point_jac * 2

    pose_jac = torch.zeros(points.shape[0], poses.shape[0], 3, 7, dtype=poses.dtype, device=poses.device)
    poses = poses.unsqueeze(0)
    pose_jac[:, :, 0, 0] = 1.0
    pose_jac[:, :, 1, 1] = 1.0
    pose_jac[:, :, 2, 2] = 1.0
    qrax = poses[:, :, 3] * points[:, :, 0]
    qray = poses[:, :, 3] * points[:, :, 1]
    qraz = poses[:, :, 3] * points[:, :, 2]
    qxax = poses[:, :, 4] * points[:, :, 0]
    qxay = poses[:, :, 4] * points[:, :, 1]
    qxaz = poses[:, :, 4] * points[:, :, 2]
    qyax = poses[:, :, 5] * points[:, :, 0]
    qyay = poses[:, :, 5] * points[:, :, 1]
    qyaz = poses[:, :, 5] * points[:, :, 2]
    qzax = poses[:, :, 6] * points[:, :, 0]
    qzay = poses[:, :, 6] * points[:, :, 1]
    qzaz = poses[:, :, 6] * points[:, :, 2]
    pose_jac[:, :, 0, 3] = -qzay + qyaz
    pose_jac[:, :, 0, 4] = qyay + qzaz
    pose_jac[:, :, 0, 5] = -2 * qyax + qxay + qraz
    pose_jac[:, :, 0, 6] = -2 * qzax - qray + qxaz
    pose_jac[:, :, 1, 3] = qzax - qxaz
    pose_jac[:, :, 1, 4] = qyax - 2 * qxay - qraz
    pose_jac[:, :, 1, 5] = qxax + qzaz
    pose_jac[:, :, 1, 6] = qrax - 2 * qzay + qyaz
    pose_jac[:, :, 2, 3] = -qyax + qxay
    pose_jac[:, :, 2, 4] = qzax + qray - 2 * qxaz
    pose_jac[:, :, 2, 5] = -qrax + qzay - 2 * qyaz
    pose_jac[:, :, 2, 6] = qxax + qyay
    # this would need to be multiplied with normalization Jac, if not normalized already

    return point_jac, pose_jac


def rotation_matrix_from_euler(euler):
    RX = torch.eye(3, dtype=euler.dtype, device=euler.device)
    RX[1, 1] = torch.cos(euler[0])
    RX[2, 2] = torch.cos(euler[0])
    RX[1, 2] = -torch.sin(euler[0])
    RX[2, 1] = torch.sin(euler[0])

    RY = torch.eye(3, dtype=euler.dtype, device=euler.device)
    RY[0, 0] = torch.cos(euler[1])
    RY[2, 2] = torch.cos(euler[1])
    RY[0, 2] = torch.sin(euler[1])
    RY[2, 0] = -torch.sin(euler[1])

    RZ = torch.eye(3, dtype=euler.dtype, device=euler.device)
    RZ[0, 0] = torch.cos(euler[2])
    RZ[1, 1] = torch.cos(euler[2])
    RZ[0, 1] = -torch.sin(euler[2])
    RZ[1, 0] = torch.sin(euler[2])

    R = torch.mm(torch.mm(RZ, RY), RX)

    return R


def quaternion_inverse(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0])


def quaternion_product(q1, q2, is_batch=False):
    if is_batch:
        q3 = torch.cat(
            [
                (q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]).unsqueeze(1),
                (q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]).unsqueeze(1),
                (q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]).unsqueeze(1),
                (q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]).unsqueeze(1),
            ],
            dim=1,
        )
    else:
        q3 = torch.tensor(
            [
                q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
                q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
                q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
            ]
        )
    return q3


def rotation_matrix_from_quaternion(q, is_batch=False):
    if is_batch:
        n_bodies = q.shape[0]
        cpm = cross_product_matrix(q[:, 1:], is_batch=True)
        R = torch.eye(3, dtype=q.dtype, device=q.device).unsqueeze(0)
        R = R.repeat(n_bodies, 1, 1)
        R = R + 2.0 * (torch.einsum("b,bij->bij", q[:, 0], cpm) + torch.einsum("bij,bjk->bik", cpm, cpm))
        return R
    else:
        cpm = cross_product_matrix(q[1:])
        R = torch.eye(3, dtype=q.dtype, device=q.device) + 2.0 * q[0] * cpm + 2.0 * torch.mm(cpm, cpm)
    return R


def euler_angles_from_quaternion(q):
    # heading = atan2(2 * qy * qw - 2 * qx * qz, 1 - 2 * qy2 - 2 * qz2)
    # attitude = asin(2 * qx * qy + 2 * qz * qw)
    # bank = atan2(2 * qx * qw - 2 * qy * qz, 1 - 2 * qx2 - 2 * qz2)

    heading = torch.atan2(2.0 * q[2] * q[0] - 2.0 * q[1] * q[3], 1.0 - 2.0 * q[2] ** 2 - 2.0 * q[3] ** 2)
    attitude = torch.asin(2.0 * q[1] * q[2] + 2.0 * q[3] * q[0])
    bank = torch.atan2(2.0 * q[1] * q[0] - 2.0 * q[2] * q[3], 1.0 - 2.0 * q[1] ** 2 - 2.0 * q[3] ** 2)

    return torch.tensor([heading, attitude, bank])


def cross_product_matrix(x, is_batch=False):
    # out = torch.tensor([[0, -x[2], x[1]],
    #                     [x[2], 0, -x[0]],
    #                     [-x[1], x[0], 0]])
    if is_batch:
        n_bodies = x.shape[0]
        out = torch.zeros(n_bodies, 3, 3, dtype=x.dtype, device=x.device)
        out[:, 0, 1] = -x[:, 2]
        out[:, 0, 2] = x[:, 1]
        out[:, 1, 0] = x[:, 2]
        out[:, 1, 2] = -x[:, 0]
        out[:, 2, 0] = -x[:, 1]
        out[:, 2, 1] = x[:, 0]
    else:
        out = torch.zeros(3, 3, dtype=x.dtype, device=x.device)
        out[0, 1] = -x[2]
        out[0, 2] = x[1]
        out[1, 0] = x[2]
        out[1, 2] = -x[0]
        out[2, 0] = -x[1]
        out[2, 1] = x[0]
    return out


def euler_covariance_from_quaternion_covariance(quaternion, C, C_lin_rot=None):
    q1, q2, q3, q4 = quaternion
    # https://stats.stackexchange.com/questions/119780/what-does-the-covariance-of-a-quaternion-mean
    dPsidQ1 = -(q3 + q2) / ((q3 + q2) ** 2 + (q4 + q1) ** 2) + (q3 - q2) / ((q3 - q2) ** 2 + (q4 - q1) ** 2)
    dPsidQ2 = (q4 + q1) / ((q3 + q2) ** 2 + (q4 + q1) ** 2) - (q4 - q1) / ((q3 - q2) ** 2 + (q4 - q1) ** 2)
    dPsidQ3 = (q4 + q1) / ((q3 + q2) ** 2 + (q4 + q1) ** 2) + (q4 - q1) / ((q3 - q2) ** 2 + (q4 - q1) ** 2)
    dPsidQ4 = -(q3 + q2) / ((q3 + q2) ** 2 + (q4 + q1) ** 2) - (q3 - q2) / ((q3 - q2) ** 2 + (q4 - q1) ** 2)

    dThetadQ1 = 2 * q4 / (1 - 4 * (q2 * q3 + q1 * q4) ** 2) ** 0.5
    dThetadQ2 = 2 * q3 / (1 - 4 * (q2 * q3 + q1 * q4) ** 2) ** 0.5
    dThetadQ3 = 2 * q2 / (1 - 4 * (q2 * q3 + q1 * q4) ** 2) ** 0.5
    dThetadQ4 = 2 * q1 / (1 - 4 * (q2 * q3 + q1 * q4) ** 2) ** 0.5

    dPhidQ1 = -(q3 + q2) / ((q3 + q2) ** 2 + (q4 + q1) ** 2) - (q3 - q2) / ((q3 - q2) ** 2 + (q4 - q1) ** 2)
    dPhidQ2 = (q4 + q1) / ((q3 + q2) ** 2 + (q4 + q1) ** 2) + (q4 - q1) / ((q3 - q2) ** 2 + (q4 - q1) ** 2)
    dPhidQ3 = (q4 + q1) / ((q3 + q2) ** 2 + (q4 + q1) ** 2) - (q4 - q1) / ((q3 - q2) ** 2 + (q4 - q1) ** 2)
    dPhidQ4 = -(q3 + q2) / ((q3 + q2) ** 2 + (q4 + q1) ** 2) + (q3 - q2) / ((q3 - q2) ** 2 + (q4 - q1) ** 2)

    G = torch.tensor(
        [
            [dPsidQ1, dPsidQ2, dPsidQ3, dPsidQ4],
            [dThetadQ1, dThetadQ2, dThetadQ3, dThetadQ4],
            [dPhidQ1, dPhidQ2, dPhidQ3, dPhidQ4],
        ]
    )

    C_euler = G.mm(C).mm(G.transpose(0, 1))

    if C_lin_rot is None:
        return C_euler

    C_lin_rot_euler = C_lin_rot.mm(G.transpose(0, 1))
    return C_euler, C_lin_rot_euler


def rotation_matrix_axis_angle(u, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    assert u.size()[0] == theta.size()[0]
    assert u.size()[1] == 3
    n = theta.size()[0]
    R = torch.zeros(n, 3, 3, dtype=u.dtype, device=u.device)

    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    c, s = torch.cos(theta), torch.sin(theta)
    R[:, 0, 0] = c + ux**2 * (1.0 - c)
    R[:, 0, 1] = ux * uy * (1.0 - c) - uz * s
    R[:, 0, 2] = ux * uz * (1.0 - c) + uy * s

    R[:, 1, 0] = uy * ux * (1.0 - c) + uz * s
    R[:, 1, 1] = c + uy**2 * (1.0 - c)
    R[:, 1, 2] = uy * uz * (1.0 - c) - ux * s

    R[:, 2, 0] = uz * ux * (1.0 - c) - uy * s
    R[:, 2, 1] = uz * uy * (1.0 - c) + ux * s
    R[:, 2, 2] = c + uz**2 * (1.0 - c)

    return R


def H_transform_to_trans_quat(H, is_batch=False):
    quat = roation_to_quat(H, is_batch=is_batch)
    if is_batch:
        translation = H[:, :3, 3]
    else:
        translation = H[:3, 3]
    return translation, quat


def roation_to_quat(H, is_batch=False):
    H = H[..., :3, :3]
    batch_dim = H.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(H.reshape(batch_dim + (9,)), dim=-1)

    tmp = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    tmp = torch.abs(tmp) + 0.1
    q_abs = torch.sqrt(tmp)
    q_abs[tmp < 0] = torch.zeros_like(q_abs[tmp < 0])

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    quat = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    return quat


def euler_to_quat(euler):
    cos = torch.cos(euler / 2.0)
    sin = torch.sin(euler / 2.0)
    quat = torch.zeros(4, dtype=euler.dtype, device=euler.device)
    quat[0] = cos[0] * cos[1] * cos[2] + sin[0] * sin[1] * sin[2]
    quat[1] = sin[0] * cos[1] * cos[2] - cos[0] * sin[1] * sin[2]
    quat[2] = cos[0] * sin[1] * cos[2] + sin[0] * cos[1] * sin[2]
    quat[3] = cos[0] * cos[1] * sin[2] - sin[0] * sin[1] * cos[2]
    return quat


def rotation_to_euler(R):
    euler = torch.zeros(3, dtype=R.dtype, device=R.device)
    euler[1] = torch.atan2(-R[2, 0], torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if torch.isclose(euler[1], torch.tensor(-np.pi / 2.0)):
        euler[0] = torch.atan2(-R[1, 2], -R[0, 2])
        euler[2] = torch.tensor([0.0])
    elif torch.isclose(euler[1], torch.tensor(np.pi / 2.0)):
        euler[0] = torch.atan2(R[1, 2], R[0, 2])
        euler[2] = torch.tensor([0.0])
    else:
        euler[0] = torch.atan2(R[1, 0], R[0, 0])
        euler[2] = torch.atan2(R[2, 1], R[2, 2])

    return euler


def affine_from_points(p, p_prime):
    # adapted from
    # https://stackoverflow.com/questions/27546081/determining-a-homogeneous-affine-transformation-matrix-from-six-points-in-3d-usi
    """
        Find the unique homogeneous affine transformation that
        maps a set of 3 points to another set of 3 points in 3D
        space:

            p_prime == np.dot(p, R) + t

        where `R` is an unknown rotation matrix, `t` is an unknown
        translation vector, and `p` and `p_prime` are the original
        and transformed set of points stored as row vectors:

            p       = np.array((p1,       p2,       p3))
            p_prime = np.array((p1_prime, p2_prime, p3_prime))

        The result of this function is an augmented 4-by-4
        matrix `A` that represents this affine transformation:

            np.column_stack((p_prime, (1, 1, 1))) == \
                np.dot(np.column_stack((p, (1, 1, 1))), A)

        Source: https://math.stackexchange.com/a/222170 (robjohn)
        """

    centroid_p = torch.mean(p, dim=0, keepdim=True)
    centroid_p_prime = torch.mean(p_prime, dim=0, keepdim=True)

    H = torch.mm(torch.transpose((p - centroid_p), 0, 1), (p_prime - centroid_p_prime))

    U, S, Vh = torch.linalg.svd(H, full_matrices=True)

    R = torch.mm(Vh, U.transpose(0, 1))

    # reflection case
    if torch.det(R) < 0:
        Vh[:, 2] *= -1
        R = torch.mm(Vh, U.transpose(0, 1))

    t = centroid_p_prime.squeeze() - torch.einsum("ij,j->i", R, centroid_p.squeeze())

    H = torch.cat((torch.cat((R, t.unsqueeze(1)), dim=1), torch.tensor([[0, 0, 0, 1]])))

    return H


def ransac_affine_from_points(p, p_prime, threshold_inlier=0.01, n_tries=100, is_remove_static_points=True):
    assert p.size() == p_prime.size()

    if is_remove_static_points:
        errors = torch.sum((p_prime - p) ** 2, dim=1) ** 0.5
        mask_nonstatic = errors > threshold_inlier
        p, p_prime = p[mask_nonstatic], p_prime[mask_nonstatic]

    n_points = p.size()[0]
    k = 3
    best_n_inliers, best_H = -1, None
    if n_points < k:
        raise ValueError("not enough non-static points:" + str(n_points) + "<" + str(k))

    for i_try in range(n_tries):
        indices = torch.tensor(random.sample(range(n_points), k))
        H = affine_from_points(p[indices], p_prime[indices])

        # transform the points using this H
        p_H = torch.einsum(
            "ij,bj->bi", H, torch.cat((p, torch.ones(n_points, 1, dtype=p.dtype, device=p.device)), dim=1)
        )[:, :3]

        errors = torch.sum((p_prime - p_H) ** 2, dim=1) ** 0.5
        mask_inliers = errors < threshold_inlier
        n_inliers = torch.sum(mask_inliers)

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_H = H
            best_inliers = mask_inliers

    if is_remove_static_points:
        best_inliers_in_all = mask_nonstatic.clone()
        best_inliers_in_all[mask_nonstatic] = best_inliers
    else:
        best_inliers_in_all = best_inliers

    return best_H, best_inliers_in_all


def l2dist(points1, points2, is_batch=False):
    powered = torch.pow(points1 - points2, 2)
    if is_batch:
        summed = torch.einsum("pbi->pb", powered)
    else:
        summed = torch.einsum("pi->p", powered)
    root = torch.sqrt(summed + 0.0000000000000000001)
    return root


def unpack_pose_quat(pose, is_batch=False):
    if is_batch:
        return pose[:, :3], pose[:, 3:]
    else:
        return pose[:3], pose[3:]


def unpack_twist(twist, is_batch=False):
    if is_batch:
        return twist[:, :3], twist[:, 3:]
    else:
        return twist[:3], twist[3:]


def twist_from_poses_quat(pose_t, pose_t_minus_1, dt, is_batch=False):
    trans1, quat1 = unpack_pose_quat(pose_t, is_batch=is_batch)
    trans2, quat2 = unpack_pose_quat(pose_t_minus_1, is_batch=is_batch)

    linvel = (trans1 - trans2) / dt
    quat_grad = (quat1 - quat2) / dt
    angular_vel = 2.0 * quaternion_product(quat_grad, quaternion_inverse(quat1), is_batch=is_batch)

    if is_batch:
        twist = torch.cat([linvel, angular_vel[:, 1:]], dim=1)
    else:
        twist = torch.cat([linvel, angular_vel[1:]], dim=0)
    return twist


def hat_operator(vector, is_batch=False):
    if is_batch:
        n_batches = vector.shape[0]
        m = torch.zeros(n_batches, 3, 3, dtype=vector.dtype, device=vector.device)
        m[:, 0, 1] = -vector[:, 2]
        m[:, 0, 2] = vector[:, 1]
        m[:, 1, 2] = -vector[:, 0]
        m[:, 1, 0] = vector[:, 2]
        m[:, 2, 0] = -vector[:, 1]
        m[:, 2, 1] = vector[:, 0]
    else:
        zero = torch.zeros(1, dtype=vector.dtype, device=vector.device)
        m = torch.cat(
            [
                torch.cat([zero, -vector[2:3], vector[1:2]]).unsqueeze(0),
                torch.cat([vector[2:3], zero, -vector[0:1]]).unsqueeze(0),
                torch.cat([-vector[1:2], vector[0:1], zero]).unsqueeze(0),
            ]
        )
    return m


def vee_operator(matrix, is_batch=False):
    if is_batch:
        n_batches = matrix.shape[0]
        v = torch.zeros(n_batches, 3, dtype=matrix.dtype, device=matrix.device)
        v[:, 0] = matrix[:, 2, 1]
        v[:, 1] = matrix[:, 0, 2]
        v[:, 2] = matrix[:, 1, 0]
    else:
        v = torch.cat([matrix[2, 1].unsqueeze(0), matrix[0, 2].unsqueeze(0), matrix[1, 0].unsqueeze(0)])
    return v


def adjoint_from_se3(se3_pose, is_batch=False):
    omega, v = unpack_se3_pose(se3_pose, is_batch=is_batch)
    omega_bracket = hat_operator(omega, is_batch=is_batch)
    v_bracket = hat_operator(v, is_batch=is_batch)
    if is_batch:
        ad = torch.zeros(se3_pose.shape[0], 6, 6, dtype=omega.dtype, device=omega.device)
        ad[:, :3, :3] = omega_bracket
        ad[:, 3:6, 3:6] = omega_bracket
        ad[:, :3, 3:6] = v_bracket
        return ad
    else:
        ad = torch.zeros(6, 6, dtype=omega.dtype, device=omega.device)
        ad[:3, :3] = omega_bracket
        ad[3:6, 3:6] = omega_bracket
        ad[:3, 3:6] = v_bracket
        return ad


def adjoint_from_SE3(SE3_pose, is_batch=False):
    R, t_SE3 = unpack_SE3_pose(SE3_pose, is_batch=is_batch)
    t_hat = hat_operator(t_SE3, is_batch=is_batch)
    if is_batch:
        n_batches = SE3_pose.shape[0]
        Ad = torch.zeros(n_batches, 6, 6, dtype=SE3_pose.dtype, device=SE3_pose.device)
        Ad[:, :3, :3] = R
        Ad[:, 3:6, 3:6] = R
        Ad[:, :3, 3:6] = torch.einsum("bij,bjk->bik", t_hat, R)
    else:
        Ad = torch.zeros(6, 6, dtype=SE3_pose.dtype, device=SE3_pose.device)
        Ad[:3, :3] = R
        Ad[3:6, 3:6] = R
        Ad[:3, 3:6] = torch.einsum("ij,jk->ik", t_hat, R)
    return Ad


def unpack_se3_pose(se3_pose, is_batch=False):
    if is_batch:
        t_se3, omega = se3_pose[:, :3], se3_pose[:, 3:6]
    else:
        t_se3, omega = se3_pose[:3], se3_pose[3:6]
    return omega, t_se3


def unpack_SE3_pose(SE3_pose, is_batch=False):
    if is_batch:
        R, t_SE3 = SE3_pose[:, :3, :3], SE3_pose[:, :3, 3]
    else:
        R, t_SE3 = SE3_pose[:3, :3], SE3_pose[:3, 3]
    return R, t_SE3


def exponential_map_so3(omega, is_batch=False, is_return_angle=False):
    omega_bracket = hat_operator(omega, is_batch=is_batch)
    if is_batch:
        n_batches = omega.shape[0]
        angle = torch.norm(omega, dim=1)
        e = torch.eye(3, dtype=omega.dtype, device=omega.device).unsqueeze(0).repeat(n_batches, 1, 1)
        difference = torch.einsum("bij,b->bij", omega_bracket, torch.sin(angle) / (angle + 0.000000001)) + torch.einsum(
            "bij,b->bij",
            torch.einsum("bij,bjk->bik", omega_bracket, omega_bracket),
            (1.0 - torch.cos(angle)) / torch.pow(angle + 0.000000001, 2),
        )
        e[angle != 0] = e[angle != 0] + difference[angle != 0]
    else:
        angle = torch.norm(omega)
        e = torch.eye(3, dtype=omega.dtype, device=omega.device)
        difference = omega_bracket * torch.sin(angle) / (angle + 0.000000001) + torch.einsum(
            "ij,jk->ik", omega_bracket, omega_bracket
        ) * (1.0 - torch.cos(angle)) / torch.pow(angle + 0.000000001, 2)
        if angle != 0:
            e += difference
    if is_return_angle:
        return e, angle
    else:
        return e


def exponential_map_se3(se3_pose, is_batch=False):
    omega, t_se3 = unpack_se3_pose(se3_pose, is_batch=is_batch)
    exp_omega, angle = exponential_map_so3(omega, is_batch=is_batch, is_return_angle=True)
    omega_hat = hat_operator(omega, is_batch=is_batch)
    V = get_se3_V(angle, omega, omega_hat, is_batch=is_batch).detach()
    V = torch.eye(3, dtype=omega.dtype, device=omega.device)
    if is_batch:
        n_batches = omega.shape[0]
        SE3_pose = torch.zeros(n_batches, 4, 4, dtype=omega.dtype, device=omega.device)
        SE3_pose[:, :3, :3] = exp_omega
        SE3_pose[:, :3, 3] = torch.einsum("bij,bj->bi", V, t_se3)
        SE3_pose[:, 3, 3] = 1.0
    else:
        # SE3_pose = torch.zeros(4, 4, dtype=omega.dtype, device=omega.device)
        # SE3_pose[:3, :3] = exp_omega
        # SE3_pose[:3, 3] = torch.einsum("ij,j->i", V, t_se3)
        # SE3_pose[3, 3] = 1.0
        SE3_pose = torch.cat(
            [
                torch.cat([exp_omega, torch.einsum("ij,j->i", V, t_se3).unsqueeze(1)], dim=1),
                torch.cat(
                    [
                        torch.zeros(3, dtype=omega.dtype, device=omega.device),
                        torch.ones(1, dtype=omega.dtype, device=omega.device),
                    ]
                ).unsqueeze(0),
            ]
        )
    return SE3_pose


def get_only_se3_V(se3_pose, is_batch=False):
    omega, t_se3 = unpack_se3_pose(se3_pose, is_batch=is_batch)
    exp_omega, angle = exponential_map_so3(omega, is_batch=is_batch, is_return_angle=True)
    omega_hat = hat_operator(omega, is_batch=is_batch)
    V = get_se3_V(angle, omega, omega_hat, is_batch=is_batch)
    return V


def get_se3_V(angle, omega, omega_hat, is_batch):
    if is_batch:
        n_batches = omega.shape[0]
        V = torch.eye(3, dtype=omega.dtype, device=omega.device).unsqueeze(0).repeat(n_batches, 1, 1)
        difference = torch.einsum(
            "bij,b->bij", omega_hat, (1.0 - torch.cos(angle)) / torch.pow(angle + 0.000000001, 2)
        ) + torch.einsum(
            "bij,b->bij",
            torch.einsum("bij,bjk->bik", omega_hat, omega_hat),
            (angle - torch.sin(angle)) / torch.pow(angle + 0.000000001, 3),
        )
        V[angle != 0] = V[angle != 0] + difference[angle != 0]
    else:
        V = torch.eye(3, dtype=omega.dtype, device=omega.device)
        difference = omega_hat * (1.0 - torch.cos(angle)) / torch.pow(angle + 0.000000001, 2) + torch.einsum(
            "ij,jk->ik", omega_hat, omega_hat
        ) * (angle - torch.sin(angle)) / torch.pow(angle + 0.000000001, 3)
        if angle != 0:
            V += difference
    return V


def log_map_so3(R, is_batch=False, is_return_angle=False):
    # R = ensure_numerical_stable_rotation_matrix(R, is_batch=is_batch)
    if is_batch:
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
        angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
        log_R = R - torch.transpose(R, -1, -2)
        log_R[angle != 0] = torch.einsum("bij,b->bij", torch.clone(log_R), angle / (2.0 * torch.sin(angle)))[angle != 0]
        log_R[angle == 0] = torch.zeros_like(log_R[angle == 0])
        omega = vee_operator(log_R, is_batch=is_batch)
    else:
        trace = torch.trace(R)
        angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
        if angle == 0.0:
            log_R = torch.zeros_like(R)
        else:
            log_R = angle / (2.0 * torch.sin(angle)) * (R - torch.transpose(R, -1, -2))
        omega = vee_operator(log_R, is_batch=is_batch)
    if is_return_angle:
        return omega, angle
    else:
        return omega


def log_map_se3(SE3_pose, is_batch=False):
    R, t_SE3 = unpack_SE3_pose(SE3_pose, is_batch=is_batch)
    omega, angle = log_map_so3(R, is_batch=is_batch, is_return_angle=True)
    omega_hat = hat_operator(omega, is_batch=is_batch)
    if is_batch:
        n_batches = omega.shape[0]
        V_inv = torch.eye(3, dtype=omega.dtype, device=omega.device).unsqueeze(0).repeat(n_batches, 1, 1)
        difference = -omega_hat * 0.5 + torch.einsum(
            "bij,b->bij",
            torch.einsum("bij,bjk->bik", omega_hat, omega_hat),
            (1.0 - (angle * torch.cos(angle / 2.0) / 2.0 / torch.sin(angle / 2.0))) / torch.pow(angle + 0.000000001, 2),
        )
        V_inv[angle != 0] = V_inv[angle != 0] + difference[angle != 0]
        t_se3 = torch.einsum("bij,bj->bi", V_inv.detach(), t_SE3)
        return torch.cat([t_se3, omega], dim=1)
    else:
        V_inv = torch.eye(3, dtype=omega.dtype, device=omega.device)
        if angle != 0.0:
            V_inv += -omega_hat * 0.5 + torch.einsum("ij,jk->ik", omega_hat, omega_hat) * (
                1.0 - (angle * torch.cos(angle / 2.0) / 2.0 / torch.sin(angle / 2.0))
            ) / torch.pow(angle + 0.000000001, 2)
        t_se3 = torch.einsum("ij,j->i", V_inv.detach(), t_SE3)
        return torch.cat([t_SE3, omega])


def ensure_numerical_stable_rotation_matrix(R, is_batch=True):
    q = roation_to_quat(R, is_batch=is_batch)
    new_R = rotation_matrix_from_quaternion(q, is_batch=is_batch)
    return new_R


def angle_between_vectors(a, b):
    inner_product = (a * b).sum(dim=-1)
    angle = torch.acos(torch.clamp(inner_product / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1)), min=-1.0, max=1.0))
    return angle
