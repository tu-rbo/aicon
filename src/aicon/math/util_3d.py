"""
Utility functions for 3D transformations and geometry operations.
This module provides functions for handling 3D transformations, rotations, and geometric operations
using PyTorch tensors.

This module in part includes code written by Manuel Baum, extended by Vito Mengers.
"""

import numpy as np
# import rospy
# import tf
import torch

gravity = torch.tensor([0., 0., -9.81])

def homogeneous_transform_inverse(H, is_batch=False):
    """
    Compute the inverse of a homogeneous transformation matrix.
    
    Args:
        H: 4x4 homogeneous transformation matrix
        is_batch: Whether H is a batch of matrices
        
    Returns:
        H_inverse: Inverse of the homogeneous transformation matrix
    """
    if is_batch:
        n_batch = H.shape[0]
        H_inverse = torch.zeros(n_batch, 4, 4, dtype=H.dtype, device=H.device)
        for i in range(n_batch):
            H_inverse[i] = homogeneous_transform_inverse(H[i], is_batch=False)
    else:
        R = H[:3,:3]
        d = H[:3, 3]

        R_inverse = R.inverse()

        H_inverse = construct_h_transform(R_inverse, -R_inverse.mv(d))
    return H_inverse

def htransform_from_linpos_rotpos(linpos, rotpos, is_batch=False):
    """
    Create a homogeneous transformation matrix from linear and rotational positions.
    
    Args:
        linpos: Linear position (translation)
        rotpos: Rotational position (quaternion)
        is_batch: Whether inputs are batches
        
    Returns:
        H: 4x4 homogeneous transformation matrix
    """
    if is_batch:
        n_bodies = linpos.shape[0]
        H = torch.eye(4, dtype=linpos.dtype, device=linpos.device).unsqueeze(0)
        H = H.repeat(n_bodies, 1, 1)
        H[:, :3, 3] = linpos
        H[:, :3, :3] = rotation_matrix_from_quaternion(rotpos, is_batch=True)
    else:
        H = torch.eye(4, dtype=linpos.dtype, device=linpos.device)
        H[:3, :3] = rotation_matrix_from_quaternion(rotpos)
        H[:3, 3] = linpos
    return H

def move_points_for_body_transforms(mu_bodies, mu_points_in_body_frames):
    n_bodies = mu_bodies.shape[0]
    n_points = mu_points_in_body_frames.shape[0]
    homogenous_points_in_frames = torch.cat([mu_points_in_body_frames, torch.ones(n_points, n_bodies, 1, dtype=mu_bodies.dtype, device=mu_bodies.device)], dim=2)
    linpos = mu_bodies[:, :3]
    rotpos = mu_bodies[:, 6:10]
    homogenous_transforms = htransform_from_linpos_rotpos(linpos, rotpos, is_batch=True)
    expected_points = torch.einsum("bij,pbj->pbi", homogenous_transforms, homogenous_points_in_frames)[:, :, :3]
    return expected_points

def construct_h_transform(R, t):
    """
    Construct a homogeneous transformation matrix from rotation matrix and translation.
    
    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector
        
    Returns:
        H: 4x4 homogeneous transformation matrix
    """
    H = torch.cat([torch.cat([R, t.unsqueeze(1)], dim=1),
                            torch.cat([torch.zeros(3, dtype=R.dtype, device=R.device),
                                       torch.ones(1, dtype=R.dtype, device=R.device)]).unsqueeze(0)])
    return H

def rotation_matrix_from_euler(euler):
    RX = torch.eye(3, dtype=euler.dtype, device=euler.device)
    RX[1,1] = torch.cos(euler[0])
    RX[2,2] = torch.cos(euler[0])
    RX[1,2] = -torch.sin(euler[0])
    RX[2,1] = torch.sin(euler[0])

    RY = torch.eye(3, dtype=euler.dtype, device=euler.device)
    RY[0,0] = torch.cos(euler[1])
    RY[2,2] = torch.cos(euler[1])
    RY[0,2] = torch.sin(euler[1])
    RY[2,0] = -torch.sin(euler[1])

    RZ = torch.eye(3, dtype=euler.dtype, device=euler.device)
    RZ[0,0] = torch.cos(euler[2])
    RZ[1,1] = torch.cos(euler[2])
    RZ[0,1] = -torch.sin(euler[2])
    RZ[1,0] = torch.sin(euler[2])

    R = torch.mm(torch.mm(RZ, RY), RX)

    return R

def quaternion_inverse(q):
    """
    Compute the inverse of a quaternion.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        q_inv: Inverse quaternion
    """
    return q*torch.tensor([1., -1., -1., -1.])

def quaternion_product(q1, q2, is_batch = False):
    """
    Compute the product of two quaternions.
    
    Args:
        q1: First quaternion
        q2: Second quaternion
        is_batch: Whether inputs are batches
        
    Returns:
        q3: Product quaternion
    """
    if is_batch:
        q3 = torch.cat([(q1[:,0] * q2[:,0] - q1[:,1] * q2[:,1] - q1[:,2] * q2[:,2] - q1[:,3] * q2[:,3]).unsqueeze(1),
                        (q1[:,0] * q2[:,1] + q1[:,1] * q2[:,0] + q1[:,2] * q2[:,3] - q1[:,3] * q2[:,2]).unsqueeze(1),
                        (q1[:,0] * q2[:,2] - q1[:,1] * q2[:,3] + q1[:,2] * q2[:,0] + q1[:,3] * q2[:,1]).unsqueeze(1),
                        (q1[:,0] * q2[:,3] + q1[:,1] * q2[:,2] - q1[:,2] * q2[:,1] + q1[:,3] * q2[:,0]).unsqueeze(1)], dim=1)
    else:
        q3 = torch.tensor([q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                           q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                           q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                           q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]])
    return q3

def rotation_matrix_from_quaternion(q, is_batch=False):
    """
    Convert a quaternion to a rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        is_batch: Whether input is a batch
        
    Returns:
        R: 3x3 rotation matrix
    """
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

def cross_product_matrix(x, is_batch=False):
    """
    Create a cross product matrix from a vector.
    
    Args:
        x: 3D vector
        is_batch: Whether input is a batch
        
    Returns:
        out: 3x3 cross product matrix
    """
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
        out = torch.zeros(3,3, dtype=x.dtype, device=x.device)
        out[0,1] = -x[2]
        out[0,2] = x[1]
        out[1,0] = x[2]
        out[1,2] = -x[0]
        out[2,0] = -x[1]
        out[2,1] = x[0]
    return out

def rotation_matrix_axis_angle(u, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    assert u.size()[0] == theta.size()[0]
    assert u.size()[1] == 3
    n = theta.size()[0]
    R = torch.zeros(n, 3, 3, dtype=u.dtype, device=u.device)

    ux, uy, uz = u[:,0], u[:,1], u[:,2]
    c, s = torch.cos(theta), torch.sin(theta)
    R[:, 0, 0] = c + ux**2 * (1. - c)
    R[:, 0, 1] = ux * uy * (1. - c) - uz * s
    R[:, 0, 2] = ux * uz * (1. - c) + uy * s

    R[:, 1, 0] = uy * ux * (1. - c) + uz * s
    R[:, 1, 1] = c + uy**2 * (1. - c)
    R[:, 1, 2] = uy * uz * (1. - c) - ux * s

    R[:, 2, 0] = uz * ux * (1. - c) - uy * s
    R[:, 2, 1] = uz * uy * (1. - c) + ux * s
    R[:, 2, 2] = c+uz**2 * (1. - c)

    return R

def htransform_tf(frame_from, frame_to, pose=None):
    if pose == None:
        tf_listener = tf.TransformListener()
        ### transform the 3d positions to camera_link
        now = rospy.Time(0)
        # now = rospy.Time.now()
        tf_listener.waitForTransform(frame_to, frame_from, now, rospy.Duration(.1))

        t = tf_listener.getLatestCommonTime(frame_from, frame_to)
        # position, quaternion = self.tf.lookupTransform(self.frame_id_measurement_3d_points, msg.header.frame_id, t)
        position, quaternion = tf_listener.lookupTransform(frame_to, frame_from, t)
    else:
        position, quaternion = pose
    Q = rotation_matrix_from_quaternion(torch.tensor([quaternion[3], quaternion[0], quaternion[1], quaternion[
        2]]))  # change order because ros has w in the back
    tau = torch.tensor(position)

    H = torch.eye(4)
    H[:3, :3] = Q
    H[:3, 3] = tau

    return H

def htransform_tf_linear_quaternion(position, quaternion):
    Q = rotation_matrix_from_quaternion(torch.tensor([quaternion[3], quaternion[0], quaternion[1], quaternion[
        2]]))  # change order because ros has w in the back
    tau = torch.tensor(position)

    H = torch.eye(4)
    H[:3, :3] = Q
    H[:3, 3] = tau

    return H

def H_transform_to_trans_quat(H, is_batch=False):
    quat = roation_to_quat(H, is_batch=is_batch)
    if is_batch:
        translation = H[:, :3, 3]
    else:
        translation = H[:3, 3]
    return translation, quat

def roation_to_quat(H, is_batch=False):
    """
    Convert a rotation matrix to a quaternion.
    
    Args:
        H: 3x3 rotation matrix or 4x4 homogeneous transform
        is_batch: Whether input is a batch
        
    Returns:
        q: Quaternion [w, x, y, z]
    """
    H = H[..., :3, :3]
    batch_dim = H.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        H.reshape(batch_dim + (9,)), dim=-1
    )

    tmp = torch.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22, ],
                    dim=-1, )
    tmp = torch.abs(tmp) + 0.1
    q_abs = torch.sqrt(tmp)

    q0 = q_abs[..., 0] / 2.0
    q1 = q_abs[..., 1] / 2.0
    q2 = q_abs[..., 2] / 2.0
    q3 = q_abs[..., 3] / 2.0

    q1 = q1 * torch.sign(m21 - m12)
    q2 = q2 * torch.sign(m02 - m20)
    q3 = q3 * torch.sign(m10 - m01)

    q = torch.stack([q0, q1, q2, q3], dim=-1)
    q = q / torch.norm(q, dim=-1, keepdim=True)
    return q

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
    euler[1] = torch.atan2(-R[2, 0], torch.sqrt(R[0,0] ** 2 + R[1, 0] **2))
    if torch.isclose(euler[1], torch.tensor(- np.pi / 2.0)):
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
    # shameless copy from here:
    # https://stackoverflow.com/questions/27546081/determining-a-homogeneous-affine-transformation-matrix-from-six-points-in-3d-usi

    '''
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
        '''

    centroid_p = torch.mean(p, dim=0, keepdim=True)
    centroid_p_prime = torch.mean(p_prime, dim=0, keepdim=True)

    H = torch.mm(torch.transpose((p - centroid_p), 0, 1), (p_prime - centroid_p_prime))

    U, S, Vh = torch.linalg.svd(H, full_matrices=True)

    R = torch.mm(Vh, U.transpose(0,1))

    # reflection case
    if torch.det(R) < 0:
        Vh[:, 2] *= -1
        R = torch.mm(Vh, U.transpose(0, 1))

    t = centroid_p_prime.squeeze() - torch.einsum("ij,j->i", R, centroid_p.squeeze())

    H = torch.cat((torch.cat((R, t.unsqueeze(1)), dim=1),
               torch.tensor([[0, 0, 0, 1]])))
    return H

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
        m[:, 0, 1] = - vector[:, 2]
        m[:, 0, 2] = vector[:, 1]
        m[:, 1, 2] = - vector[:, 0]
        m[:, 1, 0] = vector[:, 2]
        m[:, 2, 0] = - vector[:, 1]
        m[:, 2, 1] = vector[:, 0]
    else:
        zero = torch.zeros(1, dtype=vector.dtype, device=vector.device)
        m = torch.cat([torch.cat([zero, - vector[2:3], vector[1:2]]).unsqueeze(0),
                       torch.cat([vector[2:3], zero, - vector[0:1]]).unsqueeze(0),
                       torch.cat([-vector[1:2], vector[0:1], zero]).unsqueeze(0)])
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
        difference = torch.einsum("bij,b->bij", omega_bracket, torch.sin(angle) / (angle  + 0.000000001)) \
            + torch.einsum("bij,b->bij", torch.einsum("bij,bjk->bik", omega_bracket, omega_bracket), (1.0 - torch.cos(angle)) / torch.pow(angle  + 0.000000001, 2))
        e[angle != 0] = e[angle != 0] + difference[angle != 0]
    else:
        angle = torch.norm(omega)
        e = torch.eye(3, dtype=omega.dtype, device=omega.device)
        difference = omega_bracket * torch.sin(angle) / (angle  + 0.000000001)\
            + torch.einsum("ij,jk->ik", omega_bracket, omega_bracket) * (1.0 - torch.cos(angle)) / torch.pow(angle + 0.000000001, 2)
        if angle != 0:
            e += difference
    if is_return_angle:
        return e, angle
    else:
        return e

def exponential_map_se3(se3_pose, is_batch=False):
    """
    Convert a twist vector to a homogeneous transformation matrix using the exponential map.
    
    Args:
        se3_pose: Twist vector [v, w] where v is linear velocity and w is angular velocity
        is_batch: Whether input is a batch
        
    Returns:
        H: 4x4 homogeneous transformation matrix
    """
    if is_batch:
        n_bodies = se3_pose.shape[0]
        v = se3_pose[:, :3]
        w = se3_pose[:, 3:]
        theta = torch.norm(w, dim=1)
        w_hat = cross_product_matrix(w, is_batch=True)
        w_hat_sq = torch.einsum("bij,bjk->bik", w_hat, w_hat)
        R = torch.eye(3, dtype=se3_pose.dtype, device=se3_pose.device).unsqueeze(0)
        R = R.repeat(n_bodies, 1, 1)
        R = R + torch.einsum("b,bij->bij", torch.sin(theta), w_hat) / (theta + 1e-10)
        R = R + torch.einsum("b,bij->bij", 1.0 - torch.cos(theta), w_hat_sq) / (theta**2 + 1e-10)
        V = torch.eye(3, dtype=se3_pose.dtype, device=se3_pose.device).unsqueeze(0)
        V = V.repeat(n_bodies, 1, 1)
        V = V + torch.einsum("b,bij->bij", 1.0 - torch.cos(theta), w_hat) / (theta**2 + 1e-10)
        V = V + torch.einsum("b,bij->bij", theta - torch.sin(theta), w_hat_sq) / (theta**3 + 1e-10)
        t = torch.einsum("bij,bj->bi", V, v)
        H = torch.eye(4, dtype=se3_pose.dtype, device=se3_pose.device).unsqueeze(0)
        H = H.repeat(n_bodies, 1, 1)
        H[:, :3, :3] = R
        H[:, :3, 3] = t
    else:
        v = se3_pose[:3]
        w = se3_pose[3:]
        theta = torch.norm(w)
        w_hat = cross_product_matrix(w)
        w_hat_sq = torch.mm(w_hat, w_hat)
        R = torch.eye(3, dtype=se3_pose.dtype, device=se3_pose.device) + torch.sin(theta) * w_hat / (theta + 1e-10)
        R = R + (1.0 - torch.cos(theta)) * w_hat_sq / (theta**2 + 1e-10)
        V = torch.eye(3, dtype=se3_pose.dtype, device=se3_pose.device) + (1.0 - torch.cos(theta)) * w_hat / (theta**2 + 1e-10)
        V = V + (theta - torch.sin(theta)) * w_hat_sq / (theta**3 + 1e-10)
        t = torch.mv(V, v)
        H = torch.eye(4, dtype=se3_pose.dtype, device=se3_pose.device)
        H[:3, :3] = R
        H[:3, 3] = t
    return H

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
        difference = torch.einsum("bij,b->bij", omega_hat, (1.0 - torch.cos(angle)) / torch.pow(angle + 0.000000001, 2)) \
                     + torch.einsum("bij,b->bij", torch.einsum("bij,bjk->bik", omega_hat, omega_hat),
                                    (angle - torch.sin(angle)) / torch.pow(angle  + 0.000000001, 3))
        V[angle != 0] = V[angle != 0] + difference[angle != 0]
    else:
        V = torch.eye(3, dtype=omega.dtype, device=omega.device)
        difference = omega_hat * (1.0 - torch.cos(angle)) / torch.pow(angle + 0.000000001, 2) \
                     + torch.einsum("ij,jk->ik", omega_hat, omega_hat) * (angle - torch.sin(angle)) / torch.pow(angle  + 0.000000001,
                                                                                                                3)
        if angle != 0:
            V += difference
    return V

def log_map_so3(R, is_batch=False, is_return_angle=False):
    # R = ensure_numerical_stable_rotation_matrix(R, is_batch=is_batch)
    if is_batch:
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
        angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
        log_R = (R - torch.transpose(R, -1, -2))
        log_R[angle!=0] = torch.einsum("bij,b->bij", torch.clone(log_R), angle / ( 2.0 * torch.sin(angle)))[angle!=0]
        log_R[angle==0] = torch.zeros_like(log_R[angle==0])
        omega = vee_operator(log_R, is_batch=is_batch)
    else:
        trace = torch.trace(R)
        angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
        if angle == 0.0:
            log_R = torch.zeros_like(R)
        else:
            log_R = angle / ( 2.0 * torch.sin(angle)) * (R - torch.transpose(R, -1, -2))
        omega = vee_operator(log_R, is_batch=is_batch)
    if is_return_angle:
        return omega, angle
    else:
        return omega

def log_map_se3(SE3_pose, is_batch=False):
    """
    Convert a homogeneous transformation matrix to a twist vector using the logarithm map.
    
    Args:
        SE3_pose: 4x4 homogeneous transformation matrix
        is_batch: Whether input is a batch
        
    Returns:
        xi: Twist vector [v, w] where v is linear velocity and w is angular velocity
    """
    if is_batch:
        n_bodies = SE3_pose.shape[0]
        R = SE3_pose[:, :3, :3]
        t_SE3 = SE3_pose[:, :3, 3]
        theta = torch.acos((torch.einsum("bii->b", R) - 1.0) / 2.0)
        w_hat = (R - torch.transpose(R, 1, 2)) / (2.0 * torch.sin(theta).unsqueeze(1).unsqueeze(2) + 1e-10)
        w = torch.stack([w_hat[:, 2, 1], w_hat[:, 0, 2], w_hat[:, 1, 0]], dim=1)
        w_hat_sq = torch.einsum("bij,bjk->bik", w_hat, w_hat)
        V_inv = torch.eye(3, dtype=SE3_pose.dtype, device=SE3_pose.device).unsqueeze(0)
        V_inv = V_inv.repeat(n_bodies, 1, 1)
        V_inv = V_inv - 0.5 * w_hat
        V_inv = V_inv + torch.einsum("b,bij->bij", 1.0 - theta / (2.0 * torch.tan(theta / 2.0)), w_hat_sq) / (theta**2 + 1e-10)
        v = torch.einsum("bij,bj->bi", V_inv, t_SE3)
        xi = torch.cat([v, w], dim=1)
    else:
        R = SE3_pose[:3, :3]
        t_SE3 = SE3_pose[:3, 3]
        theta = torch.acos((torch.trace(R) - 1.0) / 2.0)
        w_hat = (R - R.t()) / (2.0 * torch.sin(theta) + 1e-10)
        w = torch.tensor([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
        w_hat_sq = torch.mm(w_hat, w_hat)
        V_inv = torch.eye(3, dtype=SE3_pose.dtype, device=SE3_pose.device) - 0.5 * w_hat
        V_inv = V_inv + (1.0 - theta / (2.0 * torch.tan(theta / 2.0))) * w_hat_sq / (theta**2 + 1e-10)
        v = torch.mv(V_inv, t_SE3)
        xi = torch.cat([v, w])
    return xi

def ensure_numerical_stable_rotation_matrix(R, is_batch=True):
    q = roation_to_quat(R, is_batch=is_batch)
    new_R = rotation_matrix_from_quaternion(q, is_batch=is_batch)
    return new_R

def angle_between_vectors(a, b):
    inner_product = (a * b).sum(dim=-1)
    angle = torch.acos(torch.clamp(inner_product / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1)), min=-1.0, max=1.0))
    return angle

def quaternion_derivative(quat1, quat2, dt, is_batch=False):
    """
    Compute the derivative of a quaternion.
    
    Args:
        quat1: Initial quaternion
        quat2: Final quaternion
        dt: Time step
        is_batch: Whether inputs are batches
        
    Returns:
        quat_derivative: Quaternion derivative
    """
    quat_grad = quaternion_product(quat2, quaternion_inverse(quat1), is_batch=is_batch)
    angular_vel = 2.0 * quaternion_product(quat_grad, quaternion_inverse(quat1), is_batch=is_batch)
    return angular_vel / dt

def quaternion_from_axis_angle(axis, angle, is_batch=False):
    """
    Convert axis-angle representation to quaternion.
    
    Args:
        axis: Rotation axis
        angle: Rotation angle
        is_batch: Whether inputs are batches
        
    Returns:
        q: Quaternion [w, x, y, z]
    """
    if is_batch:
        n_bodies = axis.shape[0]
        q = torch.zeros(n_bodies, 4, dtype=axis.dtype, device=axis.device)
        q[:, 0] = torch.cos(angle / 2.0)
        q[:, 1:] = torch.einsum("b,bj->bj", torch.sin(angle / 2.0), axis)
    else:
        q = torch.zeros(4, dtype=axis.dtype, device=axis.device)
        q[0] = torch.cos(angle / 2.0)
        q[1:] = torch.sin(angle / 2.0) * axis
    return q

def axis_angle_from_quaternion(q, is_batch=False):
    """
    Convert quaternion to axis-angle representation.
    
    Args:
        q: Quaternion [w, x, y, z]
        is_batch: Whether input is a batch
        
    Returns:
        axis: Rotation axis
        angle: Rotation angle
    """
    if is_batch:
        n_bodies = q.shape[0]
        angle = 2.0 * torch.acos(q[:, 0])
        axis = q[:, 1:] / torch.sin(angle / 2.0).unsqueeze(1)
    else:
        angle = 2.0 * torch.acos(q[0])
        axis = q[1:] / torch.sin(angle / 2.0)
    return axis, angle

def quaternion_from_euler(euler, is_batch=False):
    """
    Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles [roll, pitch, yaw]
        is_batch: Whether input is a batch
        
    Returns:
        q: Quaternion [w, x, y, z]
    """
    if is_batch:
        n_bodies = euler.shape[0]
        q = torch.zeros(n_bodies, 4, dtype=euler.dtype, device=euler.device)
        q[:, 0] = torch.cos(euler[:, 0] / 2.0) * torch.cos(euler[:, 1] / 2.0) * torch.cos(euler[:, 2] / 2.0) + \
                  torch.sin(euler[:, 0] / 2.0) * torch.sin(euler[:, 1] / 2.0) * torch.sin(euler[:, 2] / 2.0)
        q[:, 1] = torch.sin(euler[:, 0] / 2.0) * torch.cos(euler[:, 1] / 2.0) * torch.cos(euler[:, 2] / 2.0) - \
                  torch.cos(euler[:, 0] / 2.0) * torch.sin(euler[:, 1] / 2.0) * torch.sin(euler[:, 2] / 2.0)
        q[:, 2] = torch.cos(euler[:, 0] / 2.0) * torch.sin(euler[:, 1] / 2.0) * torch.cos(euler[:, 2] / 2.0) + \
                  torch.sin(euler[:, 0] / 2.0) * torch.cos(euler[:, 1] / 2.0) * torch.sin(euler[:, 2] / 2.0)
        q[:, 3] = torch.cos(euler[:, 0] / 2.0) * torch.cos(euler[:, 1] / 2.0) * torch.sin(euler[:, 2] / 2.0) - \
                  torch.sin(euler[:, 0] / 2.0) * torch.sin(euler[:, 1] / 2.0) * torch.cos(euler[:, 2] / 2.0)
    else:
        q = torch.zeros(4, dtype=euler.dtype, device=euler.device)
        q[0] = torch.cos(euler[0] / 2.0) * torch.cos(euler[1] / 2.0) * torch.cos(euler[2] / 2.0) + \
               torch.sin(euler[0] / 2.0) * torch.sin(euler[1] / 2.0) * torch.sin(euler[2] / 2.0)
        q[1] = torch.sin(euler[0] / 2.0) * torch.cos(euler[1] / 2.0) * torch.cos(euler[2] / 2.0) - \
               torch.cos(euler[0] / 2.0) * torch.sin(euler[1] / 2.0) * torch.sin(euler[2] / 2.0)
        q[2] = torch.cos(euler[0] / 2.0) * torch.sin(euler[1] / 2.0) * torch.cos(euler[2] / 2.0) + \
               torch.sin(euler[0] / 2.0) * torch.cos(euler[1] / 2.0) * torch.sin(euler[2] / 2.0)
        q[3] = torch.cos(euler[0] / 2.0) * torch.cos(euler[1] / 2.0) * torch.sin(euler[2] / 2.0) - \
               torch.sin(euler[0] / 2.0) * torch.sin(euler[1] / 2.0) * torch.cos(euler[2] / 2.0)
    return q

def euler_from_quaternion(q, is_batch=False):
    """
    Convert quaternion to Euler angles.
    
    Args:
        q: Quaternion [w, x, y, z]
        is_batch: Whether input is a batch
        
    Returns:
        euler: Euler angles [roll, pitch, yaw]
    """
    if is_batch:
        n_bodies = q.shape[0]
        euler = torch.zeros(n_bodies, 3, dtype=q.dtype, device=q.device)
        euler[:, 0] = torch.atan2(2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
                                  1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2]))
        euler[:, 1] = torch.asin(2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]))
        euler[:, 2] = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                  1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]))
    else:
        euler = torch.zeros(3, dtype=q.dtype, device=q.device)
        euler[0] = torch.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                               1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
        euler[1] = torch.asin(2.0 * (q[0] * q[2] - q[3] * q[1]))
        euler[2] = torch.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                               1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
    return euler












