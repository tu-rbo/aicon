"""
RGB-D Sensor for Real-Robot Object Pushing

Segments the object in the depth image and lifts its contour keypoints into the camera frame. Until the object
template is registered, the object is segmented by a flood fill seeded at the believed object position; afterwards
the belief-free object detector (object_detector.py) is used and publishes the fitted template's keypoints and
orientation.

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import math
import os
from typing import Dict, Type, Union, Tuple

import torch

import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage

from aicon_base.connections import ActiveInterconnection
from aicon_base.math.se3_and_SE3 import homogeneous_transform_inverse, exponential_map_se3
from .params import NUM_KEYPOINTS, RECORD_COLOR_TOPIC, RECORD_DEPTH_TOPIC
from . import object_detector as OD
from aicon_real_robots.sensors.camera_sensors import ROSCameraRGBDSensor

# Fewest keypoints off the image border for a contour to be used.
MIN_VALID_KEYPOINTS = 4

# Contour area range [px] accepted by the seeded segmentation (used before the detector takes over).
SEED_AREA_PX = (600, 15000)

# Throttled recording of the evaluated frames for offline analysis: every Nth depth frame (lossless PNG) and colour
# frame (JPEG with 4:4:4 chroma). 0 disables a stream.
RECORD_DEPTH_EVERY_N = 3
RECORD_COLOR_EVERY_N = 9
RECORD_PNG_LEVEL = 3
RECORD_JPEG_QUALITY = 95

# Bits of `keypoints_reject_cause`:
#   1: no contour found
#   2: contour depth differs from the believed object distance (recorded only)
#   4: seed too close to the image border (recorded only)
#   8: too few keypoints off the image border
#   16: contour area outside SEED_AREA_PX
#   32: the believed object position projects outside the image


def wrap_to_pi(a):
    """Fold an angle onto [-pi, pi)."""
    return float((float(a) + math.pi) % (2.0 * math.pi) - math.pi)


class ObjectPushingSensor(ROSCameraRGBDSensor):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        ros_topic_base_color: str,
        ros_topic_base_depth: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        scaling_factor: float = 0.25,
    ):
        super().__init__(
            name,
            connections,
            ros_topic_base_color=ros_topic_base_color,
            ros_topic_base_depth=ros_topic_base_depth,
            dtype=dtype,
            device=device,
            mockbuild=mockbuild,
            scaling_factor=scaling_factor,
        )
        self.img_shape = None
        self.P_inv = None
        self.H_ee_to_cam_inv = None
        self.last_starting_pos = None
        self.pusher_mask = cv2.imread(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "pusher_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        # per keypoint: 1 if it lies on the object boundary, 0 if on the image border
        self._last_keypoints_valid = torch.ones(
            NUM_KEYPOINTS, dtype=dtype if dtype is not None else torch.get_default_dtype()
        )
        self._detector_disabled = False
        self._record_pub_depth = None
        self._record_pub_color = None
        self._record_frame_counter = 0
        if not mockbuild:
            self._record_pub_depth = self.node.create_publisher(CompressedImage, RECORD_DEPTH_TOPIC, 2)
            self._record_pub_color = self.node.create_publisher(CompressedImage, RECORD_COLOR_TOPIC, 2)
            torch.set_default_dtype(dtype)

    def _stamp_msg(self, msg):
        """Stamp with the node clock at publication."""
        now = self.node.get_clock().now().to_msg()
        msg.header.stamp.sec = now.sec
        msg.header.stamp.nanosec = now.nanosec
        msg.header.frame_id = self.frame_id if self.frame_id is not None else ""
        return msg

    def _publish_recorded_frames(self):
        """Publish the throttled, compressed depth and colour frames this tick evaluated (see RECORD_*).

        Called before any accept/reject decision, so rejected frames are recorded too. Never raises.
        """
        if RECORD_DEPTH_EVERY_N <= 0 and RECORD_COLOR_EVERY_N <= 0:
            return
        self._record_frame_counter += 1
        n = self._record_frame_counter
        try:
            if RECORD_DEPTH_EVERY_N > 0 and n % RECORD_DEPTH_EVERY_N == 0 and self._record_pub_depth is not None:
                d = self._internal_last_depth_image
                if d.ndim == 2:
                    # metres -> uint16 millimetres; decode with cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                    mm = torch.clamp(d * 1000.0, 0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
                    ok, buf = cv2.imencode(".png", mm, [cv2.IMWRITE_PNG_COMPRESSION, RECORD_PNG_LEVEL])
                    if ok:
                        msg = self._stamp_msg(CompressedImage())
                        msg.format = "16UC1; png"
                        msg.data = buf.tobytes()
                        self._record_pub_depth.publish(msg)
            if RECORD_COLOR_EVERY_N > 0 and n % RECORD_COLOR_EVERY_N == 0 and self._record_pub_color is not None:
                c = self._internal_last_image
                if c.ndim == 3 and c.shape[2] == 3:
                    bgr = torch.clamp(c, 0, 255).to(torch.uint8).cpu().numpy()
                    ok, buf = cv2.imencode(
                        ".jpg",
                        bgr,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            RECORD_JPEG_QUALITY,
                            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                            cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
                        ],
                    )
                    if ok:
                        msg = self._stamp_msg(CompressedImage())
                        msg.format = "bgr8; jpeg q%d 4:4:4" % RECORD_JPEG_QUALITY
                        msg.data = buf.tobytes()
                        self._record_pub_color.publish(msg)
        except Exception as e:  # noqa: BLE001
            print("Frame recording failed (run continues):", e)

    def obtain_measurements(self) -> bool:
        if (
            self._internal_timestep <= 0
            or (not self._internal_first_info_received)
            or self._internal_timestep_depth <= 0
        ):
            print(
                "No camera input yet!",
                self._internal_timestep,
                self._internal_first_info_received,
                self._internal_timestep_depth,
            )
            return False
        if self.img_shape is None:
            self.img_dependent_init(self._internal_last_image)
        self._publish_recorded_frames()

        # the believed object position projected into the image, used to seed the segmentation
        connection = self.connections["Keypoints2ObjectHeight"]
        if (
            (not connection.connected_quantities_initialized["object_height"])
            or (not connection.connected_quantities_initialized["object_position"])
            or (not connection.connected_quantities_initialized["pose_ee"])
        ):
            starting_pos = self.last_starting_pos
            estimated_dist = -1.0
        else:
            with connection.lock:
                object_pos2d = connection.connected_quantities["object_position"][:2]
                object_height = connection.connected_quantities["object_height"]
                pose_ee = connection.connected_quantities["pose_ee"]
            object_3d_homog = torch.cat(
                [object_pos2d, object_height, torch.ones(1, dtype=self.dtype, device=self.device)]
            )
            relative_pos_in_frame = torch.einsum(
                "lk,ki,ij,j->l",
                self._internal_P,
                self.H_ee_to_cam_inv,
                homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                object_3d_homog,
            )
            estimated_dist = relative_pos_in_frame[2]
            starting_pos = torch.round(relative_pos_in_frame[:2] / estimated_dist * self.scaling_factor)
            if torch.any(torch.isnan(starting_pos)) or torch.any(torch.isinf(starting_pos)):
                starting_pos = self.last_starting_pos
            else:
                starting_pos = starting_pos.cpu().numpy().astype(int)

        self.last_starting_pos = starting_pos
        edge_band_width = 2
        contour = None
        mask = None
        keypoints_partial_in_CF = torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
        cause = 0
        seed_outside = (
            starting_pos[0] < edge_band_width
            or starting_pos[1] < edge_band_width
            or starting_pos[0] > self.img_shape[1] - edge_band_width
            or starting_pos[1] > self.img_shape[0] - edge_band_width
        )
        # Zero depth edges and the pusher: the edge band is what separates the object from the table.
        depth_img = self._internal_last_depth_image.cpu().numpy()
        depth_filled = cv2.medianBlur(depth_img, 5)
        depth_filled = cv2.bilateralFilter(depth_filled, 5, 5 * 2, 5 / 2, borderType=cv2.BORDER_CONSTANT)
        dx = cv2.Sobel(depth_filled, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth_filled, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(dx, dy)
        edge_thresh = 0.017
        edge_mask = grad_mag > edge_thresh
        depth_img[edge_mask] = 0
        depth_img[self.pusher_mask > 0] = 0

        use_detector = self._detector_ready()
        # NaN unless this frame's detection sets it
        self._last_object_angle_meas = float("nan")
        if seed_outside and not use_detector:
            keypoints_in_CF = torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
            contour = None
            keypoints_partial_in_CF = torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
            cause = 32
        else:
            if use_detector:
                if seed_outside:
                    cause |= 32
                res = self._detector_frame(depth_img)
            else:
                # flood fill seeded at the belief
                starting_pos = self.seed_tuple(starting_pos)
                res = self.process_img_around_point(depth_img, estimated_dist, starting_pos)
            (
                center_no_in_frame,
                contour,
                contour_wrong_size,
                depths,
                keypoints,
                keypoints_torch,
                mask,
                median_depth,
                too_few_valid_points,
                wrong_expected_dist,
            ) = res
            self.last_starting_pos = starting_pos

            if contour is not None:
                img_rgb = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                img_rgb[edge_mask] = (255, 0, 0)
                if self.frame_rejects(too_few_valid_points, contour_wrong_size):
                    img_rgb[mask > 0] = (125, 0, 0)
                cv2.drawMarker(img_rgb, (starting_pos[0], starting_pos[1]), color=(0, 0, 255), thickness=2)
                cv2.drawContours(img_rgb, [contour], -1, (0, 255, 0), 2)
                for point in keypoints:
                    x, y = point.astype(int)
                    cv2.circle(img_rgb, (x, y), 4, (0, 0, 255), -1)

                cv2.imshow("mask2", img_rgb)
                cv2.waitKey(1)

                # The contour keypoints in the camera frame, published whether or not the frame is accepted. A short
                # contour can have fewer than NUM_KEYPOINTS points; the rest stays NaN.
                n_kp = int(min(keypoints_torch.shape[0], depths.shape[0], NUM_KEYPOINTS))
                keypoints_partial_in_CF = torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
                if n_kp > 0:
                    partial_rays = torch.einsum(
                        "ij,bj->bi",
                        self.P_inv,
                        torch.cat(
                            [
                                keypoints_torch[:n_kp] / self.scaling_factor,
                                torch.ones(n_kp, 1, dtype=self.dtype, device=self.device),
                            ],
                            dim=1,
                        ),
                    )[:, :3]
                    keypoints_partial_in_CF[:n_kp] = partial_rays * depths[:n_kp].unsqueeze(1)

                cause = (cause & 32) + (
                    2 * int(bool(wrong_expected_dist))
                    + 4 * int(bool(center_no_in_frame))
                    + 8 * int(bool(too_few_valid_points))
                    + 16 * int(bool(contour_wrong_size))
                )

                if self.frame_rejects(too_few_valid_points, contour_wrong_size):
                    keypoints_in_CF = torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
                else:
                    rays = torch.einsum(
                        "ij,bj->bi",
                        self.P_inv,
                        torch.concat(
                            [
                                keypoints_torch / self.scaling_factor,
                                torch.ones(keypoints_torch.shape[0], 1, dtype=self.dtype, device=self.device),
                            ],
                            dim=1,
                        ),
                    )[:, :3]
                    # a short contour leaves the remaining rows NaN
                    keypoints_in_CF = torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
                    n_kp = min(keypoints_torch.shape[0], NUM_KEYPOINTS)
                    keypoints_in_CF[:n_kp] = (rays * depths.unsqueeze(1))[:n_kp]
            else:
                keypoints_in_CF = torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
                keypoints_partial_in_CF = torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
                cause = 1

        self.quantities["keypoints_in_CF"] = keypoints_in_CF
        self.quantities["keypoints_partial_in_CF"] = keypoints_partial_in_CF
        self.quantities["keypoints_reject_cause"] = torch.tensor([float(cause)], dtype=self.dtype, device=self.device)
        self.quantities["keypoints_valid"] = self._last_keypoints_valid.to(dtype=self.dtype, device=self.device)
        self.quantities["object_angle_meas"] = self._angle_meas_quantity(self._last_object_angle_meas, keypoints_in_CF)
        return True

    def _angle_meas_quantity(self, last_angle, keypoints_in_CF):
        """`object_angle_meas` for this tick: the detector's angle, or NaN if this frame's contour was rejected."""
        ok = not bool(torch.any(torch.isnan(keypoints_in_CF)))
        return torch.tensor([last_angle if ok else float("nan")], dtype=self.dtype, device=self.device)

    def _detector_template(self):
        """(ref, scale, min_xy, poly_centered) of `keypoints_registered` for the detector, cached; None until the
        template is registered."""
        conn = self.connections.get("TwoDimKeypointsFrom3D")
        if conn is None:
            return None
        with conn.lock:
            if not conn.connected_quantities_initialized.get("keypoints_registered", False):
                return None
            kr = conn.connected_quantities["keypoints_registered"].detach().clone()
        kr = kr.detach().to(device="cpu", dtype=torch.double)
        if not torch.all(torch.isfinite(kr)) or kr.shape[0] < 3:
            return None
        key = float(torch.sum(torch.abs(kr)))
        if getattr(self, "_det_tpl_key", None) != key:
            self._det_tpl = OD.sdf_from_registered(kr)
            self._det_tpl_key = key
        return self._det_tpl

    def _detector_ready(self):
        """Whether the detector can be used: it needs the registered template, which is only created from contours
        found by the seeded segmentation."""
        if self._detector_disabled:
            return False
        ready = (self._detector_template() is not None) and (self._detector_H_bc() is not None)
        if ready and not getattr(self, "_det_announced", False):
            self._det_announced = True
            print("Object detector active from this tick on (keypoints_registered and pose_ee are available).")
        return ready

    def _detector_H_bc(self):
        """base <- camera, as numpy double. None until pose_ee has arrived."""
        conn = self.connections.get("Keypoints2ObjectHeight")
        if conn is None:
            return None
        with conn.lock:
            if not conn.connected_quantities_initialized.get("pose_ee", False):
                return None
            pose_ee = conn.connected_quantities["pose_ee"].detach().clone()
        pose_ee = pose_ee.detach().to(device="cpu", dtype=torch.double)
        if not torch.all(torch.isfinite(pose_ee)):
            return None
        H_ec = self.connections["TwoDimKeypointsFrom3D"].H_ee_to_cam.detach().to(device="cpu", dtype=torch.double)
        return (exponential_map_se3(pose_ee) @ H_ec).numpy()

    def _detector_frame(self, depth_img):
        """Run the detector on the preprocessed depth frame [m] and return the result in the same 10-tuple format as
        `process_img_around_point`, so that the rest of `obtain_measurements` is shared.

        The published keypoints are the fitted template projected into the image (complete and consistently
        ordered even for partial views), with depths sampled inside the blob.
        """
        NOTHING = (True, None, True, None, None, None, None, None, True, True)
        tpl = self._detector_template()
        H_bc = self._detector_H_bc()
        if tpl is None or H_bc is None:
            self._last_keypoints_valid = torch.zeros(NUM_KEYPOINTS, dtype=self.dtype, device=self.device)
            return NOTHING
        P_np = self._internal_P.detach().to(device="cpu", dtype=torch.double).numpy()
        try:
            det = OD.detect(
                depth_img * 1000.0,
                P_np,
                H_bc,
                tpl,
                prev_theta=getattr(self, "_det_prev_theta", None),
                already_preprocessed=True,
            )
        except Exception as e:  # noqa: BLE001
            # After repeated failures fall back to the seeded segmentation, and say so loudly.
            self._det_fails = getattr(self, "_det_fails", 0) + 1
            if self._det_fails >= 10:
                self._detector_disabled = True
                print("=" * 78)
                print("OBJECT DETECTOR DISABLED after 10 consecutive failures: %s" % e)
                print("The sensor has fallen back to the belief-seeded segmentation.")
                print("")
                print("ORIENTATION IS NOW OPEN LOOP: the detector is the only source of orientation measurements.")
                print("ABORT THE RUN.")
                print("=" * 78)
            self._last_keypoints_valid = torch.zeros(NUM_KEYPOINTS, dtype=self.dtype, device=self.device)
            return NOTHING
        self._det_fails = 0

        if det is None:
            self._last_keypoints_valid = torch.zeros(NUM_KEYPOINTS, dtype=self.dtype, device=self.device)
            return NOTHING

        # The raw proposal feeds the detector's continuity rule next frame; the hysteresis decides what is published.
        if not hasattr(self, "_det_hyst"):
            self._det_hyst = OD.OrientationHysteresis()
        if det.theta_reliable:
            det.theta = self._det_hyst.update(det.theta)
            self._det_prev_theta = det.theta
        self._det_last = det

        # det.theta rotates base -> template, so the object's angle is its negative (wrapped to [-pi, pi)).
        self._last_object_angle_meas = wrap_to_pi(-det.theta) if det.theta_reliable else float("nan")

        h, w = depth_img.shape
        mask = np.asarray(det.mask, dtype=np.uint8)
        use_tpl = det.template_px is not None
        kp = np.asarray(det.template_px if use_tpl else det.keypoints_px, dtype=np.float32)

        # Depth at the projected pixels, only where they lie on the blob (elsewhere it would be the table's depth);
        # the remaining points take the median. The pixel coordinates themselves are not clipped.
        rows = np.clip(kp[:, 1].astype(int), 0, h - 1)
        cols = np.clip(kp[:, 0].astype(int), 0, w - 1)
        in_img = (kp[:, 0] >= 0) & (kp[:, 0] < w) & (kp[:, 1] >= 0) & (kp[:, 1] < h)
        on_blob = in_img & (mask[rows, cols] > 0)
        raw_d = np.asarray(depth_img[rows, cols], dtype=np.float32) * on_blob
        if not np.any(raw_d > 0):
            bd = depth_img[mask > 0]
            bd = bd[bd > 0]
            if bd.size == 0:
                self._last_keypoints_valid = torch.zeros(NUM_KEYPOINTS, dtype=self.dtype, device=self.device)
                return NOTHING
            raw_d = np.full(len(kp), float(np.median(bd)), dtype=np.float32)
        depths = torch.from_numpy(raw_d).to(dtype=self.dtype, device=self.device)
        median_depth = torch.nanmedian(depths[depths > 0])
        depths = torch.where(depths > 0, depths, median_depth)
        keypoints_torch = torch.from_numpy(kp).to(dtype=self.dtype, device=self.device)

        if use_tpl:
            # every point of the fitted template lies on the object boundary, none on the frame's cut
            valid = np.ones(len(kp), dtype=bool)
        else:
            valid = np.asarray(det.valid_mask)
        self._last_keypoints_valid = torch.from_numpy(np.asarray(valid)).to(dtype=self.dtype, device=self.device)
        return (False, det.contour, False, depths, kp, keypoints_torch, mask, median_depth, False, False)

    @staticmethod
    def seed_tuple(sp):
        """A seed as a plain (int, int), so seeds of different types can be compared."""
        return (int(sp[0]), int(sp[1]))

    @staticmethod
    def frame_rejects(too_few_valid_points, contour_wrong_size):
        """Whether a segmented contour is discarded. (The belief-based checks are recorded but do not reject.)"""
        if too_few_valid_points:
            return True
        return bool(contour_wrong_size)

    def process_img_around_point(self, depth_img, estimated_dist, starting_pos):
        """Flood-fill segmentation around a seed. Returns (center_no_in_frame, contour, contour_wrong_size, depths,
        keypoints, keypoints_torch, mask, median_depth, too_few_valid_points, wrong_expected_dist)."""
        mask, contour, average_depth = self.segment_object(
            depth_img, np.array([starting_pos[0], starting_pos[1], 50, 50]), depth_tol=0.01, min_size=100
        )
        if contour is None:
            self._last_keypoints_valid = torch.zeros(NUM_KEYPOINTS, dtype=self.dtype, device=self.device)
            return True, contour, True, None, None, None, None, None, True, True

        keypoints, contour = self.contour_with_fixed_keypoints(contour, n_keypoints=NUM_KEYPOINTS)

        depths = self._internal_last_depth_image[keypoints[:, 1], keypoints[:, 0]]
        median_depth = torch.nanmedian(depths[depths != 0])
        depths[depths == 0] = median_depth
        wrong_expected_dist = torch.abs(torch.mean(depths, dim=0) - estimated_dist) > 0.04 and estimated_dist != -1

        edge_band_width = 10
        center_no_in_frame = (
            starting_pos[0] < edge_band_width
            or starting_pos[1] < edge_band_width
            or starting_pos[0] > self.img_shape[1] - edge_band_width
            or starting_pos[1] > self.img_shape[0] - edge_band_width
        )

        # Keypoints on the image border lie on the frame's cut, not on the object: mark them, and reject the frame
        # only if too few others remain.
        keypoints_torch = torch.from_numpy(keypoints).to(dtype=self.dtype, device=self.device)
        on_border = torch.logical_or(
            torch.logical_or(keypoints_torch[:, 0] < edge_band_width, keypoints_torch[:, 1] < edge_band_width),
            torch.logical_or(
                keypoints_torch[:, 0] > self.img_shape[1] - edge_band_width,
                keypoints_torch[:, 1] > self.img_shape[0] - edge_band_width,
            ),
        )
        self._last_keypoints_valid = torch.logical_not(on_border).to(dtype=self.dtype)
        too_few_valid_points = int(torch.sum(self._last_keypoints_valid)) < MIN_VALID_KEYPOINTS

        area = cv2.contourArea(contour)
        contour_wrong_size = area > SEED_AREA_PX[1] or area < SEED_AREA_PX[0]
        return (
            center_no_in_frame,
            contour,
            contour_wrong_size,
            depths,
            keypoints,
            keypoints_torch,
            mask,
            median_depth,
            too_few_valid_points,
            wrong_expected_dist,
        )

    def img_dependent_init(self, img):
        self.img_shape = img.shape[-3:-1]
        self.P_inv = torch.inverse(self._internal_P[:3, :3])
        self.H_ee_to_cam_inv = homogeneous_transform_inverse(self.connections["Keypoints2ObjectHeight"].H_ee_to_cam)
        self.last_starting_pos = (int(img.shape[1] / 2), int(img.shape[0] / 2))

    def initial_definitions(self, quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None):
        self.quantities["keypoints_in_CF"] = torch.zeros(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device)
        self.quantities["keypoints_partial_in_CF"] = (
            torch.ones(NUM_KEYPOINTS, 3, dtype=self.dtype, device=self.device) * torch.nan
        )
        self.quantities["keypoints_reject_cause"] = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.quantities["keypoints_valid"] = torch.ones(NUM_KEYPOINTS, dtype=self.dtype, device=self.device)
        # the detector's object angle in the base frame, NaN when not measured this frame
        self.quantities["object_angle_meas"] = torch.ones(1, dtype=self.dtype, device=self.device) * torch.nan
        self._last_object_angle_meas = float("nan")
        self.local_quantity_names_and_shapes = {
            "keypoints_in_CF": (NUM_KEYPOINTS, 3),
            # the contour found this frame, published even if it was rejected
            "keypoints_partial_in_CF": (NUM_KEYPOINTS, 3),
            "keypoints_reject_cause": (1,),
            "keypoints_valid": (NUM_KEYPOINTS,),
            "object_angle_meas": (1,),
        }

    def segment_object(self, depth, coarse_bbox, depth_tol=0.05, min_size=500):
        """
        depth       : HxW depth map in meters (float)
        coarse_bbox : (x, y, w, h) rectangle roughly around the object
        depth_tol   : max depth difference from seed (meters)
        min_size    : minimum number of pixels to keep
        Returns:
            object_mask : HxW uint8 mask of the object
            contour     : largest 2D contour of the object
        """
        H, W = depth.shape
        x, y, w, h = coarse_bbox

        # seed: first pixel with valid depth in the ROI
        roi_depth = depth[y : y + h, x : x + w]
        valid_mask = roi_depth > 0
        if not np.any(valid_mask):
            return np.zeros_like(depth, np.uint8), None, None
        seed_local = np.argwhere(valid_mask)[0]
        seed = (x + seed_local[1], y + seed_local[0])
        seed_val = depth[seed[1], seed[0]]

        depth_u16 = (depth * 1000).astype(np.float32)  # meters -> mm
        loDiff = int(depth_tol * 1000)
        upDiff = int(depth_tol * 1000)

        mask = np.zeros((H + 2, W + 2), np.uint8)  # floodFill requires a 1 px border

        _, _, _, _ = cv2.floodFill(
            depth_u16, mask, seedPoint=(seed[0], seed[1]), newVal=0, loDiff=loDiff, upDiff=upDiff, flags=4 | (255 << 8)
        )

        object_mask = (mask[1:-1, 1:-1] > 0).astype(np.uint8)

        # fill small holes and remove speckles
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(object_mask, connectivity=8)
        cleaned_mask = np.zeros_like(object_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 1
        object_mask = cleaned_mask

        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None

        return object_mask, contour, seed_val

    def contour_with_fixed_keypoints(self, contour, n_keypoints=50, epsilon_factor=0.02):
        """
        contour       : (N,1,2) contour from cv2.findContours
        n_keypoints   : desired number of keypoints
        epsilon_factor: tolerance for approxPolyDP relative to arc length

        Returns:
            fixed_points: (<= n_keypoints, 2) array of keypoints, and the simplified contour
        """
        # simplify the contour, preserving sharp corners
        peri = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :]  # (M,2)

        diffs = np.diff(approx, axis=0, append=approx[:1])
        dists = np.linalg.norm(diffs, axis=1)

        # too many points: drop the ones with the shortest adjacent edges
        if len(approx) >= n_keypoints:
            to_remove = len(approx) - n_keypoints
            dists_next = np.roll(dists, axis=0, shift=1)
            dists_combined = dists_next + dists
            idxs_to_remove = np.argsort(dists_combined)[:to_remove]
            resampled = []
            for i, p in enumerate(approx):
                if i not in idxs_to_remove:
                    resampled.append(p)
            resampled = np.array(resampled, dtype=np.float32)
            return resampled, approx

        # too few points: insert points along long edges
        cumulative = np.concatenate([[0], np.cumsum(dists)])
        total_length = cumulative[-1]

        ideal_length = total_length / n_keypoints

        resampled = []
        for j in range(1, len(approx)):
            resampled.append(approx[j - 1])
            if dists[j - 1] > ideal_length:
                p1, p2 = approx[j - 1], approx[j]
                n_extra_points = int(dists[j - 1] / ideal_length)
                for i in range(n_extra_points):
                    if len(resampled) + len(approx[j:] + 1) < n_keypoints:
                        resampled.append(p1 + (p2 - p1) * (i + 1) / (n_extra_points + 1))

        resampled.append(approx[-1])
        resampled = np.array(resampled, dtype=np.float32)

        return np.array(resampled, dtype=np.float32), approx
