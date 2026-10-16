"""
Sensors for Drawer Manipulation with AICON

Author: Vito Mengers
License: MIT
"""

__author__ = "Vito Mengers"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import os
import time
from typing import Dict, Union, Tuple, Type

import cv2 as cv
import numpy as np
import torch
import torchvision

from aicon_base.connections import ActiveInterconnection
from .util import get_sine_of_angles
from aicon_base.math.se3_and_SE3 import exponential_map_se3, homogeneous_transform_inverse
from aicon_real_robots.model_paths import os2d_root, tapir_checkpoint
from aicon_real_robots.sensors.camera_sensors import ROSCameraRGBSensor, ROSCameraRGBDSensor

# The learned models are only needed on the machine that runs the camera sensors.
try:
    from tapnet.torch import tapir_model
except ImportError:
    tapir_model = None
try:
    from os2d.config import cfg as os2d_config
    from os2d.structures.feature_map import FeatureMapSize
    from os2d.modeling.model import build_os2d_from_config
    from os2d.utils import read_image, get_image_size_after_resize_preserving_aspect_ratio
except ImportError:
    os2d_config = None


OS2D_TEMPLATE_IMAGES = {
    "zyxel": ["zyxel1.png", "zyxel2.png", "zyxel3.png"],
    "realsense": ["realsense1.png", "realsense2.png", "realsense3.png"],
    "orange": ["orange1.png", "orange2.png"],
    "rubiks": ["rubiks1.png", "rubiks2.png", "rubiks3.png", "rubiks4.png"],
}


class DLObjectInCameraSensor(ROSCameraRGBSensor):
    state_dim = 4

    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        ros_topic_base: str,
        object_name: str = "drawer",
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        scaling_factor: float = 0.2,
        template_object: str = "rubiks",
    ):
        if not mockbuild:
            torch.set_default_dtype(dtype)
            torch.set_grad_enabled(False)
            self.dtype = dtype
            self.device = device
            os2d_path = os2d_root()
            self.cfg = os2d_config
            self.cfg.merge_from_file(os.path.join(os2d_path, "experiments", "config_training.yml"))
            self.cfg.merge_from_list(
                [
                    "model.use_inverse_geom_model",
                    "True",
                    "model.use_simplified_affine_model",
                    "False",
                    "model.backbone_arch",
                    "ResNet50",
                    "train.do_training",
                    "False",
                    "init.model",
                    os.path.join(os2d_path, "models", "os2d_v2-init.pth"),
                ]
            )
            self.cfg.is_cuda = device is not None and device.type == "cuda"
            self.cfg.eval.dataset_scales = [1900]
            self.cfg.model.class_image_size = 250
            self.score_thresh = 0.4
            self.cfg.freeze()
            self.net, self.box_coder, _, img_normalization, _ = build_os2d_from_config(self.cfg, device=self.device)
            self.net.to(device)
            self.transform_image = torchvision.transforms.Compose(
                [torchvision.transforms.Normalize(img_normalization["mean"], img_normalization["std"])]
            )
            # Template images of the object on the drawer, in <os2d>/data/manipulation_objects/
            class_images = [
                read_image(os.path.join(os2d_path, "data", "manipulation_objects", f))
                for f in OS2D_TEMPLATE_IMAGES[template_object]
            ]
            self.class_ids = list(range(len(class_images)))
            class_images_th = []
            for class_image in class_images:
                class_image_th = self.preprocess_img_for_os2d(
                    torchvision.transforms.ToTensor()(class_image), size=self.cfg.model.class_image_size
                )
                class_images_th.append(class_image_th)
            class_feature_maps = self.net.net_label_features(class_images_th)
            self.class_head = self.net.os2d_head_creator.create_os2d_head(class_feature_maps)
            self.class_head.to(device)
        self.object_name = object_name
        super().__init__(
            name, connections, ros_topic_base, dtype, device, mockbuild=mockbuild, scaling_factor=scaling_factor
        )

    def preprocess_img_for_os2d(self, class_image, size):
        h, w = get_image_size_after_resize_preserving_aspect_ratio(
            h=class_image.shape[1], w=class_image.shape[2], target_size=size
        )
        class_image = torchvision.transforms.Resize((h, w))(class_image)
        class_image_th = self.transform_image(class_image).to(device=self.device, dtype=self.dtype)
        print(class_image_th.shape)
        return class_image_th

    def obtain_measurements(self) -> bool:
        if self._internal_timestep <= 0 or (not self._internal_first_info_received):
            return False
        start = time.time()
        img_orig = self._internal_last_image
        img_orig = torch.flip(img_orig, (2,)) / 255.0
        img_orig = torch.moveaxis(img_orig, 2, 0)  # expects CxHxW
        img = self.preprocess_img_for_os2d(img_orig, self.cfg.eval.dataset_scales[0])
        img = img.unsqueeze(0)
        feature_map = self.net.net_feature_maps(img)
        loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = self.net(
            class_head=self.class_head, feature_maps=feature_map
        )

        image_loc_scores_pyramid = [loc_prediction_batch[0]]
        image_class_scores_pyramid = [class_prediction_batch[0]]
        img_size_pyramid = [FeatureMapSize(img=img)]
        transform_corners_pyramid = [transform_corners_batch[0]]
        boxes = self.box_coder.decode_pyramid(
            image_loc_scores_pyramid,
            image_class_scores_pyramid,
            img_size_pyramid,
            self.class_ids,
            nms_iou_threshold=self.cfg.eval.nms_iou_threshold,
            nms_score_threshold=self.cfg.eval.nms_score_threshold,
            transform_corners_pyramid=transform_corners_pyramid,
        )
        scores = boxes.get_field("scores")
        boxes = boxes.resize(
            FeatureMapSize(
                h=img_orig.shape[1],
                w=img_orig.shape[2],
            )
        ).bbox_xyxy
        print(scores)
        boxes = boxes[scores >= self.score_thresh]
        scores = scores[scores >= self.score_thresh]
        if torch.sum(scores) == 0:
            best_fit_idx = -1
        else:
            c = self.connections["ProjectiveGeometry"]
            if c.connected_quantities_initialized["position_drawer"] and c.connected_quantities_initialized["pose_ee"]:
                pose_ee = c.connected_quantities["pose_ee"]
                position_drawer = c.connected_quantities["position_drawer"]
                relative_pos_in_frame = torch.einsum(
                    "lk,ki,ij,j->l",
                    self._internal_P,
                    homogeneous_transform_inverse(c.H_ee_to_cam),
                    homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
                    torch.cat([position_drawer, torch.ones(1, dtype=self.dtype, device=self.device)]),
                )
                estimated_dist = relative_pos_in_frame[2]
                estimated_loc = relative_pos_in_frame[:2] / estimated_dist * self.scaling_factor
                locs = torch.cat([(boxes[:, 0:1] + boxes[:, 2:3]) / 2, (boxes[:, 1:2] + boxes[:, 3:4]) / 2], dim=1)
                dists = torch.sqrt(torch.sum(torch.pow(locs - estimated_loc.unsqueeze(0), 2), dim=1))
                dist_score = 100.0 / dists
                combined_scores = dist_score + scores
                combined_scores[dists > 1.2] = 0.0

                if torch.sum(combined_scores) > 0:
                    best_fit_idx = torch.argmax(combined_scores)
                else:
                    best_fit_idx = -1
            else:
                best_fit_idx = torch.argmax(scores)

        img_anot = torchvision.utils.draw_bounding_boxes(img_orig, boxes, colors="red")
        if best_fit_idx != -1:
            best_box = boxes[best_fit_idx]
            img_anot = torchvision.utils.draw_bounding_boxes(
                img_anot, boxes[best_fit_idx : (best_fit_idx + 1)], colors="blue", width=3
            )
        else:
            best_box = torch.zeros(2, dtype=self.dtype, device=self.device) * torch.nan
        img_anot = torch.moveaxis(img_anot, 0, 2)  # expects CxHxW
        img_anot = img_anot.cpu().numpy()
        cv.imshow("mask", img_anot)
        cv.waitKey(1)

        print(self.name + " Time", time.time() - start)
        self.timestamp = torch.clone(self._internal_timestep)
        self.quantities["coarse_bounding_box_" + self.object_name] = best_box / self.scaling_factor

        return True

    def initial_definitions(self, quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None):
        self.quantities["coarse_bounding_box_" + self.object_name] = torch.empty(
            self.state_dim, dtype=self.dtype, device=self.device
        )
        self.local_quantity_names_and_shapes = {
            "coarse_bounding_box_" + self.object_name: (-1,),
        }


class CameraPointTracking(ROSCameraRGBDSensor):
    def __init__(
        self,
        name: str,
        connections: Dict[str, Type[ActiveInterconnection]],
        ros_topic_base_color: str,
        ros_topic_base_depth: str,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        mockbuild: bool = False,
        scaling_factor: float = 0.2,
    ):
        if not mockbuild:
            torch.set_default_dtype(dtype)
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
        self.model = None
        self.img_shape = None
        if not mockbuild:
            self.init_model()

    def init_model(self):
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)

        self.model.load_state_dict(torch.load(tapir_checkpoint()))
        self.model.to(self.device)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.new_points = True
        self.get_feature_grids = lambda x: self.model.get_feature_grids(x, is_training=False)
        self.steps = 0

    def online_model_init_new_points(self, frame, feature_grids, query_points):
        """Initialize query features for the query points."""
        query_features = self.model.get_query_features(
            frame,
            is_training=False,
            query_points=query_points,
            feature_grids=feature_grids,
        )
        return query_features

    def online_model_predict(self, feature_grids, query_features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        trajectories = self.model.estimate_trajectories(
            self.img_shape,
            is_training=False,
            feature_grids=feature_grids,
            query_features=query_features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        # Take only the predictions for the final resolution.
        # For running on higher resolution, it's typically better to average across
        # resolutions.
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        uncertainty = trajectories["expected_dist"][-1]
        visibles = (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(uncertainty))
        return tracks, visibles, causal_context

    def get_point_grid(self, N_Points):
        rows = int(np.sqrt(N_Points))
        columns = int(np.ceil(N_Points / rows))
        dx = self.img_shape[1] / (columns + 1)
        dy = self.img_shape[0] / (rows + 1)
        i = torch.arange(rows, device=self.device)  # Row indices
        j = torch.arange(columns, device=self.device)  # Column indices
        grid_x, grid_y = torch.meshgrid(j + 1, i + 1, indexing="xy")
        points_x = (grid_x * dx).flatten()
        points_y = (grid_y * dy).flatten()

        # Stack and select only N points
        points = torch.stack((points_y, points_x), dim=1)[:N_Points]
        return points

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
        connection = self.connections["ProjectiveGeometry"]
        if not connection.connected_quantities_initialized["coarse_bounding_box_drawer"]:
            print("No bounding box yet!")
            return False
        start = time.time()
        img = self.preprocess_img(self._internal_last_image)
        feature_grids = self.get_feature_grids(img)

        if self.img_shape is None:
            self.img_dependent_init(connection, feature_grids, img)

        if connection.connected_quantities_initialized["pose_ee"]:
            estimated_locs, estimated_visible, visible_conservative = self.get_currently_tracked_in_frame(connection)
            estimated_visible_conservative = visible_conservative
            should_track = torch.clone(self._internal_drawer_point_mask)
            if should_track.shape[0] > estimated_visible.shape[0]:
                estimated_visible = torch.cat(
                    [
                        estimated_visible,
                        torch.ones(should_track.shape[0] - estimated_visible.shape[0], device=self.device, dtype=bool),
                    ]
                )
                estimated_visible_conservative = torch.cat(
                    [
                        estimated_visible_conservative,
                        torch.zeros(should_track.shape[0] - estimated_visible.shape[0], device=self.device, dtype=bool),
                    ]
                )
            should_track[estimated_visible] = True
        else:
            should_track = torch.ones_like(self._internal_drawer_point_mask, dtype=bool)

        current_locations, currently_visible = self.track(feature_grids, should_track)

        drawer_meas, drawer_depth = self.get_drawer_meas(current_locations, currently_visible, should_track)

        viz_points = current_locations[currently_visible].unsqueeze(0)
        img_anot = torchvision.utils.draw_keypoints(
            torch.moveaxis((img[0, 0] + 1) * 0.5, 2, 0), viz_points, colors="red", radius=4
        )
        img_anot = torchvision.utils.draw_keypoints(
            img_anot,
            viz_points[:, self._internal_drawer_point_mask[should_track][currently_visible]],
            colors="blue",
            radius=5,
        )
        img_anot = torch.moveaxis(img_anot, 0, 2)
        img_anot = img_anot.cpu().numpy()
        cv.imshow("mask", img_anot)
        cv.waitKey(1)
        tracked_and_visible = torch.zeros_like(should_track, dtype=torch.bool, device=self.device)
        tracked_and_visible[should_track] = currently_visible
        locs = current_locations[currently_visible]
        locs_long = locs.long()
        depth_values = self._internal_last_depth_image[locs_long[:, 1], locs_long[:, 0]]
        rel_pos = torch.einsum(
            "ij,bj->bi",
            self.P_inv,
            torch.concat(
                [locs / self.scaling_factor, torch.ones((locs.shape[0], 1), dtype=self.dtype, device=self.device)],
                dim=1,
            ),
        )[:, :3]
        norms = torch.norm(rel_pos, dim=1, keepdim=True)
        sin_ax = rel_pos[:, 0:1] / norms
        sin_ay = rel_pos[:, 1:2] / norms
        sin_az = rel_pos[:, 2:3] / norms
        only_angles_sin = torch.concatenate([sin_ax, sin_ay, sin_az], dim=1)

        if connection.connected_quantities_initialized["pose_ee"]:
            all_depth_values = torch.zeros(
                estimated_visible_conservative.shape[0], dtype=self.dtype, device=self.device
            )
            if estimated_locs.shape[0] > 0:
                estimated_locs_long = estimated_locs.long()
                all_depth_values[estimated_visible_conservative] = self._internal_last_depth_image[
                    estimated_locs_long[:, 1], estimated_locs_long[:, 0]
                ]
            self.add_new_points(
                currently_visible, connection, estimated_locs, should_track, feature_grids, img, current_locations
            )
        else:
            all_depth_values = torch.zeros(only_angles_sin.shape[0], dtype=self.dtype, device=self.device)

        print(all_depth_values, depth_values)
        print(self.name + " Time", time.time() - start)
        self.quantities["relative_position_in_CF_environment_points"] = only_angles_sin
        self.quantities["depth_in_CF_environment_points"] = depth_values
        self.quantities["present_measurement_points_mask"] = tracked_and_visible
        self.quantities["relative_position_in_CF_drawer"] = drawer_meas
        self.quantities["depth_in_CF_drawer"] = drawer_depth
        self.quantities["all_depth_values"] = all_depth_values
        torch.cuda.empty_cache()
        return True

    def img_dependent_init(self, c_drawer, feature_grids, img):
        self.img_shape = img.shape[-3:-1]
        self.coarse_grid = self.get_point_grid(128)
        self.fine_grid = self.get_point_grid(2056)
        self.P_inv = torch.inverse(self._internal_P[:3, :3])
        self.H_ee_to_cam_inv = homogeneous_transform_inverse(c_drawer.H_ee_to_cam)
        points = torch.tensor([[1, 1]], device=self.device)
        t = torch.zeros(1, 1, device=self.device)
        query_points = torch.concat([t, points], dim=-1).to(dtype=torch.int)
        query_features = self.online_model_init_new_points(img, feature_grids, query_points[None])
        causal_state = self.model.construct_initial_causal_state(
            query_points.shape[0], len(query_features.resolutions) - 1
        )
        lowres = []
        hires = []
        for i in range(len(query_features.lowres)):
            shape = list(query_features.lowres[i].shape)
            shape[1] = 0
            lowres.append(torch.zeros(shape, device=self.device, dtype=query_features.lowres[i].dtype))
        for i in range(len(query_features.hires)):
            shape = list(query_features.hires[i].shape)
            shape[1] = 0
            hires.append(torch.zeros(shape, device=self.device, dtype=query_features.hires[i].dtype))
        self._internal_query_features = tapir_model.QueryFeatures(
            tuple(lowres), tuple(hires), tuple(query_features.resolutions)
        )
        self._internal_causal_state = []
        for i in range(len(causal_state)):
            self._internal_causal_state.append(dict())
            for k, v in causal_state[i].items():
                shape = list(causal_state[i][k].shape)
                shape[1] = 0
                self._internal_causal_state[i][k] = torch.zeros(
                    shape, device=self.device, dtype=causal_state[i][k].dtype
                )
        self._internal_drawer_point_mask = torch.zeros((0,), dtype=bool, device=self.device)

    def add_new_points(
        self, currently_visible, connection, estimated_locs, should_track, feature_grids, img, current_locations
    ):
        points_general = torch.zeros((0, 2), dtype=self.dtype, device=self.device)
        points_drawer = torch.zeros((0, 2), dtype=self.dtype, device=self.device)
        if self.new_points:
            if estimated_locs.shape[0] > 0:
                # TODO also take those much closer than the closest estimated loc
                # probably easy by just checking local depth value and if it is much lower than estimated one,
                # mask it off
                dists = torch.sqrt(
                    torch.sum(
                        torch.pow(torch.flip(self.coarse_grid, (1,)).unsqueeze(0) - estimated_locs.unsqueeze(1), 2),
                        dim=-1,
                    )
                )
                min_dist = torch.min(dists, dim=0)[0]
                new_mask = min_dist > 20
            elif self._internal_drawer_point_mask.shape[0] == 0:
                new_mask = torch.ones(self.coarse_grid.shape[0], dtype=bool, device=self.device)
            else:
                new_mask = torch.zeros(self.coarse_grid.shape[0], dtype=bool, device=self.device)
            points_general = self.coarse_grid[new_mask]
        if (
            should_track.shape[0] == 0
            or torch.sum(self._internal_drawer_point_mask[should_track][currently_visible]) < 7
        ):
            # we might need to be smarter here, for example by only sampling close to the still there points
            drawer_bounding_box = connection.connected_quantities["coarse_bounding_box_drawer"] * self.scaling_factor
            sample_around_points = True
            current_drawer_points = current_locations[
                torch.logical_and(self._internal_drawer_point_mask[should_track], currently_visible)
            ]
            if drawer_bounding_box.shape[0] == 4:
                current_points_in_box = self.check_if_in_box(
                    drawer_bounding_box, torch.flip(current_drawer_points, (1,)), 0.0
                )
                if torch.all(current_points_in_box) and (
                    torch.sum(self._internal_drawer_point_mask[should_track][currently_visible]) > 3
                    or torch.sum(self._internal_drawer_point_mask) == 0
                ):
                    if current_drawer_points.shape[0] > 0:
                        max_point = torch.max(current_drawer_points, dim=0).values
                        min_point = torch.min(current_drawer_points, dim=0).values
                        drawer_bounding_box = torch.cat(
                            [min_point[0:1], min_point[1:2], max_point[0:1], max_point[1:2]]
                        )
                        in_bounding_box = self.check_if_in_box(drawer_bounding_box, self.fine_grid, 0.1)
                    else:
                        in_bounding_box = self.check_if_in_box(drawer_bounding_box, self.fine_grid, 0.2)
                    if torch.any(in_bounding_box):
                        points_drawer = self.fine_grid[in_bounding_box]
                        sample_around_points = False
            if sample_around_points and current_drawer_points.shape[0] > 3:
                current_drawer_point_mean = torch.mean(current_drawer_points, dim=0)
                dists = torch.sqrt(
                    torch.sum(
                        torch.pow(torch.flip(self.fine_grid, (1,)) - current_drawer_point_mean.unsqueeze(0), 2), dim=-1
                    )
                )
                points_drawer = self.fine_grid[dists < 5]
            # TODO reject points based on dist to median drawer point
        points = torch.cat([points_drawer, points_general])
        if points.shape[0] > 0:
            t = torch.zeros(points.shape[0], 1, device=self.device)
            query_points = torch.concat([t, points], dim=-1).to(dtype=torch.int)
            query_features = self.online_model_init_new_points(img, feature_grids, query_points[None])
            causal_state = self.model.construct_initial_causal_state(
                query_points.shape[0], len(query_features.resolutions) - 1
            )
            lowres = []
            hires = []
            for i in range(len(self._internal_query_features.lowres)):
                lowres.append(torch.cat([self._internal_query_features.lowres[i], query_features.lowres[i]], dim=1))
            for i in range(len(self._internal_query_features.hires)):
                hires.append(torch.cat([self._internal_query_features.hires[i], query_features.hires[i]], dim=1))
            self._internal_query_features = tapir_model.QueryFeatures(
                tuple(lowres), tuple(hires), tuple(self._internal_query_features.resolutions)
            )
            for i in range(len(self._internal_causal_state)):
                for k, v in self._internal_causal_state[i].items():
                    self._internal_causal_state[i][k] = torch.cat([v, causal_state[i][k].to(device=self.device)], dim=1)

            print("New Points", points_drawer.shape[0], points_general.shape[0])
            self._internal_drawer_point_mask = torch.cat(
                [
                    self._internal_drawer_point_mask,
                    torch.ones(points_drawer.shape[0], device=self.device, dtype=bool),
                    torch.zeros(points_general.shape[0], device=self.device, dtype=bool),
                ]
            )
        return points.shape[0]

    def check_if_in_box(self, drawer_bounding_box, points, reduce_factor):
        x_width = drawer_bounding_box[2] - drawer_bounding_box[0]
        y_width = drawer_bounding_box[3] - drawer_bounding_box[1]
        in_bounding_box = torch.logical_and(
            torch.logical_and(
                drawer_bounding_box[1] + reduce_factor * y_width < points[:, 0],
                points[:, 0] < drawer_bounding_box[3] - reduce_factor * y_width,
            ),
            torch.logical_and(
                drawer_bounding_box[0] + reduce_factor * x_width < points[:, 1],
                points[:, 1] < drawer_bounding_box[2] - reduce_factor * x_width,
            ),
        )
        return in_bounding_box

    def get_currently_tracked_in_frame(self, connection):
        environment_point_tracked = connection.connected_quantities["environment_point_tracked"].bool()
        current_env_points = connection.connected_quantities["environment_point_positions"]
        pose_ee = connection.connected_quantities["pose_ee"]
        estimated_pos = torch.einsum(
            "ki,ij,bj->bk",
            self.H_ee_to_cam_inv,
            homogeneous_transform_inverse(exponential_map_se3(pose_ee)),
            torch.cat(
                [current_env_points, torch.ones(current_env_points.shape[0], 1, dtype=self.dtype, device=self.device)],
                dim=1,
            ),
        )
        angle_x = torch.abs(
            torch.arcsin(estimated_pos[:, 0:1] / torch.norm(estimated_pos[:, [0, 2]], keepdim=True, dim=-1))
        )
        angle_y = torch.abs(
            torch.arcsin(estimated_pos[:, 1:2] / torch.norm(estimated_pos[:, [1, 2]], keepdim=True, dim=-1))
        )
        # 0.6 = 69.4 / 180 * PI / 2.0 with 69.4° FOV in x
        # 0.37 = 42.5 / 180 * PI / 2.0 with 42.5° FOV in y
        visible = torch.logical_and(
            torch.logical_and(angle_x.squeeze() < 0.55, angle_y.squeeze() < 0.35), estimated_pos[:, 2] > 0
        )
        estimated_pos_in_frame = torch.einsum("ij,bj->bi", self._internal_P, estimated_pos)
        estimated_loc = estimated_pos_in_frame[visible, :2] / estimated_pos_in_frame[visible, 2:3] * self.scaling_factor
        # safeguards because of asynchronous updating
        visible_conservative = visible
        if environment_point_tracked.shape[0] > visible.shape[0]:
            visible = torch.cat(
                [
                    visible,
                    torch.ones(environment_point_tracked.shape[0] - visible.shape[0], device=self.device, dtype=bool),
                ]
            )
            visible_conservative = torch.cat(
                [
                    visible,
                    torch.zeros(environment_point_tracked.shape[0] - visible.shape[0], device=self.device, dtype=bool),
                ]
            )
        if visible.shape[0] > environment_point_tracked.shape[0]:
            environment_point_tracked = torch.cat(
                [
                    environment_point_tracked,
                    torch.ones(visible.shape[0] - environment_point_tracked.shape[0], device=self.device, dtype=bool),
                ]
            )
        return estimated_loc, torch.logical_and(visible, environment_point_tracked), visible_conservative

    def get_drawer_meas(self, current_locations, currently_visible, should_track):
        mask = torch.logical_and(self._internal_drawer_point_mask[should_track], currently_visible)
        drawer_points = current_locations[mask]
        drawer_points_long = drawer_points.long()
        depths_values = self._internal_last_depth_image[drawer_points_long[:, 1], drawer_points_long[:, 0]]
        depths_values = depths_values[depths_values > 0]
        if depths_values.shape[0] > 2:
            depth = torch.median(depths_values, dim=0)[0]
            depth_factor = depth
        else:
            depth = torch.zeros(1, device=self.device, dtype=self.dtype)
            depth_factor = 1.0
        if drawer_points.shape[0] > 2:
            loc = torch.median(drawer_points, dim=0)[0]
            rel_pos = (
                torch.einsum(
                    "ij,j->i",
                    self.P_inv,
                    torch.concat([loc / self.scaling_factor, torch.ones(1, dtype=self.dtype, device=self.device)]),
                )[:3]
                * depth_factor
            )
            drawer_meas = get_sine_of_angles(rel_pos)
        else:
            drawer_meas = torch.ones(3, dtype=self.dtype, device=self.device) * torch.nan
        return drawer_meas, depth

    def track(self, feature_grids, estimated_visible):
        if torch.sum(estimated_visible) == 0:
            return torch.zeros(0, 2, dtype=self.dtype, device=self.device), torch.zeros(
                0, dtype=bool, device=self.device
            )
        current_causal_state = []
        for i in range(len(self._internal_causal_state)):
            current_causal_state.append({})
            for k, v in self._internal_causal_state[i].items():
                current_causal_state[i][k] = v[:, estimated_visible]
        lowres = []
        hires = []
        for i in range(len(self._internal_query_features.lowres)):
            lowres.append(self._internal_query_features.lowres[i][:, estimated_visible])
        for i in range(len(self._internal_query_features.hires)):
            hires.append(self._internal_query_features.hires[i][:, estimated_visible])
        current_query_features = tapir_model.QueryFeatures(
            tuple(lowres), tuple(hires), tuple(self._internal_query_features.resolutions)
        )
        tracks, visible_likelihood, causal_state = self.online_model_predict(
            feature_grids=feature_grids,
            query_features=current_query_features,
            causal_context=current_causal_state,
        )
        for i in range(len(self._internal_causal_state)):
            for k, v in self._internal_causal_state[i].items():
                self._internal_causal_state[i][k][:, estimated_visible] = causal_state[i][k].to(self.device)
        current_locations = tracks[0, :, 0]
        currently_visible = visible_likelihood.squeeze() > 0.45
        currently_visible[self._internal_drawer_point_mask[estimated_visible]] = (
            visible_likelihood.squeeze()[self._internal_drawer_point_mask[estimated_visible]] > 0.8
        )
        return current_locations, currently_visible

    def preprocess_img(self, img):
        # frames are expected in [1, N, H, W, 3] format with pixel range in [-1, 1], float32
        img = torch.flip(img, (2,)) / 255.0 * 2 - 1
        img = img.unsqueeze(0).unsqueeze(0)
        return img

    def initial_definitions(self, quantity_names_and_shapes: Union[Dict[str, Tuple[int]], None] = None):
        self.quantities["relative_position_in_CF_environment_points"] = torch.empty(
            0, 3, dtype=self.dtype, device=self.device
        )
        self.quantities["present_measurement_points_mask"] = torch.empty(0, dtype=self.dtype, device=self.device)
        self.quantities["relative_position_in_CF_drawer"] = torch.empty(3, dtype=self.dtype, device=self.device)
        self.quantities["depth_in_CF_drawer"] = torch.empty(1, dtype=self.dtype, device=self.device)
        self.quantities["depth_in_CF_environment_points"] = torch.empty(0, dtype=self.dtype, device=self.device)
        self.quantities["all_depth_values"] = torch.empty(0, dtype=self.dtype, device=self.device)
        self.local_quantity_names_and_shapes = {
            "relative_position_in_CF_environment_points": (-1,),
            "present_measurement_points_mask": (-1,),
            "depth_in_CF_environment_points": (-1,),
            "relative_position_in_CF_drawer": (3,),
            "depth_in_CF_drawer": (1,),
            "all_depth_values": (1,),
        }
