# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of Instant NGP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import nerfacc
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


@dataclass
class SplitSpaceInstantNGPModelConfig(ModelConfig):
    """Split Space Instant NGP Model Config"""

    _target: Type = field(
        default_factory=lambda: split_space_NGPModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4
    """Levels of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""


class split_space_NGPModel():
    """Split Space Instant NGP model
       basically split space into n^2 sub-space base on aabb splitting
       the problem is how to make it parallel

    Args:
        config: Split Space Instant NGP configuration to instantiate model
    """

    config: SplitSpaceInstantNGPModelConfig
    field: TCNNNerfactoField

    def __init__(self, config: SplitSpaceInstantNGPModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        
        self.aabb_scale = self.scene_box.aabb[1][1]
        
        self.scene_box_00 = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [0, 0, aabb_scale]], dtype=torch.float32
            )
        )
        self.scene_box_01 = SceneBox(
            aabb=torch.tensor(
                [[0, 0, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
        self.scene_box_10 = SceneBox(
            aabb=torch.tensor(
                [[0, -aabb_scale, -aabb_scale], [aabb_scale, 0, aabb_scale]], dtype=torch.float32
            )
        )
        self.scene_box_11 = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, 0, -aabb_scale], [0, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )
                
        self.field_00 = TCNNNerfactoField(
            aabb=self.scene_box_00.aabb,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
        )

        self.field_01 = TCNNNerfactoField(
            aabb=self.scene_box_01.aabb,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
        )

        self.field_10 = TCNNNerfactoField(
            aabb=self.scene_box_10.aabb,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
        )

        self.field_11 = TCNNNerfactoField(
            aabb=self.scene_box_11.aabb,
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
            spatial_distortion=scene_contraction,
        )

        self.scene_aabb_00 = Parameter(self.scene_box_00.aabb.flatten(), requires_grad=False)
        self.scene_aabb_01 = Parameter(self.scene_box_01.aabb.flatten(), requires_grad=False)
        self.scene_aabb_10 = Parameter(self.scene_box_10.aabb.flatten(), requires_grad=False)
        self.scene_aabb_11 = Parameter(self.scene_box_11.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid in a split space
            self.config.render_step_size = ((self.scene_aabb_00[3:] - self.scene_aabb_00[:3]) ** 2).sum().sqrt().item() / 1000

        # Occupancy Grid.
        self.occupancy_grid_00 = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb_00,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )
        self.occupancy_grid_01 = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb_01,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )
        self.occupancy_grid_10 = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb_10,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )
        self.occupancy_grid_11 = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb_11,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler_00 = VolumetricSampler(
            occupancy_grid=self.occupancy_grid_00,
            density_fn=self.field_00.density_fn,
        )

        self.sampler_01 = VolumetricSampler(
            occupancy_grid=self.occupancy_grid_01,
            density_fn=self.field_01.density_fn,
        )
        self.sampler_10 = VolumetricSampler(
            occupancy_grid=self.occupancy_grid_10,
            density_fn=self.field_10.density_fn,
        )
        self.sampler_11 = VolumetricSampler(
            occupancy_grid=self.occupancy_grid_11,
            density_fn=self.field_11.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid_00.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field_00.density_fn(x) * self.config.render_step_size,
            )
            self.occupancy_grid_01.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field_01.density_fn(x) * self.config.render_step_size,
            )
            self.occupancy_grid_10.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field_10.density_fn(x) * self.config.render_step_size,
            )
            self.occupancy_grid_11.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field_11.density_fn(x) * self.config.render_step_size,
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_00 is None or self.field_01 is None or self.field_10 is None or self.field_11 is None  :
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_00.parameters() + self.field_01.parameters() + self.field_10.parameters() + self.field_11.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        with torch.no_grad():
            #此处需要根据不同的field取得不同的sample结果
            ray_samples, ray_indices = self.sampler_00(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

            ray_samples, ray_indices = self.sampler_01(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

            ray_samples, ray_indices = self.sampler_10(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

            ray_samples, ray_indices = self.sampler_11(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.config.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        loss_dict = {"rgb_loss": rgb_loss}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        return metrics_dict, images_dict