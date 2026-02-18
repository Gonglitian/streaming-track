"""SMPL mesh overlay on video frames using pytorch3d."""

import numpy as np
import torch

_renderer = None  # lazy singleton


class SMPLOverlayRenderer:
    """Render SMPL mesh overlay on camera frames.

    Uses SMPLX forward pass → pytorch3d rasterization → alpha blend.
    """

    def __init__(self, smplx_model_path: str, width: int = 640, height: int = 480,
                 device: str = "cuda"):
        import smplx as smplx_lib
        from pytorch3d.renderer import (
            MeshRasterizer, RasterizationSettings,
            SoftPhongShader, PointLights,
            PerspectiveCameras, MeshRenderer,
        )
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex

        self._device = torch.device(device)
        self._width = width
        self._height = height

        # Load SMPLX model
        self._body_model = smplx_lib.create(
            smplx_model_path, model_type="smplx",
            gender="neutral", num_betas=10,
            use_pca=False, flat_hand_mean=True,
        ).to(self._device)

        # Setup pytorch3d renderer
        self._cameras = PerspectiveCameras(device=self._device)
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(
            device=self._device,
            location=[[0.0, 0.0, 3.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.6, 0.6, 0.6]],
        )
        self._renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self._cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self._device, cameras=self._cameras, lights=lights),
        )
        self._faces = torch.from_numpy(
            self._body_model.faces.astype(np.int64)
        ).to(self._device)

    @torch.no_grad()
    def render_overlay(
        self,
        frame: np.ndarray,
        smpl_params: dict,
        alpha: float = 0.6,
        focal_length: float | None = None,
    ) -> np.ndarray:
        """Render SMPL mesh overlay on a camera frame.

        Args:
            frame: BGR image (H, W, 3) uint8.
            smpl_params: Dict with keys: global_orient(1,3), body_pose(1,63),
                         betas(1,10), transl(1,3) — all as numpy or torch tensors.
            alpha: Blend factor for the mesh overlay.
            focal_length: Camera focal length in pixels. If None, estimated from width.

        Returns:
            BGR image with SMPL mesh overlaid.
        """
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex

        # Convert params to torch
        params = {}
        for k, v in smpl_params.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).float()
            if v.dim() == 1:
                v = v.unsqueeze(0)
            params[k] = v.to(self._device)

        # Pad betas to 10 if needed
        if "betas" in params and params["betas"].shape[-1] < 10:
            pad = torch.zeros(1, 10 - params["betas"].shape[-1], device=self._device)
            params["betas"] = torch.cat([params["betas"], pad], dim=-1)

        # SMPLX forward
        output = self._body_model(**params)
        verts = output.vertices  # (1, V, 3)

        # Simple vertex coloring (light blue)
        colors = torch.full_like(verts, 0.6)
        colors[..., 2] = 0.9  # blue tint

        mesh = Meshes(
            verts=verts,
            faces=self._faces.unsqueeze(0),
            textures=TexturesVertex(colors),
        )

        # Setup camera intrinsics
        h, w = frame.shape[:2]
        if focal_length is None:
            focal_length = max(w, h)  # rough estimate

        # Render
        images = self._renderer(mesh)  # (1, H, W, 4) RGBA
        rendered = images[0, ..., :3].cpu().numpy()  # (H, W, 3) float [0,1]
        mask = images[0, ..., 3].cpu().numpy()  # (H, W) alpha

        # Alpha blend
        rendered_bgr = (rendered[..., ::-1] * 255).astype(np.uint8)
        mask_3ch = (mask[..., None] * alpha).astype(np.float32)

        result = frame.astype(np.float32)
        result = result * (1 - mask_3ch) + rendered_bgr.astype(np.float32) * mask_3ch
        return result.astype(np.uint8)
