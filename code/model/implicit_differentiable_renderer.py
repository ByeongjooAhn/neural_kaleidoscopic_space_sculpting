import torch
import torch.nn as nn
import numpy as np

from utils import rend_util, carving_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def get_distance_and_gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        distance = y[:, 0]

        return distance, gradients


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_loc=0,
            multires_view=0,
            is_reflection_dir=False,
    ):
        super().__init__()

        self.mode = mode
        self.is_reflection_dir = is_reflection_dir
        dims = [d_in + feature_vector_size] + dims + [d_out]

        # PE for location
        self.embedloc_fn = None
        if multires_loc > 0:
            embedloc_fn, input_ch = get_embedder(multires_loc)
            self.embedloc_fn = embedloc_fn
            dims[0] += (input_ch - 3)

        # PE for view direction
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        # reflection
        if self.is_reflection_dir:
            view_dirs = 2 * torch.sum(normals * view_dirs, dim=1, keepdim=True) * normals - view_dirs

        # PE for location
        if self.embedloc_fn is not None:
            points = self.embedloc_fn(points)

        # PE for view direction
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x


class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.use_vh_constraint = conf.get_bool('use_vh_constraint', default=True)
        self.is_fg_carving = conf.get_bool('kaleidoscope.is_fg_carving', default=True)

    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)
        pose_all = input["pose_all"]
        img_res = input["img_res"]
        virtual_sculpting = input["virtual_sculpting"]
        bounce_hit_visualhull = input["bounce_hit_visualhull"]
        is_eval_single = input["is_eval_single"]
        if not self.use_vh_constraint:
            bounce_hit_visualhull = torch.ones_like(bounce_hit_visualhull).bool()

        no_mirror = is_eval_single or (self.training and not virtual_sculpting)
        use_mirror = not no_mirror
        if use_mirror:
            num_mirror = input["num_mirror"]
            mirrors = {}
            for i in range(num_mirror):
                mirrors[f"n_{i}"] = input[f"mirrors_n_{i}"]
                mirrors[f"d_{i}"] = input[f"mirrors_d_{i}"]
        else:
            mirrors = None

        if virtual_sculpting or is_eval_single:
            ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        else:
            ray_dirs, cam_loc = rend_util.get_camera_params_concat(uv, pose, intrinsics, img_res, pose_all)

        batch_size, num_pixels, _ = ray_dirs.shape
        if cam_loc.dim() == 2:
            cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1)

        self.implicit_network.eval()
        with torch.no_grad():
            output_raytracer = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs,
                                                                 mirrors=mirrors,
                                                                 bounce_hit_visualhull=bounce_hit_visualhull
                                               )
        self.implicit_network.train()

        points = output_raytracer["points"].reshape(-1, 3)
        dists = output_raytracer["dists"].reshape(-1, 1)
        network_object_mask = output_raytracer["network_object_mask"]
        sdf_output = self.implicit_network(points)[:, 0:1]

        # output only for evaluation
        if use_mirror:
            cam_loc = output_raytracer["loc_final"].reshape(-1, 3)
            ray_dirs = output_raytracer["dir_final"].reshape(-1, 3)
            mirror_sequence = output_raytracer["mirror_sequence"]
            eval_num_bounce = output_raytracer["eval_num_bounce"]
        else:  # single image evaluation
            ray_dirs = ray_dirs.reshape(-1, 3)
            mirror_sequence = torch.zeros_like(sdf_output).reshape(-1)
            eval_num_bounce = torch.zeros_like(sdf_output).reshape(-1)

        if self.training:
            points = (cam_loc.reshape(batch_size, -1, 3) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].reshape(-1, 1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)
            if use_mirror:
                invalid_fg_mask = output_raytracer["invalid_fg_mask"]

                # carving loss
                output_carving = carving_util.sample_points_carving(
                    output_raytracer=output_raytracer,
                    is_fg_carving=self.is_fg_carving,
                )
                point_carving = output_carving['point_carving']
                sdf_output_carving = self.implicit_network(point_carving)[:, 0:1]
            else:
                invalid_fg_mask = torch.zeros_like(object_mask).bool()
                sdf_output_carving = None

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None
            invalid_fg_mask = None
            sdf_output_carving = None

        view = -ray_dirs[surface_mask]

        rgb_values = -torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view)

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'mirror_sequence': mirror_sequence,
            'eval_num_bounce': eval_num_bounce,
            'invalid_fg_mask': invalid_fg_mask,
            'sdf_output_carving': sdf_output_carving,
            'loc_bounce': output_raytracer["loc_bounce"],
            'dir_bounce': output_raytracer["dir_bounce"],
            'dists_bounce': output_raytracer["dists_bounce"],
            'distmax_bounce': output_raytracer["distmax_bounce"],
            'is_carving_fg_bounce': output_raytracer["is_carving_fg_bounce"],
            'is_carving_bg_bounce': output_raytracer["is_carving_bg_bounce"],
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals
