import torch
import torch.nn as nn
from utils import rend_util


class RayTracing(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.5,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            n_secant_steps=8,
            max_bounce=10,
            use_lastray_only=False,
            use_firstray_only=False
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps
        self.max_bounce = max_bounce
        self.use_lastray_only = use_lastray_only
        self.use_firstray_only = use_firstray_only

    def forward(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions,
                mirrors=None,
                bounce_hit_visualhull=None,
                ):

        if mirrors is None:  # multi-view generation training
            points, network_object_mask, dists, _ = self.forward_real(sdf, cam_loc, object_mask, ray_directions)
            output = {
                "points": points,
                "network_object_mask": network_object_mask,
                "dists": dists
            }
        elif self.training:  # virtual sculpting training 
            output = self.forward_kaleidoscope_visualhull(sdf, cam_loc, object_mask, ray_directions, mirrors, bounce_hit_visualhull)

        else:  # evaluation
            output = self.forward_kaleidoscope_visualhull(sdf, cam_loc, object_mask, ray_directions, mirrors, bounce_hit_visualhull)

        return output

    def forward_kaleidoscope_visualhull(
        self,
        sdf,
        cam_loc,
        object_mask,
        ray_directions,
        mirrors,
        bounce_hit_visualhull,
        visualize_rays=False
    ):
        # set boundary (object should be in front of the camera)
        real_cam_z = cam_loc[0, 0, 2]

        # set device
        device = torch.device('cuda')

        # Read data size
        batch_size, num_pixels, _ = ray_directions.shape
        P = batch_size * num_pixels
        max_bounce = self.max_bounce

        # Initialize
        loc_final = torch.zeros(P, 3, device=device).float()
        dir_final = torch.zeros(P, 3, device=device).float()
        dists_final = torch.zeros(P, 1, device=device).float()
        network_object_mask = torch.zeros(P, device=device).bool()
        count_penetration = torch.zeros(P, device=device).int()
        eval_num_bounce = torch.zeros(P, device=device).int()

        loc_bounce = torch.zeros(P, max_bounce, 3, device=device).float()
        dir_bounce = torch.zeros(P, max_bounce, 3, device=device).float()
        dists_bounce = torch.zeros(P, max_bounce, 1, device=device).float()
        distmax_bounce = torch.zeros(P, max_bounce, 1, device=device).float()
        is_carving_fg_bounce = torch.zeros(P, max_bounce, device=device).bool()
        is_carving_bg_bounce = torch.zeros(P, max_bounce, device=device).bool()

        done_bounce = torch.zeros(P, max_bounce, device=device).bool()
        point_bounce = torch.zeros(P, max_bounce, 3, device=device)

        is_firsthit_done = torch.zeros(P, device=device).bool()
        is_penetration_done = torch.zeros(P, device=device).bool()
        is_behind_total = torch.zeros(P, device=device).bool()
        mirror_sequence = torch.zeros(P, device=device).int()
        sum_hit_object = 0
        sum_hit_mirror = 0
        sum_hit_inf = 0

        # Ray tracing
        if visualize_rays:
            print(f"num_bounce\t hit_object\t    hit_inf\t hit_mirror\t     undone\t  total")

        bounce_hit_visualhull = bounce_hit_visualhull.reshape(P, -1)
        loc_now = cam_loc.reshape(-1, 3)
        dir_now = ray_directions.reshape(-1, 3)
        loc_now[is_penetration_done, :] = 0
        dir_now[is_penetration_done, :] = 0
        for num_bounce in range(max_bounce):
            # set mask for current bounce 
            # 1. Ray-object intersection
            object_mask_now = object_mask & bounce_hit_visualhull[:, num_bounce] 
            # object_mask_now = object_mask
            point_now, is_hit_object, dist_object, inside_mirror\
                = self.forward_real(sdf,
                                      loc_now.reshape(batch_size, num_pixels, 3),
                                      object_mask_now,
                                      dir_now.reshape(batch_size, num_pixels, 3),
                                      mirrors
                                    )
            is_hit_object &= ~is_penetration_done  # hit object at this bounce

            # save each bounce (for mask loss)
            loc_bounce[:, num_bounce] = loc_now.clone()
            dir_bounce[:, num_bounce] = dir_now.clone()
            dists_bounce[:, num_bounce] = dist_object.view(-1, 1).clone()
            done_bounce[:, num_bounce] = is_penetration_done.clone()
            point_bounce[:, num_bounce] = point_now.clone()

            # save last bounce (for rgb and eikonal loss) (total)
            is_hit_object_first = is_hit_object & ~is_firsthit_done
            loc_final[is_hit_object_first] = loc_now[is_hit_object_first].clone()
            dir_final[is_hit_object_first] = dir_now[is_hit_object_first].clone()
            dists_final[is_hit_object_first] = dist_object[is_hit_object_first].view(-1, 1).clone()
            network_object_mask[is_hit_object_first] = True  # mask of last bounce

            is_vh_empty = ~bounce_hit_visualhull[:, num_bounce] | ~inside_mirror
            is_carving_fg_bounce[:, num_bounce] = object_mask & is_vh_empty & ~is_penetration_done & ~is_behind_total
            is_carving_bg_bounce[:, num_bounce] = ~object_mask & ~is_penetration_done & ~is_behind_total

            eval_num_bounce[is_hit_object_first] = num_bounce  # no bounce -> num_bounce = 1;
            mirror_sequence[is_hit_object_first] = 10 * mirror_sequence[is_hit_object_first] + 0  # 0 denotes object
            is_firsthit_done |= is_hit_object  # mirror sequence done if fg or bg hit

            # 2. Ray-mirror intersection
            loc_mirror, dir_mirror, dists_mirror, is_hit_mirror, hit_mirror_id, is_behind_camera = self.forward_mirrors(mirrors,
                                                                                                                                loc_now.reshape(batch_size, num_pixels, 3),
                                                                                                                                dir_now.reshape(batch_size, num_pixels, 3),
                                                                                                                                real_cam_z)

            # update behind camera
            is_behind_total |= is_behind_camera

            # update distmax (length of the ray segment)
            distmax_bounce[:, num_bounce] = dists_mirror.view(-1, 1)

            # update mirror sequence
            is_hit_mirror = ~is_penetration_done & is_hit_mirror
            is_hit_inf = ~is_penetration_done & ~is_hit_mirror

            # mirror sequence
            is_hit_inf_nothit = is_hit_inf & ~is_firsthit_done
            is_hit_mirror_nothit = is_hit_mirror & ~is_firsthit_done
            mirror_sequence[is_hit_mirror_nothit] = 10 * mirror_sequence[is_hit_mirror_nothit] + hit_mirror_id[is_hit_mirror_nothit]
            if (hit_mirror_id[is_hit_mirror] == 0).sum() > 0:
                print('mirror sequence wrong')
            mirror_sequence[is_hit_inf_nothit] = 10 * mirror_sequence[is_hit_inf_nothit] + 9  # 9 denotes inf
            eval_num_bounce[is_hit_inf_nothit] = num_bounce  # no bounce -> num_bounce = 1;

            # show statistics
            if visualize_rays:
                print(f"{num_bounce:10d}\t {is_hit_object.sum():10d}\t {is_hit_inf.sum():10d}\t {is_hit_mirror.sum():10d}\t {(~is_penetration_done).sum():10d}\t {is_penetration_done.shape[0]:10d}")

            # update rays
            is_penetration_done |= is_hit_inf
            loc_now = loc_mirror
            dir_now = dir_mirror
            loc_now[is_penetration_done, :] = 0
            dir_now[is_penetration_done, :] = 0

            # update stats
            sum_hit_object += is_hit_object.sum()
            sum_hit_mirror += is_hit_mirror.sum()
            sum_hit_inf += is_hit_inf.sum()

            if (~is_penetration_done).sum() == 0:
                done_bounce[:, num_bounce + 1:] = True
                break

        if visualize_rays:
            print(100 * "=")
            print(f"\t\t {sum_hit_object:10d}\t {sum_hit_inf:10d}\t {sum_hit_mirror:10d}\t {(~is_penetration_done).sum():10d}\t  {is_penetration_done.shape[0]:10d}\n")

        sdf_bounce = sdf(point_bounce.view(-1, 3))
        sdf_bounce[done_bounce.view(-1)] = torch.tensor(float('Inf'))  # set sdf to Inf if it is done
        sdf_bounce = sdf_bounce.view(-1, max_bounce)

        sdf_min_bg = sdf_bounce.min(dim=1)
        indices1_bg = sdf_min_bg.indices.reshape(-1, 1, 1).repeat(1, 1, 1)
        indices3_bg = sdf_min_bg.indices.reshape(-1, 1, 1).repeat(1, 1, 3)

        loc_closest_bg = torch.gather(loc_bounce, 1, indices3_bg).squeeze(1)  # closest point along the whole rays
        dir_closest_bg = torch.gather(dir_bounce, 1, indices3_bg).squeeze(1)  # closest point along the whole rays
        dists_closest_bg = torch.gather(dists_bounce, 1, indices1_bg).squeeze(1)  # closest point along the whole rays

        loc_final[~object_mask] = loc_closest_bg[~object_mask]
        dir_final[~object_mask] = dir_closest_bg[~object_mask]
        dists_final[~object_mask] = dists_closest_bg[~object_mask]

        # fg
        sdf_bounce = sdf(point_bounce.view(-1, 3))
        invalid_bounce = done_bounce | is_carving_fg_bounce
        invalid_fg_mask = invalid_bounce.all(dim=1)
        # if invalid_fg_mask[object_mask].any():
        #     print(f'No modeling points in {invalid_fg_mask.sum()} fg pixels !')

        sdf_bounce[invalid_bounce.view(-1)] = torch.tensor(float('Inf'))  # set sdf to Inf if it is done
        sdf_bounce = sdf_bounce.view(-1, max_bounce)

        sdf_min_fg = sdf_bounce.min(dim=1)
        indices1_fg = sdf_min_fg.indices.reshape(-1, 1, 1).repeat(1, 1, 1)
        indices3_fg = sdf_min_fg.indices.reshape(-1, 1, 1).repeat(1, 1, 3)

        loc_closest_fg = torch.gather(loc_bounce, 1, indices3_fg).squeeze(1)  # closest point along the whole rays
        dir_closest_fg = torch.gather(dir_bounce, 1, indices3_fg).squeeze(1)  # closest point along the whole rays
        dists_closest_fg = torch.gather(dists_bounce, 1, indices1_fg).squeeze(1)  # closest point along the whole rays

        modeling_mask = object_mask & ~network_object_mask
        loc_final[modeling_mask] = loc_closest_fg[modeling_mask]
        dir_final[modeling_mask] = dir_closest_fg[modeling_mask]
        dists_final[modeling_mask] = dists_closest_fg[modeling_mask]

        # set the closest point along all the bounces
        points = (loc_final + dists_final * dir_final).reshape(-1, 3)

        output = {
            "points": points,
            "network_object_mask": network_object_mask,
            "dists": dists_final,
            "loc_final": loc_final,
            "dir_final": dir_final,
            "dists_final": dists_final,
            "mirror_sequence": mirror_sequence,
            "loc_bounce": loc_bounce,
            "dir_bounce": dir_bounce,
            "dists_bounce": dists_bounce,
            "distmax_bounce": distmax_bounce,  # length of ray segment"
            "eval_num_bounce": eval_num_bounce,
            "count_penetration": count_penetration,
            "is_carving_fg_bounce": is_carving_fg_bounce,
            "is_carving_bg_bounce": is_carving_bg_bounce,
            "invalid_fg_mask": invalid_fg_mask,
        }
        return output

    def forward_mirrors(self,
                        mirrors,
                        cam_loc,
                        ray_directions,
                        real_cam_z=-20.0  # ignore the points behind this
                        ):

        # initialize output
        batch_size, num_pixels, _ = cam_loc.shape
        P = batch_size * num_pixels
        loc_reflected = cam_loc.clone()  # location of mirror-ray intersection
        ray_directions_reflected = ray_directions.clone()

        # run mirror reflection
        min_distance_to_mirror = float('inf') * torch.ones(P, device=torch.device('cuda'))
        min_mirror_id = torch.zeros(P, device=torch.device('cuda')).int()
        num_mirror = len(mirrors) // 2
        for i_mirror in range(num_mirror):
            m_id = i_mirror + 1  # id starts from 1
            # run mirror reflection for each mirror
            n = mirrors[f"n_{i_mirror}"]
            d = mirrors[f"d_{i_mirror}"]
            loc_reflected_current, ray_directions_reflected_current, distance_to_mirror_current = self.intersect_mirror_ray(n, d, cam_loc, ray_directions)

            # update if is the closest mirror
            is_min = distance_to_mirror_current < min_distance_to_mirror
            loc_reflected[:, is_min, :] = loc_reflected_current[:, is_min, :]
            ray_directions_reflected[:, is_min, :] = ray_directions_reflected_current[:, is_min, :]
            min_mirror_id[is_min] = m_id
            min_distance_to_mirror[is_min] = distance_to_mirror_current[is_min]

        # update mirror id
        is_hit_mirror = (min_mirror_id > 0)
        min_distance_to_mirror[~is_hit_mirror] = 0
        is_behind_camera = loc_reflected[0, :, 2] < real_cam_z

        return loc_reflected.reshape(-1, 3), \
               ray_directions_reflected.reshape(-1, 3), \
               min_distance_to_mirror, \
               is_hit_mirror, \
               min_mirror_id, \
               is_behind_camera


        # auxiliary functionn for forward_mirrors)
    def intersect_mirror_ray(self,
                             n,
                             d,
                             cam_loc,
                             ray_directions):

        # mirror: n'x = d
        c = cam_loc  # camera location
        v = ray_directions  # ray direction

        t = (d - c @ n) / (v @ n)
        cos_angle = -v @ n
        is_reflect = (t > 0) & (cos_angle > 0)
        is_reflect = is_reflect.reshape(-1)

        distance_to_mirror = t.reshape(-1)
        distance_to_mirror[~is_reflect] = float("inf")
        loc_reflected = cam_loc + t * v
        ray_directions_reflected = 2 * cos_angle @ n.t() + v

        return loc_reflected, \
               ray_directions_reflected, \
               distance_to_mirror

    def is_inside_mirror(
        self,
        points,
        mirrors
    ):
        device = torch.device('cuda')

        num_points = points.shape[0]
        is_outside = torch.zeros(num_points, device=device).bool()
        num_mirror = len(mirrors) // 2
        for i_mirror in range(num_mirror):
            # run mirror reflection for each mirror
            n = mirrors[f"n_{i_mirror}"]
            d = mirrors[f"d_{i_mirror}"]
            is_outside = is_outside | (points @ n < d).reshape(-1)

        return ~is_outside

    def forward_real(self,
                         sdf,
                         cam_loc,
                         object_mask,
                         ray_directions,
                         mirrors=None
                     ):
        batch_size, num_pixels, _ = ray_directions.shape
        if cam_loc.dim() == 2:
            cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1)

        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)

        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections)

        network_object_mask = (acc_start_dis < acc_end_dis)

        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2)).cuda()
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]

            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask
                                                                                )

            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        # print('----------------------------------------------------------------')
        # print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
        #       .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        # print('----------------------------------------------------------------')

        inside_mirror = torch.ones_like(network_object_mask).bool()
        # if not self.training: # TODO: uncomment this
        #     return curr_start_points, \
        #         network_object_mask, \
        #         acc_start_dis, \
        #         inside_mirror

        ray_directions = ray_directions.reshape(-1, 3)
        mask_intersect = mask_intersect.reshape(-1)

        in_mask = ~network_object_mask & object_mask & ~sampler_mask
        out_mask = ~object_mask & ~sampler_mask

        mask_left_out = (in_mask | out_mask) & ~mask_intersect
        if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
            cam_left_out = cam_loc.reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
            curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

        mask = (in_mask | out_mask) & mask_intersect

        if mask.sum() > 0:
            min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]

            min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)

            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist

        # dist should be positive
        th = 1e-3  # threshold for ensure the point is inside the mirrors
        is_dist_negative = acc_start_dis < 0
        if sum(is_dist_negative):
            curr_start_points[is_dist_negative] = cam_loc[:, is_dist_negative]
            network_object_mask[is_dist_negative] = False
            acc_start_dis[is_dist_negative] = th

        # -- outside --
        # discard if outside mirror for push and pull training
        if mirrors is not None:
            inside_mirror = self.is_inside_mirror(curr_start_points, mirrors)
            network_object_mask = network_object_mask & inside_mirror

        # point_now, is_hit_object, dist_object
        return curr_start_points, \
               network_object_mask, \
               acc_start_dis, \
               inside_mirror

        # return curr_start_points, \
        #        network_object_mask, \
        #        acc_start_dis, \

    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        batch_size, num_pixels, _ = ray_directions.shape
        if cam_loc.dim() == 2:
            cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1)
        sphere_intersections_points = cam_loc.reshape(batch_size, -1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # Initialize start current points
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:, :, 0, :].reshape(-1, 3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1, 2)[unfinished_mask_start, 0]

        # Initialize end current points
        curr_end_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:, :, 1, :].reshape(-1, 3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1, 2)[unfinished_mask_end, 1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = (cam_loc + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def ray_sampler(self, sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask):
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''

        batch_size, num_pixels, _ = ray_directions.shape
        if cam_loc.dim() == 2:
            cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1)
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1, 1, -1)

        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        points = cam_loc.reshape(batch_size, -1, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            cam_loc_secant = cam_loc.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''

        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low

        return z_pred

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = cam_loc.reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist
