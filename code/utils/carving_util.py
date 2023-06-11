import torch


def sample_points_carving(output_raytracer, is_fg_carving=True, is_stratified=False, is_innermost=True):
    loc_bounce = output_raytracer["loc_bounce"].view(-1, 3)
    dir_bounce = output_raytracer["dir_bounce"].view(-1, 3)
    dists_bounce = output_raytracer["dists_bounce"].view(-1, 1)
    distmax_bounce = output_raytracer["distmax_bounce"].view(-1)
    is_carving_fg_bounce = output_raytracer["is_carving_fg_bounce"].view(-1)
    is_carving_bg_bounce = output_raytracer["is_carving_bg_bounce"].view(-1)

    device = loc_bounce.device
    point_carving_fg = torch.zeros(0, 3, device=device)
    point_carving_bg = torch.zeros(0, 3, device=device)
    point_carving_fg_stratified = None
    point_carving_bg_stratified = None
    point_carving_fg_innermost = None
    point_carving_bg_innermost = None

    point_innermost = (loc_bounce + dir_bounce * dists_bounce).view(-1, 3)

    # background carving
    if is_stratified:
        point_carving_bg_stratified = sample_stratified(
            loc_bounce,
            dir_bounce,
            dists_bounce,
            distmax_bounce,
            is_carving_bg_bounce
        )
        point_carving_bg = torch.cat([point_carving_bg, point_carving_bg_stratified])
    if is_innermost:
        point_carving_bg_innermost = point_innermost[is_carving_bg_bounce.view(-1)]
        point_carving_bg = torch.cat([point_carving_bg, point_carving_bg_innermost])

    # foreground carving
    if is_fg_carving:
        if is_stratified:
            point_carving_fg_stratified = sample_stratified(
                loc_bounce,
                dir_bounce,
                dists_bounce,
                distmax_bounce,
                is_carving_fg_bounce
            )
            point_carving_fg = torch.cat([point_carving_fg, point_carving_fg_stratified])
        if is_innermost:
            point_carving_fg_innermost = point_innermost[is_carving_fg_bounce.view(-1)]
            point_carving_fg = torch.cat([point_carving_fg, point_carving_fg_innermost])

    output = {
        'point_carving': torch.cat([point_carving_fg, point_carving_bg]),
        'point_carving_fg': point_carving_fg,
        'point_carving_bg': point_carving_bg,
        'point_carving_fg_stratified': point_carving_fg_stratified,
        'point_carving_bg_stratified': point_carving_bg_stratified,
        'point_carving_fg_innermost': point_carving_fg_innermost,
        'point_carving_bg_innermost': point_carving_bg_innermost
    }
    return output


def sample_stratified(
    loc_bounce,
    dir_bounce,
    dists_bounce,
    distmax_bounce,
    is_carving_bounce,
    z_interval=0.25
):

    is_carving_bounce = is_carving_bounce.view(-1)

    loc_bounce = loc_bounce.view(-1, 3)[is_carving_bounce]
    dir_bounce = dir_bounce.view(-1, 3)[is_carving_bounce]
    dists_bounce = dists_bounce.view(-1, 1)[is_carving_bounce]
    distmax_bounce = distmax_bounce.view(-1)[is_carving_bounce]

    max_dist = 10
    is_inf = distmax_bounce > max_dist
    distmax_bounce[is_inf] = max_dist

    z_interval = torch.tensor(z_interval, device=distmax_bounce.device)
    max_iter = torch.ceil(max_dist / z_interval).int()
    point_list = []
    for i in range(max_iter):
        dist_remain = distmax_bounce - z_interval * i
        is_remain = dist_remain > 0
        z_val = torch.rand_like(dist_remain) * torch.min(dist_remain, z_interval)
        z_val += z_interval * i
        point_i = loc_bounce + z_val.view(-1, 1) * dir_bounce
        point_list.append(point_i[is_remain])

    point_stratified = torch.cat(point_list, dim=0)

    return point_stratified
