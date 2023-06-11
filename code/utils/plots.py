import plotly.graph_objs as go

import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from scipy.io import savemat
import matplotlib.pyplot as plt


from matplotlib import cm
from utils import rend_util, gif_util


def plot(model, indices, model_outputs, pose, ground_truth, path, path_results, epoch, img_res, plot_nimgs, max_depth, resolution, writer=None):
    # arrange data to plot
    rgb_gt = ground_truth['rgb']
    gt_num_bounce = ground_truth['gt_num_bounce']
    vh_num_bounce = ground_truth['vh_num_bounce']
    mirror_sequence_edge = ground_truth['mirror_sequence_edge']
    mirror_pixel = ground_truth['mirror_pixel']
    bounce_hit_visualhull = ground_truth['bounce_hit_visualhull']

    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    object_mask = model_outputs['object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    rgb_eval = model_outputs['rgb_values']
    rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)
    mirror_sequence = model_outputs['mirror_sequence']
    eval_num_bounce = model_outputs['eval_num_bounce']

    depth = torch.ones(batch_size * num_samples).float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = depth.reshape(batch_size, num_samples, 1)

    network_object_mask = network_object_mask.reshape(batch_size, -1)
    object_mask = object_mask.reshape(batch_size, -1)

    # plot rendered images
    psnr = plot_images(rgb_eval, rgb_gt, mirror_sequence_edge, path, epoch, plot_nimgs, mirror_pixel, img_res, path_results, writer=writer)

    # plot depth maps
    plot_depth_maps(depth, path, epoch, plot_nimgs, img_res, path_results, writer=writer)

    # plot rendered images
    mask_error = plot_mask(network_object_mask, object_mask, mirror_sequence_edge, path, epoch, plot_nimgs, mirror_pixel, img_res, path_results, writer=writer)

    # plot_mirror sequence
    mirror_sequence_cm = plot_mirror_sequence(mirror_sequence, path, epoch, img_res, path_results, writer=writer)

    # plot num_bounce
    label_error, label_error_fg, vh_valid_error = plot_num_bounce(eval_num_bounce, gt_num_bounce, vh_num_bounce, bounce_hit_visualhull, mirror_sequence_edge, object_mask, network_object_mask, path, epoch, mirror_pixel, img_res, path_results, writer=writer)

    plot_debug(model_outputs, ground_truth, epoch, path)

    # plot surface
    surface_traces, meshexport = get_surface_trace(path=path,
                                                   epoch=epoch,
                                                   sdf=lambda x: model.implicit_network(x)[:, 0],
                                                   resolution=resolution,
                                                   path_results=path_results,
                                                   writer=writer
                                                   )

    # surface rendering
    return meshexport, mirror_sequence_cm, psnr, mask_error, label_error, label_error_fg, vh_valid_error


def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_surface_trace(path, epoch, sdf, resolution=100, path_results=None, writer=None, is_intermediate_save=True):
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)
        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            opacity=1.0)]

        meshexport = trimesh.Trimesh(verts, faces, vertex_normals=-normals)
        if is_intermediate_save:
            meshexport.export('{0}/surface/surface_{1}.ply'.format(path, epoch), 'ply')

        meshexport.export(f'{path_results}/surface.ply', 'ply')

        if writer is not None:
            colors = (verts - verts.min()) / (verts.max() - verts.min())
            scale_factor = 255
            colors = (colors * scale_factor).astype(np.uint8)
            writer.add_mesh('mesh', [verts],
                            faces=[faces],
                            colors=[colors],
                            global_step=epoch)
        return traces, meshexport

    return None, None


def get_surface_high_res_mesh(sdf, resolution=100, level=0):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=level,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, vertex_normals=-normals)
    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float)
    mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.eig(s_cov, True)[1].transpose(0, 1)
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                          verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, vertex_normals=-normals)

    return meshexport


def get_grid_uniform(resolution):
    x = np.linspace(-1.2, 1.2, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}


def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def plot_depth_maps(depth_maps, path, epoch, plot_nrow, img_res, path_results, writer=None):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()

    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor.transpose(1, 2, 0))
    img.save('{0}/depth/depth_{1}.png'.format(path, epoch))
    img.save(f'{path_results}/depth.png')

    if writer is not None:
        writer.add_image('rendering/depth', tensor, epoch)


def plot_images(rgb_points, ground_true, mirror_sequence_edge, path, epoch, plot_nrow, mirror_pixel, img_res, path_results, writer=None):
    device = rgb_points.device
    ground_true = (ground_true.to(device) + 1.) / 2.
    rgb_points = (rgb_points + 1.) / 2.

    # ignore outside of mirror pixel
    mirror_pixel = mirror_pixel.reshape(-1)
    rgb_points.view(-1, 3)[~mirror_pixel, :] = 0
    ground_true.view(-1, 3)[~mirror_pixel, :] = 0

    diff = abs(ground_true - rgb_points).reshape(img_res[0], img_res[1], 3)
    plt.imsave('{0}/rendering/rendering_diff_{1}.png'.format(path, epoch), diff.cpu().detach().numpy())
    plt.imsave(f'{path_results}/rendering_diff.png', diff.cpu().detach().numpy())

    diff_noedge = diff.clone().reshape(-1, 3)
    diff_noedge[mirror_sequence_edge.reshape(-1), :] = 0
    diff_noedge = diff_noedge.reshape(img_res[0], img_res[1], 3)
    plt.imsave('{0}/rendering/rendering_diff_noedge_{1}.png'.format(path, epoch), diff_noedge.cpu().detach().numpy())
    plt.imsave(f'{path_results}/rendering_diff_noedge.png', diff_noedge.cpu().detach().numpy())

    rgb_points = rgb_points.reshape(img_res[0], img_res[1], 3)
    plt.imsave('{0}/rendering/rendering_{1}.png'.format(path, epoch), rgb_points.cpu().detach().numpy())
    plt.imsave(f'{path_results}/rendering.png', rgb_points.cpu().detach().numpy())

    valid = ~mirror_sequence_edge.reshape(-1) & mirror_pixel.reshape(-1)
    mse = (diff.reshape(-1, 3)[valid, :].reshape(-1)**2).mean()
    psnr = 10 * torch.log10(1. / mse)
    if writer is not None:
        # writer.add_image('image', tensor, epoch)
        writer.add_scalar('evaluation/psnr', 100 * psnr.item(), epoch)

    return psnr


def plot_mask(network_object_mask, object_mask, mirror_sequence_edge, path, epoch, plot_nrow, mirror_pixel, img_res, path_results, writer=None):
    mask_input = lin2img(object_mask.unsqueeze(-1), img_res)
    mask = lin2img(network_object_mask.unsqueeze(-1), img_res).float()
    mask_diff = (mask_input != mask).float()
    mask_diff_squeeze = mask_diff.squeeze().bool().cpu().detach().numpy()
    mirror_sequence_edge = mirror_sequence_edge.reshape(-1)

    tensor = torchvision.utils.make_grid(mask.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.reshape(3, -1)
    false_positive = mask_diff_squeeze.reshape(-1) & (tensor[0, :].squeeze() > 0.5)
    false_negative = mask_diff_squeeze.reshape(-1) & (tensor[0, :].squeeze() < 0.5)

    # false positive (need to be carved)
    tensor[0, false_positive] = 1
    tensor[1, false_positive] = 0
    tensor[2, false_positive] = 0

    # false negative (need to be expanded)
    tensor[0, false_negative] = 0
    tensor[1, false_negative] = 0
    tensor[2, false_negative] = 1

    # ignore outside of mirror pixel
    tensor[:, ~mirror_pixel.reshape(-1)] = 0

    tensor = tensor.reshape(3, img_res[0], img_res[1])
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor.transpose(1, 2, 0))
    img.save('{0}/mask/mask_{1}.png'.format(path, epoch))
    img.save(f'{path_results}/mask.png')

    if writer is not None:
        writer.add_image('mask', tensor, epoch)

    # visualize ignoring edge
    tensor = tensor.reshape(3, -1)
    tensor[:, mirror_sequence_edge.reshape(-1)] = 0
    tensor = tensor.reshape(3, img_res[0], img_res[1])
    img = Image.fromarray(tensor.transpose(1, 2, 0))
    img.save('{0}/mask/mask_ignoreedge_{1}.png'.format(path, epoch))
    img.save(f'{path_results}/mask_ignoreedge.png')

    valid = ~mirror_sequence_edge.reshape(-1) & mirror_pixel.reshape(-1)
    mask_diff_ignoreedge = (object_mask != network_object_mask).reshape(-1) & valid
    mask_error = mask_diff_ignoreedge.sum() / valid.sum()
    if writer is not None:
        writer.add_scalar('evaluation/mask_error', 100 * mask_error.item(), epoch)

    return mask_error


def plot_mirror_sequence(mirror_sequence, path, epoch, img_res, path_results, writer=None):
    g, id = findgroups(mirror_sequence.squeeze())
    mirror_sequence_sorted = g.reshape(img_res[0], img_res[1])

    cmap = cm.get_cmap('rainbow', len(id))
    mirror_sequence_cm = torch.tensor(cmap(mirror_sequence_sorted)).numpy()[..., :-1]
    plt.imsave(f'{path}/mirror_sequence/mirror_sequence_{epoch}.png', mirror_sequence_cm)
    plt.imsave(f'{path_results}/mirror_sequence.png', mirror_sequence_cm)

    if writer is not None:
        writer.add_image('mirror_sequence', torch.tensor(mirror_sequence_cm.transpose(2, 0, 1)), epoch)

    return mirror_sequence_cm


def plot_num_bounce(eval_num_bounce, gt_num_bounce, vh_num_bounce, bounce_hit_visualhull, mirror_sequence_edge, object_mask, network_object_mask, path, epoch, mirror_pixel, img_res, path_results, writer=None):
    mirror_sequence_edge = mirror_sequence_edge.view(-1)
    bounce_hit_visualhull = bounce_hit_visualhull.squeeze(0)

    eval_num_bounce = eval_num_bounce.reshape(img_res[0], img_res[1])
    gt_num_bounce = gt_num_bounce.reshape(img_res[0], img_res[1])
    vh_num_bounce = vh_num_bounce.reshape(img_res[0], img_res[1])

    plt.imsave(f'{path}/num_bounce/eval_num_bounce_{epoch}.png', eval_num_bounce)
    plt.imsave(f'{path}/num_bounce/gt_num_bounce_{epoch}.png', gt_num_bounce)
    plt.imsave(f'{path}/num_bounce/vh_num_bounce_{epoch}.png', vh_num_bounce)

    plt.imsave(f'{path_results}/eval_num_bounce.png', eval_num_bounce)
    plt.imsave(f'{path_results}/gt_num_bounce.png', gt_num_bounce)
    plt.imsave(f'{path_results}/vh_num_bounce.png', vh_num_bounce)

    valid = ~mirror_sequence_edge.reshape(-1) & mirror_pixel.reshape(-1)
    mask_diff_ignoreedge = (eval_num_bounce != gt_num_bounce).reshape(-1) & valid
    label_error = mask_diff_ignoreedge.sum() / valid.sum()

    valid_fg = ~mirror_sequence_edge.reshape(-1) & mirror_pixel.reshape(-1) & object_mask
    mask_diff_fg = (eval_num_bounce != gt_num_bounce).reshape(-1) & valid_fg
    label_error_fg = mask_diff_fg.sum() / valid_fg.sum()

    # check if the bounce is valid with visual hull
    b = eval_num_bounce.reshape(-1, 1).repeat(1, 1).to(torch.int64)
    is_vh_invalid = ~(torch.gather(bounce_hit_visualhull, 1, b).squeeze(1).bool()) & valid_fg  # closest point along the whole rays
    is_vh_invalid = is_vh_invalid.reshape(img_res[0], img_res[1])
    plt.imsave(f'{path_results}/is_vh_invalid.png', is_vh_invalid, cmap='gray')
    vh_valid_error = is_vh_invalid.sum() / valid_fg.sum()

    if writer is not None:
        writer.add_scalar('evaluation/bounce_error', 100 * label_error.item(), epoch)
        writer.add_scalar('evaluation/bounce_error_fg', 100 * label_error_fg.item(), epoch)
        writer.add_scalar('evaluation/vh_valid_error', 100 * vh_valid_error.item(), epoch)

    # mask difference (vs gt)
    mask_diff = (eval_num_bounce != gt_num_bounce)
    eval_large = eval_num_bounce > gt_num_bounce
    eval_small = eval_num_bounce < gt_num_bounce

    bounce_large = mask_diff.reshape(-1) & eval_large.reshape(-1)
    bounce_small = mask_diff.reshape(-1) & eval_small.reshape(-1)

    tensor = mask_diff.clone().unsqueeze(-1).repeat(1, 1, 3).reshape(-1, 3).float()
    # smaller bounce (need to be carved)
    tensor[bounce_small, 0] = 1
    tensor[bounce_small, 1] = 0
    tensor[bounce_small, 2] = 0

    # larger bounce (need to be expanded)
    tensor[bounce_large, 0] = 0
    tensor[bounce_large, 1] = 0
    tensor[bounce_large, 2] = 1

    tensor[~mirror_pixel.reshape(-1), :] = 0
    img_tensor = tensor.reshape(img_res[0], img_res[1], 3).numpy()
    plt.imsave(f'{path_results}/eval_num_bounce_error_gt.png', img_tensor)

    tensor[mirror_sequence_edge, :] = 0
    img_tensor = tensor.reshape(img_res[0], img_res[1], 3).numpy()
    plt.imsave(f'{path_results}/eval_num_bounce_error_gt_ignoreedge.png', img_tensor)

    # mask difference (vs vh)
    mask_diff = (eval_num_bounce != vh_num_bounce)
    eval_large = eval_num_bounce > vh_num_bounce
    eval_small = eval_num_bounce < vh_num_bounce

    bounce_large = mask_diff.reshape(-1) & eval_large.reshape(-1)
    bounce_small = mask_diff.reshape(-1) & eval_small.reshape(-1)

    tensor = mask_diff.clone().unsqueeze(-1).repeat(1, 1, 3).reshape(-1, 3).float()
    # smaller bounce (need to be carved)
    tensor[bounce_small, 0] = 1
    tensor[bounce_small, 1] = 0
    tensor[bounce_small, 2] = 0

    # larger bounce (need to be expanded)
    tensor[bounce_large, 0] = 0
    tensor[bounce_large, 1] = 0
    tensor[bounce_large, 2] = 1

    tensor[~mirror_pixel.reshape(-1), :] = 0
    img_tensor = tensor.reshape(img_res[0], img_res[1], 3).numpy()
    plt.imsave(f'{path_results}/eval_num_bounce_error_vh.png', img_tensor)

    tensor[mirror_sequence_edge, :] = 0
    img_tensor = tensor.reshape(img_res[0], img_res[1], 3).numpy()
    plt.imsave(f'{path_results}/eval_num_bounce_error_vh_ignoreedge.png', img_tensor)

    return label_error, label_error_fg, vh_valid_error


def plot_count_penetration(count_penetration, path, epoch, img_res, path_results, writer=None):
    count_penetration = count_penetration.reshape(img_res[0], img_res[1])
    plt.imsave(f'{path}/count_penetration/count_penetration_{epoch}.png', count_penetration)
    plt.imsave(f'{path_results}/count_penetration.png', count_penetration)

    is_unreliable = count_penetration > 1
    plt.imsave(f'{path}/count_penetration/is_unreliable_{epoch}.png', is_unreliable, cmap='gray')
    plt.imsave(f'{path_results}/is_unreliable.png', is_unreliable, cmap='gray')

def findgroups(x):
    id = set(x.tolist())
    g = -torch.ones_like(x)
    for i, id_i in enumerate(id):
        is_i = (x == id_i)
        g[is_i] = i
    assert ((g < 0).sum() == 0)

    return g, id


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


def plot_debug(model_outputs, ground_truth, epoch, path):

    # name_list = model_outputs.keys()
    name_list = ['points', 'object_mask', 'network_object_mask', 'sdf_output', 'mirror_sequence', 'rgb_values']

    mdic = {name: model_outputs[name].detach().cpu().clone().numpy() for name in name_list}
    mdic['mirror_sequence_edge'] = ground_truth['mirror_sequence_edge'].detach().cpu().clone().numpy()
    savemat(f"{path}/var_epoch{epoch}.mat", mdic)


def plot_loss(loss_test, loss_train, path, plot_freq):
    # train
    epoch_train = np.arange(len(loss_train["total"]))
    plt.clf()
    plt.plot(epoch_train, "total", data=loss_train, label="total")
    plt.plot(epoch_train, "rgb_loss", data=loss_train, label="rgb")
    plt.plot(epoch_train, "eikonal_loss", data=loss_train, label="eikonal")
    plt.plot(epoch_train, "mask_loss", data=loss_train, label="mask")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f"train loss")
    plt.legend()
    plt.savefig(f"{path}/loss_train.png")


def plot_rendering_gif(
    gif_var,
    mirror_sequence_cm,
    eval_num_bounce,
    rgb_in,
    results_dir,
    epoch,
    img_res,
    writer,
):

    # rendering
    gif_var.images_rendering = gif_util.gif_rendering(
        rgb_in,
        gif_var.images_rendering,
        results_dir,
        img_res,
        epoch
    )

    # mirror sequence
    gif_var.images_mirror_sequence = gif_util.gif_mirror_sequence(
        mirror_sequence_cm,
        gif_var.images_mirror_sequence,
        results_dir,
        img_res,
        epoch
    )

    # bounce
    gif_var.images_eval_num_bounce = gif_util.gif_num_bounce(
        eval_num_bounce,
        gif_var.images_eval_num_bounce,
        results_dir,
        img_res,
        epoch
    )

    # tensorboard
    writer.add_video(
        'video_rendering',
        gif_util.list2tensor_gif(gif_var.images_rendering),
        fps=10
    )
    # variables to keys
    return gif_var


def plot_surface(
    epoch,
    model,
    resolution,
    path_results,
    writer
):
    # plot surface
    surface_traces, meshexport = get_surface_trace(
        path=None,
        epoch=epoch,
        sdf=lambda x: model.implicit_network(x)[:, 0],
        resolution=resolution,
        path_results=path_results,
        writer=writer,
        is_intermediate_save=False
    )

    return meshexport
