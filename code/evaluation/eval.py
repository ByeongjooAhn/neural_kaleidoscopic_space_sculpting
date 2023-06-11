import sys
sys.path.append('../code')
from pytorch3d.renderer.cameras import look_at_view_transform
import imageio
from utils import rend_util
import utils.plots as plt
import utils.general as utils
import math
from PIL import Image
import cvxpy as cp
import numpy as np
import torch
from pyhocon import ConfigFactory
import os
import GPUtil
import argparse
from tqdm.auto import tqdm



def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_rendering = kwargs['eval_rendering']
    eval_surface = kwargs['eval_surface']
    eval_levelset = kwargs['eval_levelset']
    frame = kwargs['frame']
    elev_init = kwargs['elev_init']
    azim_init = kwargs['azim_init']
    scale = kwargs['scale']

    expname = conf.get_string('dataset.expname') + '-' + kwargs['expname']

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    dataset_conf = conf.get_config('dataset')
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

    # settings for camera optimization
    scale_mat = eval_dataset.get_scale_mat()

    if eval_rendering:
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
        total_pixels = eval_dataset.total_pixels
        img_res = eval_dataset.img_res

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, str(frame), 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']

    ####################################################################################################################
    print("evaluating...")

    model.eval()

    resultsdir = f"../results_sculpting/{conf.get_string('dataset.expname')}/{kwargs['expname']}/{timestamp}"
    if eval_surface:
        with torch.no_grad():

            # debug
            mesh = plt.get_surface_high_res_mesh(
                sdf=lambda x: model.implicit_network(x)[:, 0],
                resolution=kwargs['resolution']
            )

            # Taking the biggest connected component
            components = mesh.split(only_watertight=False)
            areas = np.array([c.area for c in components], dtype=float)
            mesh_clean = components[areas.argmax()]

            # Color
            cam_loc = torch.tensor([0.4749, -0.1580, -15.6970]).cuda()
            points = torch.tensor(mesh_clean.vertices).float().cuda()
            feature_vectors = model.implicit_network(points)[:, 1:]
            normals = torch.tensor(mesh_clean.vertex_normals).float().cuda()

            # label dependence
            label = torch.zeros((points.shape[0], 1), dtype=torch.long).cuda()
            color = model.rendering_network(points, normals, label, feature_vectors)
            mesh_clean.visual.vertex_colors = (255 * (color.cpu() + 1) / 2).int()
            mesh_clean.export('{0}/surface_hrtexture_label_{1}.ply'.format(evaldir, epoch), 'ply')

    # level set
    if eval_levelset:
        with torch.no_grad():
            # level_list = [-0.03, 0, 0.03, 0.06]
            level_list = [0]
            for level in level_list:
                print(f"level: {level}")
                # debug
                mesh = plt.get_surface_high_res_mesh(
                    sdf=lambda x: model.implicit_network(x)[:, 0],
                    resolution=kwargs['resolution'],
                    level=level
                )

                # Taking the biggest connected component
                components = mesh.split(only_watertight=False)
                areas = np.array([c.area for c in components], dtype=float)
                mesh_clean = components[areas.argmax()]

                # Color
                cam_loc = torch.tensor([0.4749, -0.1580, -15.6970]).cuda()
                points = torch.tensor(mesh_clean.vertices).float().cuda()
                feature_vectors = model.implicit_network(points)[:, 1:]
                normals = torch.tensor(mesh_clean.vertex_normals).float().cuda()
                color = normals

                mesh_clean.visual.vertex_colors = (255 * (color.cpu() + 1) / 2).int()
                mesh_clean.export('{0}/surface_level_{1}.ply'.format(evaldir, level), 'ply')
                mesh_clean.export('{0}/surface_level_{1}.ply'.format(resultsdir, level), 'ply')

    if eval_rendering:  # low-res rendering for view synthesis
        images_dir = '{0}/rendering'.format(evaldir)
        utils.mkdir_ifnotexists(images_dir)

        # scale = 1/4
        img_res = [500, 500]
        total_pixels = int(img_res[0] * img_res[1])

        psnrs = []
        for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda() * scale
            model_input["intrinsics"][0, -1, -1] = 1

            # scale: 1/4
            model_input["intrinsics"][0, 0, -1] = 300.0
            model_input["intrinsics"][0, 1, -1] = 200.0

            uv = np.mgrid[0:(img_res[0]), 0:(img_res[1])].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            uv = uv.reshape(2, -1).transpose(1, 0)
            uv = uv.reshape(1, -1, 2)

            model_input["uv"] = uv.cuda()
            model_input["object_mask"] = torch.ones(1, img_res[0] * img_res[1]).bool().cuda()

            # pose
            model_input['pose'] = model_input['pose'].cuda()

            # decompose
            model_input["pose_all"] = []
            model_input["img_res"] = []
            model_input["is_concat"] = False
            model_input["virtual_sculpting"] = False
            model_input["bounce_hit_visualhull"] = torch.ones_like(model_input["object_mask"]).bool()
            model_input["is_eval_single"] = True

            R_init, _ = look_at_view_transform(
                elev=elev_init,
                azim=azim_init
            )

            R_real = model_input['pose'][0, 0:3, 0:3].clone()
            T_real = model_input['pose'][0, :-1, -1].clone()
            dist = T_real.norm()
            elev = 0
            up = ((0, 1, 0),)

            # rotate
            img_list = []
            azim_list = range(-180, 180, 10)

            for i, azim in enumerate(azim_list):
                R_, _ = look_at_view_transform(
                    elev=elev, azim=azim, up=up)

                R = R_init.cuda() @ R_.cuda() @ R_real
                T = R_init.cuda() @ R_.cuda() @ T_real
                
                model_input['pose'][0, 0:3, 0:3] = R
                model_input['pose'][0, 0:3, -1] = T


                split = utils.split_input(model_input, total_pixels)
                res = []
                for s in tqdm(split, desc=f'{expname} evaluation'):
                    out = model(s)
                    res.append({
                        'rgb_values': out['rgb_values'].detach().cpu()
                    })

                batch_size = ground_truth['rgb'].shape[0]
                model_outputs = utils.merge_output(res, total_pixels, batch_size)
                rgb_eval = model_outputs['rgb_values']
                rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)

                rgb_eval = (rgb_eval + 1.) / 2.
                rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
                rgb_eval = rgb_eval.transpose(1, 2, 0)
                img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
                img.save(f'{images_dir}/eval_{i}.png')
                img_list.append(np.array(img))

                mask = model_input['object_mask']
                mask = plt.lin2img(mask.unsqueeze(-1), img_res).cpu().numpy()[0]
                mask = mask.transpose(1, 2, 0)

                print(f'rendering {i}/{len(azim_list)}: azim: {azim}')
            imageio.mimsave(f'{evaldir}/eval.gif', img_list, fps=10)
            imageio.mimsave(f'{resultsdir}/eval.gif', img_list, fps=10)
            break


def get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts,):
    ''' Align predicted pose to gt pose and print cameras accuracy'''

    # find rotation
    d = pred_Rs.shape[-1]
    n = pred_Rs.shape[0]

    Q = torch.addbmm(torch.zeros(d, d, dtype=torch.double), gt_Rs, pred_Rs.transpose(1, 2))
    Uq, _, Vq = torch.svd(Q)
    sv = torch.ones(d, dtype=torch.double)
    sv[-1] = torch.det(Uq @ Vq.transpose(0, 1))
    R_opt = Uq @ torch.diag(sv) @ Vq.transpose(0, 1)
    R_fixed = torch.bmm(R_opt.repeat(n, 1, 1), pred_Rs)

    # find translation
    pred_ts = pred_ts @ R_opt.transpose(0, 1)
    c_opt = cp.Variable()
    t_opt = cp.Variable((1, d))

    constraints = []
    obj = cp.Minimize(cp.sum(
        cp.norm(gt_ts.numpy() - (c_opt * pred_ts.numpy() + np.ones((n, 1), dtype=np.double) @ t_opt), axis=1)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    t_fixed = c_opt.value * pred_ts.numpy() + np.ones((n, 1), dtype=np.double) * t_opt.value

    # Calculate transaltion error
    t_error = np.linalg.norm(t_fixed - gt_ts.numpy(), axis=-1)
    t_error = t_error
    t_error_mean = np.mean(t_error)
    t_error_medi = np.median(t_error)

    # Calculate rotation error
    R_error = compare_rotations(R_fixed, gt_Rs)

    R_error = R_error.numpy()
    R_error_mean = np.mean(R_error)
    R_error_medi = np.median(R_error)

    print('CAMERAS EVALUATION: R error mean = {0} ; t error mean = {1} ; R error median = {2} ; t error median = {3}'
          .format("%.2f" % R_error_mean, "%.2f" % t_error_mean, "%.2f" % R_error_medi, "%.2f" % t_error_medi))

    # return alignment and aligned pose
    return R_opt.numpy(), t_opt.value, c_opt.value, R_fixed.numpy(), t_fixed


def compare_rotations(R1, R2):
    cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return cos_err.acos() * 180 / np.pi


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def get_rbg_value(model, points, feature_vectors, normals, view_dirs):
   
    rgb_vals = model.rendering_network(points, normals, view_dirs, feature_vectors)

    return rgb_vals


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/armadillo_real_scale1.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest', type=str, help='The trained model checkpoint to test')
    parser.add_argument('--resolution', default=400, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')
    parser.add_argument('--eval_rendering', default=False, action="store_true", help='If set, evaluate rendering quality.')
    parser.add_argument('--eval_surface', default=False, action="store_true", help='If set, evaluate rendering quality.')
    parser.add_argument('--eval_levelset', default=False, action="store_true", help='If set, evaluate rendering quality.')
    parser.add_argument('--frame', type=int, default=0, help='frame index')
    parser.add_argument('--elev_init', type=float, default=0.0, help='elev_init')
    parser.add_argument('--azim_init', type=float, default=-180.0, help='azim_init')
    parser.add_argument('--scale', type=float, default=1, help='cam scale')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             resolution=opt.resolution,
             eval_rendering=opt.eval_rendering,
             eval_surface=opt.eval_surface,
             eval_levelset=opt.eval_levelset,
             frame=opt.frame,
             elev_init=opt.elev_init,
             azim_init=opt.azim_init,
             scale=opt.scale
             )
