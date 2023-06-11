import os
import sys
import torch
import shutil
from datetime import datetime
from pyhocon import ConfigFactory
from types import SimpleNamespace

import utils.plots as plt
import utils.general as utils
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm


class IDRTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        # torch.set_num_threads(1)

        # read configs
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.nepochs_frame0 = self.conf.get_int('train.nepochs_frame0', default=3000)
        self.nepochs = self.conf.get_int('train.nepochs', default=200)
        self.expname = self.conf.get_string('dataset.expname') + '-' + kwargs['expname']
        self.expnum = kwargs['expname']

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], self.expname)):
                timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        # set directories
        utils.mkdir_ifnotexists(os.path.join('../', self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        # set plot dirs
        self.plots_dir_ = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir_)
        self.results_dir_ = os.path.join(self.expdir, self.timestamp, f'r_{self.expname}')
        utils.mkdir_ifnotexists(self.results_dir_)

        # set checkpoints dirs
        self.checkpoints_path_ = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path_)

        # set save dir
        utils.mkdir_ifnotexists('../results_sculpting')
        save_path_ = f"../results_sculpting/{self.conf.get_string('dataset.expname')}"
        utils.mkdir_ifnotexists(save_path_)
        save_path_ = os.path.join(save_path_, self.expnum)
        utils.mkdir_ifnotexists(save_path_)
        self.save_path = os.path.join(save_path_, self.timestamp)
        utils.mkdir_ifnotexists(self.save_path)
        self.save_iter = os.path.join(self.save_path, 'iterations.txt')

        # copy runconf
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # set GPU
        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        # set dataset
        dataset_conf = self.conf.get_config('dataset')
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)
        print('Finish loading data ...')

        # set dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        # set model
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        # set loss
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        # set optimizer
        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # set checkpoint
        self.start_epoch = 0
        frame = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints', str(frame))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'SchedulerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        # read other parameters in config
        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.num_pixels_milestones = self.conf.get_list('train.num_pixels_milestones', default=[])
        self.num_pixels_factor = self.conf.get_float('train.num_pixels_factor', default=1)
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_images_total = self.train_dataset.n_images_total
        self.pose_all = self.train_dataset.pose_all
        self.virtual_sculpting = self.train_dataset.virtual_sculpting
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.surface_freq = self.conf.get_int('train.surface_freq', default=100)
        self.label_update_start = self.conf.get_int('train.label_update_start', default=5000)
        self.plot_conf = self.conf.get_config('plot')
        self.elev_init = self.conf.get_float('visualization.elev_init', default=0.0)
        self.is_pretrain = self.conf.get_bool('train.is_pretrain')
        self.frame_independent = self.conf.get_bool('train.frame_independent')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

        # set tensorboard
        log_dir = os.path.join(self.expdir, self.timestamp)
        self.writer = SummaryWriter(log_dir=log_dir, comment=f"{self.expname}")

        # set parameters for kaleidoscope
        self.mirrors_n, self.mirrors_d = self.train_dataset.get_mirror()

        # set n_frames
        self.n_frames = self.train_dataset.get_n_frames()

        # set n_batches
        self.n_batches = len(self.train_dataloader)

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):
        # initialize debug variables
        psnr = 0
        mask_error = 0
        label_error = 0
        dist_chamfer = 0

        # tensorboard
        writer = self.writer

        # IDR
        epoch_prev = 0
        for frame in range(self.n_frames):
            # update frame in dataset
            self.train_dataset.change_frame(frame)

            # update new plots dir
            self.plots_dir = os.path.join(self.plots_dir_, str(frame))
            utils.mkdir_ifnotexists(self.plots_dir)
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'surface'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'depth'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'rendering'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'mask'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'mirror_sequence'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'num_bounce'))
            utils.mkdir_ifnotexists(os.path.join(self.plots_dir, 'count_penetration'))

            # update new results dir
            self.results_dir = os.path.join(self.results_dir_, str(frame))
            utils.mkdir_ifnotexists(self.results_dir)
            utils.mkdir_ifnotexists(os.path.join(self.results_dir, 'surface'))
            utils.mkdir_ifnotexists(os.path.join(self.results_dir, 'surface_error'))
            utils.mkdir_ifnotexists(os.path.join(self.results_dir, 'pc'))

            # update new checkpoints dir
            self.checkpoints_path = os.path.join(self.checkpoints_path_, str(frame))
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            # update nepoch
            if frame == 0:
                nepochs = self.nepochs_frame0
            else:
                nepochs = self.nepochs

            # run each frame independently
            if self.frame_independent:
                # reset model
                self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
                if torch.cuda.is_available():
                    self.model.cuda()

                # reset loss
                self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

                # reset optimizer
                self.lr = self.conf.get_float('train.learning_rate')
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
                self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)
                self.num_pixels = self.conf.get_int('train.num_pixels')

                # reset nepochs
                nepochs = self.nepochs_frame0

            print(f"training ...")

            # for loss plot
            loss_train = {}
            loss_train["total"] = []
            loss_train["rgb_loss"] = []
            loss_train["eikonal_loss"] = []
            loss_train["mask_loss"] = []
            loss_train["carving_loss"] = []

            # for gif rendering
            gif_rendering_var = SimpleNamespace(
                images_rendering=[],
                images_mirror_sequence=[],
                images_eval_num_bounce=[],
                images_count_penetration=[]
            )

            gif_surface_var = SimpleNamespace(
                images_mesh_dic=None,
                images_mesh_error_dic=None
            )

            for epoch in range(self.start_epoch, nepochs + 1):
                epoch_total = epoch + epoch_prev

                # milestones
                if self.frame_independent or frame == 0:
                    if epoch in self.alpha_milestones:
                        self.loss.alpha = self.loss.alpha * self.alpha_factor

                    if epoch in self.num_pixels_milestones:
                        self.num_pixels = int(self.num_pixels * self.num_pixels_factor)

                # checkpoints
                if (epoch == nepochs) or (epoch == nepochs // 2):
                    self.save_checkpoints(epoch)

                # evaluation
                if ((epoch % self.plot_freq == 0) or (epoch == nepochs)):  # and (epoch > 0):
                    self.model.eval()

                    # load data
                    self.train_dataset.change_sampling_idx(-1)
                    indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input["object_mask"] = model_input["object_mask"].cuda()
                    model_input["epoch"] = epoch

                    # mirror
                    model_input["num_mirror"] = len(self.mirrors_n)
                    for i in range(model_input["num_mirror"]):
                        model_input[f"mirrors_n_{i}"] = self.mirrors_n[i].cuda()
                        model_input[f"mirrors_d_{i}"] = self.mirrors_d[i].cuda()
                    model_input["mirror_sequence_edge"] = ground_truth["mirror_sequence_edge"].cuda()

                    # pose
                    model_input['pose'] = model_input['pose'].cuda()

                    # num_bounces
                    model_input["vh_num_bounce"] = ground_truth["vh_num_bounce"].cuda()
                    model_input["bounce_hit_visualhull"] = ground_truth["bounce_hit_visualhull"].cuda()

                    # other parameters
                    model_input['pose_all'] = torch.stack(self.pose_all).cuda()
                    model_input['img_res'] = self.img_res
                    model_input['virtual_sculpting'] = self.virtual_sculpting
                    model_input['is_eval_single'] = False

                    # evaluation
                    split = utils.split_input(model_input, self.total_pixels)
                    res = []
                    for s in tqdm(split, desc=f'{self.expname} evaluation [{frame}/{self.n_frames}] [{epoch}/{nepochs}]'):
                        out = self.model(s)
                        res.append({
                            'points': out['points'].detach().cpu(),
                            'rgb_values': out['rgb_values'].detach().cpu(),
                            'network_object_mask': out['network_object_mask'].detach().cpu(),
                            'object_mask': out['object_mask'].detach().cpu(),
                            'sdf_output': out['sdf_output'].detach().cpu(),  # debug start
                            'mirror_sequence': out['mirror_sequence'].detach().cpu(),
                            'eval_num_bounce': out['eval_num_bounce'].detach().cpu(),
                            'loc_bounce': out['loc_bounce'].detach().cpu(),
                            'dir_bounce': out['dir_bounce'].detach().cpu(),
                            'dists_bounce': out['dists_bounce'].detach().cpu(),
                            'distmax_bounce': out['distmax_bounce'].detach().cpu(),
                            'is_carving_fg_bounce': out['is_carving_fg_bounce'].detach().cpu(),
                            'is_carving_bg_bounce': out['is_carving_bg_bounce'].detach().cpu(),

                        })

                    batch_size = ground_truth['rgb'].shape[0]
                    model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                    # plot
                    meshexport, mirror_sequence_cm, psnr, mask_error, label_error, label_error_fg, vh_valid_error = plt.plot(
                        self.model,
                        indices,
                        model_outputs,
                        model_input['pose'].cpu(),
                        ground_truth,
                        self.plots_dir,
                        self.results_dir,
                        epoch_total,
                        self.img_res,
                        **self.plot_conf,
                        writer=writer
                    )

                    # gif
                    gif_rendering_var = plt.plot_rendering_gif(
                        gif_rendering_var,
                        mirror_sequence_cm,
                        model_outputs['eval_num_bounce'],
                        model_outputs["rgb_values"].detach().cpu(),
                        self.results_dir,
                        epoch_total,
                        self.img_res,
                        writer=writer
                    )

                    # save loss
                    model_outputs["grad_theta"] = torch.empty(0)  # skip grad_theta because of memory
                    model_outputs["sdf_output_carving"] = torch.empty(0)  # skip point carving because of different shape

                    self.model.train()

                    # update mask
                    if not self.virtual_sculpting:
                        if ((epoch >= self.label_update_start) & (epoch > 0)):
                            self.train_dataset.update_mask(
                                model_outputs['mirror_sequence']
                            )

                    # save tp save
                    filename_list = ['rendering.png', 'rendering_diff.png', 'rendering_diff_noedge.png',
                                     'mask.png', 'mask_ignoreedge.png', 'mirror_sequence.png']
                    for filename in filename_list:
                        source = os.path.join(self.results_dir, filename)
                        target = os.path.join(self.save_path, filename)
                        shutil.copyfile(source, target)
                    with open(self.save_iter, 'w') as f:
                        f.write(
                            f'iter: {epoch_total}, psnr: {psnr}, mask_error: {mask_error}, '
                            f'label_error: {label_error}, label_error_fg: {label_error_fg}, vh_valid_error: {vh_valid_error}, '
                            f'dist_chamfer: {dist_chamfer}'
                        )

                # surface evaluation
                if ((epoch % self.surface_freq == 0) or (epoch == nepochs)):
                    self.model.eval()

                    resolution = self.plot_conf.get_int('resolution')

                    meshexport = plt.plot_surface(
                        epoch_total,
                        self.model,
                        resolution,
                        self.results_dir,
                        writer=writer
                    )

                    # save to save dir
                    filename_list = ['surface.ply']
                    for filename in filename_list:
                        source = os.path.join(self.results_dir, filename)
                        target = os.path.join(self.save_path, filename)
                        shutil.copyfile(source, target)
                    with open(self.save_iter, 'w') as f:
                        f.write(
                            f'iter: {epoch_total}, psnr: {psnr}, mask_error: {mask_error}, '
                            f'label_error: {label_error}, label_error_fg: {label_error_fg}, vh_valid_error: {vh_valid_error}'
                        )

                    self.model.train()

                # train
                self.train_dataset.change_sampling_idx(self.num_pixels)

                for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input["object_mask"] = model_input["object_mask"].cuda()
                    model_input["epoch"] = epoch

                    # mirror
                    model_input["num_mirror"] = len(self.mirrors_n)
                    for i in range(model_input["num_mirror"]):
                        model_input[f"mirrors_n_{i}"] = self.mirrors_n[i].cuda()
                        model_input[f"mirrors_d_{i}"] = self.mirrors_d[i].cuda()
                    model_input["mirror_sequence_edge"] = ground_truth["mirror_sequence_edge"].cuda()

                    # camera
                    model_input['pose'] = model_input['pose'].cuda()

                    # num_bounces
                    model_input["vh_num_bounce"] = ground_truth["vh_num_bounce"].cuda()
                    model_input["gt_num_bounce"] = ground_truth["gt_num_bounce"].cuda()
                    model_input["bounce_hit_visualhull"] = ground_truth["bounce_hit_visualhull"].cuda()

                    # other parameters
                    model_input['pose_all'] = torch.stack(self.pose_all).cuda()
                    model_input['img_res'] = self.img_res
                    model_input['virtual_sculpting'] = self.virtual_sculpting
                    model_input['is_eval_single'] = False

                    # model
                    model_outputs = self.model(model_input)

                    # loss
                    loss_output = self.loss(model_outputs, ground_truth)
                    loss = loss_output['loss']
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # log
                    print(
                        f"{self.expname} [{data_index}/{self.n_batches}, {indices.item()}] [{epoch}/{nepochs}]\n"
                        f"loss = {loss.item():.06f}, "
                        f"rgb_loss = {loss_output['rgb_loss'].item():.06f}, "
                        f"eikonal_loss = {loss_output['eikonal_loss'].item():.06f}, "
                        f"mask_loss = {loss_output['mask_loss'].item():.06f}, "
                        f"carving_loss = {loss_output['carving_loss'].item():.06f}, "
                        f"alpha = {self.loss.alpha:.06f}, "
                        f"lr = {self.scheduler.get_lr()[0]:.06f}"
                    )

                    loss_train["total"].append(loss_output['loss'].item())
                    loss_train["rgb_loss"].append(loss_output['rgb_loss'].item())
                    loss_train["eikonal_loss"].append(loss_output['eikonal_loss'].item())
                    loss_train["mask_loss"].append(loss_output['mask_loss'].item())
                    loss_train["carving_loss"].append(loss_output['carving_loss'].item())

                    writer.add_scalar('loss/total', loss_output['loss'].item(), epoch_total)
                    writer.add_scalar('loss/rgb', loss_output['rgb_loss'].item(), epoch_total)
                    writer.add_scalar('loss/eikonal', loss_output['eikonal_loss'].item(), epoch_total)
                    writer.add_scalar('loss/mask', loss_output['mask_loss'].item(), epoch_total)
                    writer.add_scalar('loss/carving', loss_output['carving_loss'].item(), epoch_total)

                self.scheduler.step()

            # accumulate previous epoch for tensorboard
            epoch_prev += epoch