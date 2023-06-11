import torch
from torch import nn
from torch.nn import functional as F


class IDRLoss(nn.Module):
    def __init__(self, rgb_weight, eikonal_weight, mask_weight, carving_weight, alpha, use_bg_last=False, reflectivity=1.0):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.carving_weight = carving_weight
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.reflectivity = reflectivity
        self.use_bg_last = use_bg_last

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask, mirror_sequence_edge, n_bounce, invalid_fg_mask):
        object_mask = object_mask & ~mirror_sequence_edge & ~invalid_fg_mask
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = (rgb_values + 1.) / 2
        rgb_values *= (self.reflectivity**n_bounce).reshape(-1, 1).repeat(1, 3)
        rgb_values = 2 * rgb_values - 1

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask, mirror_sequence_edge, invalid_fg_mask):
        if self.use_bg_last:  # model and carve for last bounce
            mask = ~(network_object_mask & object_mask) & ~mirror_sequence_edge & ~invalid_fg_mask
        else:  # modeling only
            mask = ~network_object_mask & object_mask & ~mirror_sequence_edge & ~invalid_fg_mask  # modeling only
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(1), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_carving_loss(self, sdf_output_carving, object_mask, max_bounce=10):  # bg loss # TODO: use is_carving and point instead of new shape for merging
        if sdf_output_carving is None or sdf_output_carving.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        sdf_pred = -self.alpha * sdf_output_carving.view(-1)
        gt = torch.zeros_like(sdf_pred)
        carving_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred, gt, reduction='sum') / float(object_mask.shape[0] * max_bounce)

        return carving_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']
        mirror_sequence_edge = ground_truth['mirror_sequence_edge'].squeeze().cuda()
        n_bounce = ground_truth['cam_num_bounce'].float().squeeze().cuda()
        invalid_fg_mask = model_outputs['invalid_fg_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask, mirror_sequence_edge, n_bounce, invalid_fg_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask, mirror_sequence_edge, invalid_fg_mask)
        carving_loss = self.get_carving_loss(model_outputs['sdf_output_carving'], object_mask)
        # carving_loss = torch.tensor(0).cuda().float()
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        loss = self.rgb_weight * rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss + \
               self.carving_weight * carving_loss

        return {
            'loss': loss,
            'rgb_loss': self.rgb_weight * rgb_loss,
            'eikonal_loss': self.eikonal_weight * eikonal_loss,
            'mask_loss': self.mask_weight * mask_loss,
            'carving_loss': self.carving_weight * carving_loss,
        }
