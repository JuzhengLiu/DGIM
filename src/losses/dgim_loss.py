import torch
import torch.nn as nn


class DGIMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['dgim']['loss']

        # coarse-level
        self.pos_w = self.loss_config['pos_weight']
        self.neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, data, weight=None):
        """
        Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_0_to_1 (torch.Tensor): (N, HW0, HW1)
            conf_matrix_1_to_0 (torch.Tensor): (N, HW0, HW1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            }
            weight (torch.Tensor): (N, HW0, HW1)
        """
        # Only positive correspondences contribute to this term. Select them
        # before clamping instead of retaining two full [B, HW0, HW1] copies.
        b, i, j = data["conf_matrix_gt"].nonzero(as_tuple=True)
        forward = data["conf_matrix_0_to_1"][b, i, j].float().clamp(1e-6, 1 - 1e-6)
        reverse = data["conf_matrix_1_to_0"][b, i, j].float().clamp(1e-6, 1 - 1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']
        loss_pos = -alpha * ((1 - forward).pow(gamma) * forward.log()
                             + (1 - reverse).pow(gamma) * reverse.log())
        if weight is not None:
            loss_pos = loss_pos * weight[b, i, j]
        elif 'mask0' in data:
            loss_pos = loss_pos * (data['mask0'].flatten(1)[b, i] & data['mask1'].flatten(1)[b, j])
        return self.pos_w * loss_pos.sum() / max(loss_pos.numel(), 1)

    def compute_fine_loss(self, data):
        """
        Point-wise Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_fine (torch.Tensor): (N, W_f^2, W_f^2)
            conf_matrix_f_gt (torch.Tensor): (N, W_f^2, W_f^2)
            }
        """
        conf_matrix_fine = data['conf_matrix_fine'].float()
        if conf_matrix_fine.numel() == 0:
            return conf_matrix_fine.sum()
        conf_matrix_f_gt = data['conf_matrix_f_gt']
        pos_mask, neg_mask = conf_matrix_f_gt > 0, conf_matrix_f_gt == 0
        pos_w, neg_w = self.pos_w, self.neg_w

        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            neg_w = 0.

        conf_matrix_fine = torch.clamp(conf_matrix_fine, 1e-6, 1 - 1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']

        loss_pos = - alpha * torch.pow(1 - conf_matrix_fine[pos_mask], gamma) * (conf_matrix_fine[pos_mask]).log()
        # loss_pos *= conf_matrix_f_gt[pos_mask]
        loss_neg = - alpha * torch.pow(conf_matrix_fine[neg_mask], gamma) * (1 - conf_matrix_fine[neg_mask]).log()

        return pos_w * loss_pos.mean() + neg_w * loss_neg.mean()

    def compute_sub_pixel_loss(self, data):
        """
        Args:
        data (dict): {
            m_bids (torch.Tensor): (N)
            T_0to1 (torch.Tensor): (B, 3, 3)
            mkpts0_f_train (torch.Tensor): (N, 2)
            mkpts1_f_train (torch.Tensor): (N, 2)
            }
        """
        m_bids = data['m_bids']
        kpts0 = data['mkpts0_f_train']
        kpts1 = data['mkpts1_f_train']
        T_0to1 = data['T_0to1'][m_bids]

        # Geometry stays in FP32 even when the matcher uses mixed precision.
        with torch.autocast(device_type=kpts0.device.type, enabled=False):
            kpts0, kpts1 = kpts0.float(), kpts1.float()
            homogeneous = torch.cat((kpts0, torch.ones_like(kpts0[:, :1])), dim=1)
            projected = (T_0to1.float() @ homogeneous.unsqueeze(-1)).squeeze(-1)
            xy, z = projected[:, :2], projected[:, 2:]
            # Apply DGIM's five-pixel cutoff in homogeneous coordinates. A
            # point on the projective horizon must never enter the division.
            with torch.no_grad():
                residual = torch.linalg.vector_norm(xy - kpts1 * z, dim=1)
                keep = (z[:, 0] != 0) & (residual <= 5.0 * z[:, 0].abs())
            distances = torch.linalg.vector_norm(xy[keep] / z[keep] - kpts1[keep], dim=1)
            return distances.sum() / max(distances.numel(), 1)

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(data)
        loss_c *= self.loss_config['coarse_weight']
        loss = loss_c
        loss_scalars.update({"loss_c": loss_c.detach()})

        # 2. fine-level matching loss for windows
        loss_f = self.compute_fine_loss(data)
        loss_f *= self.loss_config['fine_weight']
        loss = loss + loss_f
        loss_scalars.update({"loss_f": loss_f.detach()})

        # 3. sub-pixel refinement loss
        loss_sub = self.compute_sub_pixel_loss(data)
        loss_sub *= self.loss_config['sub_weight']
        loss = loss + loss_sub
        loss_scalars.update({"loss_sub": loss_sub.detach()})

        loss_scalars.update({'loss': loss.detach()})
        data.update({"loss": loss,
                     "loss_c": loss_c,
                     "loss_f": loss_f,
                     "loss_sub": loss_sub,
                     "loss_scalars": loss_scalars})
