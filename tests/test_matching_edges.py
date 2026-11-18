"""CPU regression cases for DGIM geometry and candidate selection."""

import unittest

import torch

from src.dgim.utils.coarse_matching import CoarseMatching
from src.dgim.utils.fine_matching import FineMatching
from src.dgim.utils.supervision import spvs_coarse, spvs_fine
from src.losses.dgim_loss import DGIMLoss


GEOMETRY = {'DGIM': {'RESOLUTION': (8, 2), 'FINE_WINDOW_SIZE': 5}}


def image_pair(homography):
    return {
        'image0': torch.zeros(1, 1, 64, 64),
        'image1': torch.zeros(1, 1, 64, 64),
        'T_0to1': homography[None],
        'T_1to0': torch.linalg.inv(homography)[None],
        'hw0_c': (8, 8), 'hw1_c': (8, 8),
        'hw0_f': (32, 32), 'hw1_f': (32, 32),
        'hw0_i': (64, 64), 'pair_id': [0],
    }


def fine_matcher():
    return FineMatching({'fine': {'dsmax_temperature': 0.1, 'thr': 0.1},
                         'fine_window_size': 5, 'backbone': {'block_dims': [4]}})


def matching_loss():
    return DGIMLoss({'dgim': {'loss': {
        'pos_weight': 1.0, 'neg_weight': 1.0, 'fine_type': 'l2',
        'coarse_weight': 0.5, 'fine_weight': 0.3, 'sub_weight': 0.1,
        'focal_alpha': 0.25, 'focal_gamma': 2.0,
    }}})


class MatchingEdges(unittest.TestCase):
    def test_identity_supervises_the_origin_and_every_interior_window_cell(self):
        batch = image_pair(torch.eye(3))
        spvs_coarse(batch, GEOMETRY)
        torch.testing.assert_close(batch['conf_matrix_gt'][0], torch.eye(64))

        # Both windows are centred at coarse cell (3, 3), clear of any padding.
        batch.update(b_ids=torch.tensor([0]), i_ids=torch.tensor([27]), j_ids=torch.tensor([27]))
        spvs_fine(batch, GEOMETRY)
        torch.testing.assert_close(batch['conf_matrix_f_gt'][0], torch.eye(25))

    def test_uniform_scores_do_not_produce_a_fine_match(self):
        matcher = fine_matcher().eval()
        batch = image_pair(torch.eye(3))
        batch.update(b_ids=torch.tensor([0]), i_ids=torch.tensor([27]), j_ids=torch.tensor([27]))
        features = torch.zeros(1, 25, 4)  # each dual-softmax score is 1/625
        matcher(features, features, batch)
        self.assertEqual(batch['mconf_f'].numel(), 0)
        self.assertEqual(tuple(batch['mkpts0_f'].shape), (0, 2))

    def test_disjoint_images_with_no_candidates_have_differentiable_zero_loss(self):
        translation = torch.eye(3)
        translation[0, 2] = 1024
        batch = image_pair(translation)
        spvs_coarse(batch, GEOMETRY)
        self.assertEqual(batch['spv_b_ids'].numel(), 0)
        coarse = CoarseMatching({'thr': 0.3, 'border_rm': 2, 'train_coarse_percent': 0.3,
                                 'train_pad_num_gt_min': 200, 'dsmax_temperature': 0.1})
        features = torch.zeros(1, 64, 4, requires_grad=True)
        coarse(features, features, batch)
        self.assertEqual(batch['b_ids'].numel(), 0)
        windows = torch.empty(0, 25, 4, requires_grad=True)
        fine_matcher()(windows, windows, batch)
        spvs_fine(batch, GEOMETRY)
        matching_loss()(batch)
        torch.testing.assert_close(batch['loss'], torch.tensor(0.0))
        batch['loss'].backward()
        self.assertIsNotNone(windows.grad)
        torch.testing.assert_close(features.grad, torch.zeros_like(features))

    def test_projective_horizon_does_not_poison_gradient_of_valid_match(self):
        transform = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
        source = torch.tensor([[0., 0.], [1., 1.]], requires_grad=True)
        target = torch.tensor([[4., 2.], [3., 1.]], requires_grad=True)
        loss = matching_loss().compute_sub_pixel_loss({
            'T_0to1': transform[None], 'm_bids': torch.tensor([0, 0]),
            'mkpts0_f_train': source, 'mkpts1_f_train': target,
        })
        torch.testing.assert_close(loss, torch.tensor(1.0))
        loss.backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        torch.testing.assert_close(source.grad[0], torch.zeros(2))
        torch.testing.assert_close(target.grad[1], torch.tensor([1., 0.]))
