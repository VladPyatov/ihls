from collections import defaultdict

from kornia.geometry.epipolar import essential_from_Rt, fundamental_from_essential
from kornia.geometry.conversions import convert_points_to_homogeneous
from loguru import logger
import torch
import torch.nn as nn

from .utils.irls import FundamentalMatrixEstimator
from .utils.essential import find_essential
from .utils.fundamental import find_fundamental
from .dfe import IRLSEightPointNet, EightPointNet


def create_empty_stats(batch_size):
    EMPTY_STATS = {
        'fine_converged': torch.ones((batch_size, 1), dtype=torch.float32),
        'fine_num_steps': torch.ones((batch_size, 1), dtype=torch.float32),
        'fine_atol': torch.ones((batch_size, 1), dtype=torch.float32),
        'fine_rtol': torch.ones((batch_size, 1), dtype=torch.float32),
    }
    return EMPTY_STATS


class FundamentalMatrixEstimation(nn.Module):
    """Fundamental Matrix Estimation with IRLS"""

    def __init__(self, config, matcher_type='loftr'):
        super().__init__()
        self.matcher_type = matcher_type
        self.robust_estimator_type = config['robust_estimator_type']
        self.compute_fine = config['compute_fine']
        self.matrix_type = config['matrix_type']
        if self.robust_estimator_type == 'LIRLS':
            self.depth = 3
            self.robust_estimator = IRLSEightPointNet(
                depth=self.depth, side_info_size=4,
                matrix_type=self.matrix_type)
        elif self.robust_estimator_type == 'IRLS':
            self.depth = 3
            self.robust_estimator = FundamentalMatrixEstimator(
                config['p'], config['num_irls_steps'], config['atol'], config['rtol'], config['eps'],
                config['learn_p'], config['initial_p'], config['initial_num_irls_steps'])
        elif self.robust_estimator_type == 'DFE':
            self.depth = 5
            self.robust_estimator = EightPointNet(depth=self.depth, side_info_size=4)
        else:
            raise NotImplementedError('Unknown robust estimator')
        self.ls_estimator = find_fundamental #if self.matrix_type == 'F' else find_essential

    def compute_side_information(self, b_idx: int, data: dict):
        b_mask = b_idx == data['b_ids']
        if self.robust_estimator_type == 'IRLS':
            return data['mconf'][b_mask][None]
        elif self.matcher_type == 'loftr':
            return torch.stack([
                data['mconf'][b_mask],
                data['std'][b_mask], 
                data['mdist'][b_mask],
                data['mcosine'][b_mask]
            ])[None].permute(0, 2, 1)
    
    def compute_points(self, b_idx: int, data: dict):
        b_mask = b_idx == data['b_ids']
        pts1, pts2 = data['mkpts0_f'][b_mask][None], data['mkpts1_f'][b_mask][None]
        if self.matrix_type == 'E':
            intr1i = torch.linalg.inv(data['K0'][[b_idx]])
            intr2i = torch.linalg.inv(data['K1'][[b_idx]])
            pts1h = convert_points_to_homogeneous(pts1)
            pts2h = convert_points_to_homogeneous(pts2)
            pts1 = (intr1i @ pts1h.transpose(-1, -2)).transpose(-1, -2)[..., :2]
            pts2 = (intr2i @ pts2h.transpose(-1, -2)).transpose(-1, -2)[..., :2]
        return pts1, pts2

    def forward(self, data: dict):
        BATCH_SIZE = data['image0'].shape[0]
        data['irls_weights'] = torch.ones_like(data['mconf'])
        MIN_NUM_OF_SAMPLES = 8 if self.matrix_type == 'F' or self.training else 5
        b_ids = data['b_ids']
        fine_fms_ls = []
        fine_fms_irls = []
        stats = defaultdict(list)
        for b_idx in range(BATCH_SIZE):
            if not self.compute_fine:
                break
            b_mask = b_idx == b_ids
            if sum(b_mask) < MIN_NUM_OF_SAMPLES:
                info = 'no matches' if sum(b_mask) == 0 else 'not enough matches'
                logger.error(f'{info} to estimate fundamental matrix correctly')
                Em = essential_from_Rt(
                    torch.eye(3, dtype=torch.float32)[None].to(data['K0'].device), 
                    torch.zeros((1,3,1), dtype=torch.float32).to(data['K0'].device), 
                    torch.eye(3, dtype=torch.float32)[None].to(data['K0'].device),
                    torch.zeros((1,3,1), dtype=torch.float32).to(data['K0'].device)
                )
                fake_Fm = fundamental_from_essential(
                    Em,
                    data['K0'][b_idx][None],
                    data['K1'][b_idx][None]
                )
                # add fake values to the result lists and stats
                fine_fms_ls.append(fake_Fm)
                fake_Fm = fake_Fm.repeat(self.depth, 1, 1)[None]
                fine_fms_irls.append(fake_Fm)
                stats['fine_converged'].append(torch.tensor([0.], dtype=torch.float64))
                stats['fine_num_steps'].append(torch.tensor([0.], dtype=torch.float64))
                stats['fine_atol'].append(torch.tensor([0.], dtype=torch.float64))
                stats['fine_rtol'].append(torch.tensor([0.], dtype=torch.float64))
                continue
            
            # compute fundamental matrix from fine-level matches
            if self.compute_fine:
                side_information = self.compute_side_information(b_idx, data)
                pts1, pts2 = self.compute_points(b_idx, data)
                fm_fine_irls, fine_stats_irls, fine_weights = self.robust_estimator(pts1, pts2, side_information)
                fm_fine_ls = self.ls_estimator(pts1, pts2, data['mconf'][b_mask][None])
                if fine_weights is not None:
                    data['irls_weights'][b_mask] = fine_weights[0]
                if self.matrix_type == 'E':
                    # by convention the module returns the fundamental matrix 
                    b, n, _, _ = fm_fine_irls.shape
                    fm_fine_irls = fundamental_from_essential(fm_fine_irls.view(b*n, 3, 3), data['K0'][b_idx][None], data['K1'][b_idx][None])
                    fm_fine_irls = fm_fine_irls.view(b, n, 3, 3)
                    fm_fine_ls = fundamental_from_essential(fm_fine_ls, data['K0'][b_idx][None], data['K1'][b_idx][None])
                fine_fms_irls.append(fm_fine_irls)
                fine_fms_ls.append(fm_fine_ls)
                for k, v in fine_stats_irls.items():
                    stats['fine_' + k].append(torch.tensor([v], dtype=torch.float64))
        # aggregate batch data
        # return results + check that there were the matches
        data.update({
            'fm_fine_irls': torch.cat(fine_fms_irls) if len(fine_fms_irls) != 0 else None,
            'fm_fine_ls': torch.cat(fine_fms_ls) if len(fine_fms_ls) != 0 else None,
            'stats': {k: torch.stack(v) for k, v in stats.items()} if len(stats) != 0 else create_empty_stats(BATCH_SIZE),
        })
