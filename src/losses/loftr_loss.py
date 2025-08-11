from loguru import logger

from kornia.geometry.epipolar import (essential_from_fundamental,
                                      motion_from_essential_choose_solution)
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from src.losses.utils import epipolar_loss

def VALUE_CHECK(loss):
    if len(loss) != 0:
        loss = torch.stack(loss)
        loss = loss[~loss.isnan()].mean()
        if not torch.isnan(loss):
            return loss
        else:
            logger.warning('NaN value')
            return None
    else:
        logger.warning('no loss to return')
        return None


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftr']['loss']
        self.match_type = self.config['loftr']['match_coarse']['match_type']
        self.sparse_spvs = self.config['loftr']['match_coarse']['sparse_spvs']
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        self.coarse_pose_type = self.loss_config['coarse_pose_type']
        # fine-level
        self.fine_type = self.loss_config['fine_type']
        self.fine_pose_type = self.loss_config['fine_pose_type']
        # pose loss
        self.r_weight = self.loss_config['rotation_weight']
        self.use_angular_loss = self.loss_config['angular_loss']
        # epipolar geometry
        self.epipolar_geometry_type = self.config['fundamental_matrix']['matrix_type']
        self.e_weight = 1e-3 if self.epipolar_geometry_type == 'F' else 1e2

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                
                loss =  c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))

    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def compute_pose_loss(self, data, matching_level, solution_type='ls'):
        if matching_level == 'coarse':
            if self.coarse_pose_type is None:
                return None
            elif self.coarse_pose_type == 'cosine':
                num_of_solutions = data[f'fm_{matching_level}_{solution_type}'].shape[1]
                loss = 0
                for solution in range(num_of_solutions):
                    loss += self.pose_loss_cosine(data, data[f'fm_{matching_level}_{solution_type}'][:, solution],  matching_level)
                return loss / num_of_solutions
            else:
                raise NotImplementedError()
        elif matching_level == 'fine':
            if self.fine_pose_type is None:
                return None
            elif self.fine_pose_type == 'cosine':
                return self.pose_loss_cosine(data, data[f'fm_{matching_level}_{solution_type}'],  matching_level)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def pose_loss_cosine(self, data, Fm_pred, matching_level='fine'):
        # extract data
        if matching_level == 'coarse':
            kpts0, kpts1, K_0, K_1 = data['mkpts0_p'], data['mkpts1_p'], data['K0_c'], data['K1_c']
        else: # fine level matches
            kpts0, kpts1, K_0, K_1 = data['mkpts0_f'], data['mkpts1_f'], data['K0'], data['K1']
        b_ids, weights, T_0to1, covergence_stats = data['b_ids'], data['mconf'], data['T_0to1'], data['stats'][f'{matching_level}_converged']
        if Fm_pred is None:
            logger.error('No Fundamental Matrix to compute pose loss')
            return None
        # perform loss computation
        R_loss = []
        T_loss = []
        T_ang_loss = []
        P_loss = []
        epi_loss = []
        unique_batch_elems = b_ids.unique()
        for b_idx in unique_batch_elems:
            if covergence_stats[b_idx] != 1:
                logger.warning(f'Failed to converge for sample #{b_idx}')
                # we reached the last element and all previous samples didn't converge
                if b_idx == unique_batch_elems[-1] and len(R_loss) == 0:
                    logger.warning(f'compute loss as there are no other candidates...')
                else:
                    logger.warning(f'skip loss computation...')
                    continue
            b_mask = b_idx == b_ids
            b_F = Fm_pred[b_idx]
            num_of_solution = b_F.shape[0]
            b_kpts0 = kpts0[b_mask][None].repeat(num_of_solution, 1, 1)
            b_kpts1 = kpts1[b_mask][None].repeat(num_of_solution, 1, 1)
            if num_of_solution > 10:
                b_kpts0 = b_kpts0[:, :512]
                b_kpts1 = b_kpts1[:, :512]
            b_K0, b_K1 = K_0[b_idx][None].repeat(num_of_solution, 1, 1), K_1[b_idx][None].repeat(num_of_solution, 1, 1)
            b_T_0to1 = T_0to1[b_idx][None].repeat(num_of_solution, 1, 1)
            # estimate camera pose
            
            b_E = essential_from_fundamental(b_F, b_K0, b_K1)
            assert not b_E.isnan().any()
            (b_R, b_T, triang_points) = motion_from_essential_choose_solution(b_E, b_K0, b_K1, b_kpts0, b_kpts1)
            b_T = b_T[:, :, 0]
            # compute rotation error
            gt_R = b_T_0to1[:, :3, :3]
            R_err = (((b_R.transpose(-1, -2) @ gt_R).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2).clip(-0.99999, 0.99999).arccos().abs().mean()
            # compute translation error (angular and distance)
            gt_T = b_T_0to1[:, :3, 3]
            # compute angular error
            n = gt_T.norm(dim=-1) * b_T.norm(dim=-1)
            t_ang_err = ((gt_T * b_T).sum(-1) / n).clip(-0.99999, 0.99999).arccos()
            T_ang_err = torch.minimum(t_ang_err, np.pi - t_ang_err).mean()
            # if gt_T.norm() < 0:  # pure rotation is challenging TODO: check this challenge
            #     continue
            # compute absolute error
            normal_gt_T = normalize(gt_T)
            T_err = ((normal_gt_T - b_T) ** 2).sum(-1).mean()
            P_err = R_err*R_err + T_err
            P_loss.append(P_err)
            R_loss.append(R_err)
            T_loss.append(T_err)
            T_ang_loss.append(T_ang_err)
            # if ground-truth matches available compute epipolar loss
            try:
                epi_err = epipolar_loss(
                    b_idx, data, 
                    b_F if self.epipolar_geometry_type == 'F' else b_E, 
                    epipolar_geometry_type=self.epipolar_geometry_type)
                epi_loss.append(epi_err)
            except KeyError:
                logger.warning("No ground truth to compute epipolar loss")
        
        R_loss = VALUE_CHECK(R_loss)
        T_loss = VALUE_CHECK(T_loss)
        T_ang_loss = VALUE_CHECK(T_ang_loss)
        P_loss = VALUE_CHECK(P_loss)
        epi_loss = VALUE_CHECK(epi_loss)
        # return angular or absolute loss
        if self.use_angular_loss:
            # problems with angular loss can happen
            if R_loss is None:
                logger.warning('NaN values occured in Rotation part of the angular loss')
                R_loss = 0
            if T_ang_loss is None:
                logger.warning('NaN values occured in Translation part of the angular loss')
                T_ang_loss = 0
            if epi_loss is None:
                logger.warning('NaN values occured in epipolar regularization loss')
                epi_loss = 0
            total_loss = (self.r_weight * 10 * R_loss + T_ang_loss + self.e_weight * epi_loss) / 3
            if total_loss == 0:
                logger.warning('NaN values occured in pose loss cosine computation')
                total_loss = T_loss if self.training else torch.tensor(torch.pi, device=Fm_pred.device)
            return total_loss
        else:
            return P_loss

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        # 1. coarse-level loss
        loss = 0
        if self.loss_config['coarse_weight'] != 0:
            c_weight = self.compute_c_weight(data)
            #loss = torch.tensor(0., device=data['conf_matrix'].device)
            loss_c = self.compute_coarse_loss(
                data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                    else data['conf_matrix'],
                data['conf_matrix_gt'],
                weight=c_weight)
            loss += loss_c * self.loss_config['coarse_weight']
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
        
        # 2. fine-level loss
        if self.loss_config['fine_weight'] != 0:
            loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
            if loss_f is not None:
                loss += loss_f * self.loss_config['fine_weight']
                loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
            else:
                assert self.training is False
                loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound
        # 3. coarse-level pose loss
        if self.loss_config['coarse_pose_weight'] != 0:
            loss_pose_c = self.compute_pose_loss(data, 'coarse')
            if loss_pose_c is not None:
                loss += loss_pose_c * self.loss_config['coarse_pose_weight']
                loss_scalars.update({"loss_pose_c":  loss_pose_c.clone().detach().cpu()})
            else:
                loss_scalars.update({'loss_pose_c': torch.tensor(np.pi)})  # pi is the upper bound TODO: Check it
        # 4. fine-level pose loss
        if self.loss_config['fine_pose_weight'] != 0:
            loss_pose_f = self.compute_pose_loss(data, 'fine')
            if loss_pose_f is not None:
                loss += loss_pose_f * self.loss_config['fine_pose_weight']
                loss_scalars.update({"loss_pose_f":  loss_pose_f.clone().detach().cpu()})
            else:
                loss_scalars.update({'loss_pose_f': torch.tensor(np.pi)})  # pi is the upper bound TODO: Check it
        # 5. coarse-level pose loss
        if self.loss_config['coarse_irls_weight'] != 0:
            loss_irls_c = self.compute_pose_loss(data, 'coarse', 'irls')
            if loss_irls_c is not None:
                loss += loss_irls_c * self.loss_config['coarse_irls_weight']
                loss_scalars.update({"loss_irls_c":  loss_irls_c.clone().detach().cpu()})
            else:
                loss_scalars.update({'loss_irls_c': torch.tensor(np.pi)})  # pi is the upper bound TODO: Check it
        # 6. fine-level pose loss
        if self.loss_config['fine_irls_weight'] != 0:
            loss_irls_f = self.compute_pose_loss(data, 'fine', 'irls')
            if loss_irls_f is not None:
                loss += loss_irls_f * self.loss_config['fine_irls_weight']
                loss_scalars.update({"loss_irls_f":  loss_irls_f.clone().detach().cpu()})
            else:
                loss_scalars.update({'loss_irls_f': torch.tensor(np.pi)})  # pi is the upper bound TODO: Check it
        
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
