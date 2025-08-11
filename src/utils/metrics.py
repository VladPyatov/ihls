from collections import OrderedDict
from copy import deepcopy

import pymagsac
import torch
import cv2
import numpy as np
from einops.einops import rearrange
from loguru import logger
from kornia.geometry.epipolar import essential_from_fundamental, motion_from_essential_choose_solution
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
from sklearn.metrics import auc

from src.loftr.utils.coarse_matching import mask_border, mask_border_with_padding
# --- METRICS ---


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr or np.isnan(t_err):  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']

    epi_errs = []

    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, weights=None, shape0=None, shape1=None):
    if len(kpts0) < 8:
        return None
    # estimate fundamental matrix
    #F, mask = cv2.findFundamentalMat(kpts0, kpts1, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=conf)
    sorted_indices = np.argsort(-weights)
    weights = weights[sorted_indices]
    correspondences = np.hstack([kpts0, kpts1])[sorted_indices]
    F, mask = pymagsac.findFundamentalMatrix(
        np.ascontiguousarray(correspondences), 
        shape0[1], shape0[0], shape1[1], shape1[0],
        probabilities = None,#np.ascontiguousarray(weights),
        sampler = 0,
        use_magsac_plus_plus = True,
        sigma_th = 5.0)
    mask = mask.astype(np.uint8)[:, None]
    if F is None:
        print("\nF is None while trying to recover pose.\n")
        return None
    if F.shape[0] > 3:
        num_of_solutions = F.shape[0] // 3
        F = F.reshape(num_of_solutions, 3, 3)
        E = K1.T @ F @ K0
        E = E.reshape(num_of_solutions * 3, 3)
    else:
        E = K1.T @ F @ K0

    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    
    if E.shape == (3, 3):
        E = E.reshape(1, 3, 3)
    elif E.shape == (9, 3):
        E = E.reshape(3, 3, 3)

    for _E in E:
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n
    
    return ret


def estimate_pose_fm(F, kpts0, kpts1, K0, K1, confidence=None):
    # estimate camera pose
    if kpts0.shape[1] < 8 or F is None:
        logger.warning('not enough matches for Fundamental matrix estimation')
        return None
    # if there are several F matrices (from several iterations), select last one
    if len(F.shape) == 4:
        F = F[:, -1]
    E = essential_from_fundamental(F, K0, K1)
    (b_R, b_T, triang_points) = motion_from_essential_choose_solution(E, K0, K1, kpts0, kpts1)
    return (b_R, b_T[..., 0])


def compute_pose_errors(data, config):
    """
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "R_fp_errs" List[float]: [N]
            "t_fp_errs" List[float]: [N]
            "R_fi_errs" List[float]: [N]
            "t_fi_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({
        'R_errs': [], 't_errs': [],
        'R_fp_errs': [], 't_fp_errs': [],
        'R_fi_errs': [], 't_fi_errs': [],
        'inliers': [],
    })

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()
    weights = data['mconf'].cpu().numpy()
    shape0 = data['img_shape0'].cpu().numpy()
    shape1 = data['img_shape1'].cpu().numpy()
    
    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        # compute fine pose with RANSAC
        ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf,
                            weights=weights[mask], shape0=shape0[bs], shape1=shape1[bs])
        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)
        
        # compute fine pose from weighted Fundamental matrix
        ret = estimate_pose_fm(data['fm_fine_ls'][bs][None], data['mkpts0_f'][mask][None], data['mkpts1_f'][mask][None],
                               data['K0'][bs][None], data['K1'][bs][None], data['mconf'][mask][None]) if data['fm_fine_ls'] is not None else None
        if ret is None:
            data['R_fp_errs'].append(np.inf)
            data['t_fp_errs'].append(np.inf)
        else:
            R, t = ret
            R = R[0].cpu().numpy()
            t = t[0].cpu().numpy()
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_fp_errs'].append(R_err)
            data['t_fp_errs'].append(t_err)

        # compute fine pose with IRLS
        ret = estimate_pose_fm(data['fm_fine_irls'][bs][None], data['mkpts0_f'][mask][None], data['mkpts1_f'][mask][None],
                               data['K0'][bs][None], data['K1'][bs][None], data['irls_weights'][mask][None]) if data['fm_fine_irls'] is not None else None
        if ret is None:
            data['R_fi_errs'].append(np.inf)
            data['t_fi_errs'].append(np.inf)
        else:
            R, t = ret
            R = R[0].cpu().numpy()
            t = t[0].cpu().numpy()
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_fi_errs'].append(R_err)
            data['t_fi_errs'].append(t_err)


def compute_prediction_mask(data, thr, border_rm):
    axes_lengths = {
        'h0c': data['hw0_c'][0],
        'w0c': data['hw0_c'][1],
        'h1c': data['hw1_c'][0],
        'w1c': data['hw1_c'][1]
    }
    # 1. confidence thresholding
    mask = data['conf_matrix'] > thr
    mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes_lengths)
    if 'mask0' not in data:
        mask_border(mask, border_rm, False)
    else:
        mask_border_with_padding(mask, border_rm, False, data['mask0'], data['mask1'])
    mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)', **axes_lengths)
    return mask


def compute_precision_recall(data, config):
    data.update({'AMRs': [], 'AMPs': [], 'num_gt_matches': []})
    # no conf_matrix_gt, for ex. when training with SP+SG
    if data.get('conf_matrix_gt') is None:
        data['AMRs'] = [0] * data['image0'].shape[0]
        data['AMPs'] = [0] * data['image0'].shape[0]
        data['num_gt_matches'] = [400] * data['image0'].shape[0]
        return 

    lower_thr = config.LOFTR.MATCH_COARSE.THR
    border_rm = config.LOFTR.MATCH_COARSE.BORDER_RM
    scale = 1 / (1 - lower_thr)
    thrs = np.arange(lower_thr, 1, 0.1)
    # initialize tensor of PR and RC values for batch
    PR = torch.zeros(data['conf_matrix_gt'].shape[0], len(thrs))
    RC = torch.zeros(data['conf_matrix_gt'].shape[0], len(thrs))
    # compute PR and RC for different thresholds
    for i, thr in enumerate(thrs):
        mask = compute_prediction_mask(data, thr, border_rm)
        for bs in range(data['conf_matrix_gt'].shape[0]):
            precision = data['conf_matrix_gt'][bs][mask[bs]].sum() / mask[bs].sum()
            recall = data['conf_matrix_gt'][bs][mask[bs]].sum() / data['conf_matrix_gt'][bs].sum()
            PR[bs, i] = precision
            RC[bs, i] = recall
    # compute metrics
    PR = torch.nan_to_num(PR, nan=0.0)
    RC = torch.nan_to_num(RC, nan=0.0)
    for bs in range(data['conf_matrix_gt'].shape[0]):
        rc, pr = RC[bs].numpy(), PR[bs].numpy()
        data['AMPs'].append(auc(rc, pr))
        data['AMRs'].append(scale * auc(thrs, rc))
        data['num_gt_matches'].append(data['conf_matrix_gt'][bs].sum().cpu().numpy())


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds=[5, 10, 20], mark=''):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'{mark}auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def epidist_prec_rec(errors, num_gt_matches, thresholds, ret_dict=False, mark=''):
    precs = []
    recs = []
    for thr in thresholds:
        prec_ = []
        rec_ = []
        for errs, num_gt in zip(errors, num_gt_matches):
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
            rec_.append(np.sum(correct_mask) / num_gt if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
        recs.append(np.mean(rec_) if len(rec_) > 0 else 0)
    if ret_dict:
        return ({f'{mark}prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)},
                {f'{mark}rec@{t:.0e}': rec for t, rec in zip(thresholds, recs)})
    else:
        return precs, recs


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    angular_thresholds = [2, 5, 10, 20]
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')
    aggregated_metrics = dict()
    # pose auc
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aggregated_metrics.update(error_auc(pose_errors, angular_thresholds))  # (auc@5, auc@10, auc@20)
    # fp pose auc
    pose_errors = np.max(np.stack([metrics['R_fp_errs'], metrics['t_fp_errs']]), axis=0)[unq_ids]
    aggregated_metrics.update(error_auc(pose_errors, angular_thresholds, 'fp_'))
    # fi pose auc
    pose_errors = np.max(np.stack([metrics['R_fi_errs'], metrics['t_fi_errs']]), axis=0)[unq_ids]
    aggregated_metrics.update(error_auc(pose_errors, angular_thresholds, 'fi_'))

    # fine matching precision
    dist_thresholds = [epi_err_thr]
    precs, recs = epidist_prec_rec(
        np.array(metrics['epi_errs'], dtype=object)[unq_ids],
        np.array(metrics['num_gt_matches'])[unq_ids],
        dist_thresholds,
        True
    )  # (prec@err_thr)
    aggregated_metrics.update(precs)
    aggregated_metrics.update(recs)


    # matching mAMR and mAMP
    aggregated_metrics.update({
        'mAMR': np.array(metrics['AMRs'])[unq_ids].mean(),
        'mAMP': np.array(metrics['AMPs'])[unq_ids].mean()
    })
    # IRLS  stats
    for stats_type in ['fine']:
        aggregated_metrics.update({
            f'{stats_type}_converged': np.array(metrics[f'{stats_type}_converged'])[unq_ids].mean(),
            f'{stats_type}_num_steps': np.array(metrics[f'{stats_type}_num_steps'])[unq_ids].mean(),
            f'{stats_type}_atol': np.array(metrics[f'{stats_type}_atol'])[unq_ids].mean(),
            f'{stats_type}_rtol': np.array(metrics[f'{stats_type}_rtol'])[unq_ids].mean()
        })
    return aggregated_metrics
