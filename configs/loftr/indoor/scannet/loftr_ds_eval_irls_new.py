""" A config only for reproducing the ScanNet evaluation results.

We remove border matches by default, but the originally implemented
`remove_border()` has a bug, leading to only two sides of
all borders are actually removed. However, the [bug fix](https://github.com/zju3dv/LoFTR/commit/e9146c8144dea5f3cbdd98b225f3e147a171c216)
makes the scannet evaluation results worse (auc@10=40.8 => 39.5), which should be
caused by tiny result fluctuation of few image pairs. This config set `BORDER_RM` to 0
to be consistent with the results in our paper.

Update: This config is for testing the re-trained model with the pos-enc bug fixed.
"""

from src.config.default import _CN as cfg

cfg.LOFTR.COARSE.TEMP_BUG_FIX = True
cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 1.0
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

#cfg.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]
cfg.TRAINER.SCHEDULER = 'ExponentialLR'
cfg.TRAINER.SCHEDULER_INTERVAL = 'step'
cfg.TRAINER.WARMUP_STEP = 512  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1

cfg.TRAINER.FREEZE_MATCHER = True

cfg.LOFTR.LOSS.COARSE_WEIGHT = 0.0
cfg.LOFTR.LOSS.FINE_WEIGHT = 0.0
cfg.LOFTR.LOSS.COARSE_POSE_WEIGHT = 0.0
cfg.LOFTR.LOSS.FINE_POSE_WEIGHT = 0.0
cfg.LOFTR.LOSS.COARSE_IRLS_WEIGHT = 0.0
cfg.LOFTR.LOSS.FINE_IRLS_WEIGHT = 1.0
cfg.LOFTR.LOSS.COARSE_POSE_TYPE = 'cosine'
cfg.LOFTR.LOSS.FINE_POSE_TYPE = 'cosine'
cfg.LOFTR.LOSS.ROTATION_WEIGHT = 1.0
cfg.LOFTR.LOSS.ANGULAR_LOSS = True

cfg.FUNDAMENTAL_MATRIX.COMPUTE_FINE = True
cfg.FUNDAMENTAL_MATRIX.ROBUST_ESTIMATOR_TYPE = 'LIRLS'
