from configs.data.base import cfg

TEST_BASE_PATH = "data/megadepth/index"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/train"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_0.1_0.7"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/train_list.txt"

cfg.DATASET.MGDPT_IMG_RESIZE = 840
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
