
# Traininig IHLS

## Dataset setup
Same as in [LoFTR Training](./LoFTR_TRAINING.md).


## Training
We provide training scripts for both ScanNet and MegaDepth datasets. The results in our paper can be reproduced with 8 V100 GPUs with 32GB of RAM.

### Training on MegaDepth
``` shell
# train weight prediction network (WPN)
scripts/reproduce_train/outdoor_ds_irls.sh
# train full pipepine end to end (in the script you need to specify checkpoint from the previous step)
scripts/reproduce_train/outdoor_ds_irls_end2end.sh
```

### Training on ScanNet
``` shell
# train weight prediction network (WPN)
scripts/reproduce_train/indoor_ds_irls.sh
# train full pipepine end to end (here you need to specify checkpoint from the previous step)
scripts/reproduce_train/indoor_ds_irls_end2end.sh
```
> NOTE: For ScanNet dataset not all image pairs have enough LoFTR matches to train WPN. To mitigate this problem, we used presampled image pairs with more than 7 LoFTR matches. You can create your train index in similar fashion.
