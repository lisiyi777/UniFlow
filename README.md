# UniFlow

This repository contains the official implementation of the paper 

**UniFlow: Towards Zero-Shot LiDAR Scene Flow for Autonomous Vehicles via Cross-Domain Generalization**  [[arXiv]](https://arxiv.org/abs/2511.18254) · [[Project Page]](https://lisiyi777.github.io/UniFlow/)

---

## Installation & Data Preparation

Please follow the environment setup and data preparation instructions provided in [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow).

### Beam ID Extraction

Beam ID is required for beam-dropout augmentation during training. It is supported for:
- **Argoverse 2**
- **Waymo**

Beam ID is not supported for TruckScenes due to information availability, and nuScenes as it is already a sparse 32-beam LiDAR dataset.

### Argoverse 2

If you have not preprocessed Argoverse 2, please run:

```bash
python dataprocess/extract_av2.py \
  --av2_type sensor \
  --data_mode train \
  --argo_dir /path/to/av2 \
  --output_dir /path/to/av2/preprocess

```
If you already have preprocessed Argoverse 2 '.h5' files without beam information, add '--only_insert_beamid' to your command.

### Waymo

If you have not preprocessed Waymo, please run:
```bash
python dataprocess/extract_waymo.py \
  --mode train \
  --flow_data_dir /path/to/waymo/flowlabel \
  --map_dir /path/to/waymo/flowlabel/map \
  --output_dir /path/to/waymo/preprocess \
  --nproc <N>
```

If you already have preprocessed Waymo '.h5' files without beam information, you can insert beam IDs post-hoc using:
```bash
python dataprocess/add_beam_id_waymo.py \
  --preprocess_dir /path/to/waymo/preprocess/train \
  --tfrecord_root /path/to/waymo/flowlabel \
  --split train \
  --num_workers <N>
```


---

## Training

Joint training across multiple datasets is supported via `train.py`, with model backbone size and data augmentations controlled through configuration options.

Example:
```bash
python train.py \
  model=flow4d \
  train_data="[
    /scratch/siyili/waymo/preprocess/train/,
    /scratch/siyili/av2/preprocess/train/,
    /scratch/siyili/nuscenes/preprocess/train/
  ]" \
  val_data=/scratch/siyili/av2/preprocess/val/ \
  optimizer.lr=5e-4 \
  epochs=18 \
  batch_size=2 \
  num_frames=5 \
  model.target.backbone=xl \
  basic_augment=true \
  beam_dropout.enable=true \
  dataset_weight_mode=proportional
```

---

## Citation

If you find UniFlow useful in your research, please consider citing:

```bibtex
@misc{li2025uniflowzeroshotlidarscene,
      title={UniFlow: Towards Zero-Shot LiDAR Scene Flow for Autonomous Vehicles via Cross-Domain Generalization}, 
      author={Siyi Li and Qingwen Zhang and Ishan Khatri and Kyle Vedder and Deva Ramanan and Neehar Peri},
      year={2025},
      eprint={2511.18254},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18254}, 
}
```

---

## License and Acknowledgements

This repository is based on OpenSceneFlow and includes code adapted from Flow4D and DeFlow. Original copyright and license notices are preserved in the source files where applicable.
