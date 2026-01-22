"""
Insert beam_id into preprocessed Waymo HDF5 files.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
import multiprocessing as mp


def _parse_range_images(frame):
    """Return dict range_images[laser_name] = [ri_return1, (ri_return2?)]
       and range_image_top_pose (for TOP) — same as your extractor.
    """
    range_images = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:
            ri1 = tf.io.decode_compressed(laser.ri_return1.range_image_compressed, "ZLIB")
            ri = dataset_pb2.MatrixFloat(); ri.ParseFromString(bytearray(ri1.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                pose1 = tf.io.decode_compressed(laser.ri_return1.range_image_pose_compressed, "ZLIB")
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(bytearray(pose1.numpy()))
    return range_images, range_image_top_pose

def _beam_rows_top_return1(frame) -> np.ndarray:
    """Extract TOP lidar beam rows (channel ids) for FIRST return, matching your pipeline."""
    range_images, range_image_top_pose = _parse_range_images(frame)
    if dataset_pb2.LaserName.TOP not in range_images:
        return np.array([], dtype=np.uint8)

    ri = range_images[dataset_pb2.LaserName.TOP][0]
    ri_tensor = tf.reshape(tf.convert_to_tensor(ri.data), ri.shape.dims)
    mask = ri_tensor[..., 0] > 0
    rc = tf.where(mask)
    row_idx = tf.cast(rc[:, 0], tf.int32).numpy()
    return row_idx.astype(np.uint8, copy=False)

def _scan_scene_map(tfrecord_dir: Path, split: Optional[str] = None) -> Dict[str, str]:
    """Build mapping from scene_id (context.name) → full TFRecord path. Cache to JSON."""
    roots = [tfrecord_dir]
    if split is not None:
        roots = [tfrecord_dir / split] if (tfrecord_dir / split).exists() else roots

    candidates = []
    for r in roots:
        if r.is_file() and r.suffix == ".tfrecord":
            candidates.append(r)
        elif r.is_dir():
            candidates += list(r.rglob("*.tfrecord"))

    scene2file = {}
    for p in tqdm(candidates, desc="Indexing TFRecords", ncols=100):
        try:
            ds = tf.data.TFRecordDataset(str(p), compression_type='')
            it = ds.as_numpy_iterator()
            first = next(it, None)
            if first is None:
                continue
            frame = dataset_pb2.Frame.FromString(bytearray(first))
            scene_id = frame.context.name 
            scene2file[scene_id] = str(p)
        except Exception:
            continue
    return scene2file

def _load_or_make_scene_map(tfrecord_root: str, split: Optional[str], cache_json: Path) -> Dict[str, str]:
    if cache_json.exists():
        try:
            return json.loads(cache_json.read_text())
        except Exception:
            pass
    m = _scan_scene_map(Path(tfrecord_root), split)
    cache_json.parent.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(json.dumps(m))
    return m


def _tf_single_thread():
    """Ensure TensorFlow behaves in a single-threaded way per worker."""
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

def _process_one_scene(args):
    """
    Worker: insert beam_id for one scene file.
    Returns (scene_id, changed_groups, skipped_groups, mismatch_groups, errstr_or_None)
    """
    h5_path, tf_path, overwrite = args
    scene_id = Path(h5_path).stem
    changed = skipped = mismatched = 0
    try:
        with h5py.File(h5_path, "a") as f:
            try:
                ts_keys = sorted(int(k) for k in f.keys())
            except ValueError:
                return (scene_id, 0, 0, 0, "non-integer group keys")
            ts_set = set(ts_keys)

            for raw in tf.data.TFRecordDataset(tf_path, compression_type='').as_numpy_iterator():
                frame = dataset_pb2.Frame.FromString(bytearray(raw))
                ts = int(frame.timestamp_micros)
                if ts not in ts_set:
                    continue
                grp = f[str(ts)]
                if "beam_id" in grp and not overwrite:
                    skipped += 1
                    continue
                if "lidar" not in grp:
                    skipped += 1
                    continue

                try:
                    beam_id = _beam_rows_top_return1(frame)  # uint8
                except Exception as e:
                    skipped += 1
                    continue

                num_points = grp["lidar"].shape[0]
                if len(beam_id) != num_points:
                    mismatched += 1
                    continue

                if "beam_id" in grp:
                    del grp["beam_id"]
                grp.create_dataset("beam_id", data=beam_id, dtype="u1")
                changed += 1
        return (scene_id, changed, skipped, mismatched, None)
    except Exception as e:
        return (scene_id, changed, skipped, mismatched, str(e))

def insert_beam_id_waymo_posthoc_parallel(
    preprocess_dir: str,
    tfrecord_root: str,
    split: Optional[str] = None,
    overwrite: bool = True,
    cache_map_json: Optional[str] = None,
    num_workers: int = max(mp.cpu_count() - 1, 1),
):
    pre_dir = Path(preprocess_dir)
    if not pre_dir.exists():
        print(f"[insert_beam_id_waymo] preprocess_dir not found: {pre_dir}")
        return

    # Build or load scene_id -> tfrecord map in the parent process
    cache_json = Path(cache_map_json) if cache_map_json else pre_dir / "_scene_map.json"
    scene_map = _load_or_make_scene_map(tfrecord_root, split, cache_json)

    h5_files = sorted([p for p in pre_dir.iterdir() if p.suffix == ".h5"])
    jobs = []
    missing_tfrecord = 0
    for h5_path in h5_files:
        scene_id = h5_path.stem
        tf_path = scene_map.get(scene_id)
        if not tf_path:
            missing_tfrecord += 1
            continue
        jobs.append((str(h5_path), tf_path, overwrite))

    print(f"[insert_beam_id_waymo] Scenes to process: {len(jobs)} "
          f"(missing tfrecord for {missing_tfrecord})")
    print(f"[insert_beam_id_waymo] Using {num_workers} workers")

    changed_files = 0
    total_groups = updated_groups = skipped_groups = mismatch_groups = 0
    errors = 0

    with mp.get_context("spawn").Pool(processes=num_workers, initializer=_tf_single_thread) as pool:
        for scene_id, changed, skipped, mismatched, err in tqdm(
            pool.imap_unordered(_process_one_scene, jobs, chunksize=1),
            total=len(jobs), ncols=100, desc="Inserting beam_id (Waymo, parallel)"
        ):
            if err is None:
                total = changed + skipped + mismatched
                total_groups += total
                updated_groups += changed
                skipped_groups += skipped
                mismatch_groups += mismatched
                if changed > 0:
                    changed_files += 1
            else:
                errors += 1
                print(f"[ERR ] {scene_id}: {err}")

    print("\n[insert_beam_id_waymo] Summary (parallel)")
    print(f"  Scenes enqueued:    {len(jobs)}")
    print(f"  Files changed:      {changed_files}")
    print(f"  Groups total:       {total_groups}")
    print(f"  Groups updated:     {updated_groups}")
    print(f"  Groups skipped:     {skipped_groups}")
    print(f"  Groups mismatched:  {mismatch_groups}")
    print(f"  Missing TFRecords:  {missing_tfrecord}")
    print(f"  Scene errors:       {errors}")

if __name__ == '__main__':     
    insert_beam_id_waymo_posthoc_parallel(
        preprocess_dir='/home/lsiyi/data/waymo/preprocess/train',
        tfrecord_root='/home/lsiyi/waymo/flowlabel',
        split='train',         
        overwrite=True,
        cache_map_json=None,
        num_workers=32,          
    )