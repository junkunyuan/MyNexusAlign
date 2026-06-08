"""FID/IS metrics: Inception features vs. cached reference statistics."""

import glob
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_features_to_metric


def compute_metrics_with_cached_stats(img_folder, fid_stats_file, device, batch_size=64,
                                      inception_weights_path=None):
    """FID (vs. precomputed reference stats) and IS for PNGs in img_folder.

    Runs the Inception-v3 extractor directly and feeds its outputs into
    torch_fidelity's stats-level helpers, sidestepping the installed
    calculate_metrics() which lacks a fid_statistics_file argument.

    inception_weights_path loads local Inception weights; None downloads them.
    """
    img_paths = sorted(glob.glob(os.path.join(img_folder, "*.png")))
    assert img_paths, f"No PNGs found in {img_folder}"

    fe = FeatureExtractorInceptionV3(
        "inception-v3-compat", ["2048", "logits_unbiased"],
        feature_extractor_weights_path=inception_weights_path,
    ).to(device).eval()

    feats_2048, feats_logits = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Inception features"):
            chunk = img_paths[i: i + batch_size]
            arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in chunk])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device, dtype=torch.uint8)
            f2048, flogits = fe(x)
            feats_2048.append(f2048.cpu())
            feats_logits.append(flogits.cpu())

    feats_2048 = torch.cat(feats_2048, dim=0)
    feats_logits = torch.cat(feats_logits, dim=0)

    is_dict = isc_features_to_metric(feats_logits)

    # Reference .npz is float64; cast generated stats to match.
    stats_gen = fid_features_to_statistics(feats_2048)
    ref = np.load(fid_stats_file)
    stats_gen = {"mu": stats_gen["mu"].astype(np.float64),
                 "sigma": stats_gen["sigma"].astype(np.float64)}
    stats_ref = {"mu": ref["mu"].astype(np.float64),
                 "sigma": ref["sigma"].astype(np.float64)}
    fid_dict = fid_statistics_to_metric(stats_gen, stats_ref, verbose=True)

    return {**is_dict, **fid_dict}
