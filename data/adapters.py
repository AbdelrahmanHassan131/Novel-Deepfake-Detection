"""Metadata adapters for standard deepfake datasets.

Converts dataset-specific naming conventions and structures into the canonical
schema required for auditable grouped splitting:
  - sample_id: unique globally stable identifier
  - path: image path
  - label: 0 (real) or 1 (fake)
  - split: 'unassigned' | 'train' | 'dev' | 'internal_test' | 'external_dev' | 'external_test'
  - dataset_source: official name (e.g., 'CelebA', 'FaceForensics++', 'Celeb-DF-v2', 'DiffFace')
  - group_id: identifier ensuring connected originals and derivatives stay together
  - source_video_id: video identifier if derived from video
  - identity_id: person identity if known
  - original_id: pristine reference sample/video ID if known
  - generator: specific manipulation/synthesis method (e.g., 'ADM', 'DDIM', 'Deepfakes', 'Face2Face')
  - sha256: content hash
"""
from pathlib import Path
import re
from typing import Dict, Any, Optional
from data.manifest import known


def adapt_celeba(row: Dict[str, Any], identity_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Adapter for CelebA authentic face images."""
    path = Path(row['path'])
    filename = path.name
    sample_id = path.stem
    identity = identity_map.get(filename, 'unknown') if identity_map else 'unknown'

    row['dataset_source'] = 'CelebA'
    row['label'] = 0
    row['generator'] = 'authentic'
    row['source_video_id'] = 'none'
    row['identity_id'] = f"celeba_{identity}" if known(identity) else 'unknown'
    row['original_id'] = f"celeba_{sample_id}"
    row['group_id'] = f"celeba_identity_{identity}" if known(identity) else f"celeba_img_{sample_id}"
    return row


def adapt_ffpp(row: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter for FaceForensics++ video sequences.

    Filename / folder conventions:
      - pristine: original_sequences/youtube/.../video_id/frame.png -> label 0
      - manipulated: manipulated_sequences/<method>/.../vid1_vid2/frame.png -> label 1
    Connects vid1_vid2 with original videos vid1 and vid2 to prevent cross-split leakage.
    """
    path = Path(row['path'])
    parts = path.parts
    path_str = str(path).replace('\\', '/')

    row['dataset_source'] = 'FaceForensics++'

    # Detect manipulation method
    manipulation = 'authentic'
    label = 0
    for method in ('Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'):
        if method.lower() in path_str.lower():
            manipulation = method
            label = 1
            break

    row['label'] = label
    row['generator'] = manipulation

    # Detect video ID and partner video ID
    # e.g., '042_056' or '042'
    video_match = re.search(r'(\d{3})_(\d{3})', path_str)
    if video_match:
        src_vid = video_match.group(1)
        tgt_vid = video_match.group(2)
        row['source_video_id'] = f"ffpp_{src_vid}_{tgt_vid}"
        row['original_id'] = f"ffpp_{src_vid},ffpp_{tgt_vid}"
        # Group combines both videos so pristine and manipulated versions are connected
        row['group_id'] = f"ffpp_pair_{min(src_vid, tgt_vid)}_{max(src_vid, tgt_vid)}"
    else:
        single_vid = re.search(r'/(\d{3})/', path_str)
        if single_vid:
            vid = single_vid.group(1)
            row['source_video_id'] = f"ffpp_{vid}"
            row['original_id'] = f"ffpp_{vid}"
            row['group_id'] = f"ffpp_single_{vid}"
        else:
            row['source_video_id'] = 'unknown'
            row['original_id'] = 'unknown'
            if not known(row.get('group_id')):
                row['group_id'] = 'unknown'

    row['identity_id'] = 'unknown'
    return row


def adapt_celeb_df(row: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter for Celeb-DF-v2 dataset."""
    path = Path(row['path'])
    path_str = str(path).replace('\\', '/')

    row['dataset_source'] = 'Celeb-DF-v2'
    if 'Celeb-real' in path_str or 'YouTube-real' in path_str or '/real/' in path_str:
        row['label'] = 0
        row['generator'] = 'authentic'
    else:
        row['label'] = 1
        row['generator'] = 'Celeb-DF_synthesis'

    # Video id e.g. id0_0000 or id0_id1_0000
    video_match = re.search(r'(id\d+)_(?:(id\d+)_)?(\d+)', path_str)
    if video_match:
        id1 = video_match.group(1)
        id2 = video_match.group(2)
        vid_num = video_match.group(3)
        if id2:
            row['identity_id'] = f"celebdf_{id1},celebdf_{id2}"
            row['source_video_id'] = f"celebdf_{id1}_{id2}_{vid_num}"
            row['original_id'] = f"celebdf_{id1},celebdf_{id2}"
            row['group_id'] = f"celebdf_pair_{min(id1, id2)}_{max(id1, id2)}"
        else:
            row['identity_id'] = f"celebdf_{id1}"
            row['source_video_id'] = f"celebdf_{id1}_{vid_num}"
            row['original_id'] = f"celebdf_{id1}_{vid_num}"
            row['group_id'] = f"celebdf_identity_{id1}"
    else:
        if not known(row.get('group_id')):
            row['group_id'] = 'unknown'
        row['source_video_id'] = 'unknown'
        row['identity_id'] = 'unknown'

    return row


def adapt_diffface(row: Dict[str, Any]) -> Dict[str, Any]:
    """Adapter for DiffFace diffusion-based fake face benchmarks."""
    path = Path(row['path'])
    path_str = str(path).replace('\\', '/')

    row['dataset_source'] = 'DiffFace'
    generators = ['ADM', 'DDIM', 'DDPM', 'DiffSwap', 'LDM', 'StableDiffusion']
    gen = 'diffusion_unknown'
    for g in generators:
        if g.lower() in path_str.lower():
            gen = g
            break

    if '/real/' in path_str.lower() or 'real.tar' in path_str.lower():
        row['label'] = 0
        row['generator'] = 'authentic'
        # DiffFace Real.tar note from REJECTION_RECOVERY_PLAN: inspect overlap with CelebA
        row['group_id'] = f"diffface_real_{path.stem}"
    else:
        row['label'] = 1
        row['generator'] = gen
        row['group_id'] = f"diffface_{gen}_{path.stem}"

    row['source_video_id'] = 'none'
    row['identity_id'] = 'unknown'
    row['original_id'] = 'unknown'
    return row


ADAPTERS = {
    'celeba': adapt_celeba,
    'ffpp': adapt_ffpp,
    'celeb_df': adapt_celeb_df,
    'diffface': adapt_diffface,
}


def apply_adapter(row: Dict[str, Any], adapter_name: str, **kwargs) -> Dict[str, Any]:
    """Apply named adapter to a row dictionary."""
    name = adapter_name.lower().replace('-', '_')
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter '{adapter_name}'. Available: {list(ADAPTERS.keys())}")
    return ADAPTERS[name](row, **kwargs)
