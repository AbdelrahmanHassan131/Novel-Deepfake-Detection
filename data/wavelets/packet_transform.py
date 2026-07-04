import numpy as np
import pywt
import torch

def compute_wavelet_packet_coeffs(img, wavelet='haar', level=3, mode='reflect'):
    """
    Compute wavelet packet coefficients for an RGB image.

    Args:
        img: numpy array (H, W, 3) or torch tensor (3, H, W)
        wavelet: wavelet type (default: 'haar')
        level: decomposition level (default: 3)
        mode: signal extension mode (default: 'reflect')

    Returns:
        numpy array of shape (C, H', W') where C = 3 * 4^level
    """
    if torch.is_tensor(img):
        img = img.cpu().numpy()

    # Ensure image is (H, W, 3)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    H, W, _ = img.shape
    all_packets = []

    # Generate all wavelet packet paths at the given level
    def get_paths(level):
        """Generate all wavelet packet paths at a given level"""
        if level == 0:
            return ['']
        paths = []
        prev_paths = get_paths(level - 1)
        for path in prev_paths:
            for letter in ['a', 'h', 'v', 'd']:
                paths.append(path + letter)
        return paths

    packet_paths = get_paths(level)

    # Process each color channel
    for c in range(3):
        channel = img[:, :, c]

        # Create wavelet packet decomposition
        wp = pywt.WaveletPacket2D(
            data=channel, wavelet=wavelet, mode=mode, maxlevel=level)

        for path in packet_paths:
            coeff = wp[path].data
            all_packets.append(coeff)

    # Stack all packets: 3 channels * 4^level packets per channel
    # Shape: (3*4^level, H', W')
    all_packets = np.array(all_packets, dtype=np.float32)

    return all_packets


def log_scale_packets(packets, epsilon=1e-10):
    """
    Apply log-scaling to packet coefficients as in the paper.

    Args:
        packets: numpy array of packet coefficients
        epsilon: small value to avoid log(0)

    Returns:
        log-scaled packet coefficients
    """
    return np.sign(packets) * np.log(np.abs(packets) + epsilon)
