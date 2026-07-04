"""
Wavelet packet utility functions.

Shared between all Wolter2021 model variants.
Preserved exactly from the original implementations.
"""
import torch
import numpy as np
import pywt


def compute_wavelet_packet_coeffs(img, wavelet='haar', level=3, mode='reflect'):
    """
    Compute wavelet packet coefficients for an RGB image.
    NOTE: This function is now primarily used in validation/inference.
    During training, wavelets are computed in the DataLoader.

    Args:
        img: torch tensor of shape (3, H, W) or numpy array (H, W, 3)
        wavelet: wavelet type (default: 'haar')
        level: decomposition level (default: 3)
        mode: signal extension mode (default: 'reflect')

    Returns:
        torch tensor of shape (C, H', W') where C = 3 * 4^level
    """
    if torch.is_tensor(img):
        img = img.cpu().numpy()

    # Ensure image is (H, W, 3)
    if img.shape[0] == 3:
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
    all_packets = np.array(all_packets)  # Shape: (3*4^level, H', W')

    # Convert to tensor
    packets_tensor = torch.tensor(all_packets, dtype=torch.float32)

    return packets_tensor


def log_scale_packets(packets, epsilon=1e-10):
    """
    Apply log-scaling to packet coefficients as in the paper.

    Args:
        packets: torch tensor of packet coefficients
        epsilon: small value to avoid log(0)

    Returns:
        log-scaled packet coefficients
    """
    return torch.sign(packets) * torch.log(torch.abs(packets) + epsilon)
