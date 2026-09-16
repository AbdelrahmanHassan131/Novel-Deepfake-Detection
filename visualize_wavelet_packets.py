import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pywt

def compute_wavelet_packet_coeffs(img, wavelet='haar', level=3, mode='reflect'):
    """
    Compute wavelet packet coefficients for an RGB image.
    """
    H, W, _ = img.shape
    all_packets = []
    
    # Generate all wavelet packet paths
    def get_paths(level):
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
    
    # Stack all packets
    all_packets = np.array(all_packets, dtype=np.float32)
    return all_packets


def log_scale_packets(packets, epsilon=1e-10):
    return np.sign(packets) * np.log(np.abs(packets) + epsilon)


def main():
    parser = argparse.ArgumentParser(description="Visualize Wavelet Packets in a 4x4 Grid")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--output', type=str, default='wavelet_packets_grid.png', help="Output path")
    parser.add_argument('--use_log', action='store_true', help="Apply log scaling (optional)")
    args = parser.parse_args()

    # Load and preprocess image just like in the dataset
    print(f"Loading image from {args.image_path}...")
    img = Image.open(args.image_path).convert('RGB')
    
    # Standard resize and crop as in WaveletDataset
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
    ])
    img_cropped = transform(img)
    img_array = np.array(img_cropped)

    # Extract wavelets
    print("Computing wavelet packets (Level 3 Haar)...")
    wavelet_coeffs = compute_wavelet_packet_coeffs(img_array, wavelet='haar', level=3)
    
    if args.use_log:
        wavelet_coeffs = log_scale_packets(wavelet_coeffs)

    # Plot 4x4 grid of the first 16 channels (first 16 packets from the Red channel usually)
    print("Generating 4x4 grid plot...")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(16):
        packet = wavelet_coeffs[i]
        
        # Normalize for display
        p_min, p_max = packet.min(), packet.max()
        if p_max > p_min:
            packet = (packet - p_min) / (p_max - p_min)
        else:
            packet = np.zeros_like(packet)
            
        axes[i].imshow(packet, cmap='gray')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Saved wavelet packets visualization to {args.output}")


if __name__ == "__main__":
    main()
