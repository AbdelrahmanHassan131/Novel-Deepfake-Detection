"""
Extract Real Face Images from InsightFace RecordIO (train.rec / .zip)
Pure Python - No MXNet installation required! Compatible with Python 3.8 - 3.12+
"""

import os
import io
import struct
import zipfile
import argparse
from pathlib import Path
from PIL import Image

def extract_from_rec_stream(stream, output_dir, max_images=None, organize_by_id=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = 0
    record_idx = 0
    
    print(f"[*] Starting extraction to: {output_dir}")
    if max_images:
        print(f"[*] Extracting up to {max_images} images...")
    else:
        print("[*] Extracting ALL images (this may take a few minutes)...")

    while True:
        hdr = stream.read(8)
        if len(hdr) < 8:
            break
            
        magic, lrec = struct.unpack('II', hdr)
        if magic != 0xced7230a:
            # Reached end or padding
            break
            
        length = lrec & ((1 << 29) - 1)
        buf = stream.read(length)
        pad = (4 - (length % 4)) % 4
        if pad:
            stream.read(pad)
            
        record_idx += 1
        
        # Check if record contains image data (IRHeader is 24 bytes)
        if len(buf) > 24 and buf[24:27] == b'\xff\xd8\xff':
            flag, label, id1, id2 = struct.unpack('IfQQ', buf[:24])
            identity_id = int(label)
            img_bytes = buf[24:]
            
            if organize_by_id:
                sub_dir = output_dir / f"id_{identity_id:04d}"
                sub_dir.mkdir(exist_ok=True)
                img_path = sub_dir / f"{extracted:06d}.jpg"
            else:
                img_path = output_dir / f"real_{extracted:06d}.jpg"
                
            with open(img_path, 'wb') as f_out:
                f_out.write(img_bytes)
                
            extracted += 1
            
            if extracted % 10000 == 0:
                print(f"[+] Extracted {extracted:,} images...")
                
            if max_images and extracted >= max_images:
                print(f"[*] Reached limit of {max_images} images.")
                break

    print(f"[SUCCESS] Finished! Extracted {extracted:,} real images to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract images from faces_webface_112x112")
    parser.add_argument("--source", type=str, default=r"F:\t2itestdtaset\faces_webface_112x112.zip",
                        help="Path to faces_webface_112x112.zip or train.rec")
    parser.add_argument("--output", type=str, default=r"F:\t2itestdtaset\real_faces",
                        help="Output folder where images will be saved")
    parser.add_argument("--max", type=int, default=None,
                        help="Max images to extract (e.g. 10000, or leave empty for all)")
    parser.add_argument("--flat", action="store_true",
                        help="Save all images in one flat folder instead of id_XXXX folders")
    
    args = parser.parse_args()
    source_path = Path(args.source)
    
    if not source_path.exists():
        print(f"[ERROR] Source file not found: {source_path}")
        return

    if source_path.suffix.lower() == ".zip":
        print(f"[*] Reading directly from ZIP: {source_path}")
        with zipfile.ZipFile(source_path) as z:
            # Locate train.rec inside zip
            rec_files = [name for name in z.namelist() if name.endswith("train.rec")]
            if not rec_files:
                print("[ERROR] train.rec not found inside zip archive!")
                return
            rec_name = rec_files[0]
            print(f"[*] Found {rec_name} inside archive.")
            with z.open(rec_name) as stream:
                extract_from_rec_stream(stream, args.output, max_images=args.max, organize_by_id=not args.flat)
    else:
        print(f"[*] Reading from file: {source_path}")
        with open(source_path, "rb") as stream:
            extract_from_rec_stream(stream, args.output, max_images=args.max, organize_by_id=not args.flat)

if __name__ == "__main__":
    main()
