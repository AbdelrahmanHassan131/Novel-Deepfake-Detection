# Subcollection Inspection Review: Numeric vs Video-Pattern Strata

**Generated:** 2026-10-09T18:58:38.544261+00:00  
**Scope:** 256 deterministically selected and extracted training archive samples (64 per stratum).

## 1. Strata Overview and Quantities in Archive vs Sample

| Subcollection Stratum | Archive Train Population | Population Share | 256 Sample Count | File Extension | Decoded Format | Primary Resolutions | Mean File Size |
|---|---|---|---|---|---|---|---|
| **Numeric Real** | 162,078 | 20.1% | 64 | 100% JPG | 100% JPEG | (178, 218) | 6.83 KB |
| **Numeric Fake** | 31,111 | 3.9% | 64 | 100% PNG | 100% PNG | (256, 256) | 91.74 KB |
| **Video-Pattern Real** | 49,110 | 6.1% | 64 | 100% PNG | 100% PNG | Variable (150-372px) | 41.26 KB |
| **Video-Pattern Fake** | 563,122 | 69.9% | 64 | 100% PNG | 100% PNG | Variable (110-285px) | 38.68 KB |

## 2. Representative Member Paths

### Numeric Real
- `Preapred Dataset/train/real/192988.jpg`
- `Preapred Dataset/train/real/036097.jpg`
- `Preapred Dataset/train/real/143481.jpg`

### Numeric Fake
- `Preapred Dataset/train/fake/1011.png`
- `Preapred Dataset/train/fake/27947.png`
- `Preapred Dataset/train/fake/29923.png`

### Video-Pattern Real
- `Preapred Dataset/train/real/vid_0e766a04cb991214c77d1c6244cd6af88afeedad6e019ccf2b921a3a985b1914_face_000006_004.png`
- `Preapred Dataset/train/real/vid_bd04692c062371e576c63f9df6c27d984ce5b2a266381d30347b0415a57b3b4d_face_000042_015.png`
- `Preapred Dataset/train/real/vid_c21fdd752fe81832445bc887f2bd4c811203b3266b19a76d6e5ad98278cbcacc_face_000102_029.png`

### Video-Pattern Fake
- `Preapred Dataset/train/fake/vid_5232f960efad9604c46955f8c15d174fa491885166f0b23402b056550fd66420_face_000026_014.png`
- `Preapred Dataset/train/fake/vid_e5bbc34245253ebe495b4af3c65e87fc54fd869629fda19571e8b4d484315d52_face_000057_024.png`
- `Preapred Dataset/train/fake/vid_73889737f5c6dd7af6ae4706e91f4da0cd79bc70a8381db9e0d8d2913e63b5be_face_000052_027.png`

## 3. Visual Crop and Framing Observations (Contact Sheets)

### A. Numeric Real (`contact_sheet_numeric_real.png`)
- **Resolution & Aspect Ratio:** Uniform 178x218 pixels, portrait aspect ratio (~0.816).
- **Framing & Content:** Centered, cropped face images of celebrities/models. High visual similarity to the CelebA aligned face crop benchmark.
- **Compression & Quality:** Standard lossy JPEG compression with mild 8x8 block boundaries visible under magnification. Natural facial skin tones and clear lighting.

### B. Numeric Fake (`contact_sheet_numeric_fake.png`)
- **Resolution & Aspect Ratio:** Exactly 256x256 pixels, square aspect ratio (1.0).
- **Framing & Content:** Centered face portraits. Shows characteristic generative artifacts (e.g. hair strands blurring into background, subtle facial asymmetry, smoothed textures). Unlike the video-pattern fakes, these are NOT tight video crops with motion blur; they are full-face portraits.
- **Encoding:** 100% PNG (lossless container). No JPEG JFIF/Exif metadata markers present.

### C. Video-Pattern Real (`contact_sheet_vid_real.png`)
- **Resolution & Aspect Ratio:** Highly variable (dimensions from 40x40 to 372x372, aspect ratios between 0.82 and 1.09).
- **Framing & Content:** Tight face bounding boxes tracked across video frames (interviews, news broadcasts, talk shows). Several false positive detector crops are present (e.g., ear-only crops, background clothing, foliage).
- **Quality & Codecs:** Displays video motion blur, interlacing/codec artifacts, and camera noise characteristic of television/web video extractions.

### D. Video-Pattern Fake (`contact_sheet_vid_fake.png`)
- **Resolution & Aspect Ratio:** Variable (dimensions from 110x110 to 285x285).
- **Framing & Content:** Video face crops corresponding to video manipulation sequences. Visible blending seams and facial reenactment/swap boundaries.
- **Encoding:** 100% PNG.

## 4. Key Takeaways and Remaining Unknowns

1. **The archive contains multiple subcollections with conflicting characteristics:**
   - Numeric real images are 178x218 JPEGs (consistent with CelebA crops).
   - Numeric fake images are 256x256 PNGs (consistent with portrait synthesis crops).
   - Video real images are variable-dimension PNGs from video tracking.
   - Video fake images are variable-dimension PNGs from video manipulation tracking.
2. **The Numeric Fake subset is small (3.9% of train, 2,753 in R2 100k manifest) but visually distinct:**
   - It demonstrates that the training set was NOT exclusively video face crops.
   - However, because it is 256x256 PNG and real numeric is 178x218 JPEG, the numeric subset ALSO exhibits severe format, resolution, and aspect-ratio confounding with the target label.
3. **Unknowns:**
   - Exact source dataset names and generators remain unverified in metadata.
   - Whether numeric images were derived from CelebA / DiffFace or another collection cannot be asserted without user confirmation.
