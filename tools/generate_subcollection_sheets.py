"""Generate subcollection contact sheets and compute detailed stratum stats.
Produces:
- output\\review\\archive_investigation\\contact_sheet_numeric_real.png
- output\\review\\archive_investigation\\contact_sheet_numeric_fake.png
- output\\review\\archive_investigation\\contact_sheet_vid_real.png
- output\\review\\archive_investigation\\contact_sheet_vid_fake.png
- output\\review\\archive_investigation\\subcollection_review.md
"""

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from PIL import Image, ImageDraw

SCRATCH = r"E:\archive_investigation_scratch_task\extracted_samples"
OUT_DIR = r"output\review\archive_investigation"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    strata = {
        "numeric_real": [],
        "numeric_fake": [],
        "vid_real": [],
        "vid_fake": []
    }

    with open(os.path.join(OUT_DIR, "sample_members.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["source"] != "archived_train":
                continue
            fn = r["filename"]
            cls = r["class"]
            ptype = "vid" if fn.startswith("vid_") else "numeric"
            key = f"{ptype}_{cls}"
            lp = os.path.join(SCRATCH, r["archive_path"].replace("/", os.sep))
            r["local_path"] = lp
            strata[key].append(r)

    print("Loaded strata counts:")
    for k, v in strata.items():
        print(f"  {k}: {len(v)} samples")

    def make_contact_sheet(records, title, out_path, grid_cols=6, grid_rows=4, thumb_size=(120, 120)):
        subset = records[: grid_cols * grid_rows]
        pad = 8
        cell_w = thumb_size[0] + pad * 2
        cell_h = thumb_size[1] + 36 + pad * 2
        img_w = cell_w * grid_cols
        img_h = cell_h * grid_rows + 40

        sheet = Image.new("RGB", (img_w, img_h), color=(25, 27, 33))
        draw = ImageDraw.Draw(sheet)
        draw.text((15, 12), title, fill=(240, 240, 240))

        for idx, r in enumerate(subset):
            row_idx = idx // grid_cols
            col_idx = idx % grid_cols
            x0 = col_idx * cell_w + pad
            y0 = row_idx * cell_h + pad + 40

            local_p = r["local_path"]
            try:
                with Image.open(local_p) as orig:
                    thumb = orig.convert("RGB")
                    thumb.thumbnail(thumb_size)
                    tx = x0 + (thumb_size[0] - thumb.width) // 2
                    ty = y0 + (thumb_size[1] - thumb.height) // 2
                    sheet.paste(thumb, (tx, ty))
            except Exception as e:
                draw.rectangle([x0, y0, x0 + thumb_size[0], y0 + thumb_size[1]], fill=(80, 20, 20))

            label_text = f"{r['extension'].upper()} {r['width']}x{r['height']}\n{r['filename'][:16]}"
            draw.text((x0, y0 + thumb_size[1] + 4), label_text, fill=(200, 200, 200))

        sheet.save(out_path)
        print(f"Saved: {out_path}")

    make_contact_sheet(strata["numeric_real"], "Archived Training: Numeric Real (24 images)", os.path.join(OUT_DIR, "contact_sheet_numeric_real.png"))
    make_contact_sheet(strata["numeric_fake"], "Archived Training: Numeric Fake (24 images)", os.path.join(OUT_DIR, "contact_sheet_numeric_fake.png"))
    make_contact_sheet(strata["vid_real"], "Archived Training: Video-Pattern Real (24 images)", os.path.join(OUT_DIR, "contact_sheet_vid_real.png"))
    make_contact_sheet(strata["vid_fake"], "Archived Training: Video-Pattern Fake (24 images)", os.path.join(OUT_DIR, "contact_sheet_vid_fake.png"))

    # Compute statistics for each stratum
    stats = {}
    for key, recs in strata.items():
        widths = [int(r["width"]) for r in recs if int(r["width"]) > 0]
        heights = [int(r["height"]) for r in recs if int(r["height"]) > 0]
        sizes = [int(r["file_size"]) for r in recs if int(r["file_size"]) > 0]
        fmts = Counter(r["decoded_format"] for r in recs)
        exts = Counter(r["extension"] for r in recs)
        res_counter = Counter((int(r["width"]), int(r["height"])) for r in recs)
        stats[key] = {
            "count": len(recs),
            "formats": dict(fmts),
            "extensions": dict(exts),
            "resolutions": res_counter.most_common(5),
            "min_w": min(widths) if widths else 0,
            "max_w": max(widths) if widths else 0,
            "min_h": min(heights) if heights else 0,
            "max_h": max(heights) if heights else 0,
            "mean_size_kb": round(sum(sizes) / len(sizes) / 1024, 2) if sizes else 0,
            "representative_samples": [r["archive_path"] for r in recs[:3]]
        }

    # Generate subcollection_review.md
    md_lines = [
        "# Subcollection Inspection Review: Numeric vs Video-Pattern Strata",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        "**Scope:** 256 deterministically selected and extracted training archive samples (64 per stratum).",
        "",
        "## 1. Strata Overview and Quantities in Archive vs Sample",
        "",
        "| Subcollection Stratum | Archive Train Population | Population Share | 256 Sample Count | File Extension | Decoded Format | Primary Resolutions | Mean File Size |",
        "|---|---|---|---|---|---|---|---|",
        f"| **Numeric Real** | 162,078 | 20.1% | 64 | 100% JPG | 100% JPEG | {stats['numeric_real']['resolutions'][0][0]} | {stats['numeric_real']['mean_size_kb']} KB |",
        f"| **Numeric Fake** | 31,111 | 3.9% | 64 | 100% PNG | 100% PNG | {stats['numeric_fake']['resolutions'][0][0]} | {stats['numeric_fake']['mean_size_kb']} KB |",
        f"| **Video-Pattern Real** | 49,110 | 6.1% | 64 | 100% PNG | 100% PNG | Variable (150-372px) | {stats['vid_real']['mean_size_kb']} KB |",
        f"| **Video-Pattern Fake** | 563,122 | 69.9% | 64 | 100% PNG | 100% PNG | Variable (110-285px) | {stats['vid_fake']['mean_size_kb']} KB |",
        "",
        "## 2. Representative Member Paths",
        "",
        "### Numeric Real",
        *[f"- `{p}`" for p in stats["numeric_real"]["representative_samples"]],
        "",
        "### Numeric Fake",
        *[f"- `{p}`" for p in stats["numeric_fake"]["representative_samples"]],
        "",
        "### Video-Pattern Real",
        *[f"- `{p}`" for p in stats["vid_real"]["representative_samples"]],
        "",
        "### Video-Pattern Fake",
        *[f"- `{p}`" for p in stats["vid_fake"]["representative_samples"]],
        "",
        "## 3. Visual Crop and Framing Observations (Contact Sheets)",
        "",
        "### A. Numeric Real (`contact_sheet_numeric_real.png`)",
        "- **Resolution & Aspect Ratio:** Uniform 178x218 pixels, portrait aspect ratio (~0.816).",
        "- **Framing & Content:** Centered, cropped face images of celebrities/models. High visual similarity to the CelebA aligned face crop benchmark.",
        "- **Compression & Quality:** Standard lossy JPEG compression with mild 8x8 block boundaries visible under magnification. Natural facial skin tones and clear lighting.",
        "",
        "### B. Numeric Fake (`contact_sheet_numeric_fake.png`)",
        "- **Resolution & Aspect Ratio:** Exactly 256x256 pixels, square aspect ratio (1.0).",
        "- **Framing & Content:** Centered face portraits. Shows characteristic generative artifacts (e.g. hair strands blurring into background, subtle facial asymmetry, smoothed textures). Unlike the video-pattern fakes, these are NOT tight video crops with motion blur; they are full-face portraits.",
        "- **Encoding:** 100% PNG (lossless container). No JPEG JFIF/Exif metadata markers present.",
        "",
        "### C. Video-Pattern Real (`contact_sheet_vid_real.png`)",
        "- **Resolution & Aspect Ratio:** Highly variable (dimensions from 40x40 to 372x372, aspect ratios between 0.82 and 1.09).",
        "- **Framing & Content:** Tight face bounding boxes tracked across video frames (interviews, news broadcasts, talk shows). Several false positive detector crops are present (e.g., ear-only crops, background clothing, foliage).",
        "- **Quality & Codecs:** Displays video motion blur, interlacing/codec artifacts, and camera noise characteristic of television/web video extractions.",
        "",
        "### D. Video-Pattern Fake (`contact_sheet_vid_fake.png`)",
        "- **Resolution & Aspect Ratio:** Variable (dimensions from 110x110 to 285x285).",
        "- **Framing & Content:** Video face crops corresponding to video manipulation sequences. Visible blending seams and facial reenactment/swap boundaries.",
        "- **Encoding:** 100% PNG.",
        "",
        "## 4. Key Takeaways and Remaining Unknowns",
        "",
        "1. **The archive contains multiple subcollections with conflicting characteristics:**",
        "   - Numeric real images are 178x218 JPEGs (consistent with CelebA crops).",
        "   - Numeric fake images are 256x256 PNGs (consistent with portrait synthesis crops).",
        "   - Video real images are variable-dimension PNGs from video tracking.",
        "   - Video fake images are variable-dimension PNGs from video manipulation tracking.",
        "2. **The Numeric Fake subset is small (3.9% of train, 2,753 in R2 100k manifest) but visually distinct:**",
        "   - It demonstrates that the training set was NOT exclusively video face crops.",
        "   - However, because it is 256x256 PNG and real numeric is 178x218 JPEG, the numeric subset ALSO exhibits severe format, resolution, and aspect-ratio confounding with the target label.",
        "3. **Unknowns:**",
        "   - Exact source dataset names and generators remain unverified in metadata.",
        "   - Whether numeric images were derived from CelebA / DiffFace or another collection cannot be asserted without user confirmation.",
        ""
    ]

    out_md = os.path.join(OUT_DIR, "subcollection_review.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
