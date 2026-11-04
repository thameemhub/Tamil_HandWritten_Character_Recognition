# Tamil Letter Dataset – Comprehensive Audit Report

## 1. Executive Summary

- **Dataset Root**: `C:\Tamil_Dataset\dataset\Final Tamil Letter Dataset`
- **Total Active Image Classes**: 103
- **Total Valid Images**: 3600
- **Zip-Only Classes**: 4 (ப, பா, பி, பீ)
- **Empty Classes (Enrollable Later)**: 9
- **Corrupt / Unreadable Files**: 0
- **Exact Hash Duplicate Groups**: 35 (35 redundant files)
- **Image Formats**: {'.png': 3347, '.jpg': 250, '.jpeg': 3}

## 2. Category Breakdown

| Category | Class Count | Total Images | Min Images/Class | Max Images/Class | Mean Images/Class |
|:---|:---:|:---:|:---:|:---:|:---:|
| ஆய்தம | 1 | 46 | 46 | 46 | 46.0 |
| உயிர் | 12 | 509 | 31 | 47 | 42.4 |
| உயிர்மெய் | 72 | 2426 | 12 | 69 | 33.7 |
| மெய் | 18 | 619 | 26 | 38 | 34.4 |

## 3. Source Group Distribution

| Source Group | Count | Percentage |
|:---|:---:|:---:|
| numeric | 1868 | 51.9% |
| Screenshot | 622 | 17.3% |
| IMG | 413 | 11.5% |
| WhatsApp | 262 | 7.3% |
| resized | 251 | 7.0% |
| Scanned | 160 | 4.4% |
| PXL | 24 | 0.7% |

## 4. Zip-Only Classes (To Be Extracted to `data/interim/extracted/`)

| Class | Zip Filename | Internal Images |
|:---|:---|:---:|
| ப | `resized_images_zip (1).zip` | 34 |
| பா | `resized_images_zip.zip` | 33 |
| பி | `resized_images_zip (2).zip` | 31 |
| பீ | `resized_images_zip (3).zip` | 35 |

## 5. Empty Classes (Record as Enrollable Later)

The following folders contain 0 images in the source dataset and will not be trained on directly, but are marked for interactive enrollment:

| Folder Name | Category | Path |
|:---|:---|:---|
| ச | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\ச` |
| சா | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\சா` |
| சி | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\சி` |
| சீ | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\சீ` |
| ணே | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\ணே` |
| ழ | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\ழ` |
| ழா | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\ழா` |
| ழி | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\ழி` |
| ழீ | உயிர்மெய் | `உயிர்மெய் எழுத்துக்கள\ழீ` |

## 6. Episodic Eligibility by (K + Q) Threshold

An episode requires at least $K + Q$ samples per sampled class:

| $K+Q$ Threshold | Eligible Classes | % of Total Classes | Disqualified Classes |
|:---:|:---:|:---:|:---|
| >= 2 | 103 / 103 | 100.0% | None |
| >= 4 | 103 / 103 | 100.0% | None |
| >= 6 | 103 / 103 | 100.0% | None |
| >= 8 | 103 / 103 | 100.0% | None |
| >= 10 | 103 / 103 | 100.0% | None |
| >= 15 | 102 / 103 | 99.0% | து |
| >= 20 | 101 / 103 | 98.1% | து, மி |
| >= 25 | 98 / 103 | 95.1% | து, தோ, மி, ரி, ரீ |

## 7. Image Resolution Statistics

- **Width**: Min=6px, Max=1979px, Median=48px, Mean=60.3px
- **Height**: Min=11px, Max=1352px, Median=48px, Mean=55.5px

## 8. Near/Exact Perceptual Duplicate Groups (Samples)

Found 35 duplicate p-hash groups. First 5 groups:

**Group 1** (2 images):
- Class `னை`: `Screenshot (472).png`
- Class `னொ`: `Screenshot (472).png`

**Group 2** (2 images):
- Class `னை`: `Screenshot (473).png`
- Class `னொ`: `Screenshot (473).png`

**Group 3** (2 images):
- Class `னை`: `Screenshot (474).png`
- Class `னொ`: `Screenshot (474).png`

**Group 4** (2 images):
- Class `னை`: `Screenshot (475).png`
- Class `னொ`: `Screenshot (475).png`

**Group 5** (2 images):
- Class `னை`: `Screenshot (476).png`
- Class `னொ`: `Screenshot (476).png`

