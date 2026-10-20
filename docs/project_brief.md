# Project brief

# Road Damage Detection — Team Surface 3

## User and situation

The system is intended for road maintenance planners, who decide which road
sections are repaired (patched, resurfaced or sealed) and in what order.

Planners currently depend on manual inspection and driver reports to judge the
condition of a road surface. On a large network, a crack that has begun to
spread or a pothole that has opened may go unrecorded until an inspector
reaches it or a driver reports it.

The future system should help planners locate and categorise surface defects
by analysing images from road-facing cameras mounted on survey or ordinary
vehicles.

## What the system should do

- **Input:** Images or video frames from road-facing vehicle cameras.
- **Useful output:** For each frame, a set of bounding boxes. Each box carries
  a defect class (longitudinal crack, transverse crack, alligator crack, other
  corruption or pothole) and a confidence value.
- **Action or decision after the output:** Staff review the sections with the
  most severe or most numerous detections and decide which stretches to repair
  first.

Because the output is a box rather than a single label per frame, the system
reports *where* on the carriageway a defect lies and *how many* defects a
stretch contains. A repair decision needs both.

## Why AI may help

AI may help because the system must recognise the visual patterns of
cracking, surface disintegration and potholes, and separate them from the
many other dark, linear or patterned features on a road: tar sealant, lane
markings, shadows, joints, wet patches and repaired areas.

Road appearance varies with lighting, time of day, weather, camera position,
pavement material and country of origin. The dataset contains imagery from
Japan, Norway, the Czech Republic, India, the United States and China, and
the same defect class can look markedly different between them. Image size
also varies with the capture device: in a sample of 5,000 training images,
resolutions ranged from 512 × 512 to 4040 × 2035 px, and most were 600 × 600,
720 × 720, 640 × 640 or 512 × 512. A longitudinal crack and a tar-filled seal
may differ only in texture and edge contrast. Fixed if/then rules are
unlikely to handle this.

Other parts of the future system may include cameras, frame extraction from
video, an inference service, a planner dashboard, severity ranking per road
section, and human review of each result before a crew is sent.

## Data and first results

- **Where the data comes from:** RDD2022 (Road Damage Detection 2022), a
  public benchmark of road-surface imagery with bounding-box annotations. The
  team uses a pre-split copy of it. The team does not collect its own footage.
- **What the full split contains:** 26,869 training, 5,758 validation and
  5,758 test images. Every image has a matching label file.
  - The training labels contain 46,296 annotated objects in uneven
    proportions: longitudinal crack 18,201 (39.3%), transverse crack 8,386
    (18.1%), other corruption 7,554 (16.3%), alligator crack 7,527 (16.3%)
    and pothole 4,628 (10.0%).
  - The validation and test splits follow almost the same distribution.
  - A training image holds 1.72 objects on average (median 1, maximum 44).
  - 30.1% of training images have an empty label file, which means they
    contain no damage.
- **What was trained:** A random sample was drawn with a fixed seed
  (`random_state=42`): 5,000 training, 2,000 validation and 2,000 test images.
  - The training sample keeps the class proportions of the full training set
    to within about 0.6 percentage points.
  - The validation and test samples are within about 1.5 points.
  - Three pretrained detectors were fine-tuned on this sample at 640 px for
    50 epochs (batch 16, early-stopping patience 20, seed 42): YOLOv8n,
    YOLOv8s and YOLO11s.
  - Training ran on an Apple M4 (MPS) and took 3.1 h, 7.0 h and 10.2 h
    respectively.
- **Results on the 2,000-image test sample (3,223 objects):**

  | Model    | Precision | Recall | mAP50 | mAP50-95 | Size (MB) | ms / image* |
  |----------|-----------|--------|-------|----------|-----------|-------------|
  | YOLOv8n  | 0.516     | 0.455  | 0.453 | 0.220    | 6.21      | 48.5        |
  | YOLOv8s  | 0.552     | 0.466  | 0.477 | 0.235    | 21.48     | 82.8        |
  | YOLO11s  | 0.522     | 0.472  | 0.461 | 0.224    | 18.29     | 80.0        |

  *The timing is end-to-end `predict()` time over 100 test images, including
  image loading and pre-processing.

- **What the results show:**
  - YOLOv8s is best on precision and both mAP measures. YOLO11s has the
    highest recall.
  - All three models are close: the spread is 0.024 mAP50. YOLOv8n gives up
    that small margin while being 3–3.5 times smaller and about 1.7 times
    faster.
  - Recall stays between 0.45 and 0.47, so more than half of all defects are
    still missed at the default settings.
  - Pothole is the weakest class in every run, with a test mAP50 of 0.269
    (v8n) and 0.306 (v8s) and recall of 0.27–0.29.
  - Other corruption (mAP50 0.60–0.62) and alligator crack (0.55–0.58) are
    the strongest classes. This suggests that the "other corruption" class
    is learnable, and that the main gap lies with potholes and the two
    linear crack classes.
  - Validation mAP50 rose by only about 0.01–0.03 over the last ten epochs
    of each run, so longer training on the same 5,000 images is unlikely to
    help much. More training data is a likelier source of improvement.
- **What still needs confirmation:**
  - The split is random at the image level. It is not yet known whether
    frames from the same drive or road appear in both training and test,
    which would make these scores optimistic.
  - It is not yet known whether the low pothole recall comes from the class
    being rare, from small object size, or from confusion with other classes.
  - Thin cracks may be lost when large source images (up to 4040 px wide) are
    downscaled to 640 px.
  - The sample uses only 19% of the available training images, so it is not
    yet known how much the full training set would improve the scores.

## Three next actions

1. Check for leakage between the splits: look for images from the same drive
   or road sequence in both training and test, and, if found, move to a split
   grouped by sequence or location.
2. Investigate the low recall, focusing on potholes and transverse cracks.
   Inspect the confusion matrices and the frames with missed detections, test
   lower confidence thresholds, and try class weighting or oversampling of
   pothole images.
3. Retrain YOLOv8n and YOLOv8s on the full training set (26,869 images),
   preferably on a CUDA GPU to reduce training time, and test a higher input
   resolution (for example 1024 px) for thin cracks. Then choose between
   accuracy (YOLOv8s) and speed and size (YOLOv8n) for the first version.

---

*Detailed technical material — pipeline, model configuration, training
hyperparameters, evaluation protocol, delivery plan and the record of
methodological decisions — is kept in `README.md`.*
