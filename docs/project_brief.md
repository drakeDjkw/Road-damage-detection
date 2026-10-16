# Project brief

# Road Damage Detection — Team Surface 3

Yikwa Sandrakh, Wenda Waipo, Agata Melani
Artificial Intelligence Implementation Technologies · Baikal School of BRICS, INRTU

## User and situation

<<<<<<< HEAD
Road maintenance planners responsible for deciding which road sections are
repaired patched, resurfaced or sealed and in what order need the system.
=======
The system is intended for road maintenance planners, who decide which road
sections are repaired (patched, resurfaced or sealed) and in what order.
>>>>>>> 8669c217bad00b21021dc59ed00cab5783aa3d04

Planners currently depend on manual inspection and driver reports to judge the
condition of a road surface. On a large network, a crack that has begun to
spread or a pothole that has opened may go unrecorded until an inspector
reaches it or a driver reports it.

The future system should help planners locate and categorise surface defects
by analysing images from road-facing cameras mounted on survey or ordinary
vehicles.

## What the system should do

- **Input:** Images or video frames from road-facing vehicle cameras.
<<<<<<< HEAD
- **Useful output:** For each frame, a set of bounding boxes, each with a
  defect class longitudinal crack, transverse crack, alligator crack, other
  corruption or pothole  and a confidence value.
=======
- **Useful output:** For each frame, a set of bounding boxes. Each box carries
  a defect class (longitudinal crack, transverse crack, alligator crack, other
  corruption or pothole) and a confidence value.
>>>>>>> 8669c217bad00b21021dc59ed00cab5783aa3d04
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
720 × 720, 512 × 512 or 640 × 640. A longitudinal crack and a tar-filled seal
may differ only in texture and edge contrast. Fixed if/then rules are
unlikely to handle this.

Other parts of the future system may include cameras, frame extraction from
video, an inference service, a planner dashboard, severity ranking per road
section, and human review of each result before a crew is sent.

## Data and first results

<<<<<<< HEAD
- **Where the data comes from:** (Road Damage Detection), a
  public benchmark dataset of road-surface imagery with bounding-box
  annotations, accessed through its Kaggle mirror in a pre-split form.
- **What we can access now:** The full split is available 3,000 training,
  300 validation and 200 test images, each with a matching label file.
  The five classes are present in the training split in uneven numbers:
   longitudinal cracks, transverse cracks, other
  corruption, alligator cracks and potholes. This is large enough
  to train, not only to test the pipeline.
- **What still needs confirmation:** We need to confirm whether thin cracks
  survive downscaling to small input resolutions, whether 25 epochs is
  sufficient for the detection head to converge, whether the "other
  corruption" class is defined consistently enough to be learnable, and
  whether the class imbalance needs to be addressed by weighting or by
  merging classes.
=======
- **Where the data comes from:** RDD2022 (Road Damage Detection 2022), a
  public benchmark of road-surface imagery with bounding-box annotations. The
  team uses a pre-split copy of it. The team does not collect its own footage.
- **What the full split contains:** 31,214 training, 11,516 validation and
  11,167 test images. The training labels contain 54,415 annotated objects in
  uneven proportions: longitudinal crack 21,413 (39.4%), transverse crack
  9,826 (18.1%), other corruption 8,886 (16.3%), alligator crack 8,865
  (16.3%) and pothole 5,425 (10.0%). The validation and test splits follow
  almost the same distribution. A training image holds 1.72 objects on
  average (median 1, maximum 44). 30.2% of label files are empty, which means
  those images contain no damage.
- **What was trained:** A random sample was drawn with a fixed seed (10,000
  training, 2,000 validation and 2,000 test images). Its class proportions
  stay within about 0.5 percentage points of the full training set (the
  validation sample is within about 2 points). Three detectors were trained on
  this sample at 640 px for 50 epochs (batch 16, early-stopping patience 20)
  on a Tesla T4: YOLOv8n, YOLOv8s and YOLO11s.
- **Results on the 2,000-image test sample (3,353 objects):**

  | Model    | Precision | Recall | mAP50 | mAP50-95 | Size (MB) | ms / image* |
  |----------|-----------|--------|-------|----------|-----------|-------------|
  | YOLOv8n  | 0.585     | 0.501  | 0.512 | 0.260    | 5.96      | 29.9        |
  | YOLOv8s  | 0.573     | 0.525  | 0.524 | 0.267    | 21.48     | 36.3        |
  | YOLO11s  | 0.604     | 0.511  | 0.533 | 0.271    | 18.29     | 36.5        |

  *The timing is end-to-end `predict()` time over 100 test images, including
  image loading and pre-processing. Pure model inference reported by
  Ultralytics was 3.4 ms (v8n), 8.9 ms (v8s) and 9.9 ms (YOLO11s).

- **What the results show:** YOLO11s has the highest precision and mAP. All
  three models are close: the spread is about 0.02 mAP50, and the smallest
  model is 3–4 times lighter for that small loss. Recall stays at about 0.50
  for every model, so roughly half of all defects are still missed at the
  default settings. Pothole is the weakest class in every run (test mAP50 of
  0.306 for v8n and 0.337 for v8s; recall 0.28–0.33). Other corruption is the
  strongest (mAP50 about 0.68). This suggests that the "other corruption"
  class is learnable, and that the main gap lies with potholes and the two
  linear crack classes. Validation mAP50 rose by less than 0.01 across the
  last ten epochs of each run, so longer training on the same sample may not
  help much.
- **What still needs confirmation:**
  - The file counts do not match: training has 31,214 images but 31,609
    label files, and the train-split check found 10,401 images without a
    matching label and 10,796 labels without a matching image. Many file
    names carry a " (1)" suffix, and copying the training sample stopped once
    on a missing image (`India_003323 (1).jpg`). This points to duplicated or
    misnamed files that must be cleaned before results on the full set can be
    trusted, and the split must also be checked for duplicates across train,
    validation and test.
  - It is not yet known whether the low pothole recall comes from the class
    being rare, from small object size, or from confusion with other classes.
  - Thin cracks may still be lost when large source images (up to 4040 px
    wide) are downscaled to 640 px.
>>>>>>> 8669c217bad00b21021dc59ed00cab5783aa3d04

## Three next actions

1. Clean the dataset: resolve the image–label mismatches and the " (1)"
   duplicates, confirm that no image appears in more than one split, and
   rebuild the sample from the cleaned set.
2. Investigate the low recall, focusing on potholes and transverse cracks.
   Inspect the confusion matrices and the frames with missed detections, test
   lower confidence thresholds, and try class weighting or oversampling of
   pothole images.
3. Retrain the strongest candidates (YOLO11s and YOLOv8n) on the full cleaned
   training set, and test a higher input resolution (for example 1024 px) for
   thin cracks. Then choose between accuracy (YOLO11s) and speed and size
   (YOLOv8n) for the first version, according to where inference will run.

---

*Detailed technical material — pipeline, model configuration, training
hyperparameters, evaluation protocol, delivery plan and the record of
methodological decisions — is kept in `README.md`.*
