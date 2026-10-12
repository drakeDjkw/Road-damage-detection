# Project brief

# Road Damage Detection — Team Surface 3

Yikwa Sandrakh, Wenda Waipo, Agata Melani
Artificial Intelligence Implementation Technologies · Baikal School of BRICS, INRTU

## User and situation

Road maintenance planners responsible for deciding which road sections are
repaired — patched, resurfaced or sealed — and in what order need the system.

Currently, planners may need to rely on manual inspection and driver reports to
judge the condition of a road surface. On a large network, a crack that has
begun to spread or a pothole that has opened may not be recorded until an
inspector reaches it or a driver reports it.

The future system should help planners locate and categorise surface defects
by analysing images from road-facing cameras mounted on survey or ordinary
vehicles.

## What the system should do

- **Input:** Images or video frames from road-facing vehicle cameras.
- **Useful output:** For each frame, a set of bounding boxes, each with a
  defect class — longitudinal crack, transverse crack, alligator crack, other
  corruption or pothole — and a confidence value.
- **Action or decision after the output:** Staff review the sections with the
  most severe or most numerous detections and decide which stretches to repair
  first.

Because the output is a box rather than a single label per frame, the system
reports *where* on the carriageway a defect lies and *how many* defects a
stretch contains, which is what a repair decision needs.

## Why AI may help

AI may help because the system needs to recognise visual patterns associated
with cracking, surface disintegration and potholes, and to separate them from
the many other dark, linear or patterned features present on a road: tar
sealant, lane markings, shadows, joints, wet patches and repaired areas.

Road appearance may vary because of lighting, time of day, weather, camera
position, pavement material and country of origin. The dataset contains
imagery from Japan, Norway, the Czech Republic, India, the United States and
China, and the same defect class may look markedly different between them.
A longitudinal crack and a tar-filled seal may differ only in texture and
edge contrast. These differences may make fixed if/then rules insufficient.

Other parts of the future system may include cameras, frame extraction from
video, an inference service, a planner dashboard, severity ranking per road
section, and human review of each result before a crew is sent.

## Initial data plan

- **Where the data comes from:** RDD2022 (Road Damage Detection 2022), a
  public benchmark dataset of road-surface imagery with bounding-box
  annotations, accessed through its Kaggle mirror in a pre-split form. The
  team no longer collects its own footage.
- **What we can access now:** The full split is available — 26,869 training,
  5,758 validation and 5,758 test images, each with a matching label file.
  The five classes are present in the training split in uneven numbers:
  18,201 longitudinal cracks, 8,386 transverse cracks, 7,554 other
  corruption, 7,527 alligator cracks and 4,628 potholes. This is large enough
  to train, not only to test the pipeline.
- **What still needs confirmation:** We need to confirm whether thin cracks
  survive downscaling to small input resolutions, whether 25 epochs is
  sufficient for the detection head to converge, whether the "other
  corruption" class is defined consistently enough to be learnable, and
  whether the class imbalance needs to be addressed by weighting or by
  merging classes.

## Three next actions

1. Run the 640 px and 1024 px experiments and compare mAP, recall, model
   size and inference time across the three resolutions, since thin cracks
   are the detections most likely to be lost at 320 px.
2. Investigate the low recall: inspect the confusion matrix and the frames
   with no detections, test lower confidence thresholds, and establish
   whether 25 epochs is enough or whether the run is undertrained.
3. Decide whether the first version should detect all five classes, or begin
   with a coarser grouping — for example cracking versus pothole — given the
   imbalance and the ambiguity of the "other corruption" class.

---

*Detailed technical material — pipeline, model configuration, training
hyperparameters, evaluation protocol, delivery plan and the record of
methodological decisions — is kept in `README.md`.*
