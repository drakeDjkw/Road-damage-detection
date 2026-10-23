# Project decisions

## Lab 03 — Acceptance

### Decision after the baseline
Decision: **Narrow**

Reason: Based on the measured baseline (YOLOv8n, test mAP50 0.457, overall
recall 0.449, pothole mAP50 0.316 — `reports/baseline_metrics.json`) and the
limitation that the image-level random split may make these scores optimistic,
we decided to narrow the first version:

- **Output:** the system *proposes* candidate defects for a planner to review.
  It does not rank road sections for repair on its own, because it still
  misses more than half of all defects.
- **Operating conditions:** daylight still frames from road-facing vehicle
  cameras of the kind found in RDD2022. Night, heavy rain, snow cover and
  cameras from countries outside the six in the dataset are out of scope for
  the first version.
- **Priority class:** potholes, the class most relevant to safety-driven
  repair and the weakest class in every run.

The five-class task stays as it is; we do not drop classes.

### Required behavior
- **Normal input:** a readable daylight road-surface image (JPG/PNG) with a
  road-section ID. The system returns boxes with defect class and confidence,
  marks the frame as *checked*, and adds its detections to that section's
  summary. The planner opens the section, looks at the boxed frames, and
  decides whether to schedule an inspection or repair.
- **Invalid input:** a file that cannot be decoded, an image that is too
  small, too dark or too blurred to show the surface, or a frame without a
  section ID. The system rejects it with the reason (e.g. "image too dark"),
  logs it, and **does not count it as damage-free road**. The planner sees how
  many frames per section were rejected and can request a re-drive.
- **Valid input with an uncertain result:** a valid frame where a detection
  falls in a middle confidence band (model-score cutoff values *not set yet*),
  in particular a possible pothole. The system does not add it to the
  confirmed count; it places the frame in a *needs review* queue with the box
  shown. The planner confirms or dismisses it.

### One quality that matters
- **Processing time.** Planners review a whole survey drive after it is
  recorded, so each frame must be processed fast enough for a drive to be
  ready the same day.
  - Measured so far (notebook, not the Lab 02 pipeline): YOLOv8n takes
    48.5 ms per image, end-to-end `predict()` over 100 test images including
    loading and pre-processing, on an Apple M4 (MPS).
  - Proposed limit: **≤ 100 ms per image** on the same device and with the
    same timing method, leaving room for the input checks added above.
  - Still to measure: the number of frames in a typical survey drive, which
    decides whether this limit is enough.

### One acceptance check
- **Important failure:** a pothole in the image is not reported, so the
  planner believes the section is in better condition than it is and the
  repair is delayed.
- **Metric that would show this failure:** pothole recall (share of
  ground-truth pothole boxes matched by a pothole prediction at IoU ≥ 0.5) on
  the held-out test sample, at the confidence cutoff the system will use.
  - The Lab 02 primary metric (mAP50) does not show this failure directly: it
    averages all classes and all cutoffs. The pipeline currently records
    pothole mAP50 (0.316) but not pothole recall, so a later evaluation must
    add per-class recall to `reports/baseline_metrics.json`.
  - Current evidence: the notebook runs give pothole recall of 0.27–0.29 at
    default settings. This is a measured baseline, not a target.
- **Preliminary acceptance threshold:** pothole recall **≥ 0.60** on the test
  sample, while overall precision stays **≥ 0.40** (so the review queue is
  not flooded with false boxes). This is a proposed limit for a later
  evaluation; the baseline does not meet it and the number does not show that
  the system is ready.
- **Where will we check it?** On the held-out test sample (2,000 images, 384
  pothole boxes), and, once available, on a test split grouped by drive or
  road so that near-duplicate frames cannot inflate the score.
- **Why this number for now?** At 0.27–0.29 the model misses about seven of
  ten potholes, which gives a planner little over existing driver reports.
  Reaching 0.60 would mean most potholes are surfaced for review, while the
  human review step still catches false alarms. The value is a team judgement,
  not a requirement from planners.
- **Evidence still missing:**
  - per-class recall logged by the pipeline at a fixed cutoff;
  - a split grouped by drive or road, to check whether 0.27–0.29 is itself
    optimistic;
  - how many missed potholes a planner can tolerate (no planner input yet);
  - whether misses come from small object size, class rarity, or confusion
    with "other corruption" / "alligator crack" (confusion matrix and missed
    frames).

### Failure cases

| Area | What could go wrong and why it matters | How would we notice it? | Planned response |
| --- | --- | --- | --- |
| Data or input | Dark, wet or glare-heavy frames, or a camera type unlike RDD2022, reach the model; defects are missed and the section looks clean | Share of frames with zero detections jumps for one drive; planner spot-check of that drive finds visible damage | Brightness/blur check rejects such frames with a reason; rejected frames shown per section, never counted as clean |
| Model behavior | A pothole is missed or labelled as "other corruption" / "alligator crack" (**matches the acceptance check**) | Pothole recall below threshold on test sample; confusion matrix shows pothole→other classes; planner finds unflagged potholes | Route low-confidence pothole candidates to review; test lower pothole cutoff, pothole oversampling, higher input size |
| Integration or workflow | One pothole seen in several consecutive frames is counted several times, or frames lack a section ID; section summary is inflated or unlinked | Detections per section far above the number of distinct defects on a manual check; detections with no section ID in the log | Merge detections of the same defect across consecutive frames; frames without section ID are rejected from the summary |