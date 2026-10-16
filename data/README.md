# Data

**Dataset:** RDD2022 (Road Damage Detection 2022), five damage classes:
0 Longitudinal Crack, 1 Transverse Crack, 2 Alligator Crack,
3 Other Corruption, 4 Pothole.

**Source:** The public RDD2022 benchmark, used through a pre-split copy in
YOLO format. The copy has 26,869 train, 5,758 val and 5,758 test images, and
every image has a label file. The team does not collect its own footage. Ask
a team member for the copy; the download link is not stored in Git.

**Licence / access:** RDD2022 is released for research use. The exact licence
of the copy we use is *not yet confirmed*.

**Stored in Git?** No. The dataset is several GB and is provided separately.
`data/RDD_SPLIT/`, `data/RDD_SAMPLE/` and `data/dataset/` are in
`.gitignore`.

**Expected folder structure** (place the copy here before running):

```
data/
  RDD_SPLIT/
    train/images/*.jpg   train/labels/*.txt
    val/images/*.jpg     val/labels/*.txt
    test/images/*.jpg    test/labels/*.txt
```

Each label file has one line per object:
`class_id x_center y_center width height`, with the box values normalised to
[0, 1]. An empty label file means the image contains no damage. About 30% of
the training images are like this.

**Preparation:** None by hand. The pipeline:

- pairs images with labels by file name;
- validates every sampled label line;
- draws a fixed-seed sample into `data/RDD_SAMPLE/` (5,000 train / 2,000 val
  / 2,000 test);
- writes `data/RDD_SAMPLE/data.yaml`.
