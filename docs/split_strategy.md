# Split strategy

**Branch:** Our team trains the model. We fine-tune a pretrained YOLOv8n.

**Training and evaluation data:** We keep the three splits of the provided
RDD2022 copy and draw a fixed-seed sample (seed 42) from each one.

- **Training sample:** 10,000 images from `train`. The model is fitted on
  these.
- **Validation sample:** 2,000 images from `val`. It is used only during
  training, for early stopping and for choosing the best epoch.
- **Test sample:** 2,000 images from `test`. It is held out. It is not used
  to fit the model, choose the model, or adjust any setting, and it is scored
  once after training. Its mAP50 is the baseline result.

**Split method:** Random, at the image level, as supplied with the dataset
copy. The split is not grouped by drive, road or country. In the sample, the
class proportions stay within about 0.5 percentage points of the full
training set (the validation sample is within about 2 points).

**Does the evaluation resemble future input?** Partly. The test images come
from the same six countries and the same kinds of cameras as the training
images, which matches a planner using similar vehicle cameras. The test
images are still frames, however, not continuous video from a new road or a
new city. This is a limitation for deployment on unseen roads.

**Limitation of this first evaluation:** Some RDD images come from
consecutive frames of the same drive. A random image-level split can
therefore put near-identical scenes in both training and test, which would
make the score optimistic. The dataset also contains duplicated
" (1)" copies. The pipeline reports how many base file names appear in both
the training and test samples. A later version should group images by drive
or location, or at least remove duplicates across splits.
