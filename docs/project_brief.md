# Project brief

**Winter Road Surface Classifier** — Team Surface 3
Yikwa Sandrakh, Wenda Waipo, Agata Melani
Artificial Intelligence Implementation Technologies · Baikal School of BRICS, INRTU

## User and situation

Road maintenance dispatchers responsible for deciding which roads are treated — salted, plowed or sanded — and in what order need the system.

Currently, dispatchers may need to rely on manual inspection and driver reports to judge the condition of a road surface. On a large network, a stretch that has turned icy may not be noticed until a crew reaches it or a driver reports it.

The future system should help dispatchers identify road sections that may require urgent treatment by analysing road-facing camera images.

## What the system should do

- **Input:** Images or video frames from road-facing cameras.
- **Useful output:** A surface-condition label for each frame — dry, wet, snow-covered or icy — together with a confidence value.
- **Action or decision after the output:** Staff review the sections flagged as icy or snow-covered and decide which stretches to treat first.

## Why AI may help

AI may help because the system needs to recognise visual patterns associated with dry, wet, snow-covered and icy surfaces. These conditions are distinguished by texture, sheen and reflection rather than by any single measurable quantity.

Road appearance may vary because of lighting, time of day, weather, camera position, road material and passing traffic. Wet and icy surfaces in particular may look similar to one another, and the difference between them may be visible only in how light reflects from the surface. These differences may make fixed if/then rules insufficient.

Other parts of the future system may include cameras, frame extraction from video, an inference service, a dispatcher dashboard, alerts or notifications, and human review of each result before a crew is sent.

## Initial data plan

- **Where the data may come from:** Public dashcam and road-camera video, supplemented by footage recorded locally by the team if winter conditions and permission allow.
- **What we can access now:** Twenty source videos have been gathered and a first set of frames has been annotated into the four classes. The class counts are uneven — 114 snow, 79 icy, 20 dry and 8 wet — so the present set is large enough to test the pipeline but not to train a model.
- **What still needs confirmation:** We need to confirm whether enough icy-condition footage can be obtained within the remaining winter, whether wet and icy surfaces can be separated consistently by human annotators, and whether four classes is a small enough problem for the course.

## Three next actions

1. Collect and annotate more icy and wet examples, since these are the two conditions the system exists to tell apart and the two with the fewest examples.
2. Agree a written labelling guide for the four classes, have all three team members label the same subset of frames, and measure how often they agree before the remainder is labelled.
3. Decide whether the first version should classify all four conditions, or begin with the safety-critical split of icy versus not-icy, which the current data may support sooner.

---

*Detailed technical material — pipeline, model configuration, evaluation protocol, delivery plan and the record of methodological decisions — is kept in `README.md`.*
