# Experimental VIAME .pipe scaffold (WIP)

This folder contains an experimental `.pipe` template for teams aiming to integrate the FishingROV cascade into DIVE Web or DIVE Desktop workflows.

File:
- `cascade_teacher_yolo_swin.pipe`

Status:
- Experimental / work-in-progress.
- Intended as a starting template, not guaranteed drop-in.

## Outstanding requirements for drop-in DIVE compatibility

1. Plugin process-name alignment
- Confirm each process exists on your install (`video_input`, `image_filter`, `image_object_detector`, `refine_detections`, `merge_detection_sets`, `detection_csv_writer`).
- If names differ, replace process types and config keys with your build's equivalents.

2. Detector/classifier backend alignment
- `:detector:type = netharn` is a placeholder and may differ in your build.
- `classify_object_track` settings are build-dependent; replace with your classifier process if needed.

3. Model packaging compatibility
- The public bundle ships `.pt` weights for script-based inference.
- Many VIAME pipeline runners expect deployed archives (commonly `.zip`) rather than raw `.pt`.
- You must package/convert models into a format your VIAME deployment can load.

4. Path conventions and portability
- Ensure relative model paths resolve inside your DIVE/VIAME runtime environment.
- For DIVE Web uploads, verify file layout and upload packaging requirements (for example trained pipeline bundle layout).

5. Runtime behavior parity with script pipeline
- Match these key settings if parity is required:
  - left/right crop geometry (`xmin/xmax/ymin/ymax`)
  - right-stream x-offset shift (`840`)
  - NMS IoU (`0.45`)
- Note: normalized weighted-argmax classifier logic from the Python cascade may not be reproduced automatically by generic VIAME classifier processes.

6. Output schema and downstream import
- Validate emitted CSV is accepted by your DIVE import path.
- If labels/attributes differ, add a label-map or post-step to normalize class naming.

7. Validation checklist before calling it drop-in
- Run `kwiver runner --help-pipe` and verify all process keys used by the `.pipe`.
- Run on a short known video and compare detections with script outputs.
- Check class label names and confidences for expected semantics.
- Confirm DIVE Web/Desktop import completes without manual edits.

## Suggested adoption path

1. Keep `scripts/run_video.sh` / `scripts/run_images.sh` as the primary supported path.
2. Use this `.pipe` as an integration prototype for your target VIAME environment.
3. Once process keys and model packaging are confirmed, freeze a build-specific `stable` pipeline variant.
