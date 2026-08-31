# TRIDENT job summary

This file is updated once per run and summarizes what TRIDENT has done in this `job_dir`.

- Per-slide machine-readable state lives in `wsi_states/*.json`.
- Per-run manifests live in `runs/*.json`.

## Run 2026-08-04T17:44:38+0900 (trident 0.3.0) — run_id=2cba6f693cb2
- Tool: `run_single_slide`
- Status: **completed**
- Finished: `2026-08-04T17:46:15+0900`
- Slides with state: 0
- Args: `{"batch_size": 32, "custom_mpp_keys": null, "dump_patches": false, "dump_patches_format": "png", "dump_patches_jpeg_quality": 90, "dump_patches_max": 0, "gpu": 0, "job_dir": "../dataset/features", "mag": 20, "overlap": 0, "patch_encoder": "conch_v15", "patch_encoder_img_size": null, "patch_size": 512, "reader_type": null, "remove_artifacts": true, "remove_holes": false, "remove_penmarks": false, "seg_conf_thresh": 0.5, "segmenter": "grandqc", "slide_path": "../dataset/wsi/normal/70386.svs"}`

> No `wsi_states/*.json` found yet for this job dir, so this summary only contains run metadata.

## Run 2026-08-04T17:44:48+0900 (trident 0.3.0) — run_id=d437489bdbec
- Tool: `run_single_slide`
- Status: **completed**
- Finished: `2026-08-04T17:46:16+0900`
- Slides with state: 0
- Args: `{"batch_size": 32, "custom_mpp_keys": null, "dump_patches": false, "dump_patches_format": "png", "dump_patches_jpeg_quality": 90, "dump_patches_max": 0, "gpu": 1, "job_dir": "../dataset/features", "mag": 20, "overlap": 0, "patch_encoder": "conch_v15", "patch_encoder_img_size": null, "patch_size": 512, "reader_type": null, "remove_artifacts": true, "remove_holes": false, "remove_penmarks": false, "seg_conf_thresh": 0.5, "segmenter": "grandqc", "slide_path": "../dataset/wsi/abnormal/28096.svs"}`

> No `wsi_states/*.json` found yet for this job dir, so this summary only contains run metadata.

## Run 2026-08-04T18:28:54+0900 (trident 0.3.0) — run_id=532db88ed3e4
- Tool: `run_batch_of_slides`
- Status: **completed**
- Finished: `2026-08-05T02:54:10+0900`
- Slides with state: 1018
- Args: `{"batch_size": 64, "cache_batch_size": 32, "clear_dead_locks": false, "coords_dir": null, "custom_list_of_wsis": null, "custom_mpp_keys": null, "dead_lock_max_age_hours": 24.0, "device": "cuda:0", "dump_patches": false, "dump_patches_format": "png", "dump_patches_jpeg_quality": 90, "dump_patches_max": 0, "feat_batch_size": null, "gpu": 0, "job_dir": "../dataset/features", "mag": 20.0, "max_workers": null, "min_tissue_proportion": 0.0, "overlap": 0, "patch_encoder": "conch_v15", "patch_encoder_ckpt_path": null, "patch_encoder_img_size": null, "patch_size": 512, "reader_type": null, "remove_artifacts": true, "remove_holes": false, "remove_penmarks": false, "search_nested": true, "seg_batch_size": null, "seg_conf_thresh": 0.5, "segmenter": "grandqc", "skip_errors": false, "slide_encoder": null, "task": "all", "wsi_cache": null, "wsi_dir": "../dataset/wsi", "wsi_ext": null}`
- coords: completed: 1018
- segmentation: completed: 1018
- Patch features:
  - conch_v15: completed: 1018
