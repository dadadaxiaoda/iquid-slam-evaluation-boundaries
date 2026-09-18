Liquid Neural SLAM: Evaluation Boundaries and Validation Prerequisites
====================================================================

液态神经网络用于 SLAM 的可行性与评价边界探索

This repository preserves exploratory experiments with closed-form continuous-time
(CfC) networks, GRU controls, physical motion priors and quality-aware correction
gates around ORB-SLAM3. Its contribution is a reproducible record of evaluation
boundaries, supervision prerequisites and instrumentation, rather than a claim of
a universally better SLAM system.

Findings
--------
* A quality-gated CfC correction reduced local error by about 4.07% on V1_03,
  but increased it by about 1.77% on V2_02. A repeat on V1_03 showed about 5.09%
  improvement for CfC, 5.63% for GRU and 5.24% for fixed half attenuation.
  Fixed attenuation was a post-hoc diagnostic. A stable CfC advantage has not
  been established on these exploratory, previously inspected sequences.
* On one instrumented V2_02 replay, 395 one-second windows had 20.96 cm raw
  online-history RPE RMSE, versus 1.43 cm after reconstructing both endpoints
  consistently in the endpoint-time map. At the same 7.10 cm threshold, positive
  windows changed from eight to zero. This is a change in measurement/target,
  NOT a neural-network accuracy gain. It does not show absence of global drift;
  online output jumps can still matter to downstream controllers.
* The reliability pilot stopped before fitting because validation/test had only
  four/eight positive windows. Overlapping windows are not independent events.

Contents
--------
``experiments/`` contains successive residual, physical-prior, gating, reliability
and quality-capture rounds. ``workspace/`` preserves earlier scripts and results.
``upstream_baseline/`` preserves the ORB-SLAM3 source snapshot used for patching.
``artifacts/`` contains split archival model/prediction outputs and their hashes.
``publication_manifest.json`` distinguishes original hashes from public hashes;
machine-specific paths were normalized. Original desktop journals, datasets,
compiled libraries, caches and unrelated personal files are excluded.

Reproduction
------------
Run ``python verify_publication.py`` for a lightweight integrity/syntax check.
Run ``python unpack_artifacts.py`` to restore NPZ predictions and PT checkpoints
into ``restored_artifacts/``. This uses streaming disk I/O and does not train models.
Historical scripts retain stage-specific assumptions: adjust normalized
``/opt/slam-study`` paths and provide the corresponding EuRoC data yourself.
Rebuilding requires ORB-SLAM3 third-party dependencies and the instrumentation
scripts under the quality-capture stages. No turnkey full-system benchmark is
claimed. Recorded experiment environment: Python 3.10.12, numpy 2.2.6,
scipy 1.15.3, torch 2.14.0, ncps 1.0.1, PyYAML 6.0.3.

Interpretation limits
---------------------
Legacy monocular experiments used a ground-truth scale prefix; results are
conditional on that calibration. Stereo-inertial capture used metric poses.
The physical prior is not full bias-estimating VIO. Artificial observation drops
were applied to neural inputs, not to an independently rerun sensor-degraded SLAM.
Ground-truth oracle gates and final optimized trajectories are diagnostic only,
not online inputs. Endpoint-time reconstructed history and immutable online
history represent different targets. Quality proxies do not uniquely classify
every loop closure or bundle adjustment. No publication acceptance or broad
generalization claim follows from these experiments.

License and attribution
-----------------------
Released under GPL-3.0 as provided in LICENSE. ORB-SLAM3 code remains attributed
to its original authors (see NOTICE and source headers). ncps is an external
dependency; its implementation is not bundled. Raw EuRoC datasets are not
redistributed. See https://github.com/UZ-SLAMLab/ORB_SLAM3 and
https://github.com/mlech26l/ncps for upstream projects.
