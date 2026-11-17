# CLAUDE.md — hips_dummy_step

A dummy pipeline step used to measure throughput. For every alert it consumes, it calls an
external image service with the alert's position and receives catalog reference images of that
region of the sky. It does not use the images yet: the goal is only to find out whether the
service and the step keep up with the alert rate. Later, these images are meant as an input
for ML models.

## Shape

- One step, deployed as two instances with different configuration: one consuming the ZTF
  stream, one consuming the LSST stream, each with its own new consumer group.
- Follow the structure of the existing multisurvey steps in this repo (e.g.
  `../rubin_stamp_classifier_step/`, `../*_multisurvey_step/`), the step framework in
  `../libs/apf`, and Helm charts in `../charts/`. Stream overview: `../PIPELINE-MULTISURVEY.md`.
- Base branch: `multisurvey`.

## Rules

- This repository is **public**. Do not commit the service's URL, host, network details or a
  description of its internals. The endpoint and any credentials come from configuration that
  is not tracked by git (e.g. a `.env` file, already git-ignored, or deployment values kept
  outside this repo).
- The service may answer "busy, retry later"; the client must back off and retry rather than
  fail the alert.
