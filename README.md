# uav_landing_system

## Ardupilot SITL

<img width="432" height="618" alt="Screenshot from 2026-06-01 20-22-26" src="https://github.com/user-attachments/assets/ada126b2-962a-4e20-965f-3d207bc74d16" />

## How to run tests

### Test local video

Go into `src` directory and replace the `landing_tag.mp4` with your test video or use the existing one.

In the same directory, run:

```console
python3 test_video_pipeline.py --video landing_tag.mp4 --slow 60 --out test_result_annotated.mp4
```
`test_video_pipeline.py` has extensive documentation on usage, CL arguments, and output format.

### Unit tests

In the root project directory, run:

```console
python3 -m unittest discover -v tests
```

```console
python3 -m pytest -v tests/
```
