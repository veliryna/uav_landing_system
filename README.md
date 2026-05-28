# uav_landing_system

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