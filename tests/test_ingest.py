import time

import pytest

from basira.ingest.buffer import Frame, LatestFrameBuffer

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def make_frame(i):
    return Frame("cam_test", i, time.time(), None)


def test_buffer_drops_oldest_when_full():
    buf = LatestFrameBuffer(maxsize=2)
    for i in range(5):
        buf.put(make_frame(i))
    assert len(buf) == 2
    assert buf.dropped == 3
    assert buf.get().index == 3
    assert buf.get().index == 4


def test_buffer_get_times_out_when_empty():
    assert LatestFrameBuffer().get(timeout=0.05) is None


def test_latest_does_not_remove():
    buf = LatestFrameBuffer()
    buf.put(make_frame(7))
    assert buf.latest().index == 7
    assert len(buf) == 1


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return str(path)


def test_stream_reads_every_frame_of_a_file(tiny_video):
    from basira.config import CameraConfig
    from basira.ingest.stream import CameraStream

    cam = CameraConfig(id="cam_test", name="Test", source=tiny_video)
    stream = CameraStream(cam, buffer_size=100, loop=False, realtime=False).start()
    deadline = time.time() + 5
    while not stream.finished and time.time() < deadline:
        time.sleep(0.01)
    assert stream.finished
    assert stream.frames_read == 30
    assert stream.buffer.latest().image.shape == (48, 64, 3)


def test_manager_runs_several_cameras(tiny_video):
    from basira.config import CameraConfig
    from basira.ingest.stream import StreamManager

    cams = [CameraConfig(id=f"cam_{i}", name=str(i), source=tiny_video) for i in range(3)]
    with StreamManager(cams) as m:
        time.sleep(0.5)
        frames = m.latest_frames()
    assert set(frames) == {"cam_0", "cam_1", "cam_2"}
    assert all(f is not None for f in frames.values())


def test_missing_file_does_not_crash():
    from basira.config import CameraConfig
    from basira.ingest.stream import CameraStream

    cam = CameraConfig(id="cam_x", name="Missing", source="does_not_exist.mp4")
    stream = CameraStream(cam, loop=False).start()
    stream.stop()
    assert stream.frames_read == 0
