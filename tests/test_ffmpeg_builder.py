from ffmpeg_utils import build_stream_command


def test_build_stream_command_libx264():
    cmd = build_stream_command("STREAM", encoder="libx264")
    # Ensure video and audio inputs are mapped
    assert cmd.count("-map") == 2
    v_map_index = cmd.index("-map")
    assert cmd[v_map_index + 1] == "0:v:0"
    a_map_index = cmd.index("-map", v_map_index + 2)
    assert cmd[a_map_index + 1] == "1:a:0"
    # Encoder selection
    assert "libx264" in cmd
    assert "h264_v4l2m2m" not in cmd
    # Output tee muxer
    assert "-f" in cmd
    assert any("rtmps://a.rtmp.youtube.com/live2/STREAM" in part for part in cmd)
