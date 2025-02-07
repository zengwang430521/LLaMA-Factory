import av
video = 'data/LLaVA-Video-178K/0_30_s_academic_v0_1/academic_source/NextQA/1015/2774585614.mp4'
container = av.open(video, "r")
video_stream = next(stream for stream in container.streams if stream.type == "video")
total_frames = video_stream.frames
container.seek(0)
for frame_idx, frame in enumerate(container.decode(video_stream)):
    t = frame.to_image()
    print(t)
