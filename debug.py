frame_nums = []
for video, time_seg in zip(videos, video_time_segs):
    video_info = video_infos[video]
    real_fps = video_info["fps"]
    video_duration = video_info["duration"]
    total_frames = video_info['frame_num']

    # 计算这一段需要采样多少帧
    t_start, t_end = time_seg
    seg_duration = t_end - t_start
    frame_num = min(seg_duration * video_fps, seg_duration * real_fps)
    frame_num = min(frame_num, max_total_frame_num * seg_duration / total_duration)
    frame_num = math.floor(frame_num)
    frame_num = max(frame_num, 2)  # 最少采集2帧
    if frame_num % 2 != 0:
        # 必须是偶数
        frame_num -= 1
    frame_nums.append(frame_num)