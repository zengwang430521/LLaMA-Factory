def _get_fake_mm_inputs(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
        video_time_segs: Sequence["List"],
):
    '''图片部分正常处理'''
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    input_dict = {"images": None}  # default key
    if len(images) != 0:
        images = self._regularize_images(
            images,
            image_resolution=getattr(processor, "image_resolution", 512 * 512),
        )
        input_dict["images"] = images
    mm_inputs = {}
    if input_dict.get("images") is not None:
        mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))

    # 视频部分简单处理
    # 先收集视频信息
    video_infos = {}
    for video in videos:
        if video in video_infos.keys():
            continue
        container = av.open(video, "r")
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        frame_width, frame_height = video_stream.codec_context.width, video_stream.codec_context.height
        total_frames = video_stream.frames
        duration = float(video_stream.duration * video_stream.time_base)
        fps = float(total_frames) / duration
        video_infos[video] = {
            "width": frame_width,
            "height": frame_height,
            "duration": duration,
            "frame_num": total_frames,
            "fps": fps
        }

    # 给每段分配帧数
    video_fps = getattr(processor, "video_fps", 2.0)
    video_maxlen = getattr(processor, "video_maxlen", 64)

    # 先处理一下time_seg
    total_duration = 0
    for i in range(len(video_time_segs)):
        video = videos[i]
        video_duration = video_infos[video]["duration"]
        t_start, t_end = video_time_segs[i]
        t_start, t_end = max(t_start, 0), min(t_end, video_duration)
        total_duration += t_end - t_start
        video_time_segs[i] = [t_start, t_end]

    # 计算每段的shape和frame index
    # _regularize_videos() 中的过程
    video_grid_thw = []
    frame_times, frame_idxs = [], []
    for video, time_seg in zip(videos, video_time_segs):
        video_info = video_infos[video]
        video_duration = video_info["duration"]
        frame_width, frame_height = video_info["width"], video_info["height"]
        total_frames = video_info['frame_num']
        real_fps = video_info["fps"]

        # 先计算这一段需要采样多少帧
        t_start, t_end = time_seg
        seg_duration = t_end - t_start

        frame_num = min(seg_duration * video_fps, seg_duration * real_fps)
        frame_num = min(frame_num, video_maxlen * seg_duration / total_duration)
        frame_num = math.floor(frame_num)

        sample_times = np.linspace(t_start, t_end, frame_num, endpoint=False)
        sample_idxs = (sample_times * real_fps).round().astype(np.int32)

        sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
        sample_times = np.linspace(0, float(video_stream.duration * video_stream.time_base), sample_frames)
        sample_indices_seg, sample_times_seg = [], []
        for idx, t in zip(sample_indices, sample_times):
            if t_start <= t <= t_end:
                sample_indices_seg.append(idx)
                sample_times_seg.append(t)
        video_maxlen = getattr(processor, "video_maxlen", 64)
        if len(sample_indices_seg) > video_maxlen:
            sample_indices_seg = sample_indices_seg[-video_maxlen:]
            sample_times_seg = sample_times_seg[-video_maxlen:]

        # 不需要真的读取视频
        # frames: List["ImageObject"] = []
        # container.seek(0)
        # for frame_idx, frame in enumerate(container.decode(video_stream)):
        #     if frame_idx in sample_indices_seg:
        #         frames.append(frame.to_image())

        if len(sample_times_seg) % 2 != 0:
            sample_times_seg.append(sample_times_seg[-1])

        sample_frame_shapes = [(frame_width, frame_height)] * len(sample_times_seg)
        sample_frame_shapes = _regularize_images_shape(sample_frame_shapes,
                                                       image_resolution=getattr(processor, "video_resolution",
                                                                                128 * 128))

        # video_frame_shapes.append(sample_frame_shapes)
        # frame_times.append(sample_times_seg)

        # image_processor 过程
        new_width, new_height = sample_frame_shapes[0]
        grid_thw = _process_images_shape(len(sample_frame_shapes), new_width, new_height, image_processor)
        video_grid_thw.append(grid_thw)
        frame_times.append(sample_times_seg[::2])
    video_grid_thw = torch.tensor(video_grid_thw)
    return video_grid_thw, frame_times

    def _regularize_images_shape(image_shapes, image_resolution):
        output_shapes = []
        for width, height in image_shapes:
            if (width * height) > image_resolution:
                resize_factor = math.sqrt(image_resolution / (width * height))
                width, height = int(width * resize_factor), int(height * resize_factor)

            if min(width, height) < 28:
                width, height = max(width, 28), max(height, 28)

            if width / height > 200:
                width, height = height * 180, height

            if height / width > 200:
                width, height = width, width * 180

            output_shapes.append((width, height))

        return output_shapes

    def _process_images_shape(num_frame, width, height, image_processor):
        if num_frame == 1:
            num_frame = 2
        if image_processor.do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_processor.patch_size * image_processor.merge_size,
                min_pixels=image_processor.min_pixels,
                max_pixels=image_processor.max_pixels,
            )
        else:
            resized_height, resized_width = height, width

        grid_t = num_frame // image_processor.temporal_patch_size
        grid_h, grid_w = resized_height // image_processor.patch_size, resized_width // image_processor.patch_size
        return grid_t, grid_h, grid_w

    video_grid_thw, frame_times = get_video_grid_thw()
    mm_inputs["video_grid_thw"] = video_grid_thw
    mm_inputs["frame_times"] = frame_times
    return mm_inputs
