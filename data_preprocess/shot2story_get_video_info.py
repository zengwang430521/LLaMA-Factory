import os
import json
import subprocess
import concurrent.futures
from tqdm import tqdm


def get_video_info(video_path):
    """使用 ffprobe 获取视频的时长和分辨率"""
    # print(video_path)
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        info = json.loads(result.stdout)

        # 获取时长
        duration = float(info["format"]["duration"])

        # 获取分辨率
        width = info["streams"][0]["width"]
        height = info["streams"][0]["height"]

        # 计算帧率
        r_frame_rate = info["streams"][0]["r_frame_rate"]
        num, denom = map(int, r_frame_rate.split('/'))
        frame_rate = num / denom
        print(f'{video_path}: {duration} s, ({width}, {height}), {frame_rate} fps')
        return video_path, {"duration": duration, "width": width, "height": height, "frame_rate": frame_rate}
    except Exception as e:
        print(f'{video_path}: {str(e)}')
        return video_path, {"error": str(e)}


def process_videos_in_parallel(video_paths, max_workers=20):
    """并行处理多个视频文件"""
    video_info_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(get_video_info, path): path for path in video_paths}

        for future in concurrent.futures.as_completed(future_to_video):
            video_path, info = future.result()
            video_info_dict[video_path] = info

    return video_info_dict


# 示例用法
if __name__ == "__main__":
    src_dir = 'data/shot2story-videos'
    video_list = []
    for root, dirs, files in os.walk(src_dir):
        for file in tqdm(files):
            video_path = os.path.join(root, file)
            video_list.append(video_path)

    # get_video_info(video_path[0])

    print('begin')
    print(len(video_list))
    video_info = process_videos_in_parallel(video_list, max_workers=8)
    with open('shot2story_video_info.json', 'w') as f:
        json.dump(video_info, f, ensure_ascii=False)

