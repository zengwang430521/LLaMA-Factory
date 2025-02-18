import os
import json
import subprocess
import concurrent.futures


def get_video_info(video_path):
    """使用 ffprobe 获取视频的时长和分辨率"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        import json
        data = json.loads(result.stdout)

        duration = float(data["format"]["duration"])
        width = int(data["streams"][0]["width"])
        height = int(data["streams"][0]["height"])

        print(f'{video_path}: {duration}, ({width}, {height})')
        return video_path, {"duration": duration, "resolution": (width, height)}
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
    src_dir = 'data/LLaVA-Video-178K'
    video_list = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            video_path = os.path.join(root, file)
            video_list.append(video_path)

    video_info = process_videos_in_parallel(video_list, max_workers=8)
    with open('llava_video_info.json', 'w') as f:
        json.dump(video_info, f, ensure_ascii=False)

