import json
import os
from os.path import join
from tqdm import tqdm
import subprocess
import json
import tqdm


def get_video_duration_ffprobe(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = json.loads(result.stdout)["format"]["duration"]
    return float(duration)


video_path = "example.mp4"
print(get_video_duration_ffprobe(video_path))  # 输出秒数，例如 123.45
video_duration_dict = {}
src_dir = 'data/LLaVA-Video-178K'
for root, dirs, files in os.walk(src_dir):
    for file in tqdm(files):
        video_path = os.path.join(root, file)
        try:
            video_duration = get_video_duration_ffprobe(video_path)
            video_duration_dict[video_path] = video_duration
        except:
            print(f'ERROR: video_path')

with open('llava_video_len.json', 'w') as f:
    json.dump(f, ensure_ascii=False)