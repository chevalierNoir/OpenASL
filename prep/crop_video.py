import os
import json
import math
import cv2
import sys
import glob
import subprocess
import shutil
import tempfile
import argparse
import numpy as np
import json
from tqdm import tqdm
import pandas as pd

def crop_resize(imgs, bbox, target_size):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    if x1-x0<y1-y0:
        exp = (y1-y0-(x1-x0))/2
        x0, x1 = x0-exp, x1+exp
    else:
        exp = (x1-x0-(y1-y0))/2
        y0, y1 = y0-exp, y1+exp
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    left_expand = -x0 if x0 < 0 else 0
    up_expand = -y0 if y0 < 0 else 0
    right_expand = x1-imgs[0].shape[1]+1 if x1 > imgs[0].shape[1]-1 else 0
    down_expand = y1-imgs[0].shape[0]+1 if y1 > imgs[0].shape[0]-1 else 0
    rois = []
    for img in imgs:
        expand_img = cv2.copyMakeBorder(img, up_expand, down_expand, left_expand, right_expand, cv2.BORDER_CONSTANT, (0, 0, 0))
        roi = expand_img[y0+up_expand: y1+up_expand, x0+left_expand: x1+left_expand]
        roi = cv2.resize(roi, (target_size, target_size))
        rois.append(roi)
    return rois

def write_video_ffmpeg(rois, target_path, ffmpeg):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    decimals = 10
    fps = 25
    tmp_dir = tempfile.mkdtemp()
    for i_roi, roi in enumerate(rois):
        cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    list_fn = os.path.join(tmp_dir, "list")
    with open(list_fn, 'w') as fo:
        fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    ## ffmpeg
    if os.path.isfile(target_path):
        os.remove(target_path)
    cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', '-pix_fmt', 'yuv420p', target_path]
    pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # rm tmp dir
    shutil.rmtree(tmp_dir)
    return

def get_clip(input_video_dir, output_video_dir, tsv_fn, bbox_fn, rank, nshard, target_size=224, ffmpeg=None):
    os.makedirs(output_video_dir, exist_ok=True)
    df = pd.read_csv(tsv_fn, sep='\t')
    vid2bbox = json.load(open(bbox_fn))
    items = []
    for vid, yid, start, end in zip(df['vid'], df['yid'], df['start'], df['end']):
        if vid not in vid2bbox:
            continue
        bbox = vid2bbox[vid]
        items.append([vid, yid, start, end, bbox])
    num_per_shard = (len(items)+nshard-1)//nshard
    items = items[num_per_shard*rank: num_per_shard*(rank+1)]
    print(f"{len(items)} videos")
    for vid, yid, start_time, end_time, bbox in tqdm(items):
        input_video_whole, output_video = os.path.join(input_video_dir, yid+'.mp4'), os.path.join(output_video_dir, vid+'.mp4')
        if os.path.isfile(output_video):
            continue
        tmp_dir = tempfile.mkdtemp()
        input_video_clip = os.path.join(tmp_dir, 'tmp.mp4')
        cmd = [ffmpeg, '-ss', start_time, '-to', end_time, '-i', input_video_whole, '-c:v', 'libx264', '-crf', '20', input_video_clip]
        print(' '.join(cmd))
        subprocess.call(cmd)
        cap = cv2.VideoCapture(input_video_clip)
        frames_origin = []
        print(f"Reading video clip: {input_video_clip}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_origin.append(frame)
        shutil.rmtree(tmp_dir)
        x0, y0, x1, y1 = bbox
        W, H = frames_origin[0].shape[1], frames_origin[0].shape[0]
        bbox = [int(x0*W), int(y0*H), int(x1*W), int(y1*H)]
        print(bbox, frames_origin[0].shape, target_size)
        rois = crop_resize(frames_origin, bbox, target_size)
        print(f"Saving ROIs to {output_video}")
        write_video_ffmpeg(rois, output_video, ffmpeg=ffmpeg)
    return


def main():
    parser = argparse.ArgumentParser(description='download video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, help='data tsv file')
    parser.add_argument('--bbox', type=str, help='bbox json file')
    parser.add_argument('--raw', type=str, help='raw video dir')
    parser.add_argument('--output', type=str, help='output dir')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='path to ffmpeg')
    parser.add_argument('--target-size', type=int, default=224, help='image size')

    parser.add_argument('--slurm', action='store_true', help='slurm or not')
    parser.add_argument('--nshard', type=int, default=100, help='number of slurm jobs to launch in total')
    parser.add_argument('--slurm-argument', type=str, default='{"slurm_array_parallelism":100,"slurm_partition":"speech-cpu","timeout_min":240,"slurm_mem":"16g"}', help='slurm arguments')
    args = parser.parse_args()

    if args.slurm:
        import submitit
        nshard = args.nshard
        executor = submitit.AutoExecutor(folder='submitit')
        params = json.loads(args.slurm_argument)
        executor.update_parameters(**params)
        jobs = executor.map_array(get_clip, [args.raw]*nshard, [args.output]*nshard, [args.tsv]*nshard, [args.bbox]*nshard, list(range(0, nshard)), [nshard]*nshard, [args.target_size]*nshard, [args.ffmpeg]*nshard)
    else:
        get_clip(args.raw, args.output, args.tsv, args.bbox, 0, 1, target_size=args.target_size, ffmpeg=args.ffmpeg)
    return


if __name__ == '__main__':
    main()
