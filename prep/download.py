import os,sys,subprocess,json
from collections import OrderedDict
from tqdm import tqdm
import argparse
import pandas as pd

def download_video(vids, target_dir, youtube_dl):
    for url in tqdm(vids):
        cmd = [youtube_dl, url , "-f", "bestvideo+bestaudio[ext=m4a]/bestvideo+bestaudio/best", "--merge-output-format", "mp4", "--no-check-certificate", "--restrict-filenames", "-o", "'"+target_dir+"/%(id)s.%(ext)s"+"'"]
        print(" ".join(cmd))
        subprocess.run(" ".join(cmd), shell=True)
    return

def main():
    parser = argparse.ArgumentParser(description='download video', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, help='data tsv file')
    parser.add_argument('--dest', type=str, help='dest dir')
    parser.add_argument('--slurm', action='store_true', help='slurm or not')
    parser.add_argument('--nshard', type=int, default=100, help='number of slurm jobs to launch in total')
    parser.add_argument('--slurm-argument', type=str, default='{"slurm_array_parallelism":100,"slurm_partition":"speech-cpu","timeout_min":240,"slurm_mem":"16g"}', help='slurm arguments')

    args = parser.parse_args()
    tsv_fn = args.tsv
    df = pd.read_csv(tsv_fn, sep='\t')
    yids = sorted(list(set(df['yid'])))
    print(f"Download {len(yids)} raw videos into {args.dest}")
    os.makedirs(args.dest, exist_ok=True)
    youtube_dl = os.path.join(os.getcwd(), "youtube-dl")
    if not os.path.isfile(youtube_dl):
        cmd = f"curl -L https://github.com/ytdl-org/ytdl-nightly/releases/latest/download/youtube-dl -o {youtube_dl}\nchmod +rx {youtube_dl}"
        print(cmd)
        subprocess.call(cmd, shell=True)
    if not args.slurm:
        download_video(yids, args.dest, youtube_dl)
    else:
        import submitit
        executor = submitit.AutoExecutor(folder='submitit')
        params = json.loads(args.slurm_argument)
        executor.update_parameters(**params)
        print(f"launching slurm jobs\narguments: {params}...")
        batch_yids = []
        num_per_shard = (len(yids)+args.nshard-1)//args.nshard
        for i in range(0, len(yids), num_per_shard):
            batch_yids.append(yids[i: i+num_per_shard])
        jobs = executor.map_array(download_video, batch_yids, [args.dest]*len(batch_yids), [youtube_dl]*len(batch_yids))
        [job.result() for job in jobs]
    missing, n_complete = [], 0
    for yid in yids:
        dest_fn = f"{args.dest}/{yid}.mp4"
        if os.path.isfile(dest_fn):
            n_complete += 1
        else:
            missing.append(yid)
    print(f"{n_complete}/{len(yids)} videos downloaded successfully.")
    if len(missing) > 0:
        fn = f"{args.dest}/missing.txt"
        with open(fn, "w") as fo:
            fo.write("\n".join(missing)+"\n")
        print(f"List of undownloaded videos saved in {fn}.")
    return


if __name__ == '__main__':
    main()
