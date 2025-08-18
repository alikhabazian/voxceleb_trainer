import glob
from tqdm import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_WORKERS = 8  # Set based on your CPU and disk capacity

def convert_one(fname):
    outfile = fname.replace('.mp3', '.wav')
    cmd = [
        'ffmpeg', '-y',
        '-i', fname,
        '-ac', '1',
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        outfile
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return (fname, False)
    except Exception as e:
        return (fname, False)
    return (fname, True)

def convert_all():
    files = glob.glob('selected_clips/*.mp3')
    print(files[0])
    files.sort()
    print(f'{len(files)}')
    print(f'Converting {len(files)} files using {NUM_WORKERS} workers...')
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(convert_one, f) for f in files]
        for f in tqdm(as_completed(futures), total=len(futures)):
            fname, success = f.result()
            if not success:
                print(f"Failed to convert: {fname}")

convert_all()
