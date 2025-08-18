#!/usr/bin/env python3
import argparse, random, itertools, collections
from pathlib import Path

def parse_input(path):
    """
    Expect each line: speaker_id  accent_name  file_name
    Delimited by whitespace (tabs or spaces). file_name has no spaces.
    """
    speakers = collections.defaultdict(list)  # spk -> [file]
    accents  = collections.defaultdict(list)  # acc -> [file]
    meta     = {}                             # file -> (spk, acc)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.split()
            if len(parts) < 3:
                # Try to be forgiving: join trailing tokens as filename
                if len(parts) >= 2:
                    spk, acc = parts[0], parts[1]
                    fname = parts[-1]
                else:
                    continue
            else:
                spk, acc, fname = parts[0], parts[1], parts[2]

            # Enforce file1 != file2 later; here just collect
            speakers[spk].append(fname)
            accents[acc].append(fname)
            meta[fname] = (spk, acc)

    return speakers, accents, meta

def sample_pos_within_groups(groups, target_n, rng):
    """
    Positive pairs: same group (speaker or accent). 
    Avoid generating all combinations on huge groups by random sampling.
    """
    pos = set()
    group_keys = [k for k, files in groups.items() if len(files) >= 2]
    if not group_keys: 
        return []

    # Precompute combinations for small groups; random for big ones
    SMALL_CAP = 5000  # threshold of combinations to precompute
    for g in group_keys:
        files = groups[g]
        m = len(files)
        # number of combos mC2
        comb_count = m*(m-1)//2
        if comb_count <= SMALL_CAP:
            for a, b in itertools.combinations(files, 2):
                if a != b:
                    pos.add(tuple(sorted((a, b))))
        else:
            # random sampling: draw more than needed but cap attempts
            attempts = 0
            max_attempts = min(comb_count, target_n * 10 + 10000)
            while attempts < max_attempts and len(pos) < target_n:
                a, b = rng.sample(files, 2)
                if a != b:
                    pos.add(tuple(sorted((a, b))))
                attempts += 1

        if len(pos) >= target_n:
            break

    # truncate to target_n
    pos = list(pos)
    rng.shuffle(pos)
    return pos[:target_n]

def sample_neg_across_groups(meta, key_index, target_n, rng):
    """
    Negative pairs: different group (accent for accent task; speaker for speaker task).
    key_index = 1 for accent (compare accents),
               = 0 for speaker (compare speakers).
    """
    files = list(meta.keys())
    neg = set()
    attempts = 0
    max_attempts = max(10000, target_n * 20)

    while len(neg) < target_n and attempts < max_attempts:
        a, b = rng.sample(files, 2)
        if a == b:
            attempts += 1
            continue
        ka = meta[a][key_index]
        kb = meta[b][key_index]
        if ka != kb:
            neg.add(tuple(sorted((a, b))))
        attempts += 1

    neg = list(neg)
    rng.shuffle(neg)
    return neg[:target_n]

def write_pairs(pairs_pos, pairs_neg, out_path):
    """
    Write balanced pairs: equal number of 1s and 0s.
    Format: label  file1  file2
    """
    n = min(len(pairs_pos), len(pairs_neg))
    pos = pairs_pos[:n]
    neg = pairs_neg[:n]

    # Interleave for balance (optional)
    lines = []
    for p, n_ in itertools.zip_longest(pos, neg, fillvalue=None):
        if p:
            lines.append(f"1  {p[0]}  {p[1]}\n")
        if n_:
            lines.append(f"0  {n_[0]}  {n_[1]}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return len(pos), len(neg)

def main():
    ap = argparse.ArgumentParser(description="Build balanced accent and speaker pair files.")
    ap.add_argument("input_txt", type=Path, help="Path to input txt (speaker accent file)")
    ap.add_argument("--out-accent", default="acent_test_data.txt", help="Output for accent pairs (balanced)")
    ap.add_argument("--out-speaker", default="speaker_test_data.txt", help="Output for speaker pairs (balanced)")
    ap.add_argument("--pairs", type=int, default=0,
                    help="Desired total pairs per file (balanced, i.e., ~pairs lines). "
                         "If 0, use the maximum possible balanced count.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    speakers, accents, meta = parse_input(args.input_txt)

    # Decide targets
    # Upper bounds for positives are limited by within-group availability; negatives by across-group diversity.
    # We first try a large number, then truncate when writing.
    desired = args.pairs if args.pairs > 0 else 10**9

    # Accent file
    pos_acc = sample_pos_within_groups(accents, desired, rng)
    neg_acc = sample_neg_across_groups(meta, key_index=1, target_n=desired, rng=rng)
    npos_acc, nneg_acc = write_pairs(pos_acc, neg_acc, args.out_accent)
    print(f"Wrote {args.out_accent}: {npos_acc + nneg_acc} lines (1s={npos_acc}, 0s={nneg_acc})")

    # Speaker file
    pos_spk = sample_pos_within_groups(speakers, desired, rng)
    neg_spk = sample_neg_across_groups(meta, key_index=0, target_n=desired, rng=rng)
    npos_spk, nneg_spk = write_pairs(pos_spk, neg_spk, args.out_speaker)
    print(f"Wrote {args.out_speaker}: {npos_spk + nneg_spk} lines (1s={npos_spk}, 0s={nneg_spk})")

if __name__ == "__main__":
    main()
