#!/usr/bin/env python3
"""監看 CASTLE 多卡 extraction 的「真實進度」。

CASTLE 的 2-GPU extraction 進度條會卡在 2%(=0.02*frames) 不動（內層 progress
callback 沒接線），但其實一直在跑。這支腳本直接讀正在寫入的 temp memmap
(`*.latents.dat`)，二分搜尋每個檔最後一個「已寫入(非零)」的 frame，算出真實 %。

用法:
    python3 castle_extract_progress.py <project_dir 或 latent_dir>
    # 例:
    python3 castle_extract_progress.py \
      ~/.ai-agent/workspace/repos/castle-ai/projects/2026-06-04-...-open-field
"""
import sys, glob, os
import numpy as np

def main(root):
    pats = [os.path.join(root, "latent", "**", "*.latents.dat"),
            os.path.join(root, "**", "*.latents.dat"),
            os.path.join(root, "*.latents.dat")]
    files = []
    for p in pats:
        files = sorted(glob.glob(p, recursive=True))
        if files:
            break
    if not files:
        print("找不到 *.latents.dat（extraction 沒在跑，或已完成 → 找 .npz）")
        return
    total_rows = total_done = 0
    for f in files:
        sz = os.path.getsize(f)
        # filesize = rows * dim * 4 (fp32)；猜 dim（DINO 特徵維，含 multiscale）
        dim = None
        for d in (16128, 9216, 6912, 5376, 4608, 2304, 1536, 1152, 768, 384):
            if sz % (d * 4) == 0:
                dim = d
                break
        if dim is None:
            print(f"  {os.path.basename(f)}: 無法反推 dim（size={sz}）")
            continue
        rows = sz // (dim * 4)
        mm = np.memmap(f, dtype=np.float32, mode="r", shape=(rows, dim))
        lo, hi, last = 0, rows - 1, -1
        while lo <= hi:                       # 二分搜尋最後一個非零 row
            mid = (lo + hi) // 2
            if np.any(mm[mid] != 0):
                last = mid; lo = mid + 1
            else:
                hi = mid - 1
        done = last + 1
        del mm
        total_rows += rows; total_done += done
        mt = os.path.getmtime(f)
        print(f"  {os.path.basename(f)}: {done}/{rows} ({100*done/rows:.1f}%)  dim={dim}  mtime={__import__('time').strftime('%H:%M:%S', __import__('time').localtime(mt))}")
    if total_rows:
        print(f"\n>>> 真實總進度: {total_done}/{total_rows} = {100*total_done/total_rows:.1f}%")
        print("    (再跑一次看數字有沒有增加 = 確認仍在前進)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(os.path.expanduser(sys.argv[1]))
