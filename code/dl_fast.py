"""Range-parallel public-GCS downloader (resumes the partial files of the previous downloader).

Each object is split into 32 MB byte ranges fetched concurrently with curl; finished ranges
are appended in order. Writes <root>/../../logs-style marker files when a prefix is complete.
usage: dl_fast.py BUCKET PREFIX ROOT MARKER [WORKERS]
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

PART = 32 << 20


def list_objects(bucket, prefix):
    items, tok = [], None
    while True:
        u = (f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix)}"
             f"&fields=items(name,size),nextPageToken&maxResults=1000")
        if tok:
            u += "&pageToken=" + tok
        for k in range(20):
            try:
                d = json.load(urllib.request.urlopen(u, timeout=60))
                break
            except Exception as e:  # noqa: BLE001
                print("list retry", k, e, flush=True)
                time.sleep(3)
        items += d.get("items", [])
        tok = d.get("nextPageToken")
        if not tok:
            return items


def fetch_part(url, dst, start, end):
    want = end - start + 1
    for _ in range(200):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if have == want:
            return True
        if have > want:
            os.remove(dst)
            have = 0
        tmp = dst + ".tmp"
        subprocess.run(["curl", "-sS", "-L", "--connect-timeout", "20", "--max-time", "240",
                        "-r", f"{start + have}-{end}", "-o", tmp, url], stderr=subprocess.DEVNULL)
        if os.path.exists(tmp):
            with open(dst, "ab") as f, open(tmp, "rb") as g:
                f.write(g.read())
            os.remove(tmp)
    return os.path.exists(dst) and os.path.getsize(dst) == want


def main():
    bucket, prefix, root, marker = sys.argv[1:5]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    items = [i for i in list_objects(bucket, prefix) if not i["name"].endswith("/")]
    total = sum(int(i["size"]) for i in items)
    print(f"{len(items)} objects, {total / 1e9:.2f} GB", flush=True)
    jobs = []  # (object index, part index, url, part path, start, end)
    plan = []
    for oi, it in enumerate(items):
        name, size = it["name"], int(it["size"])
        dst = os.path.join(root, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if have > size:
            os.remove(dst)
            have = 0
        parts = []
        url = f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(name)}"
        s = have
        while s < size:
            e = min(size, s + PART) - 1
            pp = f"{dst}.p{s:012d}"
            parts.append(pp)
            jobs.append((url, pp, s, e))
            s = e + 1
        plan.append((dst, size, parts))
    print(f"{len(jobs)} ranges to fetch", flush=True)
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_part, *j): j for j in jobs}
        for f in cf.as_completed(futs):
            done += 1
            if not f.result():
                print("FAILED range", futs[f][1], flush=True)
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} ranges, {(time.time() - t0) / 60:.1f} min", flush=True)
    ok = True
    for dst, size, parts in plan:
        exp = []
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        s = have
        for pp in parts:
            e = min(size, s + PART) - 1
            exp.append(e - s + 1)
            s = e + 1
        if not all(os.path.exists(pp) and os.path.getsize(pp) == n for pp, n in zip(parts, exp)):
            ok = False
            print("parts incomplete, kept for the next run:", dst, flush=True)
            continue
        with open(dst, "ab") as f:
            for pp in parts:
                with open(pp, "rb") as g:
                    f.write(g.read())
                os.remove(pp)
        if os.path.getsize(dst) != size:
            ok = False
            print("SIZE MISMATCH", dst, os.path.getsize(dst), size, flush=True)
    if ok:
        open(marker, "w").write(time.strftime("%F %T"))
    print("complete" if ok else "incomplete", f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
