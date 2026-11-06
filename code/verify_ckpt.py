"""Verify a local openpi checkpoint against the public GCS listing (file set, sizes, CRC32C)."""
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request

bucket, prefix, local, marker = sys.argv[1:5]
u = (f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix)}"
     f"&fields=items(name,size,crc32c)&maxResults=1000")
for k in range(20):
    try:
        items = json.load(urllib.request.urlopen(u, timeout=60))["items"]
        break
    except Exception as e:  # noqa: BLE001
        print("list retry", k, e, flush=True)
        time.sleep(3)
try:
    import google_crc32c
except ImportError:
    google_crc32c = None
    print("google_crc32c not available: checking names and sizes only", flush=True)
ok = True
for it in items:
    rel = it["name"][len(prefix):]
    p = os.path.join(local, rel)
    size = int(it["size"])
    if not os.path.exists(p) or os.path.getsize(p) != size:
        print("MISSING/SIZE", rel, os.path.getsize(p) if os.path.exists(p) else None, size, flush=True)
        ok = False
        continue
    if google_crc32c is not None:
        c = google_crc32c.Checksum()
        with open(p, "rb") as f:
            while True:
                b = f.read(64 << 20)
                if not b:
                    break
                c.update(b)
        got = base64.b64encode(c.digest()).decode()
        if got != it["crc32c"]:
            print("CRC MISMATCH", rel, got, it["crc32c"], flush=True)
            ok = False
            continue
    print("ok", rel, size, it.get("crc32c"), flush=True)
print("VERIFIED" if ok else "NOT VERIFIED", flush=True)
if ok:
    open(marker, "w").write(time.strftime("%F %T"))
sys.exit(0 if ok else 1)
