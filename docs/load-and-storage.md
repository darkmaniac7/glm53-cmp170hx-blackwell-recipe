# Load times and storage rules

The checkpoint is 288 files (282 safetensors shards), 418 GiB (448,740,929,607 bytes). Where it sits decides whether a
reload takes six minutes or more than an hour.

## Rules

1. **One local, verified copy per host.** Every node reads a copy on its own host. Verify each copy with sha256 of
   all 288 files against the source list before the first load, and again after any checkpoint change.
2. **NVMe, not SATA, for the CMP stages.** A cold load from a SATA SSD took about 38 minutes for the eight CMPs alone and
   77 minutes for the two-host layout reading it over NFS: the loader processes issue small reads and hold the disk
   at its queue limit. A sequential warm-up reader running beside the load made it worse, not better.
3. **Cap container reads from a consumer NVMe at 450 MB/s.** At full speed a consumer PCIe 5.0 drive (Crucial T700
   2 TB) went from 48 to 81 C in two and a half minutes during a load of this checkpoint, and once dropped off the bus. With
   the cap, four containers reading at once peaked at 42 C on the first load and 37-38 C on reloads. Log the
   composite temperature during loads. Raise the cap only as a guarded test, never on a serving engine.
4. **Never serve the weights to the other host over NFS from that NVMe.** The NFS server runs outside the
   container, so the read cap does not apply to it.
5. **Give the containers room for page cache.** The memory limit of a container counts page cache. With the
   recipe's 112 GB limit and the weights on the SATA drive, the eight loader workers fell out of step, evicted and
   re-read shards (25 s per shard); a 300 GB limit fixed that load. The multi-node template uses 200 GB for the
   Blackwell node and 120 GB per CMP node.

The cap, as a `docker run` flag (the multi-node template sets it from `HOST_READ_BPS_DEVICE`):

```
--device-read-bps /dev/nvme0n1:450mb
```

and as a compose overlay for the single-host recipe (append it to `COMPOSE_FILE`):

```yaml
services:
  vllm:
    blkio_config:
      device_read_bps:
        - path: /dev/nvme0n1
          rate: "450mb"
```

Use the block device that holds the weights. The cap is per container.

## Measured load times

Eight CMPs alone (cebeuq's TP2 x PP4, one container, 2026-11-04), launch to ready:

| Source | Time |
|---|---|
| SATA SSD, cold | about 38 min |
| NVMe copy, cold (page cache dropped), 450 MB/s cap | 22.4 min (shards 13 min 17 s at the cap) |
| NVMe copy, warm page cache | 8 to 10 min |

Two hosts, layout `pp5bw30` (five containers, 2026-11-04), every stage reading a local copy:

| Load | Blackwell stage (170 GB, 108 shards, 8 loader threads) | CMP stages 1-3 (70.6 GB each, capped) | CMP stage 4 (94.3 GB) | Launch to health |
|---|---|---|---|---|
| First of the night | 444-449 s | 331-346 s | 382 s | 638 s (10.6 min) |
| Reloads, caches warm | 216-316 s | 242-307 s | 283-311 s | 349-411 s (5.8 to 6.9 min) |

The same layout loading both hosts from the SATA copy over NFS took 77 minutes.

- The Blackwell stage is bound by the loader, not the disk: its drive reads a shard at 4.7 GB/s, yet 170 GB took
  7.4 min cold. That is with `--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}'`,
  which took the stage's load from 11.7 min to 108 s in its first test (2026-11-03); it affects load time only.
- The CMP stages ran at about 210 MB/s per container, under the cap, with part of the copy in page cache.
- A reload inside a session costs 16 s from stop to launch plus the load.

## Copying

A plain sequential copy at low I/O priority, then sha256 on both sides: 418 GiB took 48 minutes from the SATA
source (155 MB/s, the source's sequential speed), hashing the NVMe copy took 95 s in parallel and the SATA source
47 minutes sequentially. Leave free space on the target: the copy is as large as the checkpoint.
