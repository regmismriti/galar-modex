Implementation of galar modex

## Running on GALAR from the tar shards (Nautilus)

`main_galar_c.py --tar_index <index.pkl>` reads frames directly from the 8 GALAR tar
shards on `bpil-galar-pvc` using a prebuilt name -> (shard, offset, size) index,
instead of extracted PNGs spread across several PVCs. `--image_dir` then only serves
as a fallback for frames not in any shard. Every run logs, per video, how many frames
came from the tars, came from the fallback, or were dropped, and writes that as
`*_data_report.json`.

1. `kubectl apply -f nautilus/amit-galar-check-job.yaml -n usd-djha` (CPU only): runs
   `--check_data_only`, which checks coverage and loader throughput.
2. `kubectl apply -f nautilus/amit-train-galar-job.yaml -n usd-djha`: trains on a GPU
   with 24 GB or more. It checkpoints every epoch and resumes after eviction.

Both jobs clone branch `amit` and write only to `modex-galar-out:/amit/modex-galar/`.
