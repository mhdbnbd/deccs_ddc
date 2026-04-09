#!/bin/bash
set -e

echo "=== DDECCS Sanity Check ==="
echo "Testing all 4 modes on AwA2 sample (2000 images)"
echo ""

for mode in kmeans deccs ddc ddeccs; do
    echo "--- AwA2 / $mode ---"
    python3 main_experiments.py --dataset awa2 --mode $mode --use_gpu --use_sample --sample_size 2000
    echo ""
done

echo "=== Sanity check complete ==="
echo "Check results in: results/awa2/"
echo ""

# Print summary
python3 -c "
import json, os
print(f'{\"Mode\":<10s} {\"NMI\":>8s} {\"ACC\":>8s} {\"TC\":>8s} {\"ITF\":>8s}')
print('-' * 44)
for mode in ['kmeans', 'deccs', 'ddc', 'ddeccs']:
    p = f'results/awa2/{mode}/summary.json'
    if os.path.exists(p):
        d = json.load(open(p))
        m = d['metrics']
        print(f'{mode:<10s} {m[\"nmi\"]:8.4f} {m[\"acc\"]:8.4f} {m.get(\"avg_tc\",0):8.4f} {m.get(\"avg_itf\",0):8.4f}')
"