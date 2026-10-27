"""Freeze the status plots and their audited inputs for this manual refresh."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
RUNS=BASE.parent
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=2))).strftime('%Y%m%dT%H%M')
out=RUNS/'2026-10-22-plot-refresh'/('snapshot-'+stamp)
out.mkdir(parents=True,exist_ok=False)
sources={
 'ablation-training.png':BASE/'ablation-training.png',
 'p2-controls.png':BASE/'p2-controls.png',
 'code-baselines.png':RUNS/'2026-10-11-code-apple-overlay/code-pareto-full.png',
 'ablation-training.svg':BASE/'ablation-training.svg',
 'p2-controls.svg':BASE/'p2-controls.svg',
 'code-baselines.svg':RUNS/'2026-10-11-code-apple-overlay/code-pareto-full.svg',
 'code-baseline-receipt.json':RUNS/'2026-10-11-code-apple-overlay/code-pareto-full.json',
 'code-overlay-receipt.json':RUNS/'2026-10-11-code-apple-overlay/styled-data.json',
 'shared-baseline-style.json':BASE.parents[2]/'configs/plots/baseline_comparison.json',
 'ablation-results-audited.json':BASE/'RESULTS_AUDITED.json',
 'ablation-plot-receipt.json':BASE/'plot-receipt.json',
 'code-p2-audited.json':RUNS/'2026-10-22-code16-cross-nfe/P2_AUDITED.json',
 'code-comparison-data.json':RUNS/'2026-10-11-code16-after-code8/code-comparison-data.json',
 'code32-eval-status.json':RUNS/'2026-10-22-code32-cross-nfe-i8008/RESULTS_AUDITED.json',
}
# Fixed-code16/code32 transfer is now shown on the main code baseline plot.
# Keep separate-figure files archival; do not emit them in default refreshes.
sources['code16-transfer-audited.json']=RUNS/'2026-10-22-code16-cross-nfe/TRANSFER_AUDITED.json'
manifest=dict(time=datetime.datetime.now().astimezone().isoformat(),remote_mutations=False,
              plotted_complete_seed_sets_only=True,calibration_labeled_training_only=True,files={})
for name,source in sources.items():
    shutil.copy2(source,out/name)
    manifest['files'][name]=dict(source=str(source),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out)
