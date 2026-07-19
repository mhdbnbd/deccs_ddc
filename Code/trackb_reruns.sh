#!/usr/bin/env bash
#
# trackb_reruns.sh — Track B verification reruns for the DDECCS thesis.
#
# Runs three stages, cheapest first:
#   Stage 0  CONTROL    kmeans, raw features        -> results_v2_control  (~20 min)
#   Stage 1  ABLATION   kmeans, --standardize       -> results_v2_std      (~20 min)
#   Stage 2  SEED RERUN ddeccs, seeded ensemble     -> results_v2          (~4.5 h)
#
# Stage 0 exists to prove the code edits did not change the shipped baseline:
# it must reproduce NMI 0.869 / 0.534 / 0.666 exactly. If it does not, stop and
# fix the edits before trusting stages 1 and 2.
#
# deccs is not run separately: ddeccs clustering is identical to deccs by
# construction (the ILP is post-hoc), so the deccs rows are the ddeccs rows
# minus TC/ITF. Running only ddeccs halves ~9 h to ~4.5 h.
#
# USAGE (from the Code/ directory):
#   chmod +x trackb_reruns.sh
#   ./trackb_reruns.sh                  # all stages
#   ./trackb_reruns.sh 0 1              # only stages 0 and 1
#   FORCE=1 ./trackb_reruns.sh          # redo runs that already have summary.json
#   RUNS=3 ./trackb_reruns.sh 0         # quick smoke test with 3 seeds
#
# Detached (recommended for stage 2):
#   nohup ./trackb_reruns.sh > trackb.out 2>&1 &
#   tail -f trackb.out
#
# Safety: refuses to write under results/, verifies edits A-G are applied
# before running, and checksums results/ before and after to prove it is
# untouched. Resumable — completed runs are skipped unless FORCE=1.

set -uo pipefail

RUNS="${RUNS:-10}"
FORCE="${FORCE:-0}"
GPU_FLAG="${GPU_FLAG:---use_gpu}"
OUT_CTRL="${OUT_CTRL:-results_v2_control}"
OUT_STD="${OUT_STD:-results_v2_std}"
OUT_SEED="${OUT_SEED:-results_v2}"
LOGDIR="${LOGDIR:-logs_v2}"

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(0 1 2)

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s\n' "${GRN}OK${RST}   $*"; }
warn() { printf '%s\n' "${YEL}WARN${RST} $*"; }
die()  { printf '%s\n' "${RED}FAIL${RST} $*"; exit 1; }
hdr()  { printf '\n%s\n' "${BLD}=== $* ===${RST}"; }

# ---------------------------------------------------------------- preflight

hdr "Preflight"

[ -f main_experiments.py ] && [ -f utils.py ] \
  || die "run this from the Code/ directory (main_experiments.py not found)"

for root in "$OUT_CTRL" "$OUT_STD" "$OUT_SEED"; do
  case "$root" in
    results|results/|./results|./results/)
      die "output root '$root' would overwrite the shipped results. Refusing." ;;
  esac
done
ok "output roots safe: $OUT_CTRL, $OUT_STD, $OUT_SEED"

command -v python3 >/dev/null || die "python3 not found"

# --- verify the code edits are applied -------------------------------------
COMPILE="$(python3 -m py_compile main_experiments.py utils.py 2>&1)" \
  || die "syntax error after editing:
$COMPILE"
ok "main_experiments.py and utils.py compile"

grep -Eq 'add_argument\( *"--output_root"' main_experiments.py \
  || die "edit A missing: --output_root not in argparse"
grep -Eq 'add_argument\( *"--standardize"' main_experiments.py \
  || die "edit A missing: --standardize not in argparse"
ok "edit A  argparse flags present"

grep -Eq 'output_dir *= *os\.path\.join\( *args\.output_root' main_experiments.py \
  || die "edit B missing: output_dir still hardcodes results/"
ok "edit B  output_dir uses args.output_root"

grep -Eq 'standardize *= *False' main_experiments.py \
  || die "edit C missing: run_single has no standardize parameter"
grep -q 'StandardScaler' main_experiments.py \
  || die "edit C missing: StandardScaler not used in main_experiments.py"
ok "edit C  run_single standardize branch present"

grep -Eq 'get_base_clusterings\( *feats, *n_clusters *= *K, *seed *= *seed *\)' main_experiments.py \
  || die "edit D missing: seed not propagated to get_base_clusterings.
       Without this the ensemble is identical across all $RUNS seeds and
       stage 2 would just reproduce the existing +/-0.000."
ok "edit D  seed propagated to get_base_clusterings"

grep -Eq 'standardize *= *args\.standardize' main_experiments.py \
  || die "edit E missing: run_single call does not pass standardize"
ok "edit E  run_single call passes standardize"

grep -Eq 'def get_base_clusterings\( *embeddings_np, *n_clusters *= *10, *seed *= *42 *\)' utils.py \
  || die "edit F missing: get_base_clusterings has no seed parameter"
ok "edit F  get_base_clusterings accepts seed"

LEFTOVER="$(awk '/^def get_base_clusterings/{f=1} f&&/random_state *= *42/{print NR": "$0} f&&/^def /&&!/get_base_clusterings/{f=0}' utils.py)"
[ -z "$LEFTOVER" ] || die "edit G incomplete: hardcoded random_state=42 remains inside
       get_base_clusterings. Replace with random_state=seed at:
$LEFTOVER"
NSEED="$(awk '/^def get_base_clusterings/{f=1} f&&/random_state *= *seed/{c++} f&&/^def /&&!/get_base_clusterings/{f=0} END{print c+0}' utils.py)"
[ "$NSEED" -ge 4 ] || die "edit G incomplete: expected >=4 'random_state=seed' inside
       get_base_clusterings (PCA, KMeans, Spectral, GMM), found $NSEED"
ok "edit G  $NSEED seeded estimators inside get_base_clusterings"

# --- environment smoke check (cheap; a 4.5 h job should not die on an import)
HELP="$(python3 main_experiments.py --help 2>&1)" \
  || die "the pipeline cannot start in this environment:
$(printf '%s\n' "$HELP" | tail -n 5 | sed 's/^/       /')
       Activate the right venv, or install the missing package, then rerun."
ok "environment OK (pipeline imports and parses arguments)"

# --- snapshot results/ ------------------------------------------------------
snapshot() {
  python3 - "$1" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
if not os.path.isdir(root):
    print("ABSENT"); raise SystemExit
h = hashlib.sha256()
for dp, dn, fn in sorted(os.walk(root)):
    dn.sort()
    for f in sorted(fn):
        p = os.path.join(dp, f)
        h.update(p.encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
print(h.hexdigest())
PY
}
RESULTS_BEFORE="$(snapshot results)"
ok "results/ checksum recorded (${RESULTS_BEFORE:0:16}...)"

mkdir -p "$LOGDIR"
say ""
say "runs per config : $RUNS"
say "force rerun     : $FORCE"
say "stages          : ${STAGES[*]}"

# ------------------------------------------------------------------ runner

FAILED=()
COMPLETED=0
SKIPPED=0

# run <label> <out_root> <ds_name> <mode> <expected_seconds> <extra args...>
run() {
  local label="$1" root="$2" dsname="$3" mode="$4" expect="$5"; shift 5
  local outdir="$root/$dsname/$mode"
  local log="$LOGDIR/${root}_${dsname}_${mode}.log"

  if [ "$FORCE" != "1" ] && [ -f "$outdir/summary.json" ]; then
    warn "skip $label (summary.json exists; FORCE=1 to redo)"
    SKIPPED=$((SKIPPED+1)); return 0
  fi

  say ""
  say "${BLD}-> $label${RST}   expect ~$((expect/60)) min   log: $log"
  local t0 t1 dt
  t0=$(date +%s)
  if python3 main_experiments.py --mode "$mode" --n_runs "$RUNS" \
        --output_root "$root" $GPU_FLAG "$@" > "$log" 2>&1; then
    t1=$(date +%s); dt=$((t1-t0))
    if [ -f "$outdir/summary.json" ]; then
      local nmi
      nmi=$(python3 -c "import json;d=json.load(open('$outdir/summary.json'))['metrics'];print(f\"NMI {d.get('nmi'):.4f} +/- {d.get('nmi_std','n/a')}\")" 2>/dev/null)
      ok "$label done in $((dt/60))m$((dt%60))s   $nmi"
    else
      ok "$label done in $((dt/60))m$((dt%60))s   (no summary.json written)"
    fi
    COMPLETED=$((COMPLETED+1))
  else
    t1=$(date +%s); dt=$((t1-t0))
    printf '%s\n' "${RED}FAIL${RST} $label after $((dt/60))m$((dt%60))s — last lines:"
    tail -n 15 "$log" | sed 's/^/       /'
    FAILED+=("$label")
  fi
}

# ------------------------------------------------------------------- stages

want() { for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

if want 0; then
  hdr "Stage 0 — CONTROL: kmeans on raw features (~20 min)"
  say "Must reproduce the shipped baseline exactly: 0.6660 / 0.5340 / 0.8690."
  run "control apy_15 kmeans" "$OUT_CTRL" apy_15 kmeans 115 --dataset apy --apy_15
  run "control apy    kmeans" "$OUT_CTRL" apy    kmeans 250 --dataset apy
  run "control awa2   kmeans" "$OUT_CTRL" awa2   kmeans 835 --dataset awa2
fi

if want 1; then
  hdr "Stage 1 — ABLATION: kmeans on standardized features (~20 min)"
  say "Tests whether standardization alone closes the aPY +14.4% consensus gain."
  run "std apy_15 kmeans" "$OUT_STD" apy_15 kmeans 115 --dataset apy --apy_15 --standardize
  run "std apy    kmeans" "$OUT_STD" apy    kmeans 250 --dataset apy --standardize
  run "std awa2   kmeans" "$OUT_STD" awa2   kmeans 835 --dataset awa2 --standardize
fi

if want 2; then
  hdr "Stage 2 — SEED RERUN: ddeccs with a seeded ensemble (~4.5 h)"
  say "Tests whether the near-zero variance claim survives real seed propagation."
  run "seeded apy_15 ddeccs" "$OUT_SEED" apy_15 ddeccs 495   --dataset apy --apy_15
  run "seeded apy    ddeccs" "$OUT_SEED" apy    ddeccs 2215  --dataset apy
  run "seeded awa2   ddeccs" "$OUT_SEED" awa2   ddeccs 13465 --dataset awa2
fi

# ---------------------------------------------------------------- postflight

hdr "Postflight"

RESULTS_AFTER="$(snapshot results)"
if [ "$RESULTS_BEFORE" = "$RESULTS_AFTER" ]; then
  ok "results/ unchanged — shipped results intact"
else
  printf '%s\n' "${RED}FAIL${RST} results/ CHANGED during this run. Restore it from git:"
  say "       git checkout -- Code/results"
fi

hdr "Summary"

python3 - "$OUT_CTRL" "$OUT_STD" "$OUT_SEED" <<'PY'
import json, os, sys
ctrl, std, seed = sys.argv[1:4]

# Shipped 10-run means (thesis ground truth)
shipped = {
    "awa2":   {"kmeans": 0.869, "deccs": 0.871, "std_kmeans": 0.002},
    "apy":    {"kmeans": 0.534, "deccs": 0.611, "std_kmeans": 0.014},
    "apy_15": {"kmeans": 0.666, "deccs": 0.677, "std_kmeans": 0.028},
}
label = {"awa2": "AwA2 K=50", "apy": "aPY-32", "apy_15": "aPY-15"}

def load(root, ds, mode):
    p = os.path.join(root, ds, mode, "summary.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p)).get("metrics", {})
    return m.get("nmi"), m.get("nmi_std")

print(f"\n{'dataset':<12}{'shipped':>10}{'control':>18}{'standardized':>18}")
print("-" * 58)
for ds in ("apy_15", "apy", "awa2"):
    s = shipped[ds]["kmeans"]
    c = load(ctrl, ds, "kmeans")
    z = load(std, ds, "kmeans")
    cs = "not run" if c is None else f"{c[0]:.4f}" + (" MATCH" if abs(c[0]-s) < 5e-4 else " DIFFERS")
    zs = "not run" if z is None else f"{z[0]:.4f} ({z[0]-s:+.4f})"
    print(f"{label[ds]:<12}{s:>10.4f}{cs:>18}{zs:>18}")

print("\nControl must read MATCH. If it differs, an edit changed the baseline —")
print("stop and diff before using stages 1 or 2.\n")

print(f"{'dataset':<12}{'shipped deccs':>16}{'seeded ddeccs':>26}")
print("-" * 54)
any_seed = False
for ds in ("apy_15", "apy", "awa2"):
    s = shipped[ds]["deccs"]
    r = load(seed, ds, "ddeccs")
    if r is None:
        print(f"{label[ds]:<12}{s:>16.4f}{'not run':>26}")
        continue
    any_seed = True
    nmi, sd = r
    sd = 0.0 if sd is None else sd
    note = "variance still ~0" if sd < 5e-4 else "VARIANCE APPEARED"
    print(f"{label[ds]:<12}{s:>16.4f}   {nmi:.4f} +/- {sd:.4f}  {note}")

if any_seed:
    print("\nIf variance stays ~0 the stability claim is now earned and the text stands.")
    print("If variance appeared, six passages need rewording:")
    print("  discussion.tex:14, 16, 54   experiments.tex:129, 153, 178")
    print("  conclusion.tex:9   and the abstract in main.tex")
print()
PY

say "logs: $LOGDIR/"
say "completed: $COMPLETED   skipped: $SKIPPED   failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  for f in "${FAILED[@]}"; do say "  ${RED}failed${RST}: $f"; done
  exit 1
fi

hdr "Next"
say "Send me these and I will tell you what changes in the thesis:"
say "  $OUT_CTRL/*/kmeans/summary.json"
say "  $OUT_STD/*/kmeans/summary.json"
say "  $OUT_SEED/*/ddeccs/summary.json"