#!/usr/bin/env bash
# phase0-sweep.sh — Phase 0 measurement harness for FlashNext-Q2 v2.
#
# Hard rule: this script NEVER modifies flashnext-quat.sh. It calls the frozen
# v0.11.4 baseline as a black box to start/stop servers, and orchestrates an
# end-to-end sweep whose only purpose is to separate kernel cost from MTP/QSA/
# scheduler costs before any v2 kernel work begins.
#
# Actions: check | sweep-short | sweep-long | full | report

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${Q0_SCRIPT:=${HOME}/flashnext-quat.sh}"        # frozen v0.11.4 entry point
: "${PORT:=8012}"
: "${BASE_URL:=http://127.0.0.1:${PORT}}"
: "${RESULTS_DIR:=$HERE/results}"
: "${RUNTIME_IMAGE:=vllm/vllm-openai:qwen38-flash-next}"
: "${SIDECAR_DIR:=}"
: "${SIDECAR_LAYER0:=${SIDECAR_DIR:+$SIDECAR_DIR/layer-00.safetensors}}"
: "${MODEL:=}"
: "${MAX_TOKENS:=512}"
: "${TEMPERATURE:=0.5}"
: "${REQ_TIMEOUT:=5400}"
: "${READY_TIMEOUT:=1500}"
: "${INTER_DELAY:=10}"
: "${NCU_TEST:=1}"
: "${PROFILE_RECHECK:=0}"
: "${MTP_KV_CACHE_MEMORY_BYTES:=20G}"              # applied to every cell (config parity)
: "${MTP_BATCHED_TOKENS:=8192}"
: "${MTP_MAX_NUM_SEQS:=8}"
: "${MTP_Q2_DECODE_W13_BLOCK_N:=16}"
: "${MTP_Q2_DECODE_W2_BLOCK_N:=16}"
: "${MTP_Q2_EXPERT_MAJOR_MIN_M:=64}"
: "${MEASURED_REPEATS:=3}"
: "${WARMUP_REPEATS:=1}"
: "${LONG_REPEATS:=1}"
: "${LONG_SECOND_MARGIN:=0.97}"
: "${NO_THINKING:=1}"
: "${RUN_SESSION_ID:=$(date -u +%Y%m%dT%H%M%SZ)-$$}"

SHORT_KS=( ${SHORT_KS_LIST:-0 1 2 3} )
SHORT_STREAMS=( ${SHORT_STREAMS_LIST:-1 2 4 8} )
SHORT_WORKLOADS=( ${SHORT_WORKLOADS_LIST:-prose code} )
LONG_CONTEXTS=( ${LONG_CONTEXTS_LIST:-64 4096 32768 131072 250000} )

SERVER_ACTIVE=0
MEM_SAMPLER_PID=""
SERVER_LAUNCH_PID=""
HAVE_METRICS=0
BASELINE_SHA=""
SIDECAR_SHA=""
RUNTIME_IMAGE_ID=""

log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
warn() { printf '[%s] WARN: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] FATAL: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }
sha256_of() { [[ -n "${1:-}" && -f "$1" ]] && sha256sum "$1" | awk '{print $1}' || echo ""; }

# ---------------------------------------------------------------- server ----

wait_ready() {
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if curl -s --max-time 5 "${BASE_URL}/v1/models" | jq -e '.data[0].id' >/dev/null 2>&1; then
      if [[ -z "$MODEL" ]]; then MODEL=$(curl -s --max-time 5 "${BASE_URL}/v1/models" | jq -r '.data[0].id'); fi
      curl -s --max-time 5 "${BASE_URL}/metrics" > /dev/null 2>&1 && HAVE_METRICS=1 || HAVE_METRICS=0
      return 0
    fi
    sleep 5
  done
  return 1
}

wait_down() {
  local deadline=$(( $(date +%s) + 60 ))
  while (( $(date +%s) < deadline )); do
    if ! curl -fsS --max-time 2 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

server_start() {
  # Starts the frozen v0.11.4 baseline for a given MTP k. The command arrays
  # below are the ONLY interface assumptions; override any of them via the
  # SERVER_CMD_* env vars (plain strings, eval'd) if the frozen interface
  # differs.
  local k=$1
  local cmd
  if [[ -n "${SERVER_CMD_K0:-}" && "$k" == "0" ]]; then
    cmd="${SERVER_CMD_K0}"
  elif [[ -n "${SERVER_CMD_K3:-}" && "$k" == "3" ]]; then
    cmd="${SERVER_CMD_K3}"
  elif [[ "$k" != "0" && "$k" != "3" && -n "${SERVER_CMD_KN:-}" ]]; then
    cmd="${SERVER_CMD_KN//__K__/$k}"
  else
    local mtp_enable=0 mtp_tokens=1
    if (( k > 0 )); then mtp_enable=1; mtp_tokens=$k; fi
    # Every k uses the same runtime policy. Comparing k=0 eager with k=3
    # piecewise graphs would measure configuration drift, not MTP cost.
    CMD=(env PORT="$PORT"
         MTP_ENABLE="$mtp_enable" MTP_NUM_SPECULATIVE_TOKENS="$mtp_tokens"
         MTP_REJECTION_SAMPLE_METHOD=block MTP_DRAFT_SAMPLE_METHOD=probabilistic
         RUNTIME_EXECUTION_MODE=piecewise
         MAX_NUM_BATCHED_TOKENS="$MTP_BATCHED_TOKENS" MAX_NUM_SEQS="$MTP_MAX_NUM_SEQS"
         KV_CACHE_MEMORY_BYTES="$MTP_KV_CACHE_MEMORY_BYTES"
         Q2_DECODE_W13_BLOCK_N="$MTP_Q2_DECODE_W13_BLOCK_N"
         Q2_DECODE_W2_BLOCK_N="$MTP_Q2_DECODE_W2_BLOCK_N"
         Q2_PREFILL_V8_MIN_M="$MTP_Q2_EXPERT_MAJOR_MIN_M"
         QSA_DET_TOPK_ENABLE=0 QSA_DET_TOPK_STRICT=0
         STARTUP_CONCURRENCY_PROBE=0
         "$Q0_SCRIPT" serve)
  fi
  if curl -fsS --max-time 2 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    warn "an existing server is listening at $BASE_URL; stopping it before k=$k"
    "$Q0_SCRIPT" stop >>"$RESULTS_DIR/server_stop.log" 2>&1 || true
    wait_down || die "endpoint remained live after baseline stop; refusing to benchmark a stale configuration"
  fi
  log "starting frozen baseline (k=$k)"
  if [[ -n "${cmd:-}" ]]; then
    eval "$cmd" >>"$RESULTS_DIR/server_k${k}.log" 2>&1 &
  else
    "${CMD[@]}" >>"$RESULTS_DIR/server_k${k}.log" 2>&1 &
  fi
  SERVER_LAUNCH_PID=$!
  SERVER_ACTIVE=1
  if ! wait_ready; then
    server_stop
    die "server k=$k did not become ready in ${READY_TIMEOUT}s (see $RESULTS_DIR/server_k${k}.log)"
  fi
  probe_counters "$k"
}

server_stop() {
  (( SERVER_ACTIVE )) || return 0
  log "stopping server"
  "$Q0_SCRIPT" stop >>"$RESULTS_DIR/server_stop.log" 2>&1 || warn "baseline stop returned nonzero (continuing)"
  wait_down || warn "endpoint still responds after stop"
  if [[ -n "$SERVER_LAUNCH_PID" ]]; then
    wait "$SERVER_LAUNCH_PID" 2>/dev/null || true
    SERVER_LAUNCH_PID=""
  fi
  SERVER_ACTIVE=0
  sleep "$INTER_DELAY"
}

cleanup() {
  local rc=$?
  [[ -n "$MEM_SAMPLER_PID" ]] && kill "$MEM_SAMPLER_PID" 2>/dev/null || true
  server_stop || true
  if (( rc != 0 )); then log "aborted (rc=$rc)"; fi
}

# ------------------------------------------------------------- counters ----

metrics_snapshot() {
  local tag=$1 dir=$2
  if [[ "${HAVE_METRICS:-0}" == "1" ]] && curl -s --max-time 10 "${BASE_URL}/metrics" > "$dir/metrics_${tag}.txt" 2>/dev/null && [[ -s "$dir/metrics_${tag}.txt" ]]; then
    return 0
  fi
  : > "$dir/metrics_${tag}.txt"
  return 0
}

probe_counters() {
  local k=$1 f="$RESULTS_DIR/counters_k${k}.json"
  if curl -s --max-time 10 "${BASE_URL}/metrics" > /tmp/phase0_counters_raw.$$ 2>/dev/null && [[ -s /tmp/phase0_counters_raw.$$ ]]; then
    python3 - "$k" /tmp/phase0_counters_raw.$$ "$f" <<'PY'
import json, re, sys
k, path, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
names = set(); pat = re.compile(r'^([a-zA-Z_:][\w:]*)[\{ ]')
with open(path, errors="replace") as fh:
    for line in fh:
        m = pat.match(line)
        if m: names.add(m.group(1))
json.dump({"k": k,
           "spec_decode_metrics_present": sorted(n for n in names if "spec_decode" in n),
           "total_metrics": len(names)}, open(out, "w"), indent=2)
PY
    rm -f /tmp/phase0_counters_raw.$$
    if (( k > 0 )); then
      local n; n=$(python3 -c "import json;print(len(json.load(open('$f'))['spec_decode_metrics_present']))")
      if [[ "$n" == "0" ]]; then warn "server k=$k exposes NO spec_decode counters; acceptance columns will be zero/null"; fi
    fi
  fi
}

# ------------------------------------------------------------ benchmark ----

run_streams() {
  python3 "$HERE/lib/bench_stream.py" \
    --base-url "$BASE_URL" --model "$MODEL" \
    --prompt-file "$2" --streams "$1" \
    --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" \
    --timeout "$REQ_TIMEOUT" --no-thinking "$NO_THINKING" --out "$3"
}

mem_sampler_start() {
  local f=$1
  ( while :; do
      gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      host=$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null)
      printf '%s %s %s\n' "$(date +%s)" "${gpu:-0}" "${host:-0}" >> "$f"
      sleep 1
    done ) &
  MEM_SAMPLER_PID=$!
}
mem_sampler_stop() {
  [[ -n "$MEM_SAMPLER_PID" ]] && { kill "$MEM_SAMPLER_PID" 2>/dev/null || true; wait "$MEM_SAMPLER_PID" 2>/dev/null || true; MEM_SAMPLER_PID=""; }
  return 0
}

# ------------------------------------------------------------------ run ----

run_cell() {
  local phase=$1 k=$2 streams=$3 workload=$4 ctx=$5 rep=$6 warmup=$7 prompt=$8
  local run_id="${phase}_k${k}_s${streams}_${workload}_ctx${ctx}_r${rep}_$(date +%s%N)"
  local dir="$RESULTS_DIR/runs/$run_id"
  mkdir -p "$dir"
  local form_m
  if (( k > 0 )); then form_m=$(( streams * (k + 1) )); else form_m=$streams; fi

  local is_warmup=false
  if (( warmup == 1 )); then is_warmup=true; fi
  cat > "$dir/config.json" <<JSON
{
  "run_id": "$run_id", "session_id": "$RUN_SESSION_ID",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "phase": "$phase",
  "warmup": $is_warmup, "repeat": $rep, "workload": "$workload", "k": $k, "streams": $streams,
  "form_m_estimate": $form_m, "context_tokens": $ctx,
  "prompt_sha256": "$(sha256_of "$prompt")"
}
JSON

  log "run $run_id"
  metrics_snapshot before "$dir"
  mem_sampler_start "$dir/mem_samples.txt"
  local rc=0
  run_streams "$streams" "$prompt" "$dir/streams.json" || rc=$?
  mem_sampler_stop
  metrics_snapshot after "$dir"
  python3 "$HERE/lib/metrics_delta.py" "$dir/metrics_before.txt" "$dir/metrics_after.txt" "$dir/metrics_delta.json" \
    || echo '{"counters_delta":{},"histogram_bucket_deltas":{},"counter_series_deltas":{}}' > "$dir/metrics_delta.json"

  python3 - "$dir" "$RESULTS_DIR/results.jsonl" "$BASELINE_SHA" "$SIDECAR_SHA" "$RUNTIME_IMAGE_ID" <<'PY'
import json, os, re, sys
dir_, results_path, script_sha, sidecar_sha, image_digest = sys.argv[1:6]
cfg = json.load(open(os.path.join(dir_, "config.json")))
res = json.load(open(os.path.join(dir_, "streams.json")))
md  = json.load(open(os.path.join(dir_, "metrics_delta.json")))
gpu, host_min = None, None
mp = os.path.join(dir_, "mem_samples.txt")
if os.path.exists(mp):
    for line in open(mp):
        p = line.split()
        if len(p) >= 3:
            # GB10 uses unified memory and nvidia-smi may report [N/A] for
            # memory.used. Keep the run valid and represent unavailable GPU
            # memory as null instead of aborting the entire sweep.
            try:
                g = float(p[1]) / 1024.0
                gpu = g if gpu is None else max(gpu, g)
            except (TypeError, ValueError):
                pass
            try:
                h = float(p[2]) / 1048576.0
                host_min = h if host_min is None else min(host_min, h)
            except (TypeError, ValueError):
                pass
cd = md.get("counters_delta", {})
def g(n): return int(cd.get(n, 0) or 0)
per_pos_map = {}
for name, series in md.get("counter_series_deltas", {}).items():
    if "accepted" not in name or "per_pos" not in name:
        continue
    for labels, value in series.items():
        m = re.search(r'(?:position|pos)="?(\d+)"?', labels)
        if m:
            position = int(m.group(1))
            per_pos_map[position] = per_pos_map.get(position, 0) + value
per_pos = [per_pos_map[i] for i in sorted(per_pos_map)]
rec = {
  "run_id": cfg["run_id"], "session_id": cfg["session_id"], "ts": cfg["ts"],
  "phase": cfg["phase"], "warmup": cfg["warmup"],
  "repeat": cfg["repeat"], "workload": cfg["workload"], "k": cfg["k"], "streams": cfg["streams"],
  "form_m_estimate": cfg["form_m_estimate"], "context_tokens": cfg["context_tokens"],
  "prompt_sha256": cfg["prompt_sha256"], "script_sha256": script_sha,
  "sidecar_layer0_sha256": sidecar_sha, "runtime_image_digest": image_digest,
  "output_tokens": res.get("output_tokens", 0), "decode_tokens": res.get("decode_tokens", 0),
  "decode_window_s": res.get("decode_window_s"), "aggregate_tps": res.get("aggregate_tps", 0.0),
  "request_total_tps": res.get("request_total_tps", 0.0),
  "per_stream_tps": res.get("per_stream_tps_mean", 0.0),
  "ttft": res.get("ttft_mean"), "ttft_s_mean": res.get("ttft_mean"), "ttft_s_p95": res.get("ttft_p95"),
  "prompt_tokens_actual": res.get("prompt_tokens"), "wall_time_s": res.get("wall_time_s"),
  "draft_cycles": g("vllm:spec_decode_num_drafts_total"),
  "draft_tokens":  g("vllm:spec_decode_num_draft_tokens_total"),
  "accepted_tokens": g("vllm:spec_decode_num_accepted_tokens_total"),
  "accepted_by_position": per_pos,
  "memory_peak_gib": round(gpu, 3) if gpu is not None else None,
  "mem_available_min_gib": round(host_min, 3) if host_min is not None else None,
  "streams_ok": res.get("ok_streams", 0), "streams_err": res.get("err_streams", 0),
  "valid": res.get("ok_streams", 0) == cfg["streams"] and res.get("err_streams", 0) == 0,
  "errors": res.get("errors", [])[:3],
}
json.dump(rec, open(os.path.join(dir_, "run.json"), "w"), indent=2)
with open(results_path, "a") as out:
    out.write(json.dumps(rec) + "\n")
print(f"{cfg['run_id']}: agg_tps={rec['aggregate_tps']:.2f} per_stream={rec['per_stream_tps']:.2f} "
      f"ttft={rec['ttft_s_mean']} drafts={rec['draft_cycles']} accepted={rec['accepted_tokens']} peak_gib={rec['memory_peak_gib']}")
PY
  if (( rc != 0 )); then warn "run $run_id finished with stream errors (rc=$rc)"; fi
  sleep "$INTER_DELAY"
}

# -------------------------------------------------------------- phases ----

ensure_workload_prompt() {
  local name=$1 path=$2
  mkdir -p "$HERE/prompts"
  [[ -s "$path" ]] && return 0
  case "$name" in
    prose)
      cat > "$path" <<'TXT'
You are an expert technical writer. Write an original, well-structured essay of at least 900 words about the engineering trade-offs of running large Mixture-of-Experts language models on a single unified-memory workstation, covering expert routing, weight streaming, KV-cache pressure, speculative decoding, and what changes when the disk replaces unused experts. Avoid lists; write in flowing paragraphs with a clear thesis.
TXT
      ;;
    code)
      cat > "$path" <<'TXT'
You are a senior Python engineer. Write a complete, self-contained Python module (~900 words equivalent including comments and docstrings) that implements a fixed-capacity GPU work-stealing task scheduler for grouped MoE inference, with a persistent CUDA-grid friendly interface, unit tests at the end, and careful attention to avoiding device-to-host synchronization inside the hot path.
TXT
      ;;
    *) die "unknown workload: $name" ;;
  esac
}

ensure_long_prompt() {
  local n=$1 path="$HERE/prompts/ctx$1.txt"
  [[ -s "$path" ]] && { echo "$path"; return 0; }
  mkdir -p "$HERE/prompts"
  python3 - "$n" "$path" <<'PY'
import sys
target = max(int(sys.argv[1]) - 640, 32)
words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega "
         * ((target // 10) // 27 + 1)).split()
text = " ".join(words[: target * 13 // 20 + 8])
open(sys.argv[2], "w").write(
  "The following is a long synthetic reference document assembled for context-retrieval benchmarking. "
  "Reference document begins: " + text +
  " Reference document ends. Question: According to the reference document above, what is the final marker word? Answer concisely. "
  "Marker word: ZQX-7714-DELTA.")
PY
  echo "$path"
}

sweep_short() {
  require_fingerprint
  printf '%s\n' "$RUN_SESSION_ID" > "$RESULTS_DIR/current-session.txt"
  log "=== Phase 0 short matrix: k=${SHORT_KS[*]} streams=${SHORT_STREAMS[*]} workloads=${SHORT_WORKLOADS[*]} ==="
  local k streams workload
  for k in "${SHORT_KS[@]}"; do
    server_start "$k"
    for streams in "${SHORT_STREAMS[@]}"; do
      for workload in "${SHORT_WORKLOADS[@]}"; do
        local prompt="$HERE/prompts/${workload}.txt"
        ensure_workload_prompt "$workload" "$prompt"
        for i in $(seq 1 $(( WARMUP_REPEATS + MEASURED_REPEATS ))); do
          local warmup=0 rep=$(( i - WARMUP_REPEATS ))
          if (( i <= WARMUP_REPEATS )); then warmup=1; rep=$i; fi
          run_cell short "$k" "$streams" "$workload" 0 "$rep" "$warmup" "$prompt"
        done
      done
    done
    server_stop
  done
  report || true
}

best_ks_from_results() {
  python3 - "$RESULTS_DIR/results.jsonl" "$LONG_SECOND_MARGIN" "$RUN_SESSION_ID" <<'PY'
import json, sys
path, margin, session_id = sys.argv[1], float(sys.argv[2]), sys.argv[3]
acc = {}
try:
    for line in open(path):
        r = json.loads(line)
        if (r.get("session_id") == session_id and r.get("phase") == "short"
                and not r.get("warmup") and r.get("valid", True)):
            acc.setdefault(r["k"], []).append(r["aggregate_tps"])
except FileNotFoundError:
    pass
if not acc:
    print("0 3"); sys.exit(0)
means = sorted(((sum(v) / len(v), k) for k, v in acc.items()), reverse=True)
out = [means[0][1]]
if len(means) > 1 and means[0][0] > 0 and means[1][0] >= margin * means[0][0]:
    out.append(means[1][1])
if 0 not in out:
    out.insert(0, 0)
print(" ".join(str(k) for k in out))
PY
}

sweep_long() {
  require_fingerprint
  local ks; ks=$(best_ks_from_results)
  log "=== Phase 0 long-context matrix; k set = {$ks} ==="
  local k ctx i
  for k in $ks; do
    server_start "$k"
    for ctx in "${LONG_CONTEXTS[@]}"; do
      local prompt; prompt=$(ensure_long_prompt "$ctx")
      for i in $(seq 1 $(( WARMUP_REPEATS + LONG_REPEATS ))); do
        local warmup=0 rep=$(( i - WARMUP_REPEATS ))
        if (( i <= WARMUP_REPEATS )); then warmup=1; rep=$i; fi
        run_cell long "$k" 1 long "$ctx" "$rep" "$warmup" "$prompt"
      done
    done
    server_stop
  done
  report || true
}

# -------------------------------------------------------------- check ----

require_fingerprint() {
  need_cmd curl
  need_cmd flock
  need_cmd jq
  need_cmd python3
  need_cmd sha256sum
  [[ -f "$Q0_SCRIPT" ]] || die "frozen baseline not found at Q0_SCRIPT=$Q0_SCRIPT"
  mkdir -p "$RESULTS_DIR/runs" "$HERE/prompts"
  [[ -n "$BASELINE_SHA" ]] || BASELINE_SHA="$(sha256_of "$Q0_SCRIPT")"
  if [[ -z "$SIDECAR_SHA" && -n "${SIDECAR_LAYER0:-}" && -f "$SIDECAR_LAYER0" ]]; then
    # Layer-0 can be large. Hash it once per harness process, never once per
    # benchmark cell, or the fingerprint itself perturbs the I/O experiment.
    SIDECAR_SHA="$(sha256_of "$SIDECAR_LAYER0")"
  fi
  if [[ -z "$RUNTIME_IMAGE_ID" ]]; then
    RUNTIME_IMAGE_ID="$(docker images -q "$RUNTIME_IMAGE" 2>/dev/null | head -1 || true)"
  fi
}

check() {
  require_fingerprint
  log "=== Phase 0.1: environment, fingerprints, profiler verification ==="
  log "frozen baseline sha256: $BASELINE_SHA"
  if [[ -n "$SIDECAR_SHA" ]]; then
    log "sidecar layer-0 sha256: $SIDECAR_SHA"
  else
    warn "sidecar layer-0 not found (set SIDECAR_DIR); runs will carry empty sidecar hash"
  fi
  local image_present=0
  if [[ -n "$RUNTIME_IMAGE_ID" ]] && docker inspect --format '{{.Id}}' "$RUNTIME_IMAGE" >/dev/null 2>&1; then
    image_present=1
    log "runtime image digest: $RUNTIME_IMAGE_ID"
  else
    warn "runtime image $RUNTIME_IMAGE not present locally; skipping container profiler probe (no implicit pull)"
  fi

  local prof="$RESULTS_DIR/profile_check.json"
  local do_probe=1
  if [[ -f "$prof" && "$PROFILE_RECHECK" != "1" ]]; then do_probe=0; fi
  python3 - "$prof" "$RUNTIME_IMAGE" "$NCU_TEST" "$do_probe" "$image_present" <<'PY'
import json, os, shutil, subprocess, sys
prof, image = sys.argv[1], sys.argv[2]
ncu_test, do_probe, image_present = (x == "1" for x in sys.argv[3:6])
out = {}
def run(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "rc": p.returncode, "stdout_head": (p.stdout or "")[:400], "stderr_head": (p.stderr or "")[:400]}
    except Exception as e:
        return {"cmd": " ".join(cmd), "error": str(e)}
if os.path.exists(prof):
    out = json.load(open(prof))
for tool in ("ncu", "nsys"):
    out[f"host_{tool}_present"] = shutil.which(tool) is not None
if out.get("host_ncu_present") and "ncu_version" not in out:
    out["ncu_version"] = run(["ncu", "--version"], 30)
    out["ncu_sets"] = run(["ncu", "--list-sets"], 120)
    out["ncu_query_metrics_rc"] = run(["ncu", "--query-metrics"], 300).get("rc")
if do_probe and ncu_test and image_present and shutil.which("docker"):
    # Execute ncu and the CUDA workload in the same container. Passing ncu as
    # arguments after `bash -lc` only sets shell positional parameters and
    # never profiles anything, so keep the complete probe in one command.
    probe_cmd = (
        "command -v ncu >/dev/null || exit 127; "
        "ncu --target-processes all --metrics gpu__time_duration.sum --launch-count 1 "
        "python3 -c 'import torch; x=torch.rand(1<<20, device=\"cuda\"); print(float(x.sum()))'"
    )
    probe = run(["docker", "run", "--rm", "--gpus", "all", "--cap-add", "SYS_ADMIN",
                 image, "bash", "-lc", probe_cmd], timeout=420)
    out["ncu_kernel_probe"] = probe
    out["ncu_usable"] = probe.get("rc") == 0 and "Traceback" not in probe.get("stdout_head", "") + probe.get("stderr_head", "")
if "ncu_usable" not in out:
    out["ncu_usable"] = False
if not out["ncu_usable"]:
    out["fallback"] = ["nsys timeline", "CUPTI if available", "existing CUDA events", "host timing for QSA metadata"]
json.dump(out, open(prof, "w"), indent=2)
print("ncu: PRESENT and usable (real kernel counter read OK)" if out["ncu_usable"]
      else "ncu NOT usable -> plan the CUPTI/nsys/CUDA-event fallback ladder before profiling")
PY
  log "profiler check written to $prof"
  log "check complete"
}

# -------------------------------------------------------------- report ----

report() {
  [[ -s "$RESULTS_DIR/results.jsonl" ]] || { echo "no results yet"; return 0; }
  python3 - "$RESULTS_DIR/results.jsonl" "$RESULTS_DIR/summary.csv" "$RESULTS_DIR/summary.json" <<'PY'
import json, sys, csv, statistics as st
path, csv_out, json_out = sys.argv[1:4]
rows = []
for line in open(path):
    row = json.loads(line)
    if not row.get("warmup") and row.get("valid", True):
        rows.append(row)
if not rows:
    print("no measured rows"); sys.exit(0)
groups = {}
for r in rows:
    groups.setdefault((r.get("session_id", "legacy"), r["phase"], r["k"], r["streams"],
                       r["workload"], r.get("context_tokens", 0)), []).append(r)
summary = []
for key, rs in sorted(groups.items()):
    t = [r["aggregate_tps"] for r in rs if r["aggregate_tps"]]
    tt = [r["ttft_s_mean"] for r in rs if r.get("ttft_s_mean")]
    dc = sum(r["draft_cycles"] for r in rs); at = sum(r["accepted_tokens"] for r in rs); dt = sum(r["draft_tokens"] for r in rs)
    summary.append({
        "session_id": key[0], "phase": key[1], "k": key[2], "streams": key[3],
        "workload": key[4], "context_tokens": key[5],
        "n_runs": len(rs), "mean_agg_tps": round(st.mean(t), 2) if t else None,
        "std_agg_tps": round(st.stdev(t), 2) if len(t) > 1 else 0.0,
        "mean_ttft_s": round(st.mean(tt), 3) if tt else None,
        "accepted_per_cycle": round(at / dc, 3) if dc else None,
        "accept_per_draft_token": round(at / dt, 3) if dt else None,
        "memory_peak_gib_max": max((r["memory_peak_gib"] for r in rs
                                     if r.get("memory_peak_gib") is not None), default=None),
    })
with open(csv_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
json.dump(summary, open(json_out, "w"), indent=2)
for s in summary: print(s)
PY
  log "summary written: $RESULTS_DIR/summary.csv"
}

# ---------------------------------------------------------------- main ----

main() {
  local action=${1:-full}
  mkdir -p "$RESULTS_DIR"
  exec 200>>"$RESULTS_DIR/.results.lock"
  flock -n 200 || die "another phase0-sweep run holds the lock"
  trap cleanup EXIT
  case "$action" in
    check)        check ;;
    sweep-short)  check; sweep_short ;;
    sweep-long)
      [[ -s "$RESULTS_DIR/current-session.txt" ]] \
        || die "no short-sweep session found; run sweep-short first or use full"
      RUN_SESSION_ID="$(tr -d '\r\n' < "$RESULTS_DIR/current-session.txt")"
      check; sweep_long
      ;;
    full)         check; sweep_short; sweep_long ;;
    report)       report ;;
    *) die "unknown action: $action (check|sweep-short|sweep-long|full|report)" ;;
  esac
}

main "$@"
