# Running Qwen3.6-35B-A3B Locally on a 6GB Laptop GPU, and Making It Survive a Real Agent

**A setup log, not a tutorial.** This documents the actual process — including the wrong turns — of getting a 35B-parameter MoE model running locally on consumer laptop hardware, extending its usable context to 128K–256K tokens via a custom CUDA kernel fork, and validating it against a real autonomous coding agent (Hermes) rather than synthetic benchmarks.

---

## TL;DR

| | |
|---|---|
| **Model** | Qwen3.6-35B-A3B (hybrid Mamba/SSM + attention MoE, 256 experts / 8 active) |
| **Quant** | Q3_K_XL (~15.7GB), chosen over Q4_K_XL after a controlled comparison |
| **Hardware** | RTX 4050 Laptop (6GB VRAM), i5-13450HX, 24GB DDR5 |
| **Backend** | Self-built `llama.cpp` fork (CUDA 13.3, MSVC 14.44) with custom `turbo4`/`turbo3` KV-cache quantization |
| **Final context** | 128K tokens (not 256K — see [§7](#7-the-256k-vs-128k-decision)) |
| **Sustained throughput** | ~18 t/s at shallow context, decaying to ~7–9 t/s past 100K tokens |
| **Validation method** | Real read-only audit task against a private ~40-module HRMS codebase, cross-checked against ground truth |

---

## 1. Motivation & Reference Walkthrough

This project took direct inspiration from a YouTube optimization guide by **Codacus** — [*"Run Qwen 3.6 35B at 17 tokens/sec on 8-year-old hardware"*](https://youtu.be/8F_5pdcD3HY). Codacus demonstrated a 5-step `llama.cpp` optimization sequence on minimal hardware (NVIDIA GTX 1060 6GB VRAM, Intel i3-8100 CPU, 24GB DDR4 RAM):

### Codacus's Original 5-Step Optimization Sequence:
1. **Naive Baseline (`-L 20` / `-ngl 20`):** Splitting layers 50/50 between CPU and GPU resulted in unusable performance (~3 tokens/sec) due to massive PCIe bus thrashing across MoE expert layers.
2. **Pin Experts to CPU RAM (`--n-cpu-moe 41`):** Offloaded sparse, "sleeping" MoE expert blocks to CPU RAM while keeping dense attention layers on GPU. Boosted speed from **3 t/s to 10 t/s**.
3. **Disable Memory Mapping (`--no-mmap`):** Forced preloading the entire model into system RAM up front to avoid disk page faults. Boosted speed from **10 t/s to 13.5 t/s**.
4. **Reclaim Idle VRAM (`--n-cpu-moe 35`):** Pulled 6 additional layers back onto the GPU to utilize remaining VRAM. Boosted speed from **13.5 t/s to 17 t/s** (reducing context to 64K).
5. **TurboQuant KV-Cache (`--kv-type-k turbo4 --kv-type-v turbo3`):** Applied 4-bit key / 3-bit value quantization to expand context from 64K to **256,000 tokens** at a sustained 17 t/s without running out of VRAM.
6. **Lock System RAM (`--mlock`):** Locked model weights in memory to prevent OS disk-paging over multi-day sessions.
7. **Speculative Decoding (Tested & Rejected):** Drafted with Qwen3.5 800M. Despite a 65% draft acceptance rate, speed dropped from 17 t/s down to 11 t/s because batching 8 speculative tokens thrashed 64 distinct experts over PCIe, and the 30 SSM/Mamba layers required sequential processing per step.

### Goal of This Setup
The objective was to reproduce and adapt Codacus's pipeline for a modern equivalent (**RTX 4050 Laptop GPU, 6GB VRAM, i5-13450HX, 24GB DDR5 RAM**), while adjusting flags to respect laptop hardware boundaries and validating the setup under multi-hour, multi-file **autonomous agentic workloads** (Hermes Agent) rather than simple single-turn prompts.

---

## 2. Environment

```
GPU:      NVIDIA RTX 4050 Laptop (6GB VRAM, Ada Lovelace, compute capability 8.9)
CPU:      Intel Core i5-13450HX (10P+6E threads exposed as 16 logical)
RAM:      24GB DDR5 (23.7GB visible to Windows)
OS:       Windows 11
Backend:  llama.cpp — two parallel installs:
            (a) WinGet-distributed prebuilt (Vulkan backend) — kept as fallback
            (b) Self-built from TheTom/llama-cpp-turboquant, branch
                feature/turboquant-kv-cache, CUDA backend
Toolchain: CUDA Toolkit 13.3, MSVC 14.44 (VS2022 Build Tools), CMake 4.4
Model:     Qwen/Qwen3.6-35B-A3B (unsloth GGUF conversions)
Agent:     Hermes Agent (local desktop build), OpenAI-compatible endpoint
```

---

## 3. Architecture Notes That Mattered

Before any tuning, the model's own metadata (dumped via `llama-server`'s verbose load log) turned out to be essential context, not boilerplate:

```
general.architecture:        qwen35moe
qwen35moe.block_count:        40
qwen35moe.expert_count:       256
qwen35moe.expert_used_count:  8
qwen35moe.full_attention_interval: 4   # 1 in 4 layers is full attention
qwen35moe.ssm.*                        # the other 3 in 4 are Gated Delta Net (SSM-style)
```

**Why this matters:** Qwen3.6-35B-A3B is a *hybrid* architecture — most layers are State-Space/Mamba-style (sequential-by-nature, cannot be trivially parallelized across a speculative-decoding draft window), with full attention only every 4th layer. This single fact explained two separate observed failures later in the process:

1. Speculative decoding (tested briefly, not adopted) degraded speed rather than improving it — consistent with the video's own finding, and explainable by the SSM layers' inherent sequentiality.
2. llama.cpp's prompt-cache/context-checkpoint system repeatedly logged: `forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)`. Every new agent turn beyond a simple continuation re-processed a meaningful chunk of prompt from scratch — an architecture-level tax that no flag or quant choice removes.

---

## 4. Baseline: Adapting the Video Pipeline to Laptop Hardware

Running initial tests on `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (~20.8GB) highlighted key differences between desktop and laptop hardware constraints:

### Reference Video Steps vs. Laptop Adaptation Matrix

| Step | Codacus Video (GTX 1060 Desktop) | Laptop Adaptation (RTX 4050 6GB Laptop) | Result & Technical Rationale |
|---|---|---|---|
| 1. Naive baseline | `-L 20` (3 t/s) | Skipped naive split | Unusable — MoE layers straddle the PCIe bus |
| 2. Pin experts to CPU | `--n-cpu-moe 41` (10 t/s) | Replaced with `--fit` (`-fitt 400`) | `--fit` automatically performs per-tensor VRAM search, matching/beating manual stepping |
| 3. Disable mmap | `--no-mmap` (13.5 t/s) | **Reverted to default `mmap`** | `--no-mmap` caused `std::bad_alloc` (`0xc0000409`) on 24GB Windows (see §5) |
| 4. Reclaim idle VRAM | `--n-cpu-moe 35` (17 t/s) | Handled by `--fit` | Auto-fits max dense layers on GPU without manual `--n-cpu-moe` guessing |
| 5. TurboQuant KV cache | `-ctk turbo4 -ctv turbo3` (256k) | `-ctk turbo4 -ctv turbo3` (128k default) | Mainline Vulkan lacked kernels; CUDA build enabled TurboQuant. **128k context selected for higher agent audit thoroughness** (see §7) |
| 6. Lock RAM allocation | `--mlock` | Omitted | Avoids non-swappable RAM pressure near the 24GB Windows memory limit |
| 7. Speculative decoding | Tested & Rejected (11 t/s drop) | Rejected | Confirmed SSM & MoE sequential routing penalty slows generation |

### Why `--fit` Replaced Manual `--n-cpu-moe` Stepping
Manual decrementing of `--n-cpu-moe` is error-prone. `llama-server` ships a **`--fit`** flag (on by default in recent builds) that performs this exact search automatically, per-tensor, with finer granularity than manual stepping:

```
-fit,  --fit [on|off]        adjust unset arguments to fit device memory (default: on)
-fitt, --fit-target MiB...   safety margin per device (default: 1024)
-fitc, --fit-ctx N           minimum context size fit can settle for
```

Leaving `--n-cpu-moe`/`-ngl` **unset** and letting `-fit` decide consistently matched or beat manual tuning, and its log output (`common_params_fit_impl`) reports the exact layer/VRAM tradeoff it settled on.

---

## 5. A Real Crash, and What It Taught

Early testing on Q4_K_XL with `--no-mmap` + `--load-mode mlock` (the video's steps 3 and 6) produced a hard crash:

```
Exception code: 0xc0000409  (STATUS_STACK_BUFFER_OVERRUN / __fastfail)
Faulting module: ucrtbase.dll
```

Root cause, confirmed via free-RAM measurement at idle: **11.7GB free out of 23.7GB total**, against a 20.8GB model that `--no-mmap` demands be loaded as one contiguous, non-swappable block. The crash was an unhandled `std::bad_alloc` escalating to `abort()` — the process was asking Windows for memory that didn't exist.

**Fix:** drop `--no-mmap` and `--mlock` entirely; use plain `mmap` (the default). Unlike a forced preload, `mmap` lets the OS page sections in from disk as needed, using existing file-cache-style memory management instead of one hard allocation. This does *not* reproduce the video's step 3 exactly — but it's the correct adaptation for a RAM budget the video's own reference hardware didn't have to contend with (its 24GB was on a leaner Linux baseline).

**Consequence for `mlock`:** revisited later ([§9](#9-mlock-decision)) once on the CUDA build with a larger context — the same RAM math ruled it out there too, more severely.

---

## 6. Quant Selection: Q4 vs. Q3, Measured Not Assumed

Rather than assume Q3 would be "good enough," the comparison was made explicit:

- **Community consensus** on this model family: Q3_K_XL and Q4_K_XL are "effectively similar quality" by perplexity/KL-divergence metrics.
- **Caveat surfaced during research:** quantization loss doesn't reliably track real-task performance on hybrid SSM/MoE architectures — one quant-comparison writeup found a *higher-precision* Q8 build performing *worse* than Q4 on agentic benchmarks, attributing it to quantization interacting unpredictably with SSM/expert-routing tensors specifically.
- **Decision:** move to Q3_K_XL (15.69GB actual on disk) primarily to buy VRAM/RAM headroom for longer context — not on a pure quality argument — and validate correctness empirically afterward (§8) rather than trust the theory alone.

This freed enough headroom that **64K context fit without any KV-cache compression at all**, and 128K fit with only a manageable throughput cost — see next section.

---

## 7. The 256K vs. 128K Decision

This is the part worth reading closely if you're tuning your own setup, because the obvious choice (more context = better) was wrong for this use case.

### 7.1 Getting to CUDA + TurboQuant

Mainline llama.cpp (any backend) does not have `turbo4`/`turbo3` KV cache types. Getting them required:

1. Installing CUDA Toolkit 13.3 (matched to the driver's reported `CUDA UMD Version`, not blindly grabbing "latest"), VS2022 Build Tools (C++ workload only), and CMake.
2. **A specific, easy-to-miss trap:** launching the VS Developer Shell without explicit architecture flags defaults to the **32-bit** host compiler (`Hostx86\x86\cl.exe`). Building a 64-bit CUDA target against that either fails or misbehaves. Fix: `Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64`, then verify `where.exe cl.exe` resolves to `Hostx64\x64`.
3. Cloning `TheTom/llama-cpp-turboquant` (branch `feature/turboquant-kv-cache`, confirmed as the actual default/HEAD branch), building with:
   ```
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -j
   ```
   (`89` = Ada Lovelace compute capability, matched to the RTX 4050 specifically.)
4. Full build: ~15–20 minutes of CUDA kernel template-instance compilation (the fork hand-generates dozens of `fattn-mma-turbo-instance-*` flash-attention variants per head-dim/column-count combination).
5. Verification that `turbo4`/`turbo3` were real, wired-in kernels — not stub code — via `--help` output and a source grep for `GGML_TYPE_TURBO3_0`/`GGML_TYPE_TURBO4_0` across `dequant`, `fattn` (flash attention), and `set_rows` CUDA source files. This mattered because the fork's own branch list showed active, unmerged work on turbo4 correctness (`experiment/turbo4-quality-investigation`, `fix/test-turbo-quant-inverse-wht`), signaling the feature was real but not necessarily bug-free.

### 7.2 Context scaling results (Q3_K_XL, CUDA, turbo4/turbo3)

| Context | Loaded? | VRAM free (idle) | Steady-state gen. speed | Notes |
|---|---|---|---|---|
| 64K | Yes, no clamp | ~1061 MiB | ~17 t/s | Baseline, no turbo needed at this size |
| 128K | Yes, no clamp | ~992 MiB | ~14 t/s | Real cost, correctness confirmed |
| 256K | Yes, no clamp | headroom present | ~11–13 t/s in short samples | Confirmed correct output; host RAM dropped to ~3.8GB free at idle — closest to the earlier crash threshold all session |

Correctness was checked at each step with a controlled generation (fixed prompt, `enable_thinking: false`, forced 250-token minimum to avoid short-sample noise), comparing output for coherence, factual accuracy, and absence of repetition/drift against the 64K baseline. All three passed.

### 7.3 The actual agent test

Synthetic single-turn tests can't answer "is more context worth it for real work." A real Hermes Agent session was run twice — identical prompt, identical model/quant/backend, only the context ceiling changed — against a private, read-only audit task on a real multi-thousand-file codebase (a documentation/architecture audit with independently known ground truth).

**Prompt** (abridged): *"Read-only. Explore this repo. Map planning docs, identify MVP module set, report on two specific known subsystems (multi-tenancy isolation strategy, a formula evaluator), flag contradictions between documents, and propose a consolidation outline. Do not write anything."*

**256K run:** ~22 minutes wall-clock, ~6,300 internal agent tasks/tool-calls, one report. Found 1 major documentation contradiction. Correctly identified the tenancy model (RLS-based, not the originally-planned schema-per-tenant design) and the formula evaluator's implementation. Report was accurate but comparatively shallow on cross-document consistency checking.

**128K run:** noticeably longer wall-clock (prompt-processing alone passed the 100K-token mark by the ~53-minute mark of that phase), **~3× more internal tool-calls** (task IDs reaching 18,000+ vs. ~6,300). Same correct conclusions on both target subsystems, **but found 3 major contradictions instead of 1** — including one the 256K run missed entirely: a leftover deployment-spec code snippet using a physical-schema database pattern that silently contradicted the actual (RLS-based) architecture implemented in code.

**Interpretation:** with less room to hold everything in context at once, the agent was forced into smaller, more deliberate read/verify cycles — more total work, but closer per-document scrutiny. The larger context window let it pull more material into view per pass, but appeared to skim harder once a lot was in view simultaneously. **This is not a universal claim about context size and model quality** — it's a specific, measured result for this model, this agent's tool-calling pattern, and this class of task (multi-document cross-referencing, where thoroughness matters more than speed).

**Decision: 128K as the default**, 256K kept available for the specific case a task can't be chunked (e.g., one genuinely enormous single file that must be read in one pass).

---

## 8. Correctness Checks (Non-Negotiable, Not Optional)

Repeated at every context/quant change:

- **Thinking-mode routing:** Qwen3.6's chat template defaults to opening a `<think>` block. Without `chat_template_kwargs: {enable_thinking: false}` in the request body, short `max_tokens` budgets get entirely consumed by invisible reasoning, returning empty completions — this looked like a bug the first time it happened and was actually a request-shape mismatch.
- **`finish_reason` sanity check:** a `length`-truncated response with `predicted_n` matching the token cap exactly, on what should have been a two-word answer, was the tell that the above was happening.
- **Read-only compliance audit:** after the real agent test, `git status` on the target repo showed 2 changed files. Verified via file timestamps and content inspection that both predated the session by two days (an unrelated Obsidian vault clipping and its auto-generated graph cache) — the agent's read-only instruction held; the alarm was a false positive from unrelated vault housekeeping, not an actual violation. **This kind of check should not be skipped or assumed** — "the report looks fine" is not evidence that write instructions were respected.

---

## 9. `mlock` Decision

`--mlock` (or `--load-mode mlock`) pins the model non-swappably in RAM, preventing the gradual performance degradation the source video demonstrated over multi-hour sessions (OS paging idle weights to disk and back). Tested the reasoning, not just the flag:

- At Q3_K_XL / 128K, resident RAM usage sits close enough to the 24GB ceiling (particularly with a browser and the agent's own Electron overhead also running) that forcing a non-swappable lock risks reproducing the exact `bad_alloc` crash from §5, just via a different flag.
- **Key clarification during testing:** `mlock`'s effect is scoped to the *process*, not the system — closing `llama-server.exe` immediately and fully releases any locked memory, with zero persistence. This matters for anyone worried about a locked-RAM setup interfering with other RAM-heavy use (in this case, gaming) on the same machine between sessions — it doesn't.
- **Decision:** mmap without mlock, accepting the multi-hour staleness risk, mitigated instead by a periodic scheduled restart of the server process (not yet implemented — flagged as future work, not solved).

---

## 10. Automating the Startup

Final quality-of-life step: a PowerShell wrapper that starts `llama-server` (skipping if already running), polls `/health` until ready (with a timeout fallback), then launches the agent desktop app — reducing the daily workflow from two manual steps to one shortcut double-click.

```powershell
$serverExe = "C:\llama-turboquant\build\bin\Release\llama-server.exe"
$modelPath = "<path to Q3_K_XL gguf>"
$agentExe  = "<path to agent desktop .exe>"

$existing = Get-Process llama-server -ErrorAction SilentlyContinue
if (-not $existing) {
    Start-Process -FilePath $serverExe -ArgumentList @(
        "-m", "`"$modelPath`"", "-c", "131072", "--parallel", "1",
        "--port", "8080", "-fitt", "400", "-ctk", "turbo4", "-ctv", "turbo3"
    ) -WindowStyle Minimized

    for ($i = 0; $i -lt 36; $i++) {
        Start-Sleep -Seconds 5
        try {
            Invoke-RestMethod -Uri "http://localhost:8080/health" -TimeoutSec 2 -ErrorAction Stop
            break
        } catch { }
    }
}
Start-Process -FilePath $agentExe
```

---

## 11. Final Configuration

```powershell
C:\llama-turboquant\build\bin\Release\llama-server.exe `
  -m "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf" `
  -c 131072 `
  --parallel 1 `
  --port 8080 `
  -fitt 400 `
  -ctk turbo4 -ctv turbo3
```

| Setting | Value | Rationale |
|---|---|---|
| Quant | Q3_K_XL | Equal-quality-in-practice to Q4, meaningfully more headroom |
| Context | 128K | Empirically better agent output quality than 256K for this task class |
| Loading mode | mmap (default) | mlock unworkable at this RAM budget |
| Layer placement | `--fit` (automatic) | Beat manual `--n-cpu-moe` stepping in both effort and precision |
| KV cache | turbo4 (K) / turbo3 (V) | Required to fit 128K/256K in 6GB VRAM at all |
| Backend | Self-built CUDA fork | Mainline/Vulkan cannot do TurboQuant; kept Vulkan build as fallback |

---

## 12. Open Items / Known Limitations

- **No auto-restart/watchdog** for `llama-server.exe` yet — the mitigation for `mlock`-less long-session staleness is designed but not implemented.
- **Throughput decays with depth**, not a fixed rate: ~18 t/s shallow, ~7–9 t/s past 100K tokens in-session. This is expected (larger KV cache scan per token) and should be budgeted for, not treated as a regression.
- **Prompt caching is structurally limited** on this architecture — every new agent turn beyond simple continuation re-processes a meaningful chunk of context due to the SSM/hybrid-memory layers, independent of any flag choice here.
- **TurboQuant is unmerged, actively-developed community code**, not an upstream-stable feature — correctness was spot-checked, not exhaustively verified. Treat with the same caution you'd give any fork running ahead of mainline.

---

## 13. Follow-up: The Three Remaining Levers (64K, Prefetch, MTP) — All Tested, All Rejected

After the 128K decision, three acceleration ideas still looked plausible *on paper* and deserved a real test rather than a hand-wave. All three were run on a **second self-built fork used purely as a test vehicle** — `thecodacus/llama.cpp`, branch `fable5/turboquant` — while the production setup stayed on the TheTom build from §7. Bottom line up front: **none of the three nets positive on this hardware, and the production 128K config stands unchanged.**

### 13.1 What was tested, and the reasoning that made each plausible

| Lever | Why it *might* have helped | Verdict |
|---|---|---|
| **64K context** (vs 128K) | Smaller KV cache → less per-token scan → maybe faster decode | **No speed win** |
| **Expert prefetch** | Overlap the next layer's expert H2D copy with current compute → hide PCIe latency | **Net loss** |
| **MTP self-speculation** (`Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL`) | Model's own Multi-Token-Prediction head drafts ahead — no separate draft model, sidestepping the §1-step-7 spec-decoding penalty | **Net loss** |

### 13.2 Harness

Single fixed prompt (~2,755 tokens), `<think>` enabled, a real 128-token decode measured (not just prompt-processing), `-fitt 400`, `turbo4`/`turbo3` KV, flash-attention on. Both **prefill and decode** throughput captured so the two phases can't hide behind one blended number.

### 13.3 Base-model context ladder (turbo KV, identical prompt)

| Context allocation | Prefill t/s | Decode t/s |
|---|---|---|
| 8K | 18.1 | 4.51 |
| 64K | 17.2 | 4.74 |
| 128K | 17.0 | 5.37 |

**Prefill and decode are flat across all three allocations.** The reason: only the ~2,755 actual prompt tokens ever occupy the KV cache, regardless of the `-c` ceiling, and `--fit` rebalances the CPU/GPU offload to roughly the same margin each time. Context *allocation* size barely moves per-token speed at a fixed depth — what actually costs you is real context *depth* (§7's decay past 100K), which no allocation choice removes.

### 13.4 The 64K matrix (turbo KV)

| Config | Prefill t/s | Decode t/s | VRAM free (idle) |
|---|---|---|---|
| Base, prefetch off | 17.2 | 4.74 | 873 MiB |
| Base, **prefetch on** + `-fitt 1500` (≈2 GB reserved) | 15.0 | 4.32 | 2046 MiB |
| MTP model, MTP off | 17.2 | 4.11 | 948 MiB |
| MTP model, **MTP on** (68% accept) | 16.4 | 4.58 | 878 MiB |

- **Prefetch loses even when handed 2 GB of free VRAM to work with.** Reserving that headroom (`-fitt 1500`) forces `--fit` to push more of the model onto the CPU, and the extra offload costs more than the copy/compute overlap saves (15.0 < 17.2 prefill). Prefetch only ever won under a *fixed* non-`--fit` offload that left VRAM idle; under the real `--fit`-driven config it has nothing to reclaim.
- **MTP helps its own model (+11%, 4.11 → 4.58 decode) but still lands below plain base (4.74).** The MTP variant is ~1.5 GB larger on disk → more forced offload → a size penalty that outweighs the speculation gain. Same shape of result as the §1-step-7 draft-model rejection, reached by a different route.

### 13.5 Critical build caveat — these absolutes are NOT the production numbers

The test-vehicle fork's turbo dequant is **slow**: the *same* base model at 8K measured **21.97 t/s with f16 KV vs 4.51 t/s with turbo KV — a ~5× decode penalty** on this fork specifically. The production TheTom build in §7 sustains ~14 t/s at 128K *with* turbo, i.e. TheTom's turbo dequant kernel is far better optimized than thecodacus's.

**So read §13 for the *relative* comparisons only — not the absolute t/s.** The test fork was a measurement instrument, never a production candidate; the ~4–5 t/s decode figures here are an artifact of its slow dequant, not a property of the hardware or of TurboQuant in general. **Do not migrate production to the thecodacus fork.**

### 13.6 The 64K question is about thoroughness, not speed

Speed gives **zero** reason to drop to 64K — §13.3 settles that. The only remaining argument for 64K is the §7 thoroughness effect: a smaller window forces more read/verify chunking, which for this class of multi-document audit produced *more* contradictions caught, not fewer. Whether that holds at 64K is an **agent-audit-quality experiment** (contradictions found + wall-clock on the real Hermes task), not a throughput benchmark — and the risk to watch is cross-document contradiction-catching weakening if the planning docs stop co-fitting under a 64K window. Until that experiment is run, **128K remains the default**, exactly as §7 concluded.

---

## 14. Prefill Tuning — the one free win (`-ub 2048`), and correcting the record

Prompted by a second Codacus video ("Build a Local Coding Agent on a Budget GPU"), which makes
one load-bearing point: **agent workloads are prefill-bound, not decode-bound.** An agent
re-reads its system prompt + tool defs + MCP content + files every turn — that is prompt
processing (prefill), not token generation (decode). Prior sections optimized decode; this one
measures and tunes prefill on the **production TheTom build** (not the slow codacus fork).

### 14.1 Correcting the record — the slow-fork numbers were ~5–10× low
All §13 absolute throughput came from the slow `thecodacus` fork. Measured on the real
production build (warm, `<think>` off, ~3600-token prompt, 128K context):

| Metric | Slow codacus fork (§13) | **Production TheTom build (real)** |
| --- | --- | --- |
| Prefill | ~17 t/s | **482 t/s** (`-ub 512` default) |
| Decode | ~4.5 t/s | **~26 t/s** |

The production rig was never as slow as the test-fork data implied. (A one-off cold-start
request read 147 t/s prefill — that was cold GPU clocks + first-graph build, not steady state.)

### 14.2 Clean knob sweep (production build, 128K, averaged, one server at a time)

| Config | Prefill t/s | Decode t/s | VRAM used |
| --- | --- | --- | --- |
| `-ub 512`  `-t 10` turbo3 (default) | 481.8 | 26.4 | 4920 MiB |
| `-ub 1024` `-t 10` turbo3 | 679.7 | 25.4 | 4908 MiB |
| **`-ub 2048` `-t 10` turbo3 (chosen)** | **878.0** | 24.1 | 4954 MiB |
| `-ub 1024` `-t 15` turbo3 | 681.1 | 25.8 | 4908 MiB |
| `-ub 1024` `-t 10` turbo2 | 672.0 | 23.8 | 4878 MiB |

Data-integrity note: an initial sweep was **discarded** — a script bug killed the wrong PID
(MSYS pid, not Windows pid), so up to 7 servers stacked on the 6 GB GPU and thrashed. Tell:
decode appeared to change with `-ub`, which is physically impossible (ubatch touches only
prefill). The clean re-run gates on "GPU actually freed" before each launch; decode staying
flat (24–26) across all `-ub` confirms it.

### 14.3 Findings
- **`-ub 2048` → +82% prefill** (482 → 878 t/s), decode unchanged (~26 → ~24). The video's #1
  knob works on 6 GB too. **It fits at 128K** — the KV cache is pre-allocated for `-c 131072`
  at load, so the 4954 MiB figure already includes full-context KV; no deep-context OOM risk.
- **Threads: no gain.** `-t 10` == `-t 15` (679.7 vs 681.1). Keep 10 (leave the E-cores/OS headroom).
- **turbo2 for V: no gain, slightly worse decode.** Keep turbo3. VRAM was not the binding
  constraint here, so turbo2's free-VRAM benefit never triggered (it helps only when a freed
  expert layer can move to GPU — not the case at these ubatch sizes on this model).

### 14.4 Applied change
`start_agent.ps1` now launches with `-ub 2048` (param `$UBatch`, drop to 1024 if a future
config OOMs). Settled agent config: **`-ub 2048 -t 10 -ctk turbo4 -ctv turbo3`, 128K.** This is
the first change all project that nets a real, verified speed win on this hardware — because it
targets prefill (the metric that actually governs agent latency) rather than decode.

### 14.5 Open (next session)
- **REAP-pruned model** (`Qwen3.6-28B-REAP20-A3B`): ~20% experts stripped → less offload →
  potentially faster both phases at near-equal quality. Untested; ~10 GB download. Compare
  against the current 35B-A3B on the same prefill/decode harness.

---

## Appendix: Verified vs. Assumed Claims

| Claim | Status |
|---|---|
| Qwen3.6-35B-A3B exists, hybrid SSM/MoE architecture | Verified — model metadata dump |
| TurboQuant is a community fork, not mainline | Verified — checked mainline `--help` output lacked the types |
| `--fit` outperforms manual `--n-cpu-moe` stepping | Verified — direct comparison, matching/better VRAM margins |
| Q3 ≈ Q4 quality on this architecture | Partially verified — literature-supported, spot-checked via output comparison, not exhaustively benchmarked |
| 128K > 256K for multi-document agent correctness | Verified for this specific task class — explicitly *not* claimed as universal |
| `mlock` scope is per-process, releases fully on exit | Verified — behavior confirmed conceptually, consistent with documented Windows API semantics |
| 64K offers no raw-speed win over 128K at fixed depth | Verified — prefill/decode flat across 8K/64K/128K on the test fork (§13.3) |
| Expert prefetch nets negative under `--fit` | Verified — loses even with ~2 GB reserved VRAM (§13.4) |
| MTP self-speculation nets negative | Verified — helps its own model but stays below plain base decode (§13.4) |
| thecodacus fork turbo dequant ~5× slower than TheTom's | Verified — 21.97 (f16) vs 4.51 (turbo) t/s at 8K, same model (§13.5); absolutes in §13 are relative-only |
| Agent latency is prefill-bound, not decode-bound | Adopted from Codacus video; consistent with agent re-reading full prompt each turn (§14) |
| `-ub 2048` gives +82% prefill, decode unchanged, fits 128K in 6GB | Verified — clean averaged sweep on production build, 482→878 t/s (§14.2) |
| Production real prefill/decode = ~482 / ~26 t/s (not §13's ~17 / ~4.5) | Verified — §13 numbers were slow-fork artifacts (§14.1) |
| Threads and turbo2-V give no gain on this rig | Verified — t10==t15; turbo2 slightly worse decode (§14.3) |
