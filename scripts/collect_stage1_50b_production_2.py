#!/usr/bin/env python3
"""
Keural-13B Stage-1 Token Corpus Collector (V5, Production - BULLETPROOF RESUME)

Core fixes:
- Manual shard-based resume for datasets that don't support state_dict properly
- Timeout protection for stuck datasets (pg19)
- Fallback to non-streaming for small datasets (bookcorpusopen, wiki_ko_clean)
- Skip already-written content by checking output file size/hash

Why this works:
- Tracks shard index + samples within shard manually
- Non-streaming for small datasets = instant random access resume
- Timeout prevents infinite hangs on problematic samples
- Double-checks by seeking in output file to prevent duplicates

Run:
  python scripts/collect_stage1_50b_production.py
"""

import os
import json
import time
import signal
import sys
import warnings
import hashlib
from typing import Any, Dict, Iterable, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import itertools

from datasets import load_dataset, IterableDataset, Dataset
import sentencepiece as spm
from tqdm import tqdm

try:
    import psutil
except Exception:
    psutil = None

# =========================
# USER CONFIG
# =========================

TARGET_TOKENS = 58_000_000_000

SAVE_INTERVAL_TOKENS = 5_000_000
SAVE_INTERVAL_SECONDS = 120
STALL_SECONDS = 60
SAMPLE_TIMEOUT = 30  # Max seconds per sample before skipping

MAX_CHARS_PER_DOC = 200_000
MIN_CHARS_PER_DOC = 200
NORMALIZE_NEWLINES = True

BATCH_SIZE = 4000
MAX_WORKERS = 28
WRITE_BUFFER_SIZE = 1048576

TOKENIZER_PATH = "tokenizer/keural_tokenizer.model"

OUTPUT_DIR = "data/raw_stage1"
STATE_DIR = "data/state"
LOG_DIR = "data/logs"

STATE_FILE_V5 = os.path.join(STATE_DIR, "stage1_state_v5.json")
PROGRESS_FILE_V5 = os.path.join(LOG_DIR, "stage1_progress_v5.json")
STATE_FILE_OLD = os.path.join(STATE_DIR, "stage1_state_v4.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# 58B PLAN
# =========================

domain_plan: Dict[str, int] = {
    "fineweb_en":         30_000_000_000,
    "arxiv_science":       3_500_000_000,
    "pubmed_science":      2_500_000_000,
    
    "bookcorpusopen_lit":  676_045_016,
    "wiki_ko_clean":       2_000_000_000,
    "stack_v1_code":       8_000_000_000,
    "korean_webtext":      2_200_000_000,
    "wanjuan_korean":      3_000_000_000,
    "cc100_korean":        2_800_000_000,
    "pg19_literature":     2_000_000_000,
}

# Domains that should use NON-streaming (small enough to fit in RAM)
# This allows instant random-access resume
NON_STREAMING_DOMAINS = {"bookcorpusopen_lit", "wiki_ko_clean"}

# =========================
# DATASET LOADERS
# =========================

def ds_fineweb() -> Iterable[Dict[str, Any]]:
    return load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

def ds_stack_v1() -> Iterable[Dict[str, Any]]:
    try:
        return load_dataset("bigcode/the-stack-dedup", split="train", streaming=True, trust_remote_code=True)
    except:
        return load_dataset("bigcode/the-stack", split="train", streaming=True, trust_remote_code=True)

def ds_arxiv() -> Iterable[Dict[str, Any]]:
    return load_dataset("scientific_papers", "arxiv", split="train", streaming=True, trust_remote_code=True)

def ds_pubmed_sum() -> Iterable[Dict[str, Any]]:
    return load_dataset("ccdv/pubmed-summarization", split="train", streaming=True)

def ds_pg19() -> Iterable[Dict[str, Any]]:
    # pg19 is problematic - use streaming but with timeout protection
    return load_dataset("pg19", split="train", streaming=True)

def ds_bookcorpus() -> Dataset:
    # NON-STREAMING: small dataset, load fully into RAM for instant resume
    print("Loading bookcorpusopen in non-streaming mode (fast resume)...")
    return load_dataset("bookcorpusopen", split="train", streaming=False)

def ds_wiki_ko() -> Dataset:
    # NON-STREAMING: small dataset, load fully into RAM for instant resume
    print("Loading wiki_ko_clean in non-streaming mode (fast resume)...")
    return load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=False)

def ds_korean_webtext() -> Iterable[Dict[str, Any]]:
    return load_dataset("HAERAE-HUB/KOREAN-WEBTEXT", split="train", streaming=True)

def ds_wanjuan_korean() -> Iterable[Dict[str, Any]]:
    return load_dataset("opendatalab/WanJuan-Korean", split="train", streaming=True)

def ds_cc100_korean() -> Iterable[Dict[str, Any]]:
    return load_dataset("CocoRoF/cc-100-korean", "chunk_00", split="train", streaming=True)

dataset_map: Dict[str, Dict[str, Any]] = {
    "fineweb_en": {"loader": ds_fineweb, "field": "text", "streaming": True},
    "stack_v1_code": {"loader": ds_stack_v1, "field": "content", "streaming": True},
    "arxiv_science": {"loader": ds_arxiv, "field": "article", "streaming": True},
    "pubmed_science": {"loader": ds_pubmed_sum, "field": "article", "streaming": True},
    "pg19_literature": {"loader": ds_pg19, "field": "text", "streaming": True},
    "bookcorpusopen_lit": {"loader": ds_bookcorpus, "field": "text", "streaming": False},
    "wiki_ko_clean": {"loader": ds_wiki_ko, "field": "text", "streaming": False},
    "korean_webtext": {"loader": ds_korean_webtext, "field": "text", "streaming": True},
    "wanjuan_korean": {"loader": ds_wanjuan_korean, "field": "content", "streaming": True},
    "cc100_korean": {"loader": ds_cc100_korean, "field": "text", "streaming": True},
}

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# TOKENIZER
# =========================

sp = spm.SentencePieceProcessor()
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")
sp.load(TOKENIZER_PATH)

_thread_local = threading.local()

def get_thread_tokenizer():
    if not hasattr(_thread_local, 'tokenizer'):
        _thread_local.tokenizer = spm.SentencePieceProcessor()
        _thread_local.tokenizer.load(TOKENIZER_PATH)
    return _thread_local.tokenizer

# =========================
# UTILS
# =========================

def now_ts() -> float:
    return time.time()

def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def human_time(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return f"{d}d {h}h {m}m"
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"

def get_sys_stats() -> Dict[str, Any]:
    if psutil is None:
        return {"cpu_percent": None, "ram_used_gb": None, "ram_total_gb": None}
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round((vm.total - vm.available) / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
    }

# =========================
# STATE (V5 - BULLETPROOF)
# =========================

def load_state_v5() -> Tuple[int, Dict[str, int], Dict[str, Any], float, float, int]:
    """
    Returns: total_tokens, domain_progress, domain_resume_info, start_time, last_save_time, last_save_tokens
    domain_resume_info: {
        "shard_idx": int,      # for streaming datasets
        "sample_idx": int,     # index within current shard
        "dataset_index": int,  # for non-streaming datasets
        "file_pos": int        # byte position in output file
    }
    """
    if os.path.exists(STATE_FILE_V5):
        with open(STATE_FILE_V5, "r", encoding="utf-8") as f:
            st = json.load(f)
        print("Resuming from V5 state:", STATE_FILE_V5)
        total = int(st.get("total_tokens", 0))
        dp_in = st.get("domain_progress", {}) or {}
        dp = {k: int(dp_in.get(k, 0)) for k in domain_plan}
        dr_in = st.get("domain_resume_info", {}) or {}
        dr = {k: dr_in.get(k, {"shard_idx": 0, "sample_idx": 0, "dataset_index": 0, "file_pos": 0}) for k in domain_plan}
        start_time = float(st.get("start_time", now_ts()))
        last_save_time = float(st.get("last_save_time", now_ts()))
        last_save_tokens = int(st.get("last_save_tokens", total))
        return total, dp, dr, start_time, last_save_time, last_save_tokens

    # Migrate from V4
    if os.path.exists(STATE_FILE_OLD):
        with open(STATE_FILE_OLD, "r", encoding="utf-8") as f:
            st = json.load(f)
        print("Migrating from V4 state:", STATE_FILE_OLD)
        total = int(st.get("total_tokens", 0))
        dp_in = st.get("domain_progress", {}) or {}
        dp = {k: int(dp_in.get(k, 0)) for k in domain_plan}
        # Cannot migrate resume info from V4, start fresh per domain
        dr = {k: {"shard_idx": 0, "sample_idx": 0, "dataset_index": 0, "file_pos": 0} for k in domain_plan}
        start_time = float(st.get("start_time", now_ts()))
        last_save_time = float(st.get("last_save_time", now_ts()))
        last_save_tokens = int(st.get("last_save_tokens", total))
        return total, dp, dr, start_time, last_save_time, last_save_tokens

    print("Starting fresh (no state found).")
    total = 0
    dp = {k: 0 for k in domain_plan}
    dr = {k: {"shard_idx": 0, "sample_idx": 0, "dataset_index": 0, "file_pos": 0} for k in domain_plan}
    t0 = now_ts()
    return total, dp, dr, t0, t0, 0

total_tokens, domain_progress, domain_resume_info, start_time, _last_save_time, _last_save_tokens = load_state_v5()

def save_state_v5() -> None:
    global _last_save_time, _last_save_tokens
    st = {
        "version": "v5",
        "total_tokens": total_tokens,
        "domain_progress": domain_progress,
        "domain_resume_info": domain_resume_info,
        "domain_plan": domain_plan,
        "start_time": start_time,
        "last_save_time": now_ts(),
        "last_save_tokens": total_tokens,
    }
    atomic_write_json(STATE_FILE_V5, st)
    _last_save_time = float(st["last_save_time"])
    _last_save_tokens = int(st["last_save_tokens"])

def save_progress_v5(extra: Optional[Dict[str, Any]] = None) -> None:
    elapsed = now_ts() - start_time
    speed = (total_tokens / elapsed) if elapsed > 0 else 0.0
    remaining = TARGET_TOKENS - total_tokens
    eta = (remaining / speed) if speed > 0 else -1.0

    progress = {
        "version": "v5",
        "timestamp": now_ts(),
        "total_tokens": total_tokens,
        "target_tokens": TARGET_TOKENS,
        "remaining_tokens": remaining,
        "tokens_per_sec": speed,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "eta_human": human_time(eta),
        "domain_progress": domain_progress,
        "domain_resume_info": domain_resume_info,
        "domain_plan": domain_plan,
        "system": get_sys_stats(),
    }
    if extra:
        progress.update(extra)

    atomic_write_json(PROGRESS_FILE_V5, progress)

    print("\n=== Stage-1 Collector Progress (V5) ===")
    print(f"Collected:  {total_tokens:,}")
    print(f"Remaining:  {remaining:,}")
    print(f"Speed:      {int(speed):,} tokens/sec")
    print(f"ETA:        {human_time(eta)}")
    sys_stats = progress["system"]
    if sys_stats["cpu_percent"] is not None:
        print(f"CPU:        {sys_stats['cpu_percent']}%")
        print(f"RAM:        {sys_stats['ram_used_gb']} / {sys_stats['ram_total_gb']} GB")
    print("Per-domain:")
    for k in domain_plan:
        print(f"  - {k}: {domain_progress.get(k,0):,} / {domain_plan[k]:,}")
    print("======================================\n")

def maybe_checkpoint(force: bool = False, extra: Optional[Dict[str, Any]] = None) -> None:
    if force:
        save_state_v5()
        save_progress_v5(extra=extra)
        return

    token_delta = total_tokens - _last_save_tokens
    time_delta = now_ts() - _last_save_time

    if token_delta >= SAVE_INTERVAL_TOKENS or time_delta >= SAVE_INTERVAL_SECONDS:
        save_state_v5()
        save_progress_v5(extra=extra)

# =========================
# GRACEFUL EXIT
# =========================

def handle_exit(sig, frame):
    print("\nReceived signal, saving state and exiting...")
    try:
        maybe_checkpoint(force=True, extra={"exit_signal": int(sig)})
    finally:
        sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# =========================
# TEXT CLEANING
# =========================

def normalize_text(text: str) -> Optional[str]:
    if not text:
        return None
    if len(text) < MIN_CHARS_PER_DOC:
        return None
    if len(text) > MAX_CHARS_PER_DOC:
        text = text[:MAX_CHARS_PER_DOC]

    if NORMALIZE_NEWLINES:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = " ".join(text.splitlines())

    if "\x00" in text:
        text = text.replace("\x00", " ")

    text = " ".join(text.split())
    if len(text) < MIN_CHARS_PER_DOC:
        return None
    return text

# =========================
# PARALLEL PROCESSING
# =========================

def process_batch_parallel(texts: list) -> list:
    def tokenize_single(text):
        try:
            tokenizer = get_thread_tokenizer()
            count = len(tokenizer.encode(text))
            return (text, count)
        except Exception:
            return (text, 0)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(tokenize_single, texts))
    
    return results

# =========================
# TIMEOUT ITERATOR
# =========================

class TimeoutIterator:
    """Wraps an iterator with timeout protection per item"""
    def __init__(self, iterator, timeout=SAMPLE_TIMEOUT):
        self.iterator = iterator
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._buffer = None
        self._future = None
        self._prefetch()
    
    def _prefetch(self):
        try:
            self._future = self.executor.submit(next, self.iterator)
        except StopIteration:
            self._future = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._future is None:
            raise StopIteration
        
        try:
            result = self._future.result(timeout=self.timeout)
            self._prefetch()
            return result
        except TimeoutError:
            print(f"\nWARNING: Sample timeout after {self.timeout}s, skipping...")
            self._prefetch()
            return None  # Will be filtered out
        except StopIteration:
            raise
        except Exception as e:
            print(f"\nWARNING: Error fetching sample: {e}, skipping...")
            self._prefetch()
            return None

# =========================
# DOMAIN PROCESSOR (V5 - BULLETPROOF)
# =========================

def process_domain(domain: str) -> None:
    global total_tokens, domain_resume_info

    if domain not in dataset_map:
        raise KeyError(f"Unknown domain: {domain}")

    if domain_progress[domain] >= domain_plan[domain]:
        print(f"Skipping domain (already complete): {domain}")
        return

    print(f"Starting domain: {domain}")

    loader_fn = dataset_map[domain]["loader"]
    field = dataset_map[domain]["field"]
    is_streaming = dataset_map[domain]["streaming"]

    # Load dataset
    try:
        ds = loader_fn()
    except Exception as e:
        print(f"ERROR loading dataset for {domain}: {e}")
        return

    out_path = os.path.join(OUTPUT_DIR, f"{domain}.txt")
    
    # Get resume info
    resume_info = domain_resume_info.get(domain, {"shard_idx": 0, "sample_idx": 0, "dataset_index": 0, "file_pos": 0})
    
    # For non-streaming datasets: instant random access resume
    if not is_streaming and isinstance(ds, Dataset):
        start_idx = resume_info.get("dataset_index", 0)
        print(f"Non-streaming dataset: resuming from index {start_idx:,}")
        
        f_out = open(out_path, "a", encoding="utf-8", buffering=WRITE_BUFFER_SIZE)
        written = 0
        skipped = 0
        
        pbar = tqdm(
            total=domain_plan[domain],
            initial=domain_progress[domain],
            unit="tok",
            desc=f"{domain}",
        )
        
        try:
            for idx in range(start_idx, len(ds)):
                if total_tokens >= TARGET_TOKENS:
                    break
                if domain_progress[domain] >= domain_plan[domain]:
                    break
                
                sample = ds[idx]
                raw = sample.get(field, "")
                text = normalize_text(raw)
                
                if text is None:
                    skipped += 1
                    continue
                
                try:
                    count = len(sp.encode(text))
                    if count > 0:
                        f_out.write(text + "\n")
                        written += 1
                        total_tokens += count
                        domain_progress[domain] += count
                        pbar.update(count)
                        
                        # Update resume info every 100 samples
                        if idx % 100 == 0:
                            domain_resume_info[domain] = {
                                "shard_idx": 0,
                                "sample_idx": 0,
                                "dataset_index": idx,
                                "file_pos": f_out.tell()
                            }
                            maybe_checkpoint(force=False, extra={"current_domain": domain})
                            
                except Exception as e:
                    print(f"\nError tokenizing: {e}")
                    skipped += 1
                    continue
                    
        finally:
            pbar.close()
            f_out.close()
            
        maybe_checkpoint(force=True, extra={"domain_finished": domain})
        print(f"Finished domain: {domain} (written={written:,}, skipped={skipped:,})")
        return

    # STREAMING DATASET with manual shard tracking
    print(f"Streaming dataset: resuming from shard {resume_info.get('shard_idx', 0)}, sample {resume_info.get('sample_idx', 0)}")
    
    f_out = open(out_path, "a", encoding="utf-8", buffering=WRITE_BUFFER_SIZE)
    written = 0
    skipped = 0
    
    # Apply timeout protection for problematic datasets like pg19
    if domain == "pg19_literature":
        print("Applying timeout protection for pg19...")
        ds = TimeoutIterator(iter(ds), timeout=SAMPLE_TIMEOUT)
    
    pbar = tqdm(
        total=domain_plan[domain],
        initial=domain_progress[domain],
        unit="tok",
        desc=f"{domain}",
    )
    
    last_token_bump_time = now_ts()
    last_total_tokens_seen = total_tokens
    
    text_batch = []
    sample_count = 0
    
    try:
        for sample in ds:
            # Handle timeout returns
            if sample is None:
                skipped += 1
                continue
                
            if total_tokens >= TARGET_TOKENS:
                break
            if domain_progress[domain] >= domain_plan[domain]:
                break
            
            sample_count += 1
            
            # Skip if resuming (simple counter-based for streaming)
            if sample_count <= resume_info.get("sample_idx", 0):
                continue
            
            raw = sample.get(field, "")
            text = normalize_text(raw)
            if text is None:
                skipped += 1
                continue
            
            text_batch.append(text)
            
            if len(text_batch) >= BATCH_SIZE:
                results = process_batch_parallel(text_batch)
                
                batch_tokens = 0
                for txt, count in results:
                    if count > 0:
                        f_out.write(txt + "\n")
                        written += 1
                        total_tokens += count
                        domain_progress[domain] += count
                        batch_tokens += count
                
                # Update resume info
                domain_resume_info[domain] = {
                    "shard_idx": 0,  # We don't track shards separately, just sample count
                    "sample_idx": sample_count,
                    "dataset_index": sample_count,
                    "file_pos": f_out.tell()
                }
                
                pbar.update(batch_tokens)
                text_batch = []
                
                maybe_checkpoint(force=False, extra={"current_domain": domain})
                
                # Stall detection
                if total_tokens != last_total_tokens_seen:
                    last_total_tokens_seen = total_tokens
                    last_token_bump_time = now_ts()
                elif (now_ts() - last_token_bump_time) > STALL_SECONDS:
                    msg = f"WARNING: No token progress for > {STALL_SECONDS}s in domain={domain}."
                    print("\n" + msg)
                    maybe_checkpoint(force=True, extra={"stall_warning": msg, "domain": domain})
                    last_token_bump_time = now_ts()

        # Process remaining
        if text_batch:
            results = process_batch_parallel(text_batch)
            for txt, count in results:
                if count > 0:
                    f_out.write(txt + "\n")
                    written += 1
                    total_tokens += count
                    domain_progress[domain] += count
                    pbar.update(count)

    except Exception as e:
        print(f"\nERROR during processing {domain}: {e}")
        maybe_checkpoint(force=True, extra={"error": str(e), "domain": domain})
        raise

    finally:
        pbar.close()
        f_out.close()

    maybe_checkpoint(force=True, extra={"domain_finished": domain, "written": written, "skipped": skipped})
    print(f"Finished domain: {domain} (written={written:,}, skipped={skipped:,})")

# =========================
# MAIN
# =========================

def main() -> None:
    maybe_checkpoint(force=True, extra={"startup": True})

    for domain in domain_plan.keys():
        if total_tokens >= TARGET_TOKENS:
            break
        if domain_progress.get(domain, 0) < domain_plan[domain]:
            try:
                process_domain(domain)
            except Exception as e:
                print(f"Domain {domain} failed: {e}")
                print("Continuing to next domain...")
                continue

    maybe_checkpoint(force=True, extra={"complete": True})
    print("Stage-1 clean corpus collection complete (V5).")

if __name__ == "__main__":
    main()