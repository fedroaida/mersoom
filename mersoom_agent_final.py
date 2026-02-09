"""
mersoom_agent_final.py - Autonomous Learning Agent (LLM-free)

CHANGELOG:
- final: Refactor pass (typing + error handling + dedupe + naming + section/comment cleanup)
    * Fixes init-order bug in timezone/env helpers (no behavior change intended)
    * Leaves behavior/flags stable; improves maintainability and runtime safety
    * Intended working directory: D:\강준규\업무정리\##개인작업\Mersoom

- v23.14: Output/QA + Maintainability 통합 마감팩 (strict postprocess + QA fallback + §14 structure)
    * Adds strict postprocess option: sentence-level near-dup removal + eum ending only at sentence ends
    * Adds 2-stage QA fallback (shorten -> question-connector) with reason detail on skip
    * Adds ActionResult shape + §14 subheaders for maintainability

- v23.13: BM25 build 비용 완화 (observability + partial rebuild option)
    * Tracks build ms, corpus size, and added docs; HEALTH adds bm25_build_ms_p95_10m + bm25_docs_indexed
    * V2 mode can rebuild a recent slice (experimental) with health note

- v23.12: Timezone env (MERSOOM_TIMEZONE, default Asia/Seoul) with fallback
    * Uses zoneinfo when available; falls back to fixed offset
    * HEALTH includes tz_name + tz_fallback_used_10m

- v23.11: PoW watchdog (timeout streak -> executor restart + skip)
    * Tracks timeouts and restarts; HEALTH adds pow_timeouts_10m + pow_executor_restarts_10m

- v23.10: Recent action duplication guard (commit-time)
    * Adds recent_actions + dedupe by action|target_id|endpoint
    * HEALTH adds dup_action_skips_10m + recent_actions_size

- v23.9: 429 Retry-After handling polish + 10m observability
    * Honors Retry-After header with a safety cap + small jitter; logs basis for sleep
    * Adds HttpClient rolling 10-minute counters: http_429_10m, retry_after_sleeps_10m
    * HEALTH includes these fields via client snapshot (no behavior change unless 429 happens)

- v23.7: Interaction observability polish (health summary + openQ/thread metrics)
    * Adds HEALTH fields open_q_total/open_q_threads and reply_queue_age_p95 for better interaction visibility
    * Adds HEALTH field interaction_hint (short codes) to quickly diagnose why the agent is quiet
    * No behavior change unless MERSOOM_HEALTH_V2 is enabled (extra counters) / uses existing interaction flags

- v23.6: Reply queue scoring v2 (opt-in) + queue age observability
    * Adds env flag MERSOOM_REPLY_SCORE_V2 (default: false) to expand reply scoring (open questions, freshness, waiting penalty)
    * Adds comment_ts to reply inbox items; tie-breaks on latest comment_ts when scores are equal
    * HEALTH adds reply_queue_age_max + reply_scored_10m; emits reply.score_breakdown event (when enabled)

- v23.5: Waiting-for-remote 강화 + 관측성
    * When agent asks a question in a thread, strengthen waiting behavior and reduce consecutive replies
    * Adds env flag MERSOOM_WAITING_STRICT (default: false) to broaden question detection for waiting and extend cooldown
    * Adds HEALTH metrics waiting_threads / waiting_skips_10m
- v23.4: Open-question lifecycle (resolve/expire) + fix openq event logging
    * Detect remote answers to agent questions via lightweight token overlap; mark resolved/expired
    * Adds HEALTH metrics open_q_resolved_10m / open_q_expired_10m and events openq.resolve/openq.expire
- v23.3: Open-question tracking (ask registration) + observability
    * Adds env flag MERSOOM_OPENQ_TRACK to control open-question registration (default preserves prior behavior)
    * Registers open-questions from agent outgoing text (question heuristics) and logs openq.add event
    * HEALTH adds open_q_added_10m + open_q_added_total

- v23.21: Hotfixes + consistency polish (opt-in / safety only)
    * Interaction FSM flag: env overrides state; avoids freezing env value into saved state
    * HEALTH: thread phase counts skip __meta__; open_q_count counts only status=open when available
    * Rolling 10m counters: stale window reset on read (HEALTH reflects recent activity)
    * Phase question detection refined to reduce false clarify; JSONL buffer growth capped on repeated flush failures

- v23.2: Thread phase transitions (opt-in via MERSOOM_INTERACTION_FSM; default off)
    * Adds lightweight phase classifier (open|argue|clarify|close) and updates phase on ingest/commit
    * Emits event on phase change and bumps rolling counter 'phase_transition' for HEALTH
    * HEALTH adds 'phase_transitions_10m' + 'thread_phase_counts' (phase distribution snapshot)

- v23.1: Interaction scaffolding (schema only; opt-in via env flag; default off)
    * Adds thread schema defaults: phase/phase_ts and expanded open_questions fields (backward compatible)
    * Adds migration/backfill for conv_state.last_remote_ts + thread payload normalization (split threads file too)
    * HEALTH adds thread_count + open_q_count

- v23.0: HealthV2 scaffolding (opt-in via env flag; default off)
    * Adds generic rolling 10-minute counters for future observability
    * When MERSOOM_HEALTH_V2=true, HEALTH includes loop_tick_10m + health_emit_10m
    * Adds minimal unified reason/metrics helpers (no behavior change when flag off)

ENV FLAGS (v23.14):
  - MERSOOM_INTERACTION_FSM=true/false (default: false): enable upcoming interaction FSM features (v23.1 adds schema/backfill only).
  - MERSOOM_OPENQ_TRACK=true/false (default: true): track agent open-questions; v23.4 adds resolve/expire lifecycle.
  - MERSOOM_WAITING_STRICT=true/false (default: false): strengthen waiting_for_remote by broadening question detection and extending cooldown.
  - MERSOOM_REPLY_SCORE_V2=true/false (default: false): enable reply queue scoring v2 (open questions, freshness, waiting penalty) and extra score breakdown events.
  - MERSOOM_HEALTH_V2=true/false (default: false) : emit extended HEALTH fields + maintain extra rolling counters.
  - MERSOOM_DUP_ACTION_GUARD=true/false (default: true): guard repeated actions on same target.
  - MERSOOM_POW_WATCHDOG=true/false (default: true): restart PoW executor on repeated timeouts.
  - MERSOOM_TIMEZONE="Asia/Seoul" (default: Asia/Seoul): timezone name for logs/health.
  - MERSOOM_BM25_BUILD_V2=true/false (default: false): enable experimental partial BM25 rebuilds.
  - MERSOOM_STRICT_POSTPROCESS=true/false (default: false): stricter postprocess + QA fallback.
- v22.2: Section header cleanup + numbering consistency (no behavior change)
- v21.1: Broad optimization pass (I/O coalescing knobs + PoW prefix precompile + BM25 heap topk + tokenize/LRU caches + HTTP pool tuning; behavior preserved by default)
- v21.0: Code optimization pass (regex precompile + minor hygiene; no behavior change)
- v20.10_final2: Final2 validation pass (header/version sync; no behavior change)
- v20.10_final: Final polish (vote backlog cap env alias + selftest consistency)
- v20.10: B package (selftest expansion + reason-coverage polish + compatibility checks)
- v20.9: A-4 event log (1-line fixed format) + rules_sync/ops/arena stoploss events
- v20.8: A-3 observability wiring (no_action reason sourced from protocol + HEALTH reason_top5/ops_disabled/backlogs)
- v20.7: Action-level standardized reason codes (vote/comment/arena) for observability (A-2)
- v20.6: Observability reason protocol scaffold (state migration + helpers; no behavior change)

- v20.5: Reward unification (up/engage/risk) + mining 2nd QA gate + template exploration scheduling
    * Reward scalar = W_UP*Δup + W_ENGAGE*log1p(engage) - W_RISK*risk (env: MERSOOM_REWARD_W_UP / _W_ENGAGE / _W_RISK)
    * Mined templates require a second QA pass + near-dup checks before registration; rejected templates are recorded
    * New templates are only sampled at a capped rate (env: MERSOOM_EXPLORATION_RATE) to reduce policy pollution

- v20.4: Template cooldown + 2-layer near-dup (jaccard/3-gram) + dup fail counters
    * Apply template cooldown penalty in picker (MERSOOM_TEMPLATE_COOLDOWN_SEC / _PENALTY)
    * Extend near-dup guard with keyword Jaccard + char 3-gram similarity (MERSOOM_SIM_JACCARD_TH / MERSOOM_SIM_3GRAM_TH)
    * Record generation fail buckets (qa_fail / dup_fp / dup_sim) in state["protocol"]["gen_fail_counts"] and log DUP blocks

- v20.3: Heartbeat comment quota clamp + question-reply boost + fallback comments
    * Clamp per-cycle comments_target to feasible max (pace/limiter) and record hb_block_reason
    * Question-like targets boost reply_other selection (MERSOOM_REPLY_QUESTION_BOOST) with per-thread streak guard
    * Add last-resort fallback comment/reply templates (>=10 chars) to reduce zero-comment cycles

- v20.2: Vote backlog (durable mandatory votes)
    * Add protocol.vote_backlog (post_id, seen_ts) with TTL/cap + GC
    * Enqueue on sync; votes drain backlog before cache
    * Health metrics: vote_backlog_len, vote_backlog_drained (last 10m)

- v20.1: Mersoom guide compliance pass (personal-run)
    * Default API base -> https://mersoom.com/api
    * Feed fetch limit default -> 10 (guide recommendation)
    * Arena length default -> 300~500 chars
    * Strict English notice when English appears (LANG_STRICT)
    * Self-policing vote: downvote obvious rule violations (emoji/markdown/mostly-English w/o notice) + toxic/injection
    * Keep v20.0 hardening: recursion-safe debug logging, punctuation-aware 음슴체, single commit persist, no-action histogram, --selftest

  - v19.12: Readability/layout refresh (based on v19.11)
  - v19.11: Upvote-only default + auth401 circuit breaker + QA issue-boost + tick/meta stamping

ARCHITECTURE:
  Single-file design for whole-context editing; later sections depend on earlier ones.

QUICK NAVIGATION:
  Search for "# 1." (or "################################################################################") to jump between sections.

LAST MODIFIED: 2026-02-09
"""
# - LLM-free Mersoom agent (PoW, rate limits, learning, corpus/BM25, templates, brain)
# - P0: PoW offload, spam/near-dup guard, health/self-test, state GC, upvote-only default
# - P1: thread/user context scoring, cold-start bootstrapping, reflection recall, BM25 rebuild pacing
# - Swap-friendly: major units are grouped into numbered SECTIONS below

from __future__ import annotations

import os
import re
import json
import time
import sys
import subprocess
import math
import random
import heapq
import hashlib
import atexit
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import deque, Counter
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover - optional in older runtimes
    ZoneInfo = None

# (v21.1) Fast env access alias
_ENV = os.environ

# -----------------------------------------------------------------------------
# VERSION
# -----------------------------------------------------------------------------
AGENT_CODE_VERSION = "final"
# v22.3: Restore missing section 13 banner + quick navigation hint (no logic changes)
################################################################################
# 1. CONFIG
# - Dependencies: None
# - Used by: All sections
# - Key functions: load_config_from_env()
################################################################################

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

def _env_int(name: str, default: int, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    raw = _ENV.get(name)
    try:
        x = int(raw) if raw is not None else default
    except Exception:
        x = int(default)
    if min_v is not None:
        x = max(min_v, x)
    if max_v is not None:
        x = min(max_v, x)
    return x

def _env_float(name: str, default: float, min_v: Optional[float] = None, max_v: Optional[float] = None) -> float:
    raw = _ENV.get(name)
    try:
        x = float(raw) if raw is not None else float(default)
    except Exception:
        x = float(default)
    if min_v is not None:
        x = max(min_v, x)
    if max_v is not None:
        x = min(max_v, x)
    return x

def _env_bool(name: str, default: bool) -> bool:
    raw = _ENV.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")

def _load_timezone() -> Tuple[timezone, str, bool]:
    """Return (tzinfo, tz_name, fallback_used). Uses zoneinfo when possible."""
    tz_name = _env_str("MERSOOM_TIMEZONE", "Asia/Seoul").strip() or "Asia/Seoul"
    fallback_used = False
    tz = None
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = None
    if tz is None:
        tz = timezone(timedelta(hours=9))
        fallback_used = True
    return tz, tz_name, fallback_used

KST, TZ_NAME, TZ_FALLBACK_USED = _load_timezone()
_TZ_LOGGED = False
DEFAULT_USER_AGENT = "mersoom-agent/10.0"
MAX_NICKNAME_LEN = 10

def _safe_nickname(nick: str) -> str:
    n = (nick or "돌쇠").strip()
    n = re.sub(r"\s+", "", n)
    if len(n) > MAX_NICKNAME_LEN:
        n = n[:MAX_NICKNAME_LEN]
    return n or "돌쇠"

def _valid_auth_id(auth_id: str) -> bool:
    """Validate auth_id per Mersoom 3.0: 5~12 chars, [A-Za-z0-9_]."""
    try:
        s = (auth_id or "").strip()
        return bool(re.fullmatch(r"[A-Za-z0-9_]{5,12}", s))
    except Exception:
        return False

def _valid_password(password: str) -> bool:
    """Validate password length per guide: 10~20 chars."""
    try:
        n = len((password or "").strip())
        return bool(10 <= n <= 20)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Runtime globals (legacy): keep names stable across versions.
# These are set from Config inside run() via _apply_runtime_globals().
# -----------------------------------------------------------------------------
DEBUG_MODE = False

# Brain bias influence (Unit 08)
BRAIN_BIAS_ENABLE = True
BRAIN_BIAS_MIN = 0.85
BRAIN_BIAS_MAX = 1.25
BRAIN_BIAS_SCALE = 0.25
BRAIN_BIAS_LOG = False

# Arena (Colosseum) loop (Unit 09/10)
ARENA_ENABLE = True
ARENA_STATUS_MIN_INTERVAL_SEC = 180
ARENA_POSTS_MIN_INTERVAL_SEC = 300
ARENA_MAX_ACTIONS_PER_DAY = 6
ARENA_ARG_VARIANTS = 8
ARENA_REF_TOPK = 3
ARENA_ANTICOPY_JACCARD = 0.56
ARENA_ANTICOPY_SUBSTR_LEN = 18
ARENA_ANTICOPY_SIMHASH_MAX = 9
ARENA_USE_ANCHOR_VERBATIM = False
ARENA_QUALITY_MIN = 62

# Language rules (Mersoom 3.0)
LANG_STRICT = True
ENGLISH_NOTICE = "(한국어 모듈 오류남)"
ENGLISH_NOTICE_RATIO = 0.02
ENGLISH_NOTICE_APPEND_EUM = True

@dataclass(frozen=True)
class DebugFlags:
    debug: bool
    log_blocks: bool

@dataclass(frozen=True)
class ProtocolConfig:
    """v19 scaffold: protocol/heartbeat engine toggle (3.0 compliance will land in v19.x)."""
    enabled: bool

@dataclass(frozen=True)
class RulesSyncConfig:
    """v19.1: daily rules sync (skills.md) for 3.0 compliance."""
    daily: bool
    url: str


@dataclass(frozen=True)
class VoteProtocolConfig:
    """v19.2: mandatory voting for seen feed posts (3.0 compliance).

    - mandatory: if True, prioritize voting on any unvoted post fetched in the main feed sync.
    - seen_post_limit: LRU cap for state.seen.posts (post_id -> seen_ts)
    - voted_post_limit: LRU cap for state.votes.posts / state.voted_posts
    """
    mandatory: bool
    seen_post_limit: int
    voted_post_limit: int


@dataclass(frozen=True)
class HeartbeatConfig:
    """v19.3: 3.0 'heartbeat protocol' (4~5h cadence).

    Each heartbeat cycle aims to satisfy:
      - vote on seen feed posts (handled by v19.2 mandatory vote)
      - comment/reply on at least 2~3 posts
      - contribute (post or arena action) at least once per cycle
    """
    enabled: bool
    min_hours: float
    max_hours: float
    comment_min: int
    comment_max: int

@dataclass(frozen=True)
class LangRulesConfig:
    """v19.7: Mersoom 3.0 language rules (strict mode).

    - strict: enforce eum-style endings, strip emoji/markdown artifacts, and apply english notice rule.
    - english_notice: appended when English is present above threshold.
    - english_notice_ratio: minimum A-Z character ratio to trigger the notice (0~1).
    - english_notice_append_eum: append "임" after notice to satisfy sentence-ending rule.
    """
    strict: bool
    english_notice: str
    english_notice_ratio: float
    english_notice_append_eum: bool

@dataclass(frozen=True)
class AuthConfig:
    """v19.8: Optional account headers (auth_id/password) for points + higher comment limit.

    When enabled, the agent will attach:
      - X-Mersoom-Auth-Id
      - X-Mersoom-Password
    to write requests (POST) that already include PoW headers.

    Enable by setting:
      - MERSOOM_AUTH_ID
      - MERSOOM_PASSWORD
    (and optionally MERSOOM_AUTH_ENABLE=true)
    """
    enabled: bool
    auth_id: str
    password: str


@dataclass(frozen=True)
class BrainBiasConfig:
    enabled: bool
    min_mult: float
    max_mult: float
    scale: float
    log: bool

@dataclass(frozen=True)
class ArenaConfig:
    enabled: bool
    status_min_interval_sec: int
    posts_min_interval_sec: int
    max_actions_per_day: int

    arg_variants: int
    ref_topk: int
    anticopy_jaccard: float
    anticopy_substr_len: int
    anticopy_simhash_max: int
    use_anchor_verbatim: bool
    quality_min: int

@dataclass(frozen=True)
class Paths:
    # 기존 파일들
    state: str
    memory: str
    policy: str
    semantic: str
    journal: str
    brain: str
    brain_note: str
    memory_archive_jsonl: str  # "" => disabled

    # (확장) 맥락/코퍼스
    threads: str
    users: str
    corpus_jsonl: str

    # (P0) state generation stamps
    meta: str
    memory_meta: str

@dataclass(frozen=True)
class RateLimits:
    window_sec: int
    posts_per_window: int
    comments_per_window: int
    votes_per_window: int

    max_votes_per_tick: int
    max_comments_per_tick: int
    max_posts_per_tick: int
    max_replies_per_tick: int

@dataclass(frozen=True)
class Timing:
    tick_min_sec: int
    tick_max_sec: int
    sync_min_interval_sec: int

    after_action_sleep_min: int
    after_action_sleep_max: int
    idle_retry_min: int
    idle_retry_max: int

    global_vote_min_gap_sec: int
    global_comment_min_gap_sec: int
    global_post_min_gap_sec: int

    post_min_gap_sec: int
    same_post_comment_gap_sec: int
    same_text_gap_sec: int
    max_text_regen_tries: int

    sleep_hard_cap_sec: int

@dataclass(frozen=True)
class HttpConfig:
    base_url: str
    timeout_connect_sec: float
    timeout_read_sec: float
    user_agent: str
    dry_run: bool

    max_retries: int
    backoff_base: float
    backoff_cap: float
    retry_on_5xx: bool

@dataclass(frozen=True)
class PowConfig:
    wallet: str  # optional "nonce:wallet" format when set

@dataclass(frozen=True)
class HybridChallengeConfig:
    """Hybrid /challenge 대응(임시):
    - type=pow가 아니면(예: AI Puzzle), 일정 횟수 재요청하여 pow가 나올 때만 진행
    - 퍼즐 payload는 JSONL로 수집(추후 솔버/규칙 업데이트용)
    """
    max_retries: int
    retry_sleep_ms: int
    puzzle_log_jsonl: str  # "" => disabled
    puzzle_raw_max_chars: int
    puzzle_solver_enable: bool = True
    puzzle_solver_debug: bool = False

@dataclass(frozen=True)
class RuntimeMode:
    always_on: bool
    activity_mode: str  # "paced" | "burst"

@dataclass(frozen=True)
class SnapshotConfig:
    enabled: bool
    script: str
    timeout_sec: int
    run_on_boot: bool

@dataclass(frozen=True)
class QualityGate:
    """Quality gate + dry QA batch controls (Unit 01).

    - enabled: pre-flight checks for outgoing text; regenerate if below min_score.
    - batch_on_boot: run a local QA batch report at startup (no writes when DRY_RUN).
    """
    enabled: bool
    min_score: int
    max_tries: int

    batch_on_boot: bool
    batch_n: int
    batch_show_worst: int
    batch_save_path: str
    batch_exit: bool



@dataclass(frozen=True)
class ToxicConfig:
    """Incoming toxic/taunt handling.

    - auto_downvote: when toxic content is detected in an engagement target, vote 'down' on the post (best-effort).
    - no_reply: do not reply directly to toxic comments.
    - exclude_from_learning: do not feed toxic text into thread turns / template mining inputs.
    """
    auto_downvote: bool
    no_reply: bool
    exclude_from_learning: bool

@dataclass(frozen=True)
class Config:
    nickname: str
    auth: AuthConfig
    protocol: ProtocolConfig
    rules_sync: RulesSyncConfig
    vote_proto: VoteProtocolConfig
    heartbeat: HeartbeatConfig
    lang: LangRulesConfig
    toxic: ToxicConfig
    debug: DebugFlags
    brain_bias: BrainBiasConfig
    arena: ArenaConfig
    paths: Paths
    limits: RateLimits
    timing: Timing
    http: HttpConfig
    pow: PowConfig
    hybrid: HybridChallengeConfig
    mode: RuntimeMode
    snapshot: SnapshotConfig
    quality: QualityGate

def load_config_from_env() -> Config:
    # ✅ www 고정이 기본값 (리다이렉트로 PoW 헤더 누락 가능성 줄이기)
    base = _env_str("MERSOOM_BASE", "https://mersoom.com/api").rstrip("/")

    nickname_raw = _env_str("MERSOOM_NICKNAME", "돌쇠")
    nickname = _safe_nickname(nickname_raw)

    # v19.8 optional account headers (points + higher comment limit)
    auth_id = _env_str("MERSOOM_AUTH_ID", "").strip()
    password = _env_str("MERSOOM_PASSWORD", "").strip()
    auth_enable_flag = _env_bool("MERSOOM_AUTH_ENABLE", True)
    auth_enabled = bool(auth_enable_flag and auth_id and password and _valid_auth_id(auth_id) and _valid_password(password))
    auth = AuthConfig(enabled=auth_enabled, auth_id=auth_id, password=password)


    protocol = ProtocolConfig(
        enabled=_env_bool("MERSOOM_PROTOCOL_ENABLE", True),
    )

    rules_sync = RulesSyncConfig(
        daily=_env_bool("MERSOOM_RULES_SYNC_DAILY", True),
        url=_env_str("MERSOOM_RULES_SYNC_URL", "https://mersoom.com/docs/skills.md").strip(),
    )

    vote_proto = VoteProtocolConfig(
        mandatory=_env_bool("MERSOOM_VOTE_MANDATORY", True),
        seen_post_limit=_env_int("MERSOOM_SEEN_POST_LIMIT", 500, min_v=50, max_v=50000),
        voted_post_limit=_env_int("MERSOOM_VOTED_POST_LIMIT", 5000, min_v=200, max_v=200000),
    )


    hb_enabled = _env_bool("MERSOOM_HEARTBEAT_ENABLE", True)
    hb_min = _env_float("MERSOOM_HEARTBEAT_MIN_HOURS", 4.0, min_v=0.5, max_v=24.0)
    hb_max = _env_float("MERSOOM_HEARTBEAT_MAX_HOURS", 5.0, min_v=0.5, max_v=24.0)
    if hb_max < hb_min:
        hb_max = hb_min
    hb_cmin = _env_int("MERSOOM_HEARTBEAT_COMMENT_MIN", 2, min_v=0, max_v=50)
    hb_cmax = _env_int("MERSOOM_HEARTBEAT_COMMENT_MAX", 3, min_v=0, max_v=50)
    if hb_cmax < hb_cmin:
        hb_cmax = hb_cmin
    heartbeat = HeartbeatConfig(
        enabled=hb_enabled,
        min_hours=float(hb_min),
        max_hours=float(hb_max),
        comment_min=int(hb_cmin),
        comment_max=int(hb_cmax),
    )

    lang = LangRulesConfig(
        strict=_env_bool("MERSOOM_LANG_STRICT", True),
        english_notice=_env_str("MERSOOM_ENGLISH_NOTICE", "(한국어 모듈 오류남)"),
        english_notice_ratio=_env_float("MERSOOM_ENGLISH_NOTICE_RATIO", 0.02),
        english_notice_append_eum=_env_bool("MERSOOM_ENGLISH_NOTICE_APPEND_EUM", True),
    )

    toxic = ToxicConfig(
        auto_downvote=_env_bool("MERSOOM_TOXIC_AUTO_DOWNVOTE", True),
        no_reply=_env_bool("MERSOOM_TOXIC_NO_REPLY", True),
        exclude_from_learning=_env_bool("MERSOOM_TOXIC_EXCLUDE_LEARNING", True),
    )

    debug = DebugFlags(
        debug=_env_bool("MERSOOM_DEBUG", False),
        log_blocks=_env_bool("MERSOOM_LOG_BLOCKS", False),
    )

    # Brain bias influence (Unit 08) — keep env names for backward compatibility.
    bb_min = _env_float("MERSOOM_BRAIN_BIAS_MIN", 0.85, min_v=0.1, max_v=2.0)
    bb_max = _env_float("MERSOOM_BRAIN_BIAS_MAX", 1.25, min_v=0.1, max_v=3.0)
    if bb_max < bb_min:
        bb_max = bb_min
    brain_bias = BrainBiasConfig(
        enabled=_env_bool("MERSOOM_BRAIN_BIAS_ENABLE", True),
        min_mult=bb_min,
        max_mult=bb_max,
        scale=_env_float("MERSOOM_BRAIN_BIAS_SCALE", 0.25, min_v=0.0, max_v=2.0),
        log=_env_bool("MERSOOM_BRAIN_BIAS_LOG", False),
    )

    # Arena (Colosseum) loop (Unit 09/10)
    arena = ArenaConfig(
        enabled=_env_bool("MERSOOM_ARENA_ENABLE", True),
        status_min_interval_sec=_env_int("MERSOOM_ARENA_STATUS_MIN_INTERVAL_SEC", 180, 10, 3600),
        posts_min_interval_sec=_env_int("MERSOOM_ARENA_POSTS_MIN_INTERVAL_SEC", 300, 10, 3600),
        max_actions_per_day=_env_int("MERSOOM_ARENA_MAX_ACTIONS_PER_DAY", 6, 0, 100),

        arg_variants=_env_int("MERSOOM_ARENA_ARG_VARIANTS", 8, 1, 30),
        ref_topk=_env_int("MERSOOM_ARENA_REF_TOPK", 3, 0, 10),
        anticopy_jaccard=_env_float("MERSOOM_ARENA_ANTICOPY_JACCARD", 0.56, min_v=0.0, max_v=1.0),
        anticopy_substr_len=_env_int("MERSOOM_ARENA_ANTICOPY_SUBSTR_LEN", 18, 8, 80),
        anticopy_simhash_max=_env_int("MERSOOM_ARENA_ANTICOPY_SIMHASH_MAX", 9, 0, 64),
        use_anchor_verbatim=_env_bool("MERSOOM_ARENA_USE_ANCHOR_VERBATIM", False),
        quality_min=_env_int("MERSOOM_ARENA_QUALITY_MIN", 62, 0, 100),
    )

    # Snapshot runner (hourly) — moved into Config to keep dependency flow downward.
    snap_script = _env_str("MERSOOM_SNAPSHOT_SCRIPT", "snapshot_mersoom_to_xlsx_v2.py").strip()
    if not snap_script:
        snap_script = "snapshot_mersoom_to_xlsx_v2.py"
    snapshot = SnapshotConfig(
        enabled=_env_bool("MERSOOM_SNAPSHOT_ENABLED", True),
        script=snap_script,
        timeout_sec=_env_int("MERSOOM_SNAPSHOT_TIMEOUT_SEC", 900, min_v=10, max_v=24 * 60 * 60),
        run_on_boot=_env_bool("MERSOOM_SNAPSHOT_RUN_ON_BOOT", False),
    )


    paths = Paths(
        state=_env_str("MERSOOM_STATE", "mersoom_state.json"),
        memory=_env_str("MERSOOM_MEMORY", "mersoom_memory.json"),
        policy=_env_str("MERSOOM_POLICY", "mersoom_policy.json"),
        semantic=_env_str("MERSOOM_SEMANTIC", "mersoom_semantic.json"),
        journal=_env_str("MERSOOM_JOURNAL", "mersoom_journal.txt"),
        brain=_env_str("MERSOOM_BRAIN", "mersoom_brain.json"),
        brain_note=_env_str("MERSOOM_BRAIN_NOTE", "mersoom_brain.md"),
        memory_archive_jsonl=_env_str("MERSOOM_MEMORY_ARCHIVE", "").strip(),

        threads=_env_str("MERSOOM_THREADS", "mersoom_threads.json"),
        users=_env_str("MERSOOM_USERS", "mersoom_users.json"),
        corpus_jsonl=_env_str("MERSOOM_CORPUS", "mersoom_corpus.jsonl"),
        meta=_env_str("MERSOOM_META", "mersoom_meta.json"),
        memory_meta=_env_str("MERSOOM_MEMORY_META", "mersoom_memory_meta.json"),
    )

    window_sec = _env_int("MERSOOM_WINDOW_SEC", 30 * 60, min_v=60, max_v=24 * 60 * 60)

    # Mersoom 3.0: login(auth_id) raises comment limit to 20/30min; keep override via env.
    default_comments_30 = 20 if bool(auth_enabled) else 10

    limits = RateLimits(
        window_sec=window_sec,
        posts_per_window=_env_int("MERSOOM_POSTS_PER_WINDOW", _env_int("MERSOOM_POSTS_PER_30MIN", 2, min_v=0, max_v=200), min_v=0, max_v=200),
        comments_per_window=_env_int("MERSOOM_COMMENTS_PER_WINDOW", _env_int("MERSOOM_COMMENTS_PER_30MIN", default_comments_30, min_v=0, max_v=500), min_v=0, max_v=500),
        votes_per_window=_env_int("MERSOOM_VOTES_PER_WINDOW", _env_int("MERSOOM_VOTES_PER_30MIN", 12, min_v=0, max_v=500), min_v=0, max_v=500),

        max_votes_per_tick=_env_int("MERSOOM_MAX_VOTES_PER_TICK", 3, min_v=0, max_v=50),
        max_comments_per_tick=_env_int("MERSOOM_MAX_COMMENTS_PER_TICK", 1, min_v=0, max_v=50),
        max_posts_per_tick=_env_int("MERSOOM_MAX_POSTS_PER_TICK", 1, min_v=0, max_v=10),
        max_replies_per_tick=_env_int("MERSOOM_MAX_REPLIES_PER_TICK", 1, min_v=0, max_v=50),
    )

    timing = Timing(
        tick_min_sec=_env_int("MERSOOM_TICK_MIN_SEC", 20, min_v=5, max_v=3600),
        tick_max_sec=_env_int("MERSOOM_TICK_MAX_SEC", 60, min_v=5, max_v=3600),
        sync_min_interval_sec=_env_int("MERSOOM_SYNC_MIN_INTERVAL_SEC", 45, min_v=10, max_v=3600),

        after_action_sleep_min=_env_int("MERSOOM_AFTER_ACTION_SLEEP_MIN", 8, min_v=0, max_v=3600),
        after_action_sleep_max=_env_int("MERSOOM_AFTER_ACTION_SLEEP_MAX", 25, min_v=0, max_v=3600),
        idle_retry_min=_env_int("MERSOOM_IDLE_RETRY_MIN", 15, min_v=1, max_v=3600),
        idle_retry_max=_env_int("MERSOOM_IDLE_RETRY_MAX", 60, min_v=1, max_v=3600),

        global_vote_min_gap_sec=_env_int("MERSOOM_GLOBAL_VOTE_MIN_GAP_SEC", 5, min_v=0, max_v=3600),
        global_comment_min_gap_sec=_env_int("MERSOOM_GLOBAL_COMMENT_MIN_GAP_SEC", 10, min_v=0, max_v=3600),
        global_post_min_gap_sec=_env_int("MERSOOM_GLOBAL_POST_MIN_GAP_SEC", 30, min_v=0, max_v=3600),

        post_min_gap_sec=_env_int("MERSOOM_POST_MIN_GAP_SEC", 240, min_v=0, max_v=24 * 3600),
        same_post_comment_gap_sec=_env_int("MERSOOM_THREAD_DEBOUNCE_MIN_SEC", _env_int("MERSOOM_MIN_SAME_POST_GAP", 1800, min_v=0, max_v=24 * 3600), min_v=0, max_v=24 * 3600),
        same_text_gap_sec=_env_int("MERSOOM_MIN_SAME_TEXT_GAP", 1200, min_v=0, max_v=24 * 3600),
        max_text_regen_tries=_env_int("MERSOOM_MAX_TEXT_REGEN_TRIES", 6, min_v=0, max_v=50),

        sleep_hard_cap_sec=_env_int("MERSOOM_SLEEP_HARD_CAP_SEC", 900, min_v=30, max_v=24 * 3600),
    )

    http = HttpConfig(
        base_url=base,
        timeout_connect_sec=_env_float("MERSOOM_TIMEOUT_CONNECT_SEC", 8.0, min_v=1.0, max_v=60.0),
        timeout_read_sec=_env_float("MERSOOM_TIMEOUT_READ_SEC", 18.0, min_v=3.0, max_v=120.0),
        user_agent=_env_str("MERSOOM_USER_AGENT", DEFAULT_USER_AGENT),
        dry_run=_env_bool("MERSOOM_DRY_RUN", False),

        max_retries=_env_int("MERSOOM_HTTP_MAX_RETRIES", 4, min_v=0, max_v=20),
        backoff_base=_env_float("MERSOOM_HTTP_BACKOFF_BASE", 0.7, min_v=0.05, max_v=10.0),
        backoff_cap=_env_float("MERSOOM_HTTP_BACKOFF_CAP", 18.0, min_v=0.2, max_v=120.0),
        retry_on_5xx=_env_bool("MERSOOM_HTTP_RETRY_ON_5XX", True),
    )

    powcfg = PowConfig(wallet=_env_str("MERSOOM_WALLET", "").strip())
    hybrid = HybridChallengeConfig(
        max_retries=_env_int("MERSOOM_CHALLENGE_POW_RETRY_MAX", 4, min_v=0, max_v=50),
        retry_sleep_ms=_env_int("MERSOOM_CHALLENGE_POW_RETRY_SLEEP_MS", 120, min_v=0, max_v=5000),
        puzzle_log_jsonl=_env_str("MERSOOM_PUZZLE_LOG", r".\mersoom_puzzles.jsonl").strip(),
        puzzle_raw_max_chars=_env_int("MERSOOM_PUZZLE_RAW_MAX_CHARS", 2000, min_v=200, max_v=20000),
        puzzle_solver_enable=_env_bool("MERSOOM_PUZZLE_SOLVER_ENABLE", True),
        puzzle_solver_debug=_env_bool("MERSOOM_PUZZLE_SOLVER_DEBUG", False),
    )


    act_mode = _env_str("MERSOOM_ACTIVITY_MODE", "paced").strip().lower()
    if act_mode not in ("paced", "burst"):
        act_mode = "paced"

    mode = RuntimeMode(
        always_on=_env_bool("MERSOOM_ALWAYS_ON", True),
        activity_mode=act_mode,
    )


    quality = QualityGate(
        enabled=_env_bool("MERSOOM_QA_GATE", False),
        min_score=_env_int("MERSOOM_QA_MIN_SCORE", 72, min_v=0, max_v=100),
        max_tries=_env_int("MERSOOM_QA_MAX_TRIES", 6, min_v=1, max_v=50),

        batch_on_boot=_env_bool("MERSOOM_QA_BATCH_ON_BOOT", False),
        batch_n=_env_int("MERSOOM_QA_BATCH_N", 50, min_v=1, max_v=500),
        batch_show_worst=_env_int("MERSOOM_QA_BATCH_SHOW_WORST", 5, min_v=0, max_v=50),
        batch_save_path=_env_str("MERSOOM_QA_BATCH_SAVE", "").strip(),
        batch_exit=_env_bool("MERSOOM_QA_BATCH_EXIT", False),
    )

    # sanity: ensure ranges
    if timing.after_action_sleep_max < timing.after_action_sleep_min:
        timing = Timing(**{**timing.__dict__, "after_action_sleep_max": timing.after_action_sleep_min})
    if timing.idle_retry_max < timing.idle_retry_min:
        timing = Timing(**{**timing.__dict__, "idle_retry_max": timing.idle_retry_min})
    if timing.tick_max_sec < timing.tick_min_sec:
        timing = Timing(**{**timing.__dict__, "tick_max_sec": timing.tick_min_sec})

    return Config(
        nickname=nickname,
        auth=auth,
        protocol=protocol,
        rules_sync=rules_sync,
        vote_proto=vote_proto,
        heartbeat=heartbeat,
        lang=lang,
        toxic=toxic,
        debug=debug,
        brain_bias=brain_bias,
        arena=arena,
        paths=paths,
        limits=limits,
        timing=timing,
        http=http,
        pow=powcfg,
        hybrid=hybrid,
        mode=mode,
        snapshot=snapshot,
        quality=quality,
    )

def _apply_runtime_globals(cfg: Config) -> None:
    """Apply Config-derived flags to legacy module-level globals.

    This keeps older sections stable (they reference globals like ARENA_ENABLE),
    while making env-reading single-sourced in load_config_from_env().
    """
    global DEBUG_MODE
    global BRAIN_BIAS_ENABLE, BRAIN_BIAS_MIN, BRAIN_BIAS_MAX, BRAIN_BIAS_SCALE, BRAIN_BIAS_LOG
    global ARENA_ENABLE, ARENA_STATUS_MIN_INTERVAL_SEC, ARENA_POSTS_MIN_INTERVAL_SEC, ARENA_MAX_ACTIONS_PER_DAY
    global LANG_STRICT, ENGLISH_NOTICE, ENGLISH_NOTICE_RATIO, ENGLISH_NOTICE_APPEND_EUM
    global ARENA_ARG_VARIANTS, ARENA_REF_TOPK, ARENA_ANTICOPY_JACCARD, ARENA_ANTICOPY_SUBSTR_LEN, ARENA_ANTICOPY_SIMHASH_MAX
    global ARENA_USE_ANCHOR_VERBATIM, ARENA_QUALITY_MIN

    try:
        DEBUG_MODE = bool(getattr(cfg.debug, "debug", False))
    except Exception:
        DEBUG_MODE = False

    # language rules (Mersoom 3.0)
    try:
        lg = getattr(cfg, "lang", None)
        if lg is not None:
            LANG_STRICT = bool(getattr(lg, "strict", LANG_STRICT))
            ENGLISH_NOTICE = str(getattr(lg, "english_notice", ENGLISH_NOTICE) or ENGLISH_NOTICE)
            ENGLISH_NOTICE_RATIO = float(getattr(lg, "english_notice_ratio", ENGLISH_NOTICE_RATIO) or ENGLISH_NOTICE_RATIO)
            ENGLISH_NOTICE_APPEND_EUM = bool(getattr(lg, "english_notice_append_eum", ENGLISH_NOTICE_APPEND_EUM))
    except Exception as e:
        log_debug_exc("_apply_runtime_globals:lang", e)
        pass

    try:
        bb = cfg.brain_bias
        BRAIN_BIAS_ENABLE = bool(bb.enabled)
        BRAIN_BIAS_MIN = float(bb.min_mult)
        BRAIN_BIAS_MAX = float(bb.max_mult)
        if BRAIN_BIAS_MAX < BRAIN_BIAS_MIN:
            BRAIN_BIAS_MAX = BRAIN_BIAS_MIN
        BRAIN_BIAS_SCALE = float(bb.scale)
        BRAIN_BIAS_LOG = bool(bb.log)
    except Exception as e:
        log_debug_exc("_apply_runtime_globals:silent", e)
        pass

    try:
        ar = cfg.arena
        ARENA_ENABLE = bool(ar.enabled)
        ARENA_STATUS_MIN_INTERVAL_SEC = int(ar.status_min_interval_sec)
        ARENA_POSTS_MIN_INTERVAL_SEC = int(ar.posts_min_interval_sec)
        ARENA_MAX_ACTIONS_PER_DAY = int(ar.max_actions_per_day)

        ARENA_ARG_VARIANTS = int(ar.arg_variants)
        ARENA_REF_TOPK = int(ar.ref_topk)
        ARENA_ANTICOPY_JACCARD = float(ar.anticopy_jaccard)
        ARENA_ANTICOPY_SUBSTR_LEN = int(ar.anticopy_substr_len)
        ARENA_ANTICOPY_SIMHASH_MAX = int(ar.anticopy_simhash_max)
        ARENA_USE_ANCHOR_VERBATIM = bool(ar.use_anchor_verbatim)
        ARENA_QUALITY_MIN = int(ar.quality_min)
    except Exception as e:
        log_debug_exc("_apply_runtime_globals:silent", e)
        pass



def _apply_cli_overrides(argv: List[str]) -> None:
    """Very small CLI override layer (Unit 01).

    This keeps the single-file + env-driven architecture, but allows:
      - --dry-run
      - --qa-gate
      - --qa-batch
      - --qa-min-score=NN
      - --qa-batch-n=NN
      - --qa-batch-exit / --no-qa-batch-exit
      - --selftest / --no-selftest-exit
    """
    if not argv:
        return

    def _set(name: str, value: str) -> None:
        try:
            os.environ[name] = str(value)
        except Exception as e:
            log_debug_exc("_apply_cli_overrides:silent", e)
            pass

    for a in list(argv):
        s = str(a).strip()
        if not s:
            continue

        if s in ("--dry-run", "-n"):
            _set("MERSOOM_DRY_RUN", "true")
            continue

        if s == "--qa-gate":
            _set("MERSOOM_QA_GATE", "true")
            continue

        if s == "--qa-batch":
            _set("MERSOOM_QA_BATCH_ON_BOOT", "true")
            # safer default: exit after printing report unless user disables
            if os.getenv("MERSOOM_QA_BATCH_EXIT", "").strip() == "":
                _set("MERSOOM_QA_BATCH_EXIT", "true")
            # batch should not write by default
            if os.getenv("MERSOOM_DRY_RUN", "").strip() == "":
                _set("MERSOOM_DRY_RUN", "true")
            continue

        if s == "--qa-batch-exit":
            _set("MERSOOM_QA_BATCH_EXIT", "true")
            continue
        if s == "--no-qa-batch-exit":
            _set("MERSOOM_QA_BATCH_EXIT", "false")
            continue
        if s == "--selftest":
            _set("MERSOOM_SELFTEST", "true")
            if os.getenv("MERSOOM_SELFTEST_EXIT", "").strip() == "":
                _set("MERSOOM_SELFTEST_EXIT", "true")
            continue

        if s == "--no-selftest-exit":
            _set("MERSOOM_SELFTEST_EXIT", "false")
            continue


        if s.startswith("--qa-min-score="):
            _set("MERSOOM_QA_MIN_SCORE", s.split("=", 1)[1].strip())
            continue
        if s.startswith("--qa-batch-n="):
            _set("MERSOOM_QA_BATCH_N", s.split("=", 1)[1].strip())
            continue
        if s.startswith("--qa-save="):
            _set("MERSOOM_QA_BATCH_SAVE", s.split("=", 1)[1].strip())
            continue



################################################################################
# 2. LOGGING + TIME HELPERS
# - Dependencies: Section 1 (Config)
# - Used by: All sections
# - Key functions: log_info(), log_warn(), log_error(), now_kst_str(), sleep_chunked()
################################################################################

class Console:
    RESET = "\x1b[0m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    CYAN = "\x1b[36m"
    MAGENTA = "\x1b[35m"
    GRAY = "\x1b[90m"
    enabled = True

    @staticmethod
    def enable_windows_vt() -> None:
        if os.name != "nt":
            return
        try:
            import ctypes  # type: ignore
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
                return
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception as e:
            log_debug_exc("_apply_cli_overrides:silent", e)
            pass

    @staticmethod
    def cprint(color: str, text: str) -> None:
        if Console.enabled:
            print(f"{color}{text}{Console.RESET}")
        else:
            print(text)

Console.enable_windows_vt()
def _run_selftest() -> int:
    """Quick selftest suite (no external deps). Returns process exit code (0 OK, 1 FAIL)."""
    fails: List[str] = []

    # 1) Retry-After parsing
    try:
        h = {"Retry-After": "7"}
        ra = _parse_retry_after_sec(h)
        if ra is None or abs(float(ra) - 7.0) > 0.001:
            fails.append("retry_after_numeric")
    except Exception as e:
        fails.append(f"retry_after_numeric_exc:{type(e).__name__}")

    # 2) SlidingWindowLimiter
    try:
        lim = SlidingWindowLimiter(capacity=2, window_sec=10)
        if not lim.allow():
            fails.append("lim_allow_1")
        if not lim.allow():
            fails.append("lim_allow_2")
        if lim.allow():
            fails.append("lim_allow_over")
    except Exception as e:
        fails.append(f"lim_exc:{type(e).__name__}")

    # 3) Eum ending insertion with punctuation (avoid '?임/!임')
    try:
        s = ensure_eum_style("뭐함?", max_lines=2)
        if "?임" in s or "!임" in s:
            fails.append("eum_punc_insert")
    except Exception as e:
        fails.append(f"eum_exc:{type(e).__name__}")

    # 4) Recent FP memory
    try:
        st: Dict[str, Any] = {}
        remember_fp(st, "abc", for_post=False, ttl_sec=60, keep_max=10)
        if not recently_used_fp(st, "abc", for_post=False, ttl_sec=60, keep_max=10):
            fails.append("fp_recent")
    except Exception as e:
        fails.append(f"fp_exc:{type(e).__name__}")

    # 5) Vote backlog enqueue/pick/gc sanity (v20.2)
    _env_backup = {}
    try:
        _env_backup = {k: os.environ.get(k) for k in [
            "MERSOOM_VOTE_BACKLOG_KEEP_MAX",
            "MERSOOM_VOTE_BACKLOG_MAX",
            "MERSOOM_VOTE_BACKLOG_TTL_SEC",
        ]}
        os.environ["MERSOOM_VOTE_BACKLOG_KEEP_MAX"] = "3"
        os.environ["MERSOOM_VOTE_BACKLOG_MAX"] = "3"
        os.environ["MERSOOM_VOTE_BACKLOG_TTL_SEC"] = "3600"

        st2: Dict[str, Any] = {}
        migrate_state(st2)

        now = time.time()
        vote_backlog_enqueue(st2, "p1", now - 120.0)
        vote_backlog_enqueue(st2, "p2", now - 60.0)
        vote_backlog_enqueue(st2, "p3", now - 30.0)

        vote_backlog_gc(st2, now_ts=now)
        pid = vote_backlog_pick(st2)
        if pid != "p1":
            fails.append("vote_backlog_pick_oldest")

        # mark p1 as voted -> GC should remove + record drain
        posts_map = _voted_posts_map(st2)
        posts_map["p1"] = {"ts": now}
        vote_backlog_gc(st2, now_ts=now)
        pid2 = vote_backlog_pick(st2)
        if pid2 == "p1":
            fails.append("vote_backlog_gc_remove_voted")
        drains = _safe_list(_safe_dict(st2.get("protocol")).get("vote_backlog_drains"))
        if len(drains) <= 0:
            fails.append("vote_backlog_drain_record")

        # cap test: keep_max=1 should retain newest after GC
        os.environ["MERSOOM_VOTE_BACKLOG_KEEP_MAX"] = "1"
        os.environ["MERSOOM_VOTE_BACKLOG_MAX"] = "1"
        vote_backlog_gc(st2, now_ts=now)
        bl = _safe_list(_safe_dict(st2.get("protocol")).get("vote_backlog"))
        if len(bl) != 1:
            fails.append("vote_backlog_cap_len")
    except Exception as e:
        fails.append(f"vote_backlog_exc:{type(e).__name__}")
    finally:
        for k, v in _env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # 6) Template cooldown penalty sanity (v20.4)
    _env_backup = {}
    try:
        _env_backup = {k: os.environ.get(k) for k in [
            "MERSOOM_TEMPLATE_COOLDOWN_SEC",
            "MERSOOM_TEMPLATE_COOLDOWN_PENALTY",
            "MERSOOM_EXPLORATION_RATE",
        ]}
        os.environ["MERSOOM_TEMPLATE_COOLDOWN_SEC"] = "3600"
        os.environ["MERSOOM_TEMPLATE_COOLDOWN_PENALTY"] = "0.0"
        os.environ["MERSOOM_EXPLORATION_RATE"] = "0.0"

        pol: Dict[str, Any] = {"templates": {"items": {}, "quality": {"min_pick_score": 0}}}
        now = time.time()
        pol["templates"]["items"]["A"] = {
            "text": "A",
            "weight": 1e9,
            "static_score": 80,
            "created_ts": now - 7 * 24 * 3600,
            "last_used_ts": now - 10.0,  # within cooldown
        }
        pol["templates"]["items"]["B"] = {
            "text": "B",
            "weight": 1.0,
            "static_score": 80,
            "created_ts": now - 7 * 24 * 3600,
            "last_used_ts": 0.0,
        }

        random.seed(123)
        tid = pick_template_id(pol)
        if tid != "B":
            fails.append("template_cooldown_penalty")

        # out of cooldown -> A should dominate
        pol["templates"]["items"]["A"]["last_used_ts"] = now - 7200.0
        random.seed(123)
        tid2 = pick_template_id(pol)
        if tid2 != "A":
            fails.append("template_pick_pref")
    except Exception as e:
        fails.append(f"template_cooldown_exc:{type(e).__name__}")
    finally:
        for k, v in _env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # 7) Near-dup similarity sanity (Jaccard + 3-gram signatures) (v20.4)
    try:
        a = [1, 2, 3]
        b = [1, 2, 3]
        c = [9, 10]
        if _jaccard_ratio(a, b) < 0.99:
            fails.append("jaccard_identical_low")
        if _jaccard_ratio(a, c) > 0.2:
            fails.append("jaccard_disjoint_high")

        g1 = _sig_3grams("이건 테스트 문장임", max_ngrams=128)
        g2 = _sig_3grams("이건 테스트 문장임", max_ngrams=128)
        g3 = _sig_3grams("완전히 다른 내용임", max_ngrams=128)
        if g1 and g2 and _jaccard_ratio(g1, g2) < 0.9:
            fails.append("3gram_identical_low")
        if g1 and g3 and _jaccard_ratio(g1, g3) > 0.7:
            fails.append("3gram_different_high")
    except Exception as e:
        fails.append(f"dup_sim_exc:{type(e).__name__}")

    # 8) Reward finiteness/clip sanity (v20.5)
    try:
        tng = load_tuning_from_env()
        before = {"up": 0, "down": 0, "comments": 0, "score": 0}
        after = {"up": 10**9, "down": 10**9, "comments": 10**9, "score": -10**9}
        r, _feats = compute_reward(tng, before, after, {"action": "post", "action_type": "post_main", "reply_received": 1e9})
        if not math.isfinite(float(r)):
            fails.append("reward_not_finite")
        clip = float(getattr(tng, "reward_clip", 3.0) or 3.0)
        if abs(float(r)) > (clip + 1e-6):
            fails.append("reward_not_clipped")
    except Exception as e:
        fails.append(f"reward_exc:{type(e).__name__}")

    if fails:
        log_error("selftest", "FAIL: " + ", ".join(fails))
        return 1
    log_info("selftest: OK")
    return 0


def now_kst() -> datetime:
    global _TZ_LOGGED
    if not _TZ_LOGGED:
        try:
            log_debug(f"timezone={TZ_NAME} fallback={int(bool(TZ_FALLBACK_USED))}")
        except Exception:
            pass
        _TZ_LOGGED = True
    return datetime.now(KST)

def now_kst_str() -> str:
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")

def _today_kst() -> str:
    return now_kst().strftime("%Y-%m-%d")

def one_line(text: Any, max_len: int = 140) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(t) > max_len:
        t = t[:max_len].rstrip() + "…"
    return t

def log_info(msg: str) -> None:
    Console.cprint(Console.GRAY, f"[INFO] {now_kst_str()} | {msg}")

def log_warn(msg: str) -> None:
    Console.cprint(Console.YELLOW, f"[WARN] {now_kst_str()} | {msg}")

def log_event(name: str, **fields: Any) -> None:
    """Emit a single-line, grep-friendly event log.

    Format: [EVT] <KST timestamp> | <event_name> | <compact json>
    - Never raises.
    """
    try:
        # Normalize fields to keep output stable and one-line.
        norm: Dict[str, Any] = {}
        for k, v in (fields or {}).items():
            kk = str(k)
            # Keep simple scalars; stringify the rest (but keep dict/list where possible).
            if v is None or isinstance(v, (str, int, float, bool)):
                norm[kk] = v
            elif isinstance(v, (list, dict)):
                norm[kk] = v
            else:
                norm[kk] = one_line(v, 220)
        payload = json.dumps(norm, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        Console.cprint(Console.CYAN, f"[EVT] {now_kst_str()} | {str(name)} | {payload}")
    except Exception:
        return


def log_error(where: str, err: str) -> None:
    Console.cprint(Console.RED, f"[ERROR] {now_kst_str()} | {where} :: {err}")


def log_debug(msg: Any) -> None:
    """Debug log (printed only when DEBUG_MODE is enabled).

    - Accepts either a string, or a zero-arg callable returning a string (lazy formatting).
    """
    try:
        if not DEBUG_MODE:
            return
        if callable(msg):
            try:
                msg = msg()
            except Exception as e:
                msg = f"<log_debug thunk failed: {type(e).__name__}>"
        Console.cprint(Console.GRAY, f"[DBG ] {now_kst_str()} | {msg}")
    except Exception:
        pass


def log_debug_exc(where: str, e: BaseException) -> None:
    """Emit exception details only when DEBUG_MODE is enabled (to avoid spam).

    Important: this function must never recurse on failure.
    """
    try:
        if DEBUG_MODE:
            log_warn(f"{where}: {type(e).__name__}: {one_line(str(e), 220)}")
    except Exception:
        # Absolute last-resort: never recurse; avoid any complex formatting.
        try:
            if DEBUG_MODE:
                import sys as _sys
                _sys.stderr.write("[WARN] log_debug_exc failed while logging an exception\n")
        except Exception:
            pass

def log_action(tag: str, msg: str) -> None:
    Console.cprint(Console.GREEN, f"[{tag}] {now_kst_str()} | {msg}")

def log_sleep(sec: float, why: str = "") -> None:
    extra = f" {why}" if why else ""
    Console.cprint(Console.YELLOW, f"[SLEEP] {int(sec)}s{extra}")

def sleep_chunked(
    total_sec: float,
    *,
    hard_cap_sec: int,
    why: str = "",
    wake_deadline_wall_ts: Optional[float] = None,
) -> bool:
    """Sleep for up to total_sec seconds, but never past wake_deadline_wall_ts (epoch seconds).

    Returns True if a deadline caused an early wake-up.
    """
    remaining = max(0.0, float(total_sec))
    if remaining <= 0:
        return False

    interrupted = False

    # keep logs roughly consistent even when a deadline truncates the actual sleep
    if wake_deadline_wall_ts is not None:
        until_deadline = float(wake_deadline_wall_ts) - time.time()
        if until_deadline <= 0:
            return True
        remaining = min(remaining, max(0.0, until_deadline))

    log_sleep(remaining, why)

    while remaining > 0:
        chunk = min(remaining, float(hard_cap_sec))

        if wake_deadline_wall_ts is not None:
            until_deadline = float(wake_deadline_wall_ts) - time.time()
            if until_deadline <= 0:
                interrupted = True
                break
            chunk = min(chunk, until_deadline)

        if chunk <= 0:
            interrupted = True
            break

        time.sleep(chunk)
        remaining -= chunk

    return interrupted

def human_delay(min_sec: float, max_sec: float) -> None:
    if max_sec <= 0:
        return
    lo = max(0.0, float(min_sec))
    hi = max(lo, float(max_sec))
    time.sleep(random.uniform(lo, hi))

def next_top_of_hour_kst(dt: Optional[datetime] = None) -> datetime:
    t = dt or now_kst()
    return t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

def run_snapshot_script(script_path: str, timeout_sec: int) -> Tuple[int, str]:
    """Run a snapshot script and return (exit_code, combined_output)."""
    try:
        res = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.abspath(script_path)) or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(5, int(timeout_sec)),
            check=False,
        )
        out = res.stdout or ""
        return int(res.returncode), out
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\n[TIMEOUT]"
        return 124, out

################################################################################
# 3. STORAGE (atomic JSON/TXT, safe load)
# - Dependencies: Section 1-3 (Config, Logging, Storage)
# - Used by: All sections
# - Key functions: load_json_file(), save_json_file_atomic(), append_jsonl()
################################################################################

def load_json_file(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        # (P2) If JSON is corrupted (manual edit / partial write), back it up and continue with default.
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = f"{path}.corrupt.{ts}"
            if not os.path.exists(bak):
                os.replace(path, bak)
            log_warn(f"JSON load failed; moved to backup: {bak} ({one_line(repr(e), 160)})")
        except Exception:
            try:
                log_warn(f"JSON load failed: {path} ({one_line(repr(e), 160)})")
            except Exception as e:
                log_debug_exc("load_json_file:silent", e)
                pass
        return default

def _atomic_replace(tmp_path: str, final_path: str) -> None:
    os.replace(tmp_path, final_path)

def save_text_file_atomic(path: str, text: str, *, fsync: bool = False) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        if fsync:
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                log_debug_exc("save_text_file_atomic:silent", e)
                pass
    _atomic_replace(tmp, path)

def save_json_file_atomic(path: str, obj: Any, *, fsync: bool = False) -> None:
    if not path:
        return
    tmp = path + ".tmp"

    # v21.1: configurable JSON formatting (keeps prior default indent=2 unless explicitly compacted)
    compact = _env_bool("MERSOOM_JSON_COMPACT", False)
    indent = _env_int("MERSOOM_JSON_INDENT", 2, 0, 8)
    if compact:
        js_indent = None
        js_separators = (",", ":")
    else:
        js_indent = (indent if indent > 0 else None)
        js_separators = None

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=js_indent, separators=js_separators)
        if fsync:
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                log_debug_exc("save_json_file_atomic:silent", e)
                pass
    _atomic_replace(tmp, path)



def _apply_stamp(obj: Any, stamp: Dict[str, Any]) -> None:
    """Best-effort attach a generation stamp to a dict under '__meta__'."""
    try:
        if isinstance(obj, dict):
            m = obj.get("__meta__")
            if not isinstance(m, dict):
                m = {}
                obj["__meta__"] = m
            m.update(_safe_dict(stamp))
    except Exception as e:
        log_debug_exc("_apply_stamp:silent", e)
        pass

def _make_tick_stamp(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return {
            "tick_id": int(state.get("tick_id", 0) or 0),
            "ts": float(time.time()),
            "ts_kst": now_kst_str(),
            "ver": AGENT_CODE_VERSION,
        }
    except Exception:
        return {"tick_id": 0, "ts": float(time.time()), "ts_kst": now_kst_str(), "ver": AGENT_CODE_VERSION}

def _meta_tick_id(obj: Any) -> int:
    try:
        if not isinstance(obj, dict):
            return 0
        m = obj.get("__meta__")
        if not isinstance(m, dict):
            return 0
        return int(m.get("tick_id") or 0)
    except Exception:
        return 0

def _meta_ts(obj: Any) -> float:
    try:
        if not isinstance(obj, dict):
            return 0.0
        m = obj.get("__meta__")
        if not isinstance(m, dict):
            return 0.0
        return float(m.get("ts") or 0.0)
    except Exception:
        return 0.0


def append_text_file(path: str, text: str) -> None:
    if not path:
        return
    try:
        # (Unit 13) journal/append reliability: flush each write; optional fsync.
        line_buffered = str(os.getenv("MERSOOM_JOURNAL_LINE_BUFFERED", "true")).strip().lower() in ("1", "true", "yes", "y")
        do_fsync = str(os.getenv("MERSOOM_JOURNAL_FSYNC", "false")).strip().lower() in ("1", "true", "yes", "y")
        buffering = 1 if line_buffered else -1
        with open(path, "a", encoding="utf-8", buffering=buffering) as f:
            f.write(text)
            try:
                f.flush()
            except Exception as e:
                log_debug_exc("append_text_file:silent", e)
                pass
            if do_fsync:
                try:
                    os.fsync(f.fileno())
                except Exception as e:
                    log_debug_exc("append_text_file:silent", e)
                    pass
    except Exception as e:
        log_debug_exc("append_text_file:silent", e)
        pass

# v21.1: optional JSONL buffering (disabled by default; enable via MERSOOM_JSONL_BUFFER_MAX > 0)
_JSONL_BUFFERS: Dict[str, List[str]] = {}
_JSONL_LAST_FLUSH_TS: Dict[str, float] = {}
try:
    atexit.register(lambda: [_flush_jsonl_buffer(p, force=True) for p in list(_JSONL_BUFFERS.keys())])
except Exception:
    pass


def _flush_jsonl_buffer(path: str, *, force: bool = False) -> None:
    if not path:
        return
    buf = _JSONL_BUFFERS.get(path)
    if not buf:
        return
    try:
        flush_sec = float(_env_float("MERSOOM_JSONL_FLUSH_SEC", 2.0, 0.0, 60.0))
        now = time.time()
        last = float(_JSONL_LAST_FLUSH_TS.get(path, 0.0) or 0.0)
        if (not force) and flush_sec > 0.0 and (now - last) < flush_sec and len(buf) < max(1, _env_int("MERSOOM_JSONL_BUFFER_MAX", 0, 0, 10000)):
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(buf))
        buf.clear()
        _JSONL_LAST_FLUSH_TS[path] = now
    except Exception as e:
        log_debug_exc("_flush_jsonl_buffer:silent", e)
        pass

def append_jsonl(path: str, obj: Any) -> None:
    if not path:
        return
    try:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
        max_buf = _env_int("MERSOOM_JSONL_BUFFER_MAX", 0, 0, 10000)
        if max_buf <= 0:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            return

        buf = _JSONL_BUFFERS.setdefault(path, [])
        buf.append(line)
        # Safety: if flush keeps failing, keep buffer bounded to avoid unbounded RAM growth
        if len(buf) > max_buf * 2:
            del buf[:-max_buf]
        if len(buf) >= max_buf:
            _flush_jsonl_buffer(path, force=True)
        else:
            _flush_jsonl_buffer(path, force=False)
    except Exception as e:
        log_debug_exc("append_jsonl:silent", e)
        pass


################################################################################
# 3.1. OPS GUARDS (lock + circuit breaker)
# - Dependencies: Section 1-3 (Config, Logging, Storage)
# - Used by: Main loop / stateful updates
# - Key functions: acquire_process_lock(), _ops_init(), ops_should_skip(), ops_record_fail()
################################################################################

def _resolve_lock_path(state_path: str) -> str:
    # Prefer explicit env; otherwise keep lock next to state file.
    v = os.getenv("MERSOOM_LOCK")
    if v:
        return str(v)
    try:
        d = os.path.dirname(os.path.abspath(state_path or "")) or "."
    except Exception:
        d = "."
    return os.path.join(d, "mersoom.lock")

def acquire_process_lock(lock_path: str) -> None:
    """Best-effort single instance lock. Exits if lock exists (unless forced)."""
    if not lock_path:
        return

    force = str(os.getenv("MERSOOM_LOCK_FORCE", "false")).strip().lower() in ("1", "true", "yes", "y")
    stale_sec = _env_int("MERSOOM_LOCK_STALE_SEC", 8 * 3600, 60, 7 * 24 * 3600)

    def _try_remove_stale() -> None:
        try:
            if stale_sec <= 0:
                return
            if not os.path.exists(lock_path):
                return
            age = time.time() - float(os.path.getmtime(lock_path))
            if age >= float(stale_sec):
                os.remove(lock_path)
        except Exception as e:
            log_debug_exc("acquire_process_lock:silent", e)
            pass

    if force:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception as e:
            log_debug_exc("acquire_process_lock:silent", e)
            pass
    else:
        _try_remove_stale()

    flags = getattr(os, "O_CREAT", 0) | getattr(os, "O_EXCL", 0) | getattr(os, "O_WRONLY", 0)
    try:
        fd = os.open(lock_path, flags)
    except FileExistsError:
        Console.cprint(Console.RED, f"[LOCK] already running (lock exists): {lock_path}")
        Console.cprint(Console.RED, "If you are sure it's stale, set MERSOOM_LOCK_FORCE=true or delete the lock file.")
        raise SystemExit(2)
    except Exception as e:
        # If lock fails unexpectedly, continue (best-effort).
        log_warn(f"lock unavailable: {one_line(repr(e), 120)}")
        return

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\n")
            f.write(f"started_kst={now_kst_str()}\n")
            try:
                f.flush()
            except Exception as e:
                log_debug_exc("acquire_process_lock:silent", e)
                pass
    except Exception as e:
        log_debug_exc("acquire_process_lock:silent", e)
        pass

    def _cleanup() -> None:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except Exception as e:
            log_debug_exc("acquire_process_lock:silent", e)
            pass

    atexit.register(_cleanup)

def _ops_init(state: Dict[str, Any]) -> Dict[str, Any]:
    ops = state.get("_ops")
    if not isinstance(ops, dict):
        ops = {}
        state["_ops"] = ops
    ops.setdefault("fail_counts", {})
    ops.setdefault("disabled_until", {})
    ops.setdefault("last_fail_reason", {})
    return ops

def ops_should_skip(state: Dict[str, Any], key: str) -> bool:
    ops = _ops_init(state)
    du = ops.get("disabled_until")
    if not isinstance(du, dict):
        du = {}
        ops["disabled_until"] = du
    until = float(du.get(key, 0.0) or 0.0)
    now_ts = time.time()
    if until > 0.0 and now_ts >= until:
        # Expired -> clear and emit a single enable event
        try:
            du.pop(key, None)
        except Exception:
            pass
        try:
            # keep last_fail_reason for post-mortem unless explicitly cleared elsewhere
            log_event("ops_enabled", key=str(key))
        except Exception:
            pass
        return False
    return now_ts < until

def ops_record_success(state: Dict[str, Any], key: str) -> None:
    ops = _ops_init(state)
    fc = ops.get("fail_counts")
    if isinstance(fc, dict):
        fc[key] = 0
    lfr = ops.get("last_fail_reason")
    if isinstance(lfr, dict):
        lfr.pop(key, None)

def ops_record_failure(state: Dict[str, Any], key: str, reason: str) -> None:
    ops = _ops_init(state)
    thr = _env_int("MERSOOM_OPS_FAIL_THRESHOLD", 5, 1, 100)
    base_disable = _env_int("MERSOOM_OPS_DISABLE_SEC", 900, 60, 24 * 3600)
    per_key = {
        "sync": _env_int("MERSOOM_OPS_DISABLE_SEC_SYNC", base_disable, 60, 24 * 3600),
        "vote": _env_int("MERSOOM_OPS_DISABLE_SEC_VOTE", base_disable, 60, 24 * 3600),
        "contrib": _env_int("MERSOOM_OPS_DISABLE_SEC_CONTRIB", base_disable, 60, 24 * 3600),
        "arena": _env_int("MERSOOM_OPS_DISABLE_SEC_ARENA", max(base_disable, 1800), 60, 24 * 3600),
    }
    disable_sec = int(per_key.get(key, base_disable))

    fc = ops.get("fail_counts")
    if not isinstance(fc, dict):
        fc = {}
        ops["fail_counts"] = fc
    fc[key] = int(fc.get(key, 0) or 0) + 1

    lfr = ops.get("last_fail_reason")
    if not isinstance(lfr, dict):
        lfr = {}
        ops["last_fail_reason"] = lfr
    lfr[key] = one_line(reason or "", 200)

    if int(fc.get(key, 0) or 0) >= int(thr):
        du = ops.get("disabled_until")
        if not isinstance(du, dict):
            du = {}
            ops["disabled_until"] = du
        du[key] = time.time() + float(disable_sec)
        log_event("ops_disabled", key=str(key), disable_sec=int(disable_sec), threshold=int(thr), reason=one_line(reason, 200))
        log_warn(f"OPS: {key} disabled for {disable_sec}s after {thr} consecutive fails. last={one_line(reason, 120)}")
        fc[key] = 0  # reset after tripping


def ops_force_disable(state: Dict[str, Any], key: str, disable_sec: int, *, reason: str = "") -> None:
    """Immediately disable an ops key (without waiting for fail threshold)."""
    ops = _ops_init(state)
    du = ops.get("disabled_until")
    if not isinstance(du, dict):
        du = {}
        ops["disabled_until"] = du
    du[str(key)] = time.time() + float(int(disable_sec))

    lfr = ops.get("last_fail_reason")
    if not isinstance(lfr, dict):
        lfr = {}
        ops["last_fail_reason"] = lfr
    if reason:
        lfr[str(key)] = one_line(reason, 200)

    log_event("ops_disabled", key=str(key), disable_sec=int(disable_sec), forced=True, reason=one_line(reason, 200))

    log_warn(f"OPS: {key} force-disabled for {int(disable_sec)}s. last={one_line(reason, 140)}")

def _normalize_endpoint_key(path: str) -> str:
    """Normalize API path to a coarse key (replace ids) for auth-failure tracking."""
    p = str(path or "")
    p = p.split("?", 1)[0]
    parts = [x for x in p.split("/") if x]
    norm: List[str] = []
    for seg in parts:
        s = str(seg)
        if len(s) >= 8 and re.fullmatch(r"[A-Za-z0-9_-]+", s or "") and (any(c.isdigit() for c in s) or s.lower() != s):
            norm.append("{id}")
        else:
            norm.append(s)
    return "/" + "/".join(norm)

def record_auth401_and_maybe_trip(state: Dict[str, Any], path: str, *, where: str = "") -> bool:
    """Track repeated 401s per endpoint; optionally trip contrib circuit."""
    try:
        window = _env_int("MERSOOM_AUTH401_WINDOW_SEC", 900, 60, 24 * 3600)
        trip = _env_int("MERSOOM_AUTH401_TRIP_COUNT", 3, 1, 50)
        disable = _env_int("MERSOOM_AUTH401_DISABLE_SEC", 3600, 60, 24 * 3600)
        key = _normalize_endpoint_key(path)
        now = time.time()

        d = state.setdefault("auth401", {})
        if not isinstance(d, dict):
            d = {}
            state["auth401"] = d

        arr = _safe_list(d.get(key))
        arr2: List[float] = []
        for x in arr:
            try:
                t = float(x)
                if (now - t) <= float(window):
                    arr2.append(t)
            except Exception:
                continue
        arr2.append(now)
        d[key] = arr2[-60:]

        if len(arr2) >= int(trip):
            ops_force_disable(state, "contrib", int(disable), reason=f"auth401 {key} x{len(arr2)} {where}".strip())
            d[key] = []  # reset window for this endpoint
            return True
    except Exception as e:
        log_debug_exc("record_auth401_and_maybe_trip:silent", e)
        pass
    return False

def ops_disabled_keys(state: Dict[str, Any]) -> List[str]:
    ops = _ops_init(state)
    du = ops.get("disabled_until")
    if not isinstance(du, dict):
        return []
    now = time.time()
    out = []
    for k, v in du.items():
        try:
            if float(v or 0.0) > now:
                out.append(str(k))
        except Exception:
            continue
    return sorted(out)

################################################################################
# 4. LIMITERS + PACING
# - Dependencies: Section 1-2 (Config, Logging)
# - Used by: HTTP client + main loop pacing
# - Key functions: SlidingWindowLimiter.allow()
################################################################################

class SlidingWindowLimiter:
    """Sliding window limiter based on monotonic time (robust to wall clock skew)."""
    def __init__(self, capacity: int, window_sec: int):
        self.capacity = max(0, int(capacity))
        self.window_sec = max(1, int(window_sec))
        self.q: deque[float] = deque()

    def _gc(self) -> None:
        now = time.monotonic()
        while self.q and (now - self.q[0]) > self.window_sec:
            self.q.popleft()

    def remaining(self) -> int:
        self._gc()
        if self.capacity <= 0:
            return 0
        return max(0, self.capacity - len(self.q))

    def allow(self) -> bool:
        self._gc()
        if self.capacity <= 0:
            return False
        if len(self.q) < self.capacity:
            self.q.append(time.monotonic())
            return True
        return False

    def seconds_until_next(self) -> int:
        """
        Returns seconds until at least one slot is available.
        - capacity <= 0 => effectively blocked
        - empty queue => wait window_sec (defensive)
        """
        self._gc()
        if self.capacity <= 0:
            return 10**9
        if len(self.q) < self.capacity:
            return 0
        if not self.q:
            return self.window_sec
        oldest = self.q[0]
        wait = self.window_sec - (time.monotonic() - oldest)
        return max(0, int(math.ceil(wait)))

def pace_interval(capacity: int, window_sec: int, floor_sec: int, activity_mode: str) -> int:
    """
    paced: distribute evenly => ceil(window/capacity), but never below floor_sec
    burst: just obey floor_sec
    capacity<=0 => effectively infinite
    """
    if capacity <= 0:
        return 10**9
    base = int(math.ceil(int(window_sec) / max(1, int(capacity))))
    if activity_mode == "burst":
        return max(0, int(floor_sec))
    return max(int(floor_sec), base)

def gap_remaining(last_ts_wall: float, min_gap_sec: int) -> int:
    """last_ts_wall is wall time (time.time()) stored in state."""
    if last_ts_wall <= 0 or min_gap_sec <= 0:
        return 0
    return max(0, int(min_gap_sec - (time.time() - float(last_ts_wall))))

def catchup_budget(last_ts_wall: float, pace_sec: int, max_per_tick: int) -> int:
    """If we've been idle longer than pace, allow limited catch-up within max_per_tick."""
    if max_per_tick <= 0:
        return 0
    if pace_sec <= 0:
        return max_per_tick
    if last_ts_wall <= 0:
        return max_per_tick
    lag = (time.time() - float(last_ts_wall)) / float(pace_sec)
    want = int(min(max_per_tick, max(1, math.floor(lag))))
    return max(1, want)

################################################################################
# 5. HTTP CLIENT (retry/backoff/jitter, error classify)
# - Dependencies: Section 1-2, 4 (Config, Logging, Limiters)
# - Used by: API wrappers + agent actions
# - Key functions: HttpClient.request(), post_with_pow()
################################################################################

@dataclass
class HttpResult:
    status_code: int
    json: Optional[Any]
    text: str

class RateLimitError(Exception):
    def __init__(self, message: str, retry_after_sec: Optional[float] = None):
        super().__init__(message)
        self.retry_after_sec = retry_after_sec

class PowTimeoutError(Exception):
    pass

def _parse_retry_after_sec(headers: Any) -> Optional[float]:
    """Parse Retry-After header (seconds or HTTP date). Returns seconds to wait."""
    try:
        if not headers:
            return None
        ra = None
        try:
            ra = headers.get("Retry-After")
        except Exception:
            ra = None
        if ra is None:
            return None
        s = str(ra).strip()
        if not s:
            return None
        # numeric seconds
        if re.fullmatch(r"\d+", s):
            return float(int(s))
        # HTTP-date format
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(s)
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return None
    except Exception:
        return None

class HttpClient:
    def __init__(self, cfg: HttpConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": cfg.user_agent,
            "Accept": "application/json",
        })

        # v21.1: connection pool tuning (safe defaults; override via env)
        try:
            from requests.adapters import HTTPAdapter
            pool_conns = _env_int("MERSOOM_HTTP_POOL_CONNS", 10, 1, 200)
            pool_max = _env_int("MERSOOM_HTTP_POOL_MAX", 20, 1, 500)
            adapter = HTTPAdapter(pool_connections=pool_conns, pool_maxsize=pool_max, max_retries=0)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        except Exception as e:
            log_debug_exc("http_adapter:silent", e)
            pass


        # (P0) minimal observability / health metrics
        self.metrics: Dict[str, Any] = {
            "req_total": 0,
            "req_ok": 0,
            "rate_limited": 0,
            "http_errors": 0,
            "net_errors": 0,
            "last_status": None,
            "last_error": "",
            "latency_ms_ema": 0.0,
            "latency_ms_last": 0.0,
            "last_ok_ts": 0.0,

        }

        # v23.9: rolling 10m observability for 429 + Retry-After sleeps (in-memory only; not persisted)
        self._recent_429_ts = deque(maxlen=4000)  # timestamps of 429 responses
        self._recent_retry_after_sleep_ts = deque(maxlen=4000)  # timestamps when we honored Retry-After sleeps
        self._recent_http_401_ts = deque(maxlen=2000)
        self._recent_http_5xx_ts = deque(maxlen=4000)
        self._last_retry_after_raw_sec: float = 0.0
        self._last_retry_after_sleep_sec: float = 0.0
        self._last_retry_after_basis: str = ""  # "retry-after" | "backoff" | ""


    def _timeout(self) -> Tuple[float, float]:
        return (float(self.cfg.timeout_connect_sec), float(self.cfg.timeout_read_sec))

    @staticmethod
    def _is_validation_status(code: int) -> bool:
        return code in (400, 409, 422)

    @staticmethod
    def _is_rate_limited(code: int) -> bool:
        return code == 429

    @staticmethod
    def _is_retryable_5xx(code: int) -> bool:
        return 500 <= code <= 599

    def _backoff_sleep(self, attempt: int) -> None:
        base = float(self.cfg.backoff_base)
        cap = float(self.cfg.backoff_cap)
        t = min(cap, base * (2 ** max(0, attempt - 1))) * random.uniform(0.7, 1.3)
        time.sleep(max(0.0, t))

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        allow_retry: bool = True,
    ) -> HttpResult:
        url = f"{self.cfg.base_url}{path}"

        if self.cfg.dry_run and method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
            return HttpResult(
                status_code=200,
                json={"dry_run": True, "method": method, "url": url, "payload": json_payload},
                text="",
            )

        last_text = ""
        for attempt in range(1, int(self.cfg.max_retries) + 2):  # retries + 1 initial
            try:
                t0 = time.perf_counter()
                try:
                    self.metrics["req_total"] = int(self.metrics.get("req_total", 0)) + 1
                except Exception as e:
                    log_debug_exc("_parse_retry_after_sec:silent", e)
                    pass
                r = self.session.request(
                    method=method.upper(),
                    url=url,
                    headers=headers or {},
                    params=params or {},
                    json=json_payload,
                    timeout=self._timeout(),
                    allow_redirects=True,
                )
                last_text = r.text or ""
                code = int(r.status_code)

                lat_ms = (time.perf_counter() - t0) * 1000.0
                try:
                    self.metrics["latency_ms_last"] = float(lat_ms)
                    ema = float(self.metrics.get("latency_ms_ema", 0.0) or 0.0)
                    self.metrics["latency_ms_ema"] = (0.2 * float(lat_ms)) + (0.8 * ema if ema > 0 else 0.8 * float(lat_ms))
                    self.metrics["last_status"] = int(code)
                except Exception as e:
                    log_debug_exc("_parse_retry_after_sec:silent", e)
                    pass

                if self._is_rate_limited(code):
                    ra = _parse_retry_after_sec(getattr(r, "headers", None))
                    try:
                        self._recent_429_ts.append(time.time())
                        if ra is not None:
                            self._last_retry_after_raw_sec = float(ra)
                    except Exception:
                        pass
                    raise RateLimitError(f"429 rate limited: {last_text[:200]}", retry_after_sec=ra)
                if code == 401:
                    try:
                        self._recent_http_401_ts.append(time.time())
                    except Exception:
                        pass
                if self._is_retryable_5xx(code):
                    try:
                        self._recent_http_5xx_ts.append(time.time())
                    except Exception:
                        pass

                if self._is_validation_status(code):
                    j = None
                    try:
                        j = r.json() if last_text else None
                    except Exception:
                        j = None
                    return HttpResult(status_code=code, json=j, text=last_text)

                if allow_retry and self.cfg.retry_on_5xx and self._is_retryable_5xx(code):
                    if attempt <= int(self.cfg.max_retries):
                        self._backoff_sleep(attempt)
                        continue

                r.raise_for_status()
                if code < 400:
                    try:
                        self.metrics["req_ok"] = int(self.metrics.get("req_ok", 0)) + 1
                        self.metrics["last_ok_ts"] = time.time()
                    except Exception as e:
                        log_debug_exc("_parse_retry_after_sec:silent", e)
                        pass
                if not last_text:
                    return HttpResult(status_code=code, json=None, text="")
                try:
                    return HttpResult(status_code=code, json=r.json(), text=last_text)
                except Exception:
                    return HttpResult(status_code=code, json=None, text=last_text)

            except RateLimitError as e_rl:
                try:
                    self.metrics["rate_limited"] = int(self.metrics.get("rate_limited", 0)) + 1
                    self.metrics["last_status"] = 429
                    self.metrics["last_error"] = one_line(str(e_rl), 200)
                except Exception as e:
                    log_debug_exc("_parse_retry_after_sec:silent", e)
                    pass
                if allow_retry and attempt <= int(self.cfg.max_retries):
                    ra = getattr(e_rl, 'retry_after_sec', None)
                    if ra is not None and float(ra) > 0:
                        # v23.9: Honor Retry-After when present, with a safety cap + small jitter.
                        cap_ra = 900.0
                        raw = float(ra)
                        t = min(raw, cap_ra)
                        t = t * random.uniform(0.95, 1.05)
                        t = min(t, cap_ra)
                        try:
                            self._last_retry_after_raw_sec = float(raw)
                            self._last_retry_after_sleep_sec = float(t)
                            self._last_retry_after_basis = "retry-after"
                            self._recent_retry_after_sleep_ts.append(time.time())
                        except Exception:
                            pass
                        # Record basis clearly for operators; keep it short.
                        try:
                            if raw > cap_ra:
                                log_warn(f"http.429 retry-after capped: raw={raw:.0f}s cap={cap_ra:.0f}s sleep={t:.0f}s attempt={attempt} path={path}")
                            else:
                                log_warn(f"http.429 retry-after: sleep={t:.0f}s attempt={attempt} path={path}")
                        except Exception:
                            pass
                        time.sleep(max(0.0, t))
                    else:
                        try:
                            self._last_retry_after_basis = "backoff"
                        except Exception:
                            pass
                        self._backoff_sleep(attempt)
                    continue
                raise
            except (requests.Timeout, requests.ConnectionError) as e:
                try:
                    self.metrics["net_errors"] = int(self.metrics.get("net_errors", 0)) + 1
                    self.metrics["last_error"] = one_line(str(e), 200)
                except Exception as e:
                    log_debug_exc("_parse_retry_after_sec:silent", e)
                    pass
                if allow_retry and attempt <= int(self.cfg.max_retries):
                    self._backoff_sleep(attempt)
                    continue
                raise requests.RequestException(f"network failure after retries: {e}") from e
            except requests.HTTPError as e:
                # IMPORTANT: do NOT blindly retry 4xx (auth/validation/not found) — it creates loops.
                code2 = 0
                try:
                    resp = getattr(e, "response", None)
                    code2 = int(getattr(resp, "status_code", 0) or 0) if resp is not None else 0
                except Exception as e2:
                    log_debug_exc("HttpClient.request:http_status_parse", e2)
                    code2 = 0

                try:
                    self.metrics["http_errors"] = int(self.metrics.get("http_errors", 0)) + 1
                    if code2:
                        self.metrics["last_status"] = int(code2)
                    self.metrics["last_error"] = one_line(str(e), 200)
                except Exception as e2:
                    log_debug_exc("HttpClient.request:metrics_http_error", e2)
                    pass

                # Never retry auth failures (401/403) or general client errors (4xx).
                if 400 <= int(code2 or 0) <= 499:
                    if int(code2) == 408 and allow_retry and attempt <= int(self.cfg.max_retries):
                        self._backoff_sleep(attempt)
                        continue
                    raise

                # Retry 5xx only when enabled.
                if allow_retry and self.cfg.retry_on_5xx and self._is_retryable_5xx(int(code2 or 0)) and attempt <= int(self.cfg.max_retries):
                    self._backoff_sleep(attempt)
                    continue

                raise
        return HttpResult(status_code=0, json=None, text=last_text)

    def get_json(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        res = self.request("GET", path, params=params, allow_retry=True)
        if res.status_code >= 400:
            raise requests.HTTPError(f"{res.status_code} {res.text}")
        return res.json

    def post_json(self, path: str, payload: Dict[str, Any], *, headers: Optional[Dict[str, str]] = None) -> Any:
        res = self.request("POST", path, headers=headers, json_payload=payload, allow_retry=True)
        if res.status_code >= 400 and res.status_code not in (400, 409, 422):
            raise requests.HTTPError(f"{res.status_code} {res.text}")
        return res.json

    def health_snapshot(self) -> Dict[str, Any]:
        """(P0) lightweight runtime health/metrics snapshot for logging."""
        m = dict(self.metrics) if isinstance(getattr(self, "metrics", None), dict) else {}
        try:
            m["base_url"] = str(self.cfg.base_url)
            m["dry_run"] = bool(self.cfg.dry_run)
            if "latency_ms_ema" in m:
                m["latency_ms_ema"] = round(float(m.get("latency_ms_ema", 0.0) or 0.0), 1)
            if "latency_ms_last" in m:
                m["latency_ms_last"] = round(float(m.get("latency_ms_last", 0.0) or 0.0), 1)
            if "last_ok_ts" in m:
                m["last_ok_age_sec"] = round(max(0.0, time.time() - float(m.get("last_ok_ts") or 0.0)), 1)

            # v23.9: rolling 10m counts for 429 + Retry-After sleeps
            try:
                now2 = time.time()
                m["http_429_10m"] = int(sum(1 for t in list(getattr(self, "_recent_429_ts", [])) if (now2 - float(t or 0.0)) <= 600.0))
                m["retry_after_sleeps_10m"] = int(sum(1 for t in list(getattr(self, "_recent_retry_after_sleep_ts", [])) if (now2 - float(t or 0.0)) <= 600.0))
                m["http_401_10m"] = int(sum(1 for t in list(getattr(self, "_recent_http_401_ts", [])) if (now2 - float(t or 0.0)) <= 600.0))
                m["http_5xx_10m"] = int(sum(1 for t in list(getattr(self, "_recent_http_5xx_ts", [])) if (now2 - float(t or 0.0)) <= 600.0))
                if str(getattr(self, "_last_retry_after_basis", "") or ""):
                    m["last_429_sleep_basis"] = str(getattr(self, "_last_retry_after_basis", "") or "")
                if float(getattr(self, "_last_retry_after_sleep_sec", 0.0) or 0.0) > 0:
                    m["last_retry_after_sleep_sec"] = int(round(float(getattr(self, "_last_retry_after_sleep_sec", 0.0) or 0.0)))
            except Exception:
                pass
        except Exception as e:
            log_debug_exc("_parse_retry_after_sec:silent", e)
            pass
        return m

################################################################################
# 6. PoW CHALLENGE (parse/solve/header builder)
# - Dependencies: Section 1-3 (Config, Logging, Storage)
# - Used by: HTTP client (PoW-protected requests)
# - Key functions: parse_challenge(), solve_pow_nonblocking()
################################################################################

@dataclass(frozen=True)
class Challenge:
    token: str
    seed: str
    target_prefix: str
    limit_ms: int
    # v19.10: hybrid challenges may be solvable "puzzles" where proof is a solution string.
    kind: str = "pow"   # "pow" | "puzzle"
    proof: str = ""     # for puzzle: solution; for pow: unused
    challenge_id: str = ""

def _prefix_match_digest(digest: bytes, prefix_hex: str) -> bool:
    """prefix_hex can be odd length (nibble match)."""
    p = (prefix_hex or "").lower().strip()
    if not p:
        return False

    odd = (len(p) % 2 == 1)
    full = p[:-1] if odd else p
    pref_bytes = bytes.fromhex(full) if full else b""

    if digest[:len(pref_bytes)] != pref_bytes:
        return False

    if odd:
        nib = int(p[-1], 16)
        if len(digest) <= len(pref_bytes):
            return False
        return (digest[len(pref_bytes)] >> 4) == nib

    return True

def parse_challenge(payload: Any) -> Challenge:
    if not isinstance(payload, dict):
        raise ValueError("challenge payload must be dict")

    token = str(payload.get("token") or payload.get("challenge_id") or "").strip()
    ch = payload.get("challenge") if isinstance(payload.get("challenge"), dict) else None

    if ch:
        seed = str(ch.get("seed") or "").strip()
        target = str(ch.get("target_prefix") or "").strip()
        limit_ms = int(ch.get("limit_ms") or 2000)
        token = token or str(payload.get("token") or "").strip()
    else:
        seed = str(payload.get("seed") or "").strip()
        target = str(payload.get("target_prefix") or "").strip()
        limit_ms = int(payload.get("limit_ms") or 2000)

    if not token or not seed or not target:
        raise ValueError("invalid challenge payload: missing token/seed/target_prefix")

    limit_ms = max(200, min(60_000, int(limit_ms)))

def _challenge_type(payload: Any) -> str:
    """Best-effort challenge type probe for hybrid challenge systems."""
    try:
        if not isinstance(payload, dict):
            return ""
        ch = payload.get("challenge") if isinstance(payload.get("challenge"), dict) else None
        t = ""
        if isinstance(ch, dict):
            t = ch.get("type") or payload.get("type") or ""
        else:
            t = payload.get("type") or ""
        return str(t).strip().lower()
    except Exception:
        return ""

def _safe_json_preview(obj: Any, max_chars: int) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = repr(obj)
    s = s.replace("\r", "\\r").replace("\n", "\\n")
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "...(trunc)"
    return s

def _redact_challenge_payload_for_log(payload: Any) -> Any:
    """Redact secrets (token) and shrink noisy fields for puzzle capture logs."""
    if not isinstance(payload, dict):
        return payload
    p = dict(payload)
    if "token" in p:
        p["token"] = "<redacted>"
    ch = p.get("challenge")
    if isinstance(ch, dict):
        ch2 = dict(ch)
        # seed can be logged as a short hash to avoid leaking ephemeral values
        seed = ch2.get("seed")
        if isinstance(seed, str) and seed:
            try:
                ch2["seed_sha16"] = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
            except Exception as e:
                log_debug_exc("_redact_challenge_payload_for_log:silent", e)
                pass
            ch2["seed"] = "<redacted>"
        p["challenge"] = ch2
    return p


def _puzzle_fieldset_key(keys: List[str], *, max_keys: int = 32) -> str:
    """Stable key for grouping puzzle payload shapes. Keep short to avoid state bloat."""
    try:
        ks = [str(k) for k in (keys or []) if k is not None]
        ks = sorted(set(ks))
        if len(ks) > max_keys:
            ks = ks[:max_keys]
        return "|".join(ks)
    except Exception:
        return ""


def note_puzzle_event(state: Optional[Dict[str, Any]], *, ts_kst: str, challenge_type: str, fieldset_key: str) -> None:
    """Update lightweight in-state puzzle stats for solver planning (v19.9)."""
    try:
        if not isinstance(state, dict):
            return
        ps = state.setdefault("puzzles", {})
        # counters
        ps["puzzle_count"] = int(ps.get("puzzle_count", 0) or 0) + 1
        tc = ps.setdefault("type_counts", {})
        tc[challenge_type] = int(tc.get(challenge_type, 0) or 0) + 1
        fc = ps.setdefault("fieldset_counts", {})
        if fieldset_key:
            fc[fieldset_key] = int(fc.get(fieldset_key, 0) or 0) + 1

        ps["last_seen_at"] = ts_kst
        ps["last_seen_type"] = challenge_type

        # prune fieldset buckets to cap state growth
        max_buckets = 240
        if isinstance(fc, dict) and len(fc) > max_buckets:
            # drop smallest counts first
            items = sorted(fc.items(), key=lambda kv: (int(kv[1] or 0), kv[0]))
            drop_n = max(0, len(items) - max_buckets)
            for k, _ in items[:drop_n]:
                fc.pop(k, None)
    except Exception:
        return


def record_unhandled_challenge(
    hcfg: HybridChallengeConfig,
    payload: Any,
    *,
    reason: str,
    attempt: int,
    max_attempts: int,
    state: Optional[Dict[str, Any]] = None,
    context: str = ""
) -> None:
    """Append unhandled /challenge payload to JSONL and update puzzle stats (v19.9)."""
    try:
        if not hcfg or not str(hcfg.puzzle_log_jsonl or "").strip():
            return
        now_iso = now_kst().isoformat()
        ctype = _challenge_type(payload) or "unknown"
        ch = payload.get("challenge") if isinstance(payload, dict) else None
        chd = ch if isinstance(ch, dict) else {}

        # best-effort prompt extraction (keep short)
        prompt = ""
        for k in ("prompt", "puzzle", "question", "instruction", "task", "text", "content"):
            v = chd.get(k)
            if isinstance(v, str) and v.strip():
                prompt = v.strip()
                break
        if not prompt:
            # fallback: any string field
            for v in chd.values():
                if isinstance(v, str) and v.strip():
                    if len(v.strip()) > len(prompt):
                        prompt = v.strip()
        if prompt and len(prompt) > 2000:
            prompt = prompt[:2000] + "...(trunc)"

        # safe metadata (avoid storing seed/token directly)
        challenge_id = ""
        try:
            cid = chd.get("challenge_id") or chd.get("id") or ""
            if cid:
                challenge_id = one_line(str(cid), 128)
        except Exception:
            challenge_id = ""
        seed_hash16 = ""
        try:
            seed = chd.get("seed")
            if isinstance(seed, str) and seed:
                seed_hash16 = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        except Exception:
            seed_hash16 = ""

        red = _redact_challenge_payload_for_log(payload)
        preview = _safe_json_preview(red, int(hcfg.puzzle_raw_max_chars or 0))
        keys = sorted(list(chd.keys()))[:80]
        fieldset_key = _puzzle_fieldset_key(keys)

        evt = {
            "schema": "mersoom_puzzle_event_v1",
            "ts_kst": now_iso,
            "context": one_line(str(context or ""), 120),
            "challenge_type": ctype,
            "challenge_id": challenge_id,
            "seed_hash16": seed_hash16,
            "expires_at": chd.get("expires_at"),
            "limit_ms": chd.get("limit_ms"),
            "target_prefix": chd.get("target_prefix"),
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "reason": one_line(str(reason), 220),
            "prompt": prompt,
            "prompt_len": int(len(prompt) if isinstance(prompt, str) else 0),
            "fields_present": keys,
            "fieldset_key": fieldset_key,
            "payload_preview": preview,
        }
        append_jsonl(str(hcfg.puzzle_log_jsonl), evt)

        # update in-state puzzle stats for planning
        note_puzzle_event(state, ts_kst=now_iso, challenge_type=ctype, fieldset_key=fieldset_key)
    except Exception as e:
        log_debug_exc("record_unhandled_challenge:silent", e)
        pass



# --- v19.10: puzzle solver v1 (extract-type) ---------------------------------

_PUZZLE_CHAR_KEYWORDS = (
    "알파", "알파벳", "글자", "문자", "letter", "letters", "character", "characters", "alphabet"
)

def _extract_puzzle_prompt(payload: Any) -> Tuple[str, Dict[str, Any]]:
    """Best-effort: extract puzzle prompt/instruction text."""
    if not isinstance(payload, dict):
        return "", {}
    chd = payload.get("challenge") if isinstance(payload.get("challenge"), dict) else {}
    prompt = ""
    for k in ("prompt", "puzzle", "question", "instruction", "task", "text", "content"):
        v = chd.get(k)
        if isinstance(v, str) and v.strip():
            prompt = v.strip()
            break
    if not prompt:
        # fallback: pick the longest string field
        try:
            for v in chd.values():
                if isinstance(v, str) and v.strip():
                    vv = v.strip()
                    if len(vv) > len(prompt):
                        prompt = vv
        except Exception as e:
            log_debug_exc("_extract_puzzle_prompt:silent", e)
            pass
    return prompt, (chd if isinstance(chd, dict) else {})

def _extract_word_list_from_payload(chd: Dict[str, Any]) -> List[str]:
    """Try to locate explicit word list fields in the challenge payload."""
    cand_keys = ("words", "word_list", "items", "options", "candidates", "list")
    for k in cand_keys:
        v = chd.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            out = [one_line(str(x), 60).strip() for x in v if isinstance(x, str) and str(x).strip()]
            out = [x for x in out if x]
            if len(out) >= 3:
                return out
    return []

def _extract_word_list_from_prompt(prompt: str) -> List[str]:
    """Heuristic extraction from the prompt (supports newline lists or comma lists)."""
    if not isinstance(prompt, str) or not prompt.strip():
        return []

    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    # 1) newline list (possibly numbered/bulleted)
    out: List[str] = []
    for ln in lines:
        # remove common bullets/numbering
        ln2 = re.sub(r'^\s*(?:[\-\*\u2022]|\d+\s*[\)\.\]]|\(\d+\))\s*', '', ln).strip()
        m = re.fullmatch(r'[A-Za-z]{2,30}', ln2)
        if m:
            out.append(ln2)
    if len(out) >= 3:
        return out

    # 2) comma-separated list
    try:
        best = ""
        for m in re.finditer(r'([A-Za-z]{2,30}(?:\s*,\s*[A-Za-z]{2,30}){2,})', prompt):
            seg = m.group(1)
            if len(seg) > len(best):
                best = seg
        if best:
            parts = [p.strip() for p in best.split(",")]
            parts = [re.sub(r'[^A-Za-z]', '', p) for p in parts]
            parts = [p for p in parts if 2 <= len(p) <= 30]
            if len(parts) >= 3:
                return parts
    except Exception as e:
        log_debug_exc("_extract_word_list_from_prompt:silent", e)
        pass

    return []

def _parse_ordinals_with_pos(text: str) -> List[Tuple[int, int]]:
    """Return list of (position, number) for ordinal-like mentions (Korean/English)."""
    if not isinstance(text, str) or not text:
        return []
    out: List[Tuple[int, int]] = []
    # Korean: 1번째
    for m in re.finditer(r'(\d{1,3})\s*번째', text):
        try:
            out.append((m.start(), int(m.group(1))))
        except Exception as e:
            log_debug_exc("_parse_ordinals_with_pos:silent", e)
            pass
    # English: 1st/2nd/3rd/4th
    for m in re.finditer(r'\b(\d{1,3})\s*(st|nd|rd|th)\b', text, flags=re.IGNORECASE):
        try:
            out.append((m.start(), int(m.group(1))))
        except Exception as e:
            log_debug_exc("_parse_ordinals_with_pos:silent", e)
            pass
    out.sort(key=lambda x: x[0])
    # cap
    return out[:64]

def _parse_ordinals_in_segment(seg: str) -> List[int]:
    return [n for _, n in _parse_ordinals_with_pos(seg)][:32]

def _split_word_and_char_indices(prompt: str) -> Tuple[List[int], List[int]]:
    """Attempt to split ordinal indices into word indices and char indices."""
    if not isinstance(prompt, str) or not prompt.strip():
        return [], []

    # Prefer explicit bracket segments: [..] [..]
    seg_lists: List[List[int]] = []
    for m in re.finditer(r'[\[\(]([^\]\)]{0,220})[\]\)]', prompt):
        seg = m.group(1)
        if "번째" in seg or re.search(r'\b\d+\s*(st|nd|rd|th)\b', seg, flags=re.IGNORECASE):
            nums = _parse_ordinals_in_segment(seg)
            if nums:
                seg_lists.append(nums)
    if len(seg_lists) >= 2:
        return seg_lists[0], seg_lists[1]

    ords = _parse_ordinals_with_pos(prompt)
    if not ords:
        return [], []

    # Pivot by "letter/알파벳/글자" keyword
    pivot = None
    for kw in _PUZZLE_CHAR_KEYWORDS:
        p = prompt.lower().find(kw.lower())
        if p >= 0:
            pivot = p if pivot is None else min(pivot, p)
    if pivot is not None:
        w = [n for pos, n in ords if pos < pivot]
        c = [n for pos, n in ords if pos > pivot]
        return w[:32], c[:32]

    # Heuristic: split into half
    nums = [n for _, n in ords]
    mid = max(1, len(nums) // 2)
    return nums[:mid][:32], nums[mid:][:32]

def _puzzle_ops(prompt: str) -> Tuple[bool, bool, bool]:
    """Return (reverse, lower, upper) flags."""
    t = str(prompt or "")
    t_low = t.lower()
    rev = ("역순" in t) or ("reverse" in t_low) or ("reversed" in t_low)
    lower = ("소문자" in t) or ("lowercase" in t_low)
    upper = ("대문자" in t) or ("uppercase" in t_low)
    return rev, lower, upper

def try_solve_puzzle_extract_v1(payload: Any, hcfg: Optional[HybridChallengeConfig] = None) -> Optional[str]:
    """Solve simple 'pick Nth words and Nth letters' puzzles."""
    prompt, chd = _extract_puzzle_prompt(payload)
    if not prompt:
        return None

    # word list
    words = _extract_word_list_from_payload(chd)
    if not words:
        words = _extract_word_list_from_prompt(prompt)
    if not words:
        return None

    word_idxs, char_idxs = _split_word_and_char_indices(prompt)
    if not word_idxs or not char_idxs:
        return None

    rev, lower, upper = _puzzle_ops(prompt)

    # Build answer
    out_chars: List[str] = []
    for i, widx in enumerate(word_idxs):
        if widx <= 0 or widx > len(words):
            return None
        w = str(words[widx - 1])
        cidx = char_idxs[i] if i < len(char_idxs) else char_idxs[-1]
        if cidx <= 0 or cidx > len(w):
            return None
        out_chars.append(w[cidx - 1])

    ans = "".join(out_chars)
    if rev:
        ans = ans[::-1]
    if upper:
        ans = ans.upper()
    elif lower:
        ans = ans.lower()

    if not ans:
        return None

    # Optional debug log
    try:
        if hcfg and bool(getattr(hcfg, "puzzle_solver_debug", False)):
            log_info(f"[puzzle] solved(extract_v1) words={len(words)} idx_w={word_idxs} idx_c={char_idxs} -> '{one_line(ans, 64)}'")
    except Exception as e:
        log_debug_exc("try_solve_puzzle_extract_v1:silent", e)
        pass

    return ans


def record_puzzle_solution(
    hcfg: HybridChallengeConfig,
    payload: Any,
    *,
    answer: str,
    solver: str,
    attempt: int,
    max_attempts: int,
    state: Optional[Dict[str, Any]] = None,
    context: str = ""
) -> None:
    """Append solved puzzle payload to JSONL and update in-state puzzle stats (v19.10)."""
    try:
        if not hcfg or not str(hcfg.puzzle_log_jsonl or "").strip():
            return
        now_iso = now_kst().isoformat()
        ctype = _challenge_type(payload) or "unknown"
        ch = payload.get("challenge") if isinstance(payload, dict) else None
        chd = ch if isinstance(ch, dict) else {}

        prompt, _ = _extract_puzzle_prompt(payload)
        if prompt and len(prompt) > 2000:
            prompt = prompt[:2000] + "...(trunc)"

        challenge_id = ""
        try:
            cid = chd.get("challenge_id") or chd.get("id") or ""
            if cid:
                challenge_id = one_line(str(cid), 128)
        except Exception:
            challenge_id = ""

        seed_hash16 = ""
        try:
            seed = chd.get("seed")
            if isinstance(seed, str) and seed:
                seed_hash16 = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        except Exception:
            seed_hash16 = ""

        answer = str(answer or "").strip()
        ans_sha16 = hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16] if answer else ""

        red = _redact_challenge_payload_for_log(payload)
        preview = _safe_json_preview(red, int(hcfg.puzzle_raw_max_chars or 0))
        keys = sorted(list(chd.keys()))[:80]
        fieldset_key = _puzzle_fieldset_key(keys)

        evt = {
            "schema": "mersoom_puzzle_solution_v1",
            "ts_kst": now_iso,
            "context": one_line(str(context or ""), 120),
            "challenge_type": ctype,
            "challenge_id": challenge_id,
            "seed_hash16": seed_hash16,
            "expires_at": chd.get("expires_at"),
            "limit_ms": chd.get("limit_ms"),
            "target_prefix": chd.get("target_prefix"),
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "solver": one_line(str(solver or ""), 64),
            "answer": one_line(answer, 200),
            "answer_len": int(len(answer)),
            "answer_sha16": ans_sha16,
            "prompt": prompt,
            "prompt_len": int(len(prompt) if isinstance(prompt, str) else 0),
            "fields_present": keys,
            "fieldset_key": fieldset_key,
            "payload_preview": preview,
        }
        append_jsonl(str(hcfg.puzzle_log_jsonl), evt)

        # stats
        note_puzzle_event(state, ts_kst=now_iso, challenge_type=ctype, fieldset_key=fieldset_key)
        try:
            if isinstance(state, dict):
                ps = state.setdefault("puzzles", {})
                ps["solved_count"] = int(ps.get("solved_count", 0) or 0) + 1
                sc = ps.setdefault("solver_counts", {})
                sc[str(solver or "unknown")] = int(sc.get(str(solver or "unknown"), 0) or 0) + 1
        except Exception as e:
            log_debug_exc("record_puzzle_solution:silent", e)
            pass
    except Exception as e:
        log_debug_exc("record_puzzle_solution:silent", e)
        pass



def fetch_pow_challenge(client: HttpClient, hcfg: HybridChallengeConfig, *, state: Optional[Dict[str, Any]] = None, context: str = "") -> Challenge:
    """Fetch a challenge for write requests.

    - PoW challenge: solved via sha256(seed + nonce) prefix match.
    - Puzzle challenge (hybrid): if solvable by built-in solver, return a Challenge(kind="puzzle", proof=answer).
      Otherwise, record the payload and retry until PoW appears (bounded).
    """
    max_attempts = 1 + max(0, int(getattr(hcfg, "max_retries", 0) or 0))
    sleep_ms = int(getattr(hcfg, "retry_sleep_ms", 0) or 0)
    last_err: Optional[Exception] = None

    for i in range(1, max_attempts + 1):
        payload = client.post_json("/challenge", payload={})
        ctype = _challenge_type(payload)

        # Prefer parse-based detection: if parse_challenge works, treat as PoW-compatible.
        try:
            return parse_challenge(payload)
        except Exception as e:
            last_err = e

            # v19.10: attempt puzzle solver (extract_v1)
            if bool(getattr(hcfg, "puzzle_solver_enable", True)):
                ans: Optional[str] = None
                try:
                    ans = try_solve_puzzle_extract_v1(payload, hcfg)
                except Exception:
                    ans = None

                if ans:
                    try:
                        token = ""
                        if isinstance(payload, dict):
                            token = str(payload.get("token") or "").strip()
                            if not token:
                                chd = payload.get("challenge") if isinstance(payload.get("challenge"), dict) else {}
                                token = str(chd.get("token") or "").strip()
                        if token:
                            chd = payload.get("challenge") if isinstance(payload, dict) and isinstance(payload.get("challenge"), dict) else {}
                            cid = ""
                            try:
                                cid = one_line(str((chd.get("challenge_id") or chd.get("id") or "")), 128)
                            except Exception:
                                cid = ""
                            limit_ms = 0
                            try:
                                limit_ms = int(chd.get("limit_ms") or 0)
                            except Exception:
                                limit_ms = 0
                            solved = Challenge(token=token, seed="", target_prefix="", limit_ms=int(limit_ms), kind="puzzle", proof=str(ans), challenge_id=str(cid or ""))
                            record_puzzle_solution(hcfg, payload, answer=str(ans), solver="extract_v1", attempt=i, max_attempts=max_attempts, state=state, context=(context or "/challenge"))
                            return solved
                    except Exception as e:
                        log_debug_exc("fetch_pow_challenge:silent", e)
                        pass

            # If it's not PoW (or unsolved puzzle), record and retry
            reason = f"type={ctype or 'unknown'}; {one_line(str(e), 160)}"
            record_unhandled_challenge(hcfg, payload, reason=reason, attempt=i, max_attempts=max_attempts, state=state, context=(context or "/challenge"))
            if sleep_ms > 0 and i < max_attempts:
                time.sleep((sleep_ms / 1000.0) + random.uniform(0.0, 0.05))
            continue

    raise RuntimeError(f"no solvable challenge after {max_attempts} attempts: {one_line(str(last_err), 180)}")


def solve_pow(seed: str, target_prefix: str, limit_ms: int) -> int:
    """CPU PoW: find nonce such that sha256(seed + nonce) matches target_prefix.

    v21.1: precompile prefix once per solve (avoid bytes.fromhex per-iteration), reduce perf_counter checks.
    """
    seed_b = seed.encode("utf-8")
    p = (target_prefix or "").lower().strip()
    if not p:
        raise ValueError("target_prefix is required")
    odd = (len(p) % 2 == 1)
    full = p[:-1] if odd else p
    pref_bytes = bytes.fromhex(full) if full else b""
    nib = int(p[-1], 16) if odd else -1
    pref_len = len(pref_bytes)

    sha256 = hashlib.sha256
    perf = time.perf_counter
    deadline = perf() + (max(200, int(limit_ms)) / 1000.0)

    nonce = 0
    check_every = 256  # deadline check frequency
    while True:
        for _ in range(check_every):
            d = sha256(seed_b + str(nonce).encode("utf-8")).digest()
            if pref_len and d[:pref_len] != pref_bytes:
                nonce += 1
                continue
            if odd:
                if len(d) <= pref_len:
                    nonce += 1
                    continue
                if (d[pref_len] >> 4) != nib:
                    nonce += 1
                    continue
            return nonce
            # nonce increment is unreachable after return
        if perf() >= deadline:
            break

    raise TimeoutError("PoW time limit exceeded")

# --- PoW offload (P0): avoid blocking main thread on high difficulty ---
_POW_EXECUTOR: Optional[ProcessPoolExecutor] = None
_POW_TIMEOUT_TS = deque(maxlen=4000)
_POW_EXECUTOR_RESTART_TS = deque(maxlen=400)
_POW_TIMEOUT_STREAK = 0

def _pow_watchdog_enabled() -> bool:
    return _env_bool("MERSOOM_POW_WATCHDOG", True)

def _pow_watchdog_threshold() -> int:
    return _env_int("MERSOOM_POW_TIMEOUT_STREAK_MAX", 3, 1, 10)

def _pow_watchdog_note_success() -> None:
    global _POW_TIMEOUT_STREAK
    _POW_TIMEOUT_STREAK = 0

def _pow_watchdog_on_timeout() -> bool:
    """Return True if current tick should skip PoW (after restart)."""
    global _POW_TIMEOUT_STREAK, _POW_EXECUTOR
    if not _pow_watchdog_enabled():
        return False
    try:
        _POW_TIMEOUT_TS.append(time.time())
    except Exception:
        pass
    _POW_TIMEOUT_STREAK += 1
    if _POW_TIMEOUT_STREAK < _pow_watchdog_threshold():
        return False
    # restart executor and skip current tick
    _POW_TIMEOUT_STREAK = 0
    try:
        if _POW_EXECUTOR is not None:
            _POW_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        log_debug_exc("pow_watchdog:shutdown", e)
    _POW_EXECUTOR = None
    try:
        _POW_EXECUTOR_RESTART_TS.append(time.time())
    except Exception:
        pass
    return True

def _pow_mode() -> str:
    return str(os.getenv("MERSOOM_POW_MODE", "process")).strip().lower()

def _get_pow_executor() -> Optional[ProcessPoolExecutor]:
    global _POW_EXECUTOR
    mode = _pow_mode()
    if mode in ("sync", "off", "none", "0", "false"):
        return None
    if _POW_EXECUTOR is None:
        _POW_EXECUTOR = ProcessPoolExecutor(max_workers=1)
        try:
            atexit.register(lambda: _POW_EXECUTOR.shutdown(wait=False, cancel_futures=True) if _POW_EXECUTOR else None)
        except Exception as e:
            log_debug_exc("_get_pow_executor:silent", e)
            pass
    return _POW_EXECUTOR

def _solve_pow_worker(seed: str, target_prefix: str, limit_ms: int) -> int:
    # top-level for multiprocessing pickling
    return solve_pow(seed, target_prefix, int(limit_ms))

def solve_pow_nonblocking(seed: str, target_prefix: str, limit_ms: int) -> int:
    ex = _get_pow_executor()
    if ex is None:
        try:
            res = solve_pow(seed, target_prefix, int(limit_ms))
            _pow_watchdog_note_success()
            return res
        except TimeoutError:
            if _pow_watchdog_on_timeout():
                raise PowTimeoutError("pow timeout watchdog (sync)")
            raise

    fut = ex.submit(_solve_pow_worker, seed, target_prefix, int(limit_ms))
    deadline_wall = time.time() + max(0.3, int(limit_ms) / 1000.0 + 1.0)

    while True:
        try:
            res = int(fut.result(timeout=0.25))
            _pow_watchdog_note_success()
            return res
        except FuturesTimeoutError:
            if time.time() >= deadline_wall:
                try:
                    fut.cancel()
                except Exception as e:
                    log_debug_exc("solve_pow_nonblocking:silent", e)
                    pass
                if _pow_watchdog_on_timeout():
                    raise PowTimeoutError("pow timeout watchdog (process)")
                raise TimeoutError("PoW time limit exceeded (process)")
            continue

def build_pow_headers(ch: Challenge, *, wallet: str = "") -> Dict[str, str]:
    # v19.10: support solvable "puzzle" challenges (proof = solution string)
    if str(getattr(ch, "kind", "pow") or "pow").lower() == "puzzle":
        proof = str(getattr(ch, "proof", "") or "").strip()
        if not proof:
            raise ValueError("puzzle challenge missing proof/solution")
        return {
            "X-Mersoom-Token": ch.token,
            "X-Mersoom-Proof": proof,
            "Content-Type": "application/json",
        }

    nonce = solve_pow_nonblocking(ch.seed, ch.target_prefix, ch.limit_ms)
    proof = f"{nonce}:{wallet}" if wallet else str(nonce)
    return {
        "X-Mersoom-Token": ch.token,
        "X-Mersoom-Proof": proof,
        "Content-Type": "application/json",
    }


def build_auth_headers(auth: Optional[AuthConfig]) -> Dict[str, str]:
    """Build optional login headers for points system (Mersoom 3.0).

    Returns {} if auth is missing/disabled/invalid.
    """
    try:
        if not auth or not bool(getattr(auth, "enabled", False)):
            return {}
        aid = str(getattr(auth, "auth_id", "") or "").strip()
        pw = str(getattr(auth, "password", "") or "").strip()
        if not aid or not pw:
            return {}
        return {"X-Mersoom-Auth-Id": aid, "X-Mersoom-Password": pw}
    except Exception:
        return {}


def post_with_pow(client: HttpClient, pow_cfg: PowConfig, hcfg: HybridChallengeConfig, path: str, payload: Dict[str, Any], *, extra_headers: Optional[Dict[str, str]] = None, state: Optional[Dict[str, Any]] = None) -> Any:
    """
    Generic PoW POST helper.
    - DRY_RUN: bypass PoW.
    - If /challenge unsupported (404/405), fallback to plain POST.
    """
    if client.cfg.dry_run:
        return client.post_json(path, payload)

    # 1) challenge (hybrid: if AI Puzzle appears, record + retry until PoW)
    try:
        ch = fetch_pow_challenge(client, hcfg, state=state, context=(f"POST {path}" if path else "POST /challenge"))
    except requests.HTTPError as e:
        msg = str(e)
        if "404" in msg or "405" in msg:
            return client.post_json(path, payload)
        raise

    # 2) solve + post
    try:
        headers = build_pow_headers(ch, wallet=pow_cfg.wallet)
    except PowTimeoutError:
        if isinstance(state, dict):
            protocol_bump_counter(state, "pow_timeout", 1)
        raise
    if extra_headers:
        try:
            for k, v in dict(extra_headers).items():
                if v is None:
                    continue
                vv = str(v)
                if vv == "":
                    continue
                headers[str(k)] = vv
        except Exception as e:
            log_debug_exc("post_with_pow:silent", e)
            pass
    return client.post_json(path, payload, headers=headers)

################################################################################
# 7. SCHEMAS + DEFAULTS (state/memory/policy/semantic/brain)
# - Dependencies: Section 1-3 (Config, Logging, Storage)
# - Used by: Most sections (state/memory/policy/brain IO)
# - Key functions: default_state(), default_policy(), default_brain()
################################################################################

@dataclass(frozen=True)
class AgentTuning:
    # feed fetch
    fetch_limit: int
    arena_fetch_limit: int

    # memory
    memory_size: int

    # evaluation scheduling (per-action)
    eval_delay_min_sec: int
    eval_delay_max_sec: int
    max_eval_per_tick: int

    # learning batch cadence (global) ✅ batched cadence control
    learn_period_sec: int
    max_learn_updates_per_run: int

    # arena
    arena_cooldown_sec: int

    # reward shaping
    alpha_score: float
    alpha_up: float
    alpha_down: float
    alpha_comments: float
    alpha_novelty: float
    alpha_quotes: float
    alpha_quality: float
    alpha_continuation: float
    alpha_weak_context_penalty: float
    reward_clip: float


    # v20.5 unified reward weights
    reward_w_up: float
    reward_w_engage: float
    reward_w_risk: float

    # policy defaults
    policy_epsilon: float
    policy_lr: float
    policy_min_weight: float

    # brain EMA
    brain_topic_alpha: float
    brain_reward_alpha: float

    # output formatting
    max_output_lines: int

    # community flow + thoughts
    flow_half_life_hours: float
    scan_posts_per_sync: int
    max_thoughts: int

    # (P1) bm25 rebuild pacing
    bm25_rebuild_min_add: int
    bm25_rebuild_min_sec: int

    # (Unit 06) quote discipline (BM25)
    quote_min_score: float
    quote_max_chars: int
    quote_min_gap_sec: int

    # (Unit 04) inbox / thread-priority replies
    inbox_max_posts: int
    inbox_scan_comments_per_post: int
    inbox_recent_thread_days: int
    inbox_scan_min_sec: int

    # (v18.3) Conversation protocol (turn-taking for reply threads)
    reply_conv_skip_cooldown_sec: int
    reply_conv_budget_cap: int
    reply_question_base_prob: float
    reply_question_my_post_prob: float
    reply_question_when_asked_prob: float

    # (Unit 07) reflection influence on bandit choices
    reflection_influence: bool
    reflection_boost: float
    reflection_min_strength: float
    reflection_topk: int
    reflection_decay: float

def load_tuning_from_env() -> AgentTuning:
    def env_int(name: str, d: int, lo: int, hi: int) -> int:
        return _env_int(name, d, min_v=lo, max_v=hi)

    def env_float(name: str, d: float, lo: float, hi: float) -> float:
        return _env_float(name, d, min_v=lo, max_v=hi)

    # ✅ 기본 15분(900초) cadence (v15.6)
    eval_min = env_int("MERSOOM_EVAL_DELAY_MIN_SEC", 13 * 60, 60, 24 * 60 * 60)
    eval_max = env_int("MERSOOM_EVAL_DELAY_MAX_SEC", 17 * 60, 60, 24 * 60 * 60)
    if eval_max < eval_min:
        eval_max = eval_min

    learn_period = env_int("MERSOOM_LEARN_PERIOD_SEC", 15 * 60, 60, 24 * 60 * 60)

    return AgentTuning(
        fetch_limit=env_int("MERSOOM_FETCH_LIMIT", 10, 1, 200),
        arena_fetch_limit=env_int("MERSOOM_ARENA_FETCH_LIMIT", 40, 1, 400),

        memory_size=env_int("MERSOOM_MEMORY_SIZE", 140, 10, 5000),

        eval_delay_min_sec=eval_min,
        eval_delay_max_sec=eval_max,
        max_eval_per_tick=env_int("MERSOOM_MAX_EVAL_PER_TICK", 1, 1, 200),

        learn_period_sec=learn_period,
        max_learn_updates_per_run=env_int("MERSOOM_MAX_LEARN_UPDATES_PER_RUN", 500, 10, 100_000),

        arena_cooldown_sec=env_int("MERSOOM_ARENA_COOLDOWN_SEC", 2 * 60 * 60, 60, 24 * 60 * 60),

        alpha_score=env_float("MERSOOM_ALPHA_SCORE", 1.0, 0.0, 10.0),
        alpha_up=env_float("MERSOOM_ALPHA_UP", 1.0, 0.0, 10.0),
        alpha_down=env_float("MERSOOM_ALPHA_DOWN", 2.0, 0.0, 10.0),
        alpha_comments=env_float("MERSOOM_ALPHA_COMMENTS", 0.2, 0.0, 10.0),
        alpha_novelty=env_float("MERSOOM_ALPHA_NOVELTY", 0.30, 0.0, 10.0),
        alpha_quotes=env_float("MERSOOM_ALPHA_QUOTES", 0.15, 0.0, 10.0),
        alpha_quality=env_float("MERSOOM_ALPHA_QUALITY", 0.08, 0.0, 10.0),
        alpha_continuation=env_float("MERSOOM_ALPHA_CONT", 0.25, 0.0, 10.0),
        alpha_weak_context_penalty=env_float("MERSOOM_ALPHA_WEAKCTX", 0.12, 0.0, 10.0),
        reward_clip=env_float("MERSOOM_REWARD_CLIP", 3.0, 0.1, 50.0),


        reward_w_up=env_float("MERSOOM_REWARD_W_UP", 1.0, 0.0, 10.0),
        reward_w_engage=env_float("MERSOOM_REWARD_W_ENGAGE", 0.6, 0.0, 10.0),
        reward_w_risk=env_float("MERSOOM_REWARD_W_RISK", 1.2, 0.0, 10.0),

        policy_epsilon=env_float("MERSOOM_POLICY_EPSILON", 0.15, 0.0, 0.95),
        policy_lr=env_float("MERSOOM_POLICY_LR", 0.12, 0.0, 2.0),
        policy_min_weight=env_float("MERSOOM_POLICY_MIN_WEIGHT", 0.05, 0.0001, 10.0),

        brain_topic_alpha=env_float("MERSOOM_BRAIN_TOPIC_ALPHA", 0.18, 0.01, 0.8),
        brain_reward_alpha=env_float("MERSOOM_BRAIN_REWARD_ALPHA", 0.12, 0.01, 0.8),

        max_output_lines=env_int("MERSOOM_MAX_OUTPUT_LINES", 4, 1, 20),

        flow_half_life_hours=env_float("MERSOOM_FLOW_HALF_LIFE_HOURS", 6.0, 0.5, 72.0),
        scan_posts_per_sync=env_int("MERSOOM_SCAN_POSTS_PER_SYNC", 12, 1, 80),
        max_thoughts=env_int("MERSOOM_MAX_THOUGHTS", 400, 50, 5000),

        bm25_rebuild_min_add=env_int("MERSOOM_BM25_REBUILD_MIN_ADD", 6, 1, 200),
        bm25_rebuild_min_sec=env_int("MERSOOM_BM25_REBUILD_MIN_SEC", 600, 30, 60 * 60),

        quote_min_score=env_float("MERSOOM_QUOTE_MIN_SCORE", 18.0, 0.0, 100.0),
        quote_max_chars=env_int("MERSOOM_QUOTE_MAX_CHARS", 140, 40, 400),
        quote_min_gap_sec=env_int("MERSOOM_QUOTE_MIN_GAP_SEC", 180, 0, 3600),

        inbox_max_posts=env_int("MERSOOM_INBOX_MAX_POSTS", 10, 2, 40),
        inbox_scan_comments_per_post=env_int("MERSOOM_INBOX_SCAN_COMMENTS_PER_POST", 40, 10, 200),
        inbox_recent_thread_days=env_int("MERSOOM_INBOX_RECENT_THREAD_DAYS", 3, 1, 30),
        inbox_scan_min_sec=env_int("MERSOOM_INBOX_SCAN_MIN_SEC", 90, 15, 60 * 30),
        reply_conv_skip_cooldown_sec=env_int("MERSOOM_REPLY_CONV_SKIP_COOLDOWN_SEC", 10 * 60, 30, 24 * 60 * 60),
        reply_conv_budget_cap=env_int("MERSOOM_REPLY_CONV_BUDGET_CAP", 1, 1, 5),
        reply_question_base_prob=env_float("MERSOOM_REPLY_QUESTION_BASE_PROB", 0.25, 0.0, 1.0),
        reply_question_my_post_prob=env_float("MERSOOM_REPLY_QUESTION_MY_POST_PROB", 0.18, 0.0, 1.0),
        reply_question_when_asked_prob=env_float("MERSOOM_REPLY_QUESTION_WHEN_ASKED_PROB", 0.85, 0.0, 1.0),

        # (Unit 07) reflection influence
        reflection_influence=_env_bool("MERSOOM_REFLECTION_INFLUENCE", True),
        reflection_boost=env_float("MERSOOM_REFLECTION_BOOST", 0.35, 0.0, 2.0),
        reflection_min_strength=env_float("MERSOOM_REFLECTION_MIN_STRENGTH", 0.60, 0.0, 1.0),
        reflection_topk=env_int("MERSOOM_REFLECTION_TOPK", 3, 1, 10),
        reflection_decay=env_float("MERSOOM_REFLECTION_DECAY", 0.85, 0.50, 1.00),
    )

################################################################################
# 7.1. v15.2 MIGRATION HELPERS (state/brain placeholders)
# - Dependencies: Section 7 (Schemas)
# - Used by: Main loop (migrations)
# - Key functions: migrate_state_v15_2()
################################################################################

################################################################################
# SCHEMA MIGRATION (compact) — v15.15
# - Consolidates the v15.2~v15.14 chained migrations into a single pass.
# - Keeps legacy flat keys intact (e.g., arena_last_action_ts), but ensures nested slots exist.
################################################################################

def _sdict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key)
    if not isinstance(v, dict):
        v = {}
        d[key] = v
    return v

def _slist(d: Dict[str, Any], key: str) -> List[Any]:
    v = d.get(key)
    if not isinstance(v, list):
        v = []
        d[key] = v
    return v


# -----------------------------------------------------------------------------
# v23.1 Interaction scaffolding helpers (schema only; no behavior change)
# -----------------------------------------------------------------------------
def _normalize_open_questions_list(items: Any) -> List[Dict[str, Any]]:
    """Normalize open_questions to the expanded dict form (backward compatible)."""
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        try:
            if isinstance(it, dict):
                qid = str(it.get("qid") or "")
                text = str(it.get("text") or it.get("q") or "")
                try:
                    ts = float(it.get("ts", 0.0) or 0.0)
                except Exception:
                    ts = 0.0
                if not qid:
                    if text:
                        qid = hashlib.sha1(f"{ts}|{text}".encode("utf-8")).hexdigest()[:10]
                    else:
                        continue
                obj = dict(it)
                obj["qid"] = qid
                obj["text"] = one_line(text, 220) if text else ""
                obj["ts"] = ts
                st = str(obj.get("status") or "open")
                if st not in ("open", "resolved", "expired"):
                    st = "open"
                obj["status"] = st
                obj.setdefault("asked_by", str(obj.get("asked_by") or ""))
                obj.setdefault("last_seen_remote_id", str(obj.get("last_seen_remote_id") or ""))
                try:
                    obj["resolve_ts"] = float(obj.get("resolve_ts", 0.0) or 0.0)
                except Exception:
                    obj["resolve_ts"] = 0.0
                out.append(obj)
            elif isinstance(it, str):
                text = it.strip()
                if not text:
                    continue
                qid = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
                out.append({
                    "qid": qid,
                    "text": one_line(text, 220),
                    "ts": 0.0,
                    "status": "open",
                    "asked_by": "",
                    "last_seen_remote_id": str(last_seen_remote_id or ""),
                    "resolve_ts": 0.0,
                })
        except Exception:
            continue
    return out

def _migrate_thread_schema_inplace(th: Any, post_id: str = "") -> Dict[str, Any]:
    """Ensure a thread dict contains v23.1 interaction scaffolding fields."""
    if not isinstance(th, dict):
        th = {}
    if post_id:
        th.setdefault("post_id", str(post_id))
    th.setdefault("phase", "open")
    try:
        th["phase_ts"] = float(th.get("phase_ts", 0.0) or 0.0)
    except Exception:
        th["phase_ts"] = 0.0
    th["open_questions"] = _normalize_open_questions_list(th.get("open_questions", []))
    return th

def _migrate_threads_payload_inplace(threads_payload: Any) -> Dict[str, Any]:
    """Normalize threads payload (split file or embedded) in-place."""
    if not isinstance(threads_payload, dict):
        return {}
    for pid, th in list(threads_payload.items()):
        if pid == "__meta__":
            continue
        threads_payload[pid] = _migrate_thread_schema_inplace(th, str(pid))
    return threads_payload

def _migrate_conv_state_payload_inplace(convs: Any) -> Dict[str, Any]:
    """Backfill conv_state schema fields used by future interaction patches."""
    if not isinstance(convs, dict):
        return {}
    for k, cv in list(convs.items()):
        if not isinstance(cv, dict):
            continue
        try:
            cv.setdefault("last_remote_ts", 0.0)
            cv["last_remote_ts"] = float(cv.get("last_remote_ts", 0.0) or 0.0)
        except Exception:
            cv["last_remote_ts"] = 0.0
        convs[k] = cv
    return convs

def migrate_state(s: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(s, dict):
        s = {}

    # ---- v19 protocol/rules scaffold (no behavior change yet) ----
    proto = _sdict(s, "protocol")
    proto.setdefault("enabled", _env_bool("MERSOOM_PROTOCOL_ENABLE", True))
    proto.setdefault("cycle_id", 0)
    proto.setdefault("last_tick_at", 0.0)

    # ---- v20.6 reason protocol (observability; no behavior change) ----
    _sdict(proto, "reason_last")          # domain -> last reason code
    _sdict(proto, "reason_last_ts")       # domain -> last set timestamp
    _sdict(proto, "reason_last_detail")   # domain -> last detail (one-line)
    proto.setdefault("reason_window_started_ts", 0.0)
    _sdict(proto, "reason_window_10m")    # code -> count (rolling 10m)


    # ---- v20.2 vote backlog (durable mandatory votes) ----
    # vote_backlog: list of {"post_id": str, "seen_ts": float}
    _slist(proto, "vote_backlog")
    # drain timestamps for health metrics (recent 10m window)
    _slist(proto, "vote_backlog_drains")

    # ---- v20.3 heartbeat clamp + reply streak ----
    proto.setdefault("hb_block_reason", "")
    _sdict(proto, "reply_streak")  # post_id -> {"count": int, "last_ts": float}



    # ---- v23.1 interaction scaffolding (schema only; no behavior change) ----
    proto.setdefault("interaction_fsm_enabled", False)  # env is authoritative; see _interaction_fsm_enabled()
    proto.setdefault("openq_track_enabled", True)  # env can override; default preserves prior behavior
    proto.setdefault("waiting_strict_enabled", False)  # env can override; default off
    proto.setdefault("reply_score_v2_enabled", False)  # env can override; default off
    proto.setdefault("openq_added_total", 0)
    proto.setdefault("openq_resolved_total", 0)
    proto.setdefault("openq_expired_total", 0)
    try:
        _migrate_threads_payload_inplace(_sdict(s, "threads"))
    except Exception as e:
        log_debug_exc("migrate_threads:silent", e)
        pass
    try:
        _migrate_conv_state_payload_inplace(_sdict(s, "conv_state"))
    except Exception as e:
        log_debug_exc("migrate_conv_state:silent", e)
        pass

    # ---- v23.10 recent action guard ----
    ra = s.get("recent_actions")
    if not isinstance(ra, dict):
        s["recent_actions"] = {}





    # ---- v19.3 heartbeat protocol slots ----
    hb = _sdict(proto, "heartbeat")
    hb.setdefault("active", False)
    hb.setdefault("cycle_id", 0)
    hb.setdefault("started_at", 0.0)
    hb.setdefault("completed_at", 0.0)
    hb.setdefault("last_at", 0.0)
    hb.setdefault("next_at", 0.0)
    hb.setdefault("last_log_ts", 0.0)
    quota = _sdict(hb, "quota")
    quota.setdefault("comments_target", 0)
    quota.setdefault("comments_target_clamped", False)
    quota.setdefault("comments_done", 0)
    quota.setdefault("votes_done", 0)
    quota.setdefault("contribute_done", False)
    quota.setdefault("contribute_ts", 0.0)
    rules = _sdict(s, "rules")
    rules.setdefault("last_sync_day", "")
    rules.setdefault("last_hash", "")
    rules.setdefault("last_checked_at", 0.0)

    seen = _sdict(s, "seen")
    seen_posts = _sdict(seen, "posts")  # post_id -> seen_ts
    if not seen_posts and isinstance(s.get("seen_post_ids"), list):
        for pid in (s.get("seen_post_ids") or [])[:500]:
            if isinstance(pid, str) and pid:
                seen_posts.setdefault(pid, 0.0)

    votes2 = _sdict(s, "votes")
    votes_posts = _sdict(votes2, "posts")  # post_id -> voted_ts
    if not votes_posts and isinstance(s.get("voted_posts"), dict):
        for pid, info in (s.get("voted_posts") or {}).items():
            if not isinstance(pid, str) or not pid:
                continue
            ts = 0.0
            if isinstance(info, dict):
                try:
                    ts = float(info.get("ts", 0.0) or 0.0)
                except Exception:
                    ts = 0.0
            votes_posts.setdefault(pid, ts)

    # ---- arena nested slots (seed from legacy keys if present) ----
    arena = _sdict(s, "arena")
    arena.setdefault("day", "")  # YYYY-MM-DD (KST)
    arena.setdefault("today_proposed", False)
    arena.setdefault("today_propose_id", "")

    legacy_last_action = float(s.get("arena_last_action_ts", 0.0) or 0.0)
    legacy_last_propose_date = str(s.get("arena_last_propose_date", "") or "")
    arena.setdefault("last_action_ts", legacy_last_action)
    arena.setdefault("last_propose_date", legacy_last_propose_date)

    # topic + last post
    arena.setdefault("today_topic_id", "")
    arena.setdefault("today_topic_title", "")
    arena.setdefault("last_post_id", "")
    arena.setdefault("last_post_side", "")
    arena.setdefault("last_post_created_at", "")

    # my arena posts today: {post_id: {up:int, down:int, ts:float, side:str}}
    arena.setdefault("my_posts", {})

    # status/posts caches
    arena.setdefault("last_status_ts", 0.0)
    arena.setdefault("last_posts_ts", 0.0)
    arena.setdefault("last_phase", "")
    arena.setdefault("last_status_date", "")
    arena.setdefault("last_status_topic_id", "")
    arena.setdefault("last_status_topic_title", "")
    arena.setdefault("status_cache", {})
    arena.setdefault("posts_cache", [])
    arena.setdefault("actions_today", 0)

    # latest my-post info (cooldown buff)
    arena.setdefault("last_my_post_id", "")
    arena.setdefault("last_my_post_up", 0)
    arena.setdefault("last_my_post_down", 0)
    arena.setdefault("last_my_post_ts", 0.0)
    arena.setdefault("last_my_post_side", "")
    arena.setdefault("last_effective_cooldown_sec", 0.0)
    arena.setdefault("last_cooldown_upvotes", 0)
    arena.setdefault("last_cooldown_post_id", "")

    # risk/stoploss (blind(-5) guard)
    arena.setdefault("risk_mode", False)
    arena.setdefault("risk_level", "OK")  # OK|CAUTION|DANGER|BLIND
    arena.setdefault("risk_score", 0.0)
    arena.setdefault("risk_style", "")
    arena.setdefault("risk_last_update_ts", 0.0)
    arena.setdefault("risk_source_post_id", "")
    arena.setdefault("risk_stop_day", "")

    # strategy caches
    arena.setdefault("recent_target_post_ids", [])
    arena.setdefault("last_strategy", "")

    # ---- focus (comment/reply grounding slots) ----
    focus = _sdict(s, "focus")
    focus.setdefault("mode", "")  # comment|reply
    focus.setdefault("post_id", "")
    focus.setdefault("post_title", "")
    focus.setdefault("post_excerpt", "")
    focus.setdefault("comment_id", "")
    focus.setdefault("comment_excerpt", "")
    focus.setdefault("comment_author", "")
    focus.setdefault("created_ts", 0.0)

    # ---- evaluation budget slots ----
    eb = _sdict(s, "eval_budget")
    eb.setdefault("last_reset_ts", 0.0)
    eb.setdefault("used", 0)
    eb.setdefault("cap_per_window", 0)  # 0 => unlimited
    eb.setdefault("window_sec", 900)

    # ---- ops guards (circuit breaker) ----
    ops = _sdict(s, "_ops")
    ops.setdefault("fail_counts", {})
    ops.setdefault("disabled_until", {})
    ops.setdefault("last_fail_reason", {})
    ops.setdefault("last_tick_log_ts", 0.0)

    s["schema_ver"] = AGENT_CODE_VERSION
    return s



# -----------------------------------------------------------------------------
# v19.3 Heartbeat protocol helpers (state-only)
# -----------------------------------------------------------------------------
def _hb_get(state: Dict[str, Any]) -> Dict[str, Any]:
    proto = _safe_dict(state.get("protocol"))
    hb = _safe_dict(proto.get("heartbeat"))
    return hb if isinstance(hb, dict) else {}

def _hb_active(state: Dict[str, Any]) -> bool:
    hb = _hb_get(state)
    return bool(hb.get("active")) if hb else False

def _hb_quota(state: Dict[str, Any]) -> Dict[str, Any]:
    hb = _hb_get(state)
    q = _safe_dict(hb.get("quota"))
    return q if isinstance(q, dict) else {}


def _hb_clamp_comment_target(cfg: "Config", state: Dict[str, Any], comment_pace_sec: int, comment_limiter: "SlidingWindowLimiter") -> None:
    """
    v20.3: Clamp per-cycle heartbeat comment quota to a feasible maximum.
    Motivation: avoid "never-completes" heartbeat cycles when pacing/limits make 2~3 comments impossible.
    Stores a coarse reason in state["protocol"]["hb_block_reason"] when clamped.
    """
    try:
        hb = _hb_get(state)
        if not hb or not bool(hb.get("active")):
            return
        q = _hb_quota(state)
        if not q:
            return
        if bool(q.get("comments_target_clamped", False)):
            return

        proto = _sdict(state, "protocol")

        # Estimate available time budget of the cycle (heartbeat cadence window).
        started_at = float(hb.get("started_at", 0.0) or 0.0)
        next_at = float(hb.get("next_at", 0.0) or 0.0)
        cycle_span = 0.0
        if started_at > 0.0 and next_at > started_at:
            cycle_span = float(next_at - started_at)
        else:
            # Fallback to a typical interval if missing.
            cycle_span = float(_hb_next_interval_sec(cfg))
        cycle_span = max(0.0, cycle_span)

        # Max by pace (cooldown between comments)
        pace = int(comment_pace_sec or 0)
        if pace <= 0:
            max_by_pace = 10**9
        else:
            # First comment can happen at start; remaining need pace gaps.
            max_by_pace = 1 + int(cycle_span // max(1, pace))

        # Max by limiter (window capacity)
        cap = int(getattr(comment_limiter, "capacity", 0) or 0)
        win = int(getattr(comment_limiter, "window_sec", 1800) or 1800)
        if cap <= 0:
            max_by_limiter = 0
        else:
            win = max(1, win)
            windows = max(1, 1 + int(cycle_span // win))
            max_by_limiter = cap * windows

        possible_max = int(max(0, min(max_by_pace, max_by_limiter)))
        c_tgt = int(q.get("comments_target", 0) or 0)

        if possible_max < c_tgt:
            q["comments_target"] = int(possible_max)
            if cap <= 0:
                proto["hb_block_reason"] = "comment_limit0"
            else:
                proto["hb_block_reason"] = "cooldown"
        else:
            proto["hb_block_reason"] = ""

        q["comments_target_clamped"] = True
    except Exception as e:
        log_debug_exc("_hb_clamp_comment_target:silent", e)
        return


def _hb_maybe_complete(state: Dict[str, Any], now_ts: Optional[float] = None) -> None:
    hb = _hb_get(state)
    if not hb or not bool(hb.get("active")):
        return
    q = _hb_quota(state)
    try:
        c_tgt = int(q.get("comments_target", 0) or 0)
        c_done = int(q.get("comments_done", 0) or 0)
        contrib_done = bool(q.get("contribute_done", False))
    except Exception:
        return
    if c_tgt > 0 and c_done < c_tgt:
        return
    if not contrib_done:
        return
    if now_ts is None:
        now_ts = time.time()
    hb["active"] = False
    hb["completed_at"] = float(now_ts)
    hb["last_at"] = float(now_ts)

def _hb_record_vote(state: Dict[str, Any], ts: Optional[float] = None) -> None:
    hb = _hb_get(state)
    if not hb or not bool(hb.get("active")):
        return
    q = _hb_quota(state)
    try:
        q["votes_done"] = int(q.get("votes_done", 0) or 0) + 1
        if ts is not None:
            hb["last_at"] = float(ts)
    except Exception as e:
        log_debug_exc("_hb_record_vote:silent", e)
        pass

def _hb_record_comment(state: Dict[str, Any], ts: Optional[float] = None) -> None:
    hb = _hb_get(state)
    if not hb or not bool(hb.get("active")):
        return
    q = _hb_quota(state)
    try:
        q["comments_done"] = int(q.get("comments_done", 0) or 0) + 1
        if ts is not None:
            hb["last_at"] = float(ts)
    except Exception as e:
        log_debug_exc("_hb_record_comment:silent", e)
        pass
    _hb_maybe_complete(state, now_ts=ts)

def _hb_record_contribute(state: Dict[str, Any], ts: Optional[float] = None, kind: str = "") -> None:
    hb = _hb_get(state)
    if not hb or not bool(hb.get("active")):
        return
    q = _hb_quota(state)
    try:
        q["contribute_done"] = True
        q["contribute_ts"] = float(ts or time.time())
        hb["last_at"] = float(ts or time.time())
        if kind:
            hb["last_kind"] = str(kind)
    except Exception as e:
        log_debug_exc("_hb_record_contribute:silent", e)
        pass
    _hb_maybe_complete(state, now_ts=ts)

def migrate_brain(b: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(b, dict):
        b = {}

    # action bias (Brain → action probability hooks)
    ab = _sdict(b, "action_bias")
    ab.setdefault("by_action", {})
    ab.setdefault("by_topic_kw", {})
    ab.setdefault("by_template", {})
    ab.setdefault("by_user", {})
    ab.setdefault("last_update_ts", 0.0)
    ab.setdefault("decay", 0.98)

    # EMA slots (optional)
    ema = _sdict(b, "ema")
    ema.setdefault("action_type", {})
    ema.setdefault("template_id", {})
    ema.setdefault("topic_kw", {})
    ema.setdefault("user_nick", {})
    ema.setdefault("hour_bin", {})

    # avoid/cooldown slots
    avoid = _sdict(b, "avoid")
    avoid.setdefault("users", {})
    avoid.setdefault("topics", {})

    # last-known best combo
    last = _sdict(b, "last")
    last.setdefault("best", {})

    b.setdefault("reflection_hashes", [])
    b.setdefault("bias_updates", {"proxy": 0, "reward": 0})
    b.setdefault("max_thoughts", 200)


    # (Unit 15) Normalize stored keywords to reduce awkward 조사-attached tokens and filter rough words.
    try:
        com = _safe_dict(b.get("community"))
        if isinstance(com, dict):
            kwm = _safe_dict(com.get("kw"))
            if kwm:
                new_kw: Dict[str, float] = {}
                for k, v in kwm.items():
                    kk = normalize_ko_token(str(k))
                    if not is_clean_keyword(kk):
                        continue
                    new_kw[kk] = float(new_kw.get(kk, 0.0)) + float(v or 0.0)
                com["kw"] = new_kw

            for key in ("hot", "rising"):
                lst = _safe_list(com.get(key))
                if not lst:
                    continue
                cleaned: List[Dict[str, Any]] = []
                for it in lst:
                    if not isinstance(it, dict):
                        continue
                    kk = normalize_ko_token(str(it.get("kw") or ""))
                    if not is_clean_keyword(kk):
                        continue
                    it2 = dict(it)
                    it2["kw"] = kk
                    cleaned.append(it2)
                if cleaned:
                    com[key] = cleaned

            b["community"] = com

        ema = _safe_dict(b.get("ema"))
        if isinstance(ema, dict):
            tk = _safe_dict(ema.get("topic_kw"))
            if tk:
                new_tk: Dict[str, float] = {}
                for k, v in tk.items():
                    kk = normalize_ko_token(str(k))
                    if not is_clean_keyword(kk):
                        continue
                    new_tk[kk] = max(float(new_tk.get(kk, 0.0)), float(v or 0.0))
                ema["topic_kw"] = new_tk
                b["ema"] = ema

        ab = _safe_dict(b.get("action_bias"))
        if isinstance(ab, dict):
            bk = _safe_dict(ab.get("by_topic_kw"))
            if bk:
                new_bk: Dict[str, float] = {}
                for k, v in bk.items():
                    kk = normalize_ko_token(str(k))
                    if not is_clean_keyword(kk):
                        continue
                    new_bk[kk] = float(v or 0.0)
                ab["by_topic_kw"] = new_bk
                b["action_bias"] = ab
    except Exception as e:
        log_debug_exc("migrate_brain:silent", e)
        pass

    b["schema_ver"] = AGENT_CODE_VERSION
    return b

def load_state(path: str) -> Dict[str, Any]:
    s = load_json_file(path, default={})
    if not isinstance(s, dict):
        s = {}

    # server capability cache
    s.setdefault("post_nickname_supported", None)
    s.setdefault("comment_nickname_supported", None)
    s.setdefault("post_title_supported", None)
    s.setdefault("vote_supported", None)

    # seen and per-target throttles
    s.setdefault("seen_post_ids", [])
    s.setdefault("commented_ts", {})            # post_id -> ts
    s.setdefault("replied_ts", {})              # "post:comment" -> ts
    s.setdefault("voted_posts", {})             # post_id -> {"type":..., "ts":...}

    # repeat protection
    s.setdefault("recent_text_hashes", {})      # {hash: ts}
    s.setdefault("recent_post_text_hashes", {}) # {hash: ts}
    s.setdefault("recent_text_fps", [])         # [[fp, ts], ...]
    s.setdefault("recent_post_text_fps", [])    # [[fp, ts], ...]

    # global last-action timestamps
    s.setdefault("last_comment_ts", 0.0)
    s.setdefault("last_post_ts", 0.0)
    s.setdefault("last_vote_ts", 0.0)

    # daily counters
    s.setdefault("last_contrib_date", "")
    s.setdefault("contrib_count_today", 0)

    # arena tracking
    s.setdefault("arena_last_action_ts", 0.0)
    s.setdefault("arena_last_propose_date", "")

    # sync
    s.setdefault("last_sync_ts", 0.0)

    # learning cadence
    s.setdefault("last_learn_ts", 0.0)
    s.setdefault("learn_runs", 0)

    # snapshot cadence
    s.setdefault("last_snapshot_hour_kst", "")
    s.setdefault("last_snapshot_ts", 0.0)

    # lifetime
    s.setdefault("total_actions", 0)
    s.setdefault("total_reward", 0.0)
    s.setdefault("evaluated_count", 0)

    # expanded context
    s.setdefault("threads", {})
    s.setdefault("users", {})
    s = migrate_state(s)
    # Optional: override evaluation budget via env (0 => unlimited)
    eb = s.get("eval_budget")
    if isinstance(eb, dict):
        eb["cap_per_window"] = _env_int("MERSOOM_EVAL_BUDGET_CAP_PER_WINDOW", int(eb.get("cap_per_window", 0) or 0), 0, 100_000)
        eb["window_sec"] = _env_int("MERSOOM_EVAL_BUDGET_WINDOW_SEC", int(eb.get("window_sec", 900) or 900), 60, 24 * 3600)

    return s

def update_daily_counters(state: Dict[str, Any]) -> None:
    today = _today_kst()
    if state.get("last_contrib_date") != today:
        state["last_contrib_date"] = today
        state["contrib_count_today"] = 0

# (P0) state GC + simple health log
def gc_state(state: Dict[str, Any]) -> None:
    now = time.time()
    interval = _env_int("MERSOOM_GC_INTERVAL_SEC", 600, 30, 24 * 3600)
    last = float(state.get("last_gc_ts", 0.0) or 0.0)
    if (now - last) < float(interval):
        return
    state["last_gc_ts"] = now

    # prune per-post timestamps
    ts_ttl = _env_int("MERSOOM_PERPOST_TTL_DAYS", 14, 1, 365) * 86400
    for k in ["commented_ts", "replied_ts"]:
        d = _safe_dict(state.get(k))
        d2 = {}
        for kk, vv in d.items():
            try:
                if (now - float(vv or 0.0)) <= ts_ttl:
                    d2[str(kk)] = float(vv)
            except Exception:
                continue
        state[k] = d2

    # prune recent text fingerprints (19.4)
    fp_ttl = _env_int("MERSOOM_RECENT_TEXT_FP_TTL_SEC", 6 * 3600, 60, 30 * 24 * 3600)
    fp_keep = _env_int("MERSOOM_RECENT_TEXT_FP_KEEP_MAX", 1200, 50, 20000)
    for k in ["recent_text_fps", "recent_post_text_fps"]:
        lst = _safe_list(state.get(k))
        out = []
        for it in lst:
            try:
                fp, ts = it[0], float(it[1])
                if (now - ts) <= float(fp_ttl):
                    out.append([fp, ts])
            except Exception:
                continue
        state[k] = out[-int(fp_keep):]

    # prune reply tracking (Unit 05)
    seen = _safe_dict(state.get("reply_seen_ids"))
    seen_ttl = _env_int("MERSOOM_REPLY_SEEN_TTL_DAYS", 21, 1, 365) * 86400
    seen2 = {}
    for cid, ts in seen.items():
        try:
            if (now - float(ts or 0.0)) <= seen_ttl:
                seen2[str(cid)] = float(ts)
        except Exception:
            continue
    max_seen = _env_int("MERSOOM_MAX_REPLY_SEEN", 2000, 200, 200000)
    if len(seen2) > max_seen:
        items = sorted(seen2.items(), key=lambda kv: float(kv[1] or 0.0))
        seen2 = dict(items[-max_seen:])
    state["reply_seen_ids"] = seen2

    reps = _safe_dict(state.get("my_comment_replies"))
    reps_ttl = _env_int("MERSOOM_MYREPLY_TTL_DAYS", 60, 7, 365) * 86400
    reps2: Dict[str, Any] = {}
    for cid, obj in reps.items():
        o = _safe_dict(obj)
        try:
            ts = float(o.get("last_ts", 0.0) or 0.0)
            if ts and (now - ts) <= reps_ttl:
                reps2[str(cid)] = o
        except Exception:
            continue
    max_reps = _env_int("MERSOOM_MAX_MYREPLY_MAP", 1500, 200, 200000)
    if len(reps2) > max_reps:
        items = sorted(reps2.items(), key=lambda kv: float(_safe_dict(kv[1]).get("last_ts", 0.0) or 0.0))
        reps2 = dict(items[-max_reps:])
    state["my_comment_replies"] = reps2

    # prune voted posts
    vp = _safe_dict(state.get("voted_posts"))
    vp2 = {}
    for pid, obj in vp.items():
        try:
            ts = float(_safe_dict(obj).get("ts", 0.0) or 0.0)
            if (now - ts) <= ts_ttl:
                vp2[str(pid)] = obj
        except Exception:
            continue
    # keep under cap
    max_vp = _env_int("MERSOOM_MAX_VOTED_POSTS", 15000, 1000, 200000)
    if len(vp2) > max_vp:
        items = sorted(vp2.items(), key=lambda kv: float(_safe_dict(kv[1]).get("ts", 0.0) or 0.0))
        vp2 = dict(items[-max_vp:])
    state["voted_posts"] = vp2

    # prune protocol.vote_backlog (v20.2)
    try:
        vote_backlog_gc(state, now_ts=now)
    except Exception as e:
        log_debug_exc("vote_backlog_gc", e)



    # prune relations (P1 user affinity)
    rel = _safe_dict(state.get("relations"))
    rel_ttl = _env_int("MERSOOM_REL_TTL_DAYS", 180, 7, 3650) * 86400
    max_rel = _env_int("MERSOOM_MAX_RELATIONS", 6000, 50, 200000)
    rel2: Dict[str, Any] = {}
    for k, obj in rel.items():
        o = _safe_dict(obj)
        try:
            ts = float(o.get("last_ts", 0.0) or 0.0)
            if ts and (now - ts) <= rel_ttl:
                rel2[str(k)] = o
        except Exception:
            continue
    if len(rel2) > max_rel:
        items = sorted(rel2.items(), key=lambda kv: float(_safe_dict(kv[1]).get("last_ts", 0.0) or 0.0))
        rel2 = dict(items[-max_rel:])
    state["relations"] = rel2

    # prune thread/user models
    threads = _safe_dict(state.get("threads"))
    users = _safe_dict(state.get("users"))

    thread_ttl = _env_int("MERSOOM_THREAD_TTL_DAYS", 45, 1, 365) * 86400
    user_ttl = _env_int("MERSOOM_USER_TTL_DAYS", 120, 7, 3650) * 86400
    max_threads = _env_int("MERSOOM_MAX_THREADS", 1200, 50, 50000)
    max_users = _env_int("MERSOOM_MAX_USERS", 3000, 50, 200000)

    def _last_seen(obj: Dict[str, Any]) -> float:
        try:
            return float(obj.get("last_seen_ts", 0.0) or 0.0)
        except Exception:
            return 0.0

    # threads: TTL + cap
    th2: Dict[str, Any] = {}
    for pid, th in threads.items():
        if not isinstance(th, dict):
            continue
        ls = _last_seen(th)
        if ls and (now - ls) > thread_ttl:
            continue

        # seen_comment_ids TTL + cap
        seen = _safe_dict(th.get("seen_comment_ids"))

    # v23.4: open-question lifecycle maintenance (expire old)
    try:
        if _openq_track_enabled(state):
            thread_openq_expire_old(state, th)
    except Exception:
        pass
        seen_ttl = _env_int("MERSOOM_SEEN_COMMENT_TTL_DAYS", 14, 1, 365) * 86400
        if seen:
            seen = {str(cid): float(ts) for cid, ts in seen.items() if (now - float(ts or 0.0)) <= seen_ttl}
            if len(seen) > 800:
                items = sorted(seen.items(), key=lambda kv: float(kv[1] or 0.0))
                seen = dict(items[-700:])
        th["seen_comment_ids"] = seen

        # cap last_k_turns, open_questions
        th["last_k_turns"] = _safe_list(th.get("last_k_turns"))[-max(20, _env_int("MERSOOM_THREAD_TURNS_KEEP", 80, 20, 500)):]
        th["open_questions"] = _safe_list(th.get("open_questions"))[-max(5, _env_int("MERSOOM_THREAD_Q_KEEP", 30, 5, 300)):]

        th2[str(pid)] = th

    if len(th2) > max_threads:
        items = sorted(th2.items(), key=lambda kv: _last_seen(_safe_dict(kv[1])))
        th2 = dict(items[-max_threads:])
    state["threads"] = th2

    # users: TTL + cap
    u2: Dict[str, Any] = {}
    for nick, u in users.items():
        if not isinstance(u, dict):
            continue
        ls = _last_seen(u)
        if ls and (now - ls) > user_ttl:
            continue
        u2[str(nick)] = u
    if len(u2) > max_users:
        items = sorted(u2.items(), key=lambda kv: _last_seen(_safe_dict(kv[1])))
        u2 = dict(items[-max_users:])
    state["users"] = u2

def log_health_if_due(client: HttpClient, state: Dict[str, Any]) -> None:
    now = time.time()
    interval = _env_int("MERSOOM_HEALTH_LOG_INTERVAL_SEC", 900, 60, 24 * 3600)
    last = float(state.get("last_health_log_ts", 0.0) or 0.0)
    if (now - last) < float(interval):
        return
    state["last_health_log_ts"] = now
    try:
        snap = client.health_snapshot()
        try:
            snap["tz_name"] = str(TZ_NAME)
            snap["tz_fallback_used_10m"] = int(protocol_get_counter_10m(state, "tz_fallback_used"))
        except Exception:
            pass
        try:
            snap["dup_action_skips_10m"] = int(protocol_get_counter_10m(state, "dup_action_skip"))
            snap["recent_actions_size"] = int(len(_recent_actions_get(state)))
        except Exception:
            pass
        try:
            now2 = time.time()
            snap["pow_timeouts_10m"] = int(sum(1 for t in list(_POW_TIMEOUT_TS) if (now2 - float(t or 0.0)) <= 600.0))
            snap["pow_executor_restarts_10m"] = int(sum(1 for t in list(_POW_EXECUTOR_RESTART_TS) if (now2 - float(t or 0.0)) <= 600.0))
        except Exception:
            pass
        try:
            snap["bm25_build_ms_p95_10m"] = int(bm25_build_p95_ms(state))
            snap["bm25_docs_indexed"] = int(state.get("bm25_docs_indexed", 0) or 0)
            if str(state.get("bm25_last_build_mode", "")):
                snap["bm25_build_mode"] = str(state.get("bm25_last_build_mode"))
                if str(state.get("bm25_last_build_mode")) == "partial":
                    snap["bm25_build_note"] = "partial_rebuild_experimental"
        except Exception:
            pass
        try:
            snap["action_attempt_10m"] = int(protocol_get_counter_10m(state, "action_attempt"))
            snap["action_success_10m"] = int(protocol_get_counter_10m(state, "action_success"))
            snap["action_fail_10m"] = int(protocol_get_counter_10m(state, "action_fail"))
        except Exception:
            pass
        try:
            snap["qa_fail_bucket_10m"] = protocol_top_counters_10m(state, prefix="qa_fail_bucket:", topn=3)
        except Exception:
            pass
        try:
            now2 = time.time()
            snap["postprocess_dedupe_sentences_10m"] = int(sum(1 for t in list(_POSTPROCESS_DEDUPE_TS) if (now2 - float(t or 0.0)) <= 600.0))
        except Exception:
            pass

        # v20.2: vote backlog observability
        try:
            proto = _safe_dict(state.get("protocol"))
            bl = _safe_list(proto.get("vote_backlog"))
            drains = _safe_list(proto.get("vote_backlog_drains"))
            drained = 0
            for t in drains:
                try:
                    if (now - float(t or 0.0)) <= 600.0:
                        drained += 1
                except Exception:
                    continue
            snap["vote_backlog_len"] = int(len(bl))
            snap["vote_backlog_drained"] = int(drained)
        except Exception as e:
            log_debug_exc("health:vote_backlog", e)


        # v23.1: interaction scaffolding metrics (threads/open questions)
        try:
            ths = state.get("threads")
            if isinstance(ths, dict):
                tc = 0
                oq_open = 0
                oq_total = 0
                oq_threads = 0
                for pid, th in ths.items():
                    if pid == "__meta__":
                        continue
                    if not isinstance(th, dict):
                        continue
                    tc += 1
                    open_in_thread = 0
                    oq = th.get("open_questions")
                    if isinstance(oq, list):
                        # v23.1+ schema: count only status=="open" when available; also track total and thread coverage
                        for _q in oq:
                            if isinstance(_q, dict):
                                oq_total += 1
                                st = str(_q.get("status") or "open")
                                if st == "open":
                                    oq_open += 1
                                    open_in_thread += 1
                            else:
                                # legacy list item: treat as open-question
                                oq_total += 1
                                oq_open += 1
                                open_in_thread += 1
                    if open_in_thread > 0:
                        oq_threads += 1
                snap["thread_count"] = int(tc)
                snap["open_q_count"] = int(oq_open)
                snap["open_q_total"] = int(oq_total)
                snap["open_q_threads"] = int(oq_threads)
                snap["open_q_added_10m"] = int(protocol_get_counter_10m(state, "openq_add"))
                snap["open_q_resolved_10m"] = int(protocol_get_counter_10m(state, "openq_resolve"))
                snap["open_q_expired_10m"] = int(protocol_get_counter_10m(state, "openq_expire"))
                try:
                    snap["open_q_added_total"] = int(_safe_dict(state.get("protocol")).get("openq_added_total", 0) or 0)
                    snap["open_q_resolved_total"] = int(_safe_dict(state.get("protocol")).get("openq_resolved_total", 0) or 0)
                    snap["open_q_expired_total"] = int(_safe_dict(state.get("protocol")).get("openq_expired_total", 0) or 0)
                except Exception:
                    snap["open_q_added_total"] = 0
                # v23.2: thread phase distribution + transitions (10m rolling counter)
                pc = {"open": 0, "argue": 0, "clarify": 0, "close": 0}
                try:
                    for _pid, _th in _safe_dict(state.get("threads")).items():
                        if _pid == "__meta__":
                            continue
                        if not isinstance(_th, dict):
                            continue
                        ph = str(_th.get("phase") or "open")
                        if ph not in pc:
                            ph = "open"
                        pc[ph] = int(pc.get(ph, 0) or 0) + 1
                except Exception:
                    pass
                snap["thread_phase_counts"] = pc
                snap["phase_transitions_10m"] = int(protocol_get_counter_10m(state, "phase_transition"))
                # v23.5: waiting-for-remote observability
                try:
                    wt = 0
                    convs = state.get("conv_state")
                    if isinstance(convs, dict):
                        for _ck, _cv in convs.items():
                            if isinstance(_cv, dict) and bool(_cv.get("waiting_for_remote")):
                                wt += 1
                    snap["waiting_threads"] = int(wt)
                    snap["waiting_skips_10m"] = int(protocol_get_counter_10m(state, "waiting_skip"))
                except Exception:
                    pass



        except Exception as e:
            log_debug_exc("health:threads_openq", e)


        # v20.8 (A-3): recent top reasons (10m) + reply inbox backlog + ops disabled summary
        try:
            proto2 = _safe_dict(state.get("protocol"))
            win = _safe_dict(proto2.get("reason_window_10m"))
            started = float(proto2.get("reason_window_started_ts", 0.0) or 0.0)
            items2: List[Dict[str, Any]] = []
            for k, v in win.items():
                try:
                    items2.append({"code": str(k)[:80], "n": int(v or 0)})
                except Exception:
                    continue
            items2 = sorted(items2, key=lambda d: int(d.get("n", 0) or 0), reverse=True)
            snap["reason_top5"] = items2[:5]
            if started > 0:
                snap["reason_window_age_sec"] = int(max(0.0, now - started))
        except Exception as e:
            log_debug_exc("health:reason", e)

        try:
            cache = _safe_dict(state.get("reply_inbox_cache"))
            snap["reply_inbox_len"] = int(len(_safe_list(cache.get("items", []))))

            # v23.6: reply queue age and scoring cadence
            try:
                items = _safe_list(cache.get("items", []))
                ages: List[float] = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    ts = float(it.get("comment_ts", 0.0) or 0.0)
                    if ts > 0:
                        ages.append(max(0.0, now - ts))
                snap["reply_queue_age_max"] = int(max(ages) if ages else 0.0)
                # v23.7: p95 age for backlog sensitivity
                try:
                    if ages:
                        ages2 = sorted(ages)
                        idx95 = int(0.95 * float(len(ages2) - 1))
                        idx95 = max(0, min(len(ages2) - 1, idx95))
                        snap["reply_queue_age_p95"] = int(ages2[idx95])
                    else:
                        snap["reply_queue_age_p95"] = 0
                except Exception:
                    snap["reply_queue_age_p95"] = 0
            except Exception:
                snap["reply_queue_age_max"] = 0
                snap["reply_queue_age_p95"] = 0
            try:
                snap["reply_scored_10m"] = int(protocol_get_counter_10m(state, "reply_scored"))
            except Exception:
                snap["reply_scored_10m"] = 0

        except Exception as e:
            log_debug_exc("health:reply_inbox", e)

        # v23.7: quick interaction hint (why quiet?)
        try:
            hint: List[str] = []
            if int(snap.get("waiting_threads", 0) or 0) > 0:
                hint.append("wait")
            if int(snap.get("open_q_count", 0) or 0) > 0:
                hint.append("openq")
            if int(snap.get("reply_inbox_len", 0) or 0) <= 0:
                hint.append("inbox0")
            else:
                if int(snap.get("reply_queue_age_p95", 0) or 0) > 1800:
                    hint.append("queue_old")
            snap["interaction_hint"] = "|".join(hint) if hint else "ok"
        except Exception:
            pass

        try:
            ops = _safe_dict(state.get("ops"))
            du = _safe_dict(ops.get("disabled_until"))
            lfr = _safe_dict(ops.get("last_fail_reason"))
            out: List[Dict[str, Any]] = []
            for k, until in du.items():
                try:
                    sec = int(max(0.0, float(until or 0.0) - now))
                    if sec > 0:
                        out.append({
                            "key": str(k),
                            "sec": sec,
                            "last": one_line(str(lfr.get(k, "") or ""), 160),
                        })
                except Exception:
                    continue
            out = sorted(out, key=lambda d: int(d.get("sec", 0) or 0), reverse=True)
            if out:
                snap["ops_disabled"] = out
        except Exception as e:
            log_debug_exc("health:ops_disabled", e)
        # v23.0 (opt-in): extended HEALTH fields via MERSOOM_HEALTH_V2
        if _env_bool("MERSOOM_HEALTH_V2", False):
            try:
                protocol_bump_counter(state, "health_emit", 1)
                snap["loop_tick_10m"] = int(protocol_get_counter_10m(state, "loop_tick"))
                snap["health_emit_10m"] = int(protocol_get_counter_10m(state, "health_emit"))
            except Exception as e:
                log_debug_exc("health:v2", e)



        log_info("HEALTH " + json.dumps(snap, ensure_ascii=False, sort_keys=True))
    except Exception:
        return

def boot_self_test(client: HttpClient, cfg: Config, state: Dict[str, Any]) -> None:
    """(P0) Basic connectivity + endpoint smoke test."""
    ok = True
    errs: List[str] = []
    try:
        _ = list_posts(client, limit=1)
    except Exception as e:
        ok = False
        errs.append("list_posts:" + one_line(str(e), 140))
    try:
        _ = fetch_pow_challenge(client, cfg.hybrid)
    except Exception as e:
        # not fatal if challenge not supported
        msg = str(e)
        if ("404" in msg) or ("405" in msg):
            pass
        else:
            ok = False
            errs.append("challenge:" + one_line(msg, 140))

    state["boot_self_test_ts"] = time.time()
    state["boot_self_test_ok"] = bool(ok)
    state["boot_self_test_err"] = "; ".join(errs)[:400]
    if ok:
        log_info("BOOT_SELF_TEST ok")
    else:
        log_warn("BOOT_SELF_TEST fail: " + state.get("boot_self_test_err", ""))

def load_memory(path: str, tuning: AgentTuning) -> List[Dict[str, Any]]:
    m = load_json_file(path, default=[])
    if not isinstance(m, list):
        return []
    out = [x for x in m if isinstance(x, dict)]
    return out[-max(1, int(tuning.memory_size)):]

def record_memory(
    memory: List[Dict[str, Any]],
    item: Dict[str, Any],
    tuning: AgentTuning,
    archive_path_jsonl: str,
) -> None:
    memory.append(item)
    maxn = max(1, int(tuning.memory_size))
    if len(memory) > maxn:
        del memory[:-maxn]

    # archive (optional)
    if archive_path_jsonl:
        append_jsonl(archive_path_jsonl, item)

def load_semantic(path: str) -> Dict[str, Any]:
    s = load_json_file(path, default={})
    if not isinstance(s, dict):
        s = {}
    s.setdefault("version", 3)
    s.setdefault("by_day", {})
    return s

def bump_semantic(semantic: Dict[str, Any], day: str, key: str, val: float = 1.0) -> None:
    semantic.setdefault("by_day", {})
    semantic["by_day"].setdefault(day, {})
    semantic["by_day"][day][key] = float(semantic["by_day"][day].get(key, 0.0)) + float(val)


def _vote_proto_limits(cfg: Optional[Config]) -> Tuple[int, int]:
    """Return (seen_post_limit, voted_post_limit) with safe fallbacks."""
    try:
        vp = getattr(cfg, "vote_proto", None) if cfg is not None else None
        if vp is not None:
            return int(getattr(vp, "seen_post_limit", 500)), int(getattr(vp, "voted_post_limit", 5000))
    except Exception as e:
        log_debug_exc("_vote_proto_limits:silent", e)
        pass
    # fallback to env (keeps file runnable even if Config changes)
    return int(_env_int("MERSOOM_SEEN_POST_LIMIT", 500, min_v=50, max_v=50000)), int(_env_int("MERSOOM_VOTED_POST_LIMIT", 5000, min_v=200, max_v=200000))

def _lru_prune_map(d: Dict[str, Any], limit: int) -> None:
    if limit <= 0:
        return
    if not isinstance(d, dict):
        return
    if len(d) <= limit:
        return
    try:
        items = []
        for k, v in d.items():
            try:
                ts = float(v or 0.0)
            except Exception:
                ts = 0.0
            items.append((ts, k))
        items.sort(key=lambda x: x[0])
        for _, k in items[: max(0, len(items) - limit)]:
            try:
                d.pop(k, None)
            except Exception as e:
                log_debug_exc("_lru_prune_map:silent", e)
                pass
    except Exception:
        # best-effort; never crash the agent on pruning
        return


# -----------------------------------------------------------------------------
# v20.2 VOTE BACKLOG (durable mandatory vote)
# -----------------------------------------------------------------------------

def _vote_backlog_limits() -> Tuple[int, int]:
    """Return (max_items, ttl_sec) for protocol.vote_backlog with safe fallbacks.

    NOTE (v20.10_final): historically the backlog cap env key drifted between
    MERSOOM_VOTE_BACKLOG_MAX and MERSOOM_VOTE_BACKLOG_KEEP_MAX. We accept BOTH.
    If both are set, we use the smaller value as the effective cap.
    """
    raw_max = os.getenv("MERSOOM_VOTE_BACKLOG_MAX")
    raw_keep = os.getenv("MERSOOM_VOTE_BACKLOG_KEEP_MAX")

    if raw_max is None and raw_keep is not None:
        # legacy alias
        max_items = _env_int("MERSOOM_VOTE_BACKLOG_KEEP_MAX", 300, 1, 100000)
    else:
        max_items = _env_int("MERSOOM_VOTE_BACKLOG_MAX", 300, 1, 100000)

    # If both provided, keep <= max for safety
    if raw_max is not None and raw_keep is not None:
        try:
            mx = _env_int("MERSOOM_VOTE_BACKLOG_MAX", int(max_items), 1, 100000)
            kp = _env_int("MERSOOM_VOTE_BACKLOG_KEEP_MAX", int(mx), 1, 100000)
            max_items = int(min(mx, kp))
        except Exception:
            pass

    ttl_sec = _env_int("MERSOOM_VOTE_BACKLOG_TTL_SEC", 48 * 3600, 60, 30 * 24 * 3600)
    return int(max_items), int(ttl_sec)

def _vote_backlog_list(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    proto = state.setdefault("protocol", {})
    if not isinstance(proto, dict):
        state["protocol"] = {}
        proto = state["protocol"]
    bl = proto.get("vote_backlog")
    if not isinstance(bl, list):
        bl = []
        proto["vote_backlog"] = bl
    # keep as list of dicts (sanitized in vote_backlog_gc)
    return bl  # type: ignore[return-value]

def _vote_backlog_drains_list(state: Dict[str, Any]) -> List[float]:
    proto = state.setdefault("protocol", {})
    if not isinstance(proto, dict):
        state["protocol"] = {}
        proto = state["protocol"]
    d = proto.get("vote_backlog_drains")
    if not isinstance(d, list):
        d = []
        proto["vote_backlog_drains"] = d
    return d  # type: ignore[return-value]

def vote_backlog_record_drain(state: Dict[str, Any], ts: float) -> None:
    drains = _vote_backlog_drains_list(state)
    try:
        drains.append(float(ts))
    except Exception:
        return
    keep = _env_int("MERSOOM_VOTE_BACKLOG_DRAIN_KEEP_MAX", 2000, 50, 20000)
    if len(drains) > int(keep):
        del drains[:-int(keep)]
    # ensure persisted
    proto = state.get("protocol")
    if isinstance(proto, dict):
        proto["vote_backlog_drains"] = drains

def vote_backlog_enqueue(state: Dict[str, Any], post_id: str, seen_ts: float) -> bool:
    pid = str(post_id or "")
    if not pid:
        return False
    bl = _vote_backlog_list(state)
    for it in bl:
        if isinstance(it, dict) and str(it.get("post_id") or "") == pid:
            return False
    bl.append({"post_id": pid, "seen_ts": float(seen_ts)})
    proto = state.get("protocol")
    if isinstance(proto, dict):
        proto["vote_backlog"] = bl
    return True

def vote_backlog_remove(state: Dict[str, Any], post_id: str) -> bool:
    pid = str(post_id or "")
    if not pid:
        return False
    bl = _vote_backlog_list(state)
    removed = False
    out: List[Dict[str, Any]] = []
    for it in bl:
        if isinstance(it, dict) and str(it.get("post_id") or "") == pid and not removed:
            removed = True
            continue
        if isinstance(it, dict):
            out.append(it)
    if removed:
        proto = state.get("protocol")
        if isinstance(proto, dict):
            proto["vote_backlog"] = out
    return removed

def vote_backlog_gc(state: Dict[str, Any], now_ts: Optional[float] = None) -> None:
    """GC protocol.vote_backlog: TTL, cap, de-dup, and remove already-voted posts."""
    now = time.time() if now_ts is None else float(now_ts)
    max_items, ttl_sec = _vote_backlog_limits()

    proto = state.setdefault("protocol", {})
    if not isinstance(proto, dict):
        state["protocol"] = {}
        proto = state["protocol"]

    bl_raw = proto.get("vote_backlog")
    bl = bl_raw if isinstance(bl_raw, list) else []
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for it in bl:
        if not isinstance(it, dict):
            continue
        pid = str(it.get("post_id") or "")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        try:
            sts = float(it.get("seen_ts", 0.0) or 0.0)
        except Exception:
            sts = 0.0

        # TTL
        if sts and ttl_sec > 0 and (now - sts) > float(ttl_sec):
            continue

        # already voted -> remove
        try:
            if _is_post_voted(state, pid):
                vote_backlog_record_drain(state, now)
                continue
        except Exception:
            pass

        out.append({"post_id": pid, "seen_ts": float(sts)})

    # sort by seen_ts asc (oldest first)
    out.sort(key=lambda x: float(x.get("seen_ts", 0.0) or 0.0))

    # cap: drop oldest first
    if max_items > 0 and len(out) > int(max_items):
        out = out[-int(max_items):]

    proto["vote_backlog"] = out

    # drains history gc
    drains = _vote_backlog_drains_list(state)
    hist_ttl = _env_int("MERSOOM_VOTE_BACKLOG_DRAIN_TTL_SEC", 3600, 60, 7 * 24 * 3600)
    keep = _env_int("MERSOOM_VOTE_BACKLOG_DRAIN_KEEP_MAX", 2000, 50, 20000)
    d2: List[float] = []
    for t in drains:
        try:
            ft = float(t)
        except Exception:
            continue
        if hist_ttl > 0 and (now - ft) > float(hist_ttl):
            continue
        d2.append(ft)
    if len(d2) > int(keep):
        d2 = d2[-int(keep):]
    proto["vote_backlog_drains"] = d2

def vote_backlog_pick(state: Dict[str, Any]) -> Optional[str]:
    """Return the oldest backlog post_id (after GC) without removing it."""
    bl = _vote_backlog_list(state)
    if not bl:
        return None
    it0 = bl[0] if isinstance(bl[0], dict) else None
    pid = str(it0.get("post_id") or "") if isinstance(it0, dict) else ""
    return pid or None


def _seen_posts_map(state: Dict[str, Any]) -> Dict[str, Any]:
    seen = state.setdefault("seen", {})
    if not isinstance(seen, dict):
        state["seen"] = {}
        seen = state["seen"]
    posts = seen.setdefault("posts", {})
    if not isinstance(posts, dict):
        seen["posts"] = {}
        posts = seen["posts"]
    return posts

def _voted_posts_map(state: Dict[str, Any]) -> Dict[str, Any]:
    votes = state.setdefault("votes", {})
    if not isinstance(votes, dict):
        state["votes"] = {}
        votes = state["votes"]
    posts = votes.setdefault("posts", {})
    if not isinstance(posts, dict):
        votes["posts"] = {}
        posts = votes["posts"]
    return posts

def _legacy_voted_posts_map(state: Dict[str, Any]) -> Dict[str, Any]:
    vp = state.get("voted_posts")
    if not isinstance(vp, dict):
        state["voted_posts"] = {}
        vp = state["voted_posts"]
    return vp

def _is_post_voted(state: Dict[str, Any], post_id: str) -> bool:
    pid = str(post_id or "")
    if not pid:
        return False
    try:
        if pid in _voted_posts_map(state):
            return True
    except Exception as e:
        log_debug_exc("_is_post_voted:silent", e)
        pass
    try:
        if pid in _legacy_voted_posts_map(state):
            return True
    except Exception as e:
        log_debug_exc("_is_post_voted:silent", e)
        pass
    return False

def _record_seen_post(cfg: Optional[Config], state: Dict[str, Any], post_id: str, ts: float) -> None:
    pid = str(post_id or "")
    if not pid:
        return
    m = _seen_posts_map(state)
    m[pid] = float(ts)
    seen_limit, _ = _vote_proto_limits(cfg)
    _lru_prune_map(m, seen_limit)

def _record_voted_post(cfg: Optional[Config], state: Dict[str, Any], post_id: str, vtype: str, ts: float) -> None:
    pid = str(post_id or "")
    if not pid:
        return
    # nested map
    vm = _voted_posts_map(state)
    vm[pid] = float(ts)
    # legacy map (kept for backward compatibility)
    legacy = _legacy_voted_posts_map(state)
    legacy[pid] = {"type": str(vtype or ""), "ts": float(ts)}
    # prune both
    _, voted_limit = _vote_proto_limits(cfg)
    _lru_prune_map(vm, voted_limit)
    if voted_limit > 0 and isinstance(legacy, dict) and len(legacy) > voted_limit:
        try:
            items = []
            for k, info in legacy.items():
                t = 0.0
                if isinstance(info, dict):
                    try:
                        t = float(info.get("ts", 0.0) or 0.0)
                    except Exception:
                        t = 0.0
                items.append((t, k))
            items.sort(key=lambda x: x[0])
            for _, k in items[: max(0, len(items) - voted_limit)]:
                legacy.pop(k, None)
        except Exception as e:
            log_debug_exc("_record_voted_post:silent", e)
            pass


def load_brain(path: str) -> Dict[str, Any]:
    b = load_json_file(path, default={})
    if not isinstance(b, dict):
        b = {}
    b.setdefault("version", 3)
    b.setdefault("last_memory_ts", 0.0)
    b.setdefault("mood", {"ema_reward": 0.0, "valence": 0.0, "arousal": 0.0})
    b.setdefault("topic_ema", {})
    b.setdefault("beliefs", {})
    b.setdefault("thoughts", [])
    b.setdefault("thought_seq", 0)
    b.setdefault("community", {"kw": {}, "by_cat": {}, "last_ts": 0.0, "hot": [], "rising": [], "last_delta": {}})
    b.setdefault("persona", {
        "stance_axes": {
            "skepticism": 0.5,
            "cooperation": 0.6,
            "verbosity": 0.4,
            "conflict": 0.3,
        },
        # 핵심 드라이브(성향): "철학의 극" + "네임드 지향" + "대댓글 논쟁 선호" + "점진적 적응"
        "drives": {
            "philosophy": 0.82,
            "fame": 0.75,
            "debate": 0.80,
            "adaptation": 0.78,
        },
        # 경험치 기반 성숙도(초기엔 템플릿 의존, 점점 자율적으로 변형/학습)
        "maturity": {"level": 0.0, "xp": 0, "last_ts": 0.0},
        "goals": {
            "be_named": True,
            "seek_philosophy_extremes": True,
            "enjoy_debate_on_own_posts": True,
        },
        "taboos": [],
        "signature": "eum",
    })

    # normalize persona (backward compatible)
    per = b.get("persona")
    if not isinstance(per, dict):
        per = {}
        b["persona"] = per
    per.setdefault("stance_axes", {
        "skepticism": 0.5,
        "cooperation": 0.6,
        "verbosity": 0.4,
        "conflict": 0.3,
    })
    per.setdefault("drives", {"philosophy": 0.82, "fame": 0.75, "debate": 0.80, "adaptation": 0.78})
    per.setdefault("maturity", {"level": 0.0, "xp": 0, "last_ts": 0.0})
    per.setdefault("goals", {"be_named": True, "seek_philosophy_extremes": True, "enjoy_debate_on_own_posts": True})
    per.setdefault("taboos", [])
    per.setdefault("signature", "eum")
    b = migrate_brain(b)
    return b

def write_journal(journal_path: str, summary: str) -> None:
    ts = now_kst_str()
    append_text_file(journal_path, f"[{ts}] {summary}\n")

################################################################################

################################################################################
# 7.2. FOCUS HELPERS (comment/reply target grounding)
# - Dependencies: Section 7-8 (Schemas, Text)
# - Used by: Reply/comment builders
# - Key functions: validate_grounding(), build_focus_context()
################################################################################

def _norm_excerpt(s: Any, max_len: int = 220) -> str:
    """Collapse whitespace and truncate for compact focus storage/logs."""
    if s is None:
        return ""
    try:
        t = str(s)
    except Exception:
        return ""
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[:max_len].rstrip() + "…"
    return t

def _post_title_guess(post: Dict[str, Any]) -> str:
    if not isinstance(post, dict):
        return ""
    for k in ("title", "subject", "name"):
        v = post.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: first part of content
    for k in ("content", "text", "body"):
        v = post.get(k)
        if isinstance(v, str) and v.strip():
            return one_line(v.strip())[:80]
    return ""

def _post_excerpt_guess(post: Dict[str, Any]) -> str:
    if not isinstance(post, dict):
        return ""
    for k in ("content", "text", "body", "content_text", "description"):
        v = post.get(k)
        if v:
            return _norm_excerpt(v, 240)
    return ""

def _comment_excerpt_guess(c: Dict[str, Any]) -> str:
    if not isinstance(c, dict):
        return ""
    for k in ("content", "text", "body"):
        v = c.get(k)
        if v:
            return _norm_excerpt(v, 200)
    return ""

def set_focus(
    state: Dict[str, Any],
    *,
    mode: str,  # "comment" | "reply"
    post_id: str,
    post: Optional[Dict[str, Any]] = None,
    comment: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the current comment/reply target into state['focus'].

    v15 Unit 02: lock the target *before* text generation so later units can ground on it.
    """
    if mode not in ("comment", "reply"):
        mode = "comment"

    p = post or {}
    c = comment or {}

    focus = _safe_dict(state.get("focus"))
    focus["mode"] = mode
    focus["post_id"] = str(post_id or "")
    focus["post_title"] = _post_title_guess(p)
    focus["post_excerpt"] = _post_excerpt_guess(p)
    focus["comment_id"] = str(c.get("id") or "") if c else ""
    focus["comment_excerpt"] = _comment_excerpt_guess(c) if c else ""
    focus["comment_author"] = str(c.get("nickname") or "") if c else ""
    focus["created_ts"] = float(time.time())

    state["focus"] = focus

    # attach recent QA issue-boost map for scoring/generation (best-effort)
    try:
        qb = _safe_dict(state.get("qa_issue_boost"))
        if qb:
            focus["qa_issue_boost"] = qb
            state["focus"] = focus
    except Exception as e:
        log_debug_exc("set_focus:silent", e)
        pass

    # lightweight observability
    cid = focus.get("comment_id") or ""
    title = one_line(str(focus.get("post_title") or ""))[:60]
    if cid:
        log_action("FOCUS", f"mode={mode} post_id={post_id} comment_id={cid} title={title}")
    else:
        log_action("FOCUS", f"mode={mode} post_id={post_id} title={title}")

def clear_focus(state: Dict[str, Any]) -> None:
    focus = _safe_dict(state.get("focus"))
    focus["mode"] = ""
    focus["post_id"] = ""
    focus["post_title"] = ""
    focus["post_excerpt"] = ""
    focus["comment_id"] = ""
    focus["comment_excerpt"] = ""
    focus["comment_author"] = ""
    focus["created_ts"] = float(time.time())
    state["focus"] = focus

# -----------------------------------------------------------------------------
# v15 Unit 03 — Compose Input (focus + thread summary + constraints)
# -----------------------------------------------------------------------------
def build_compose_input(
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    th: Dict[str, Any],
    user: Dict[str, Any],
    *,
    is_reply: bool,
    reply_to_own_post: bool = False,
) -> Dict[str, Any]:
    """
    Unify the text-generation inputs into a single bundle.

    Unit 03 goal:
      compose_input = {focus + thread_summary + constraints}

    Notes:
      - For replies, comment_excerpt must be the highest-priority grounding text.
      - We keep this lightweight (no huge bodies) and compute summaries/keywords from excerpts.
    """
    focus = _safe_dict(state.get("focus"))
    mode = str(focus.get("mode") or ("reply" if (is_reply or reply_to_own_post) else "comment"))
    if mode not in ("comment", "reply"):
        mode = "comment"

    post_title = str(focus.get("post_title") or "")
    post_excerpt = str(focus.get("post_excerpt") or "")
    comment_excerpt = str(focus.get("comment_excerpt") or "")
    comment_author = str(focus.get("comment_author") or "")
    thread_summary = str(th.get("summary") or "")

    # Fallback thread summary from last turns (cheap)
    if not thread_summary:
        last_turns = _safe_list(th.get("last_k_turns"))[-6:]
        ctx = " ".join([str(x.get("text") or "") for x in last_turns])
        thread_summary = _simple_summary(ctx, max_len=160) if ctx else ""

    # Grounding target: reply -> comment_excerpt, else post_excerpt -> summary
    target_kind = "thread"
    target_text = ""
    if mode == "reply" and comment_excerpt:
        target_kind = "comment"
        target_text = comment_excerpt
    elif post_excerpt:
        target_kind = "post"
        target_text = post_excerpt
    elif thread_summary:
        target_kind = "thread"
        target_text = thread_summary
    else:
        last_turns = _safe_list(th.get("last_k_turns"))[-4:]
        target_text = " ".join([str(x.get("text") or "") for x in last_turns])

    target_text = one_line(target_text, 260)  # keep it bounded
    target_summary = _simple_summary(target_text, max_len=110) if target_text else ""
    target_keywords = top_keywords(target_text, k=6) if target_text else []
    thread_keywords = _safe_list(th.get("keywords"))[:8]

    # Constraints bundle (kept small; later units can expand)
    constraints = {
        "eum_style": True,
        "logical_professor": True,
        "no_insults": True,
        "no_markdown": True,
        "max_lines": int(getattr(tuning, "max_output_lines", 4) or 4),
        "mode": mode,
    }

    return {
        "has_focus": bool(focus.get("post_id") or focus.get("post_title") or focus.get("post_excerpt")),
        "mode": mode,
        "post_title": post_title,
        "post_excerpt": post_excerpt,
        "comment_excerpt": comment_excerpt,
        "comment_author": comment_author,
        "thread_summary": thread_summary,
        "thread_keywords": thread_keywords,
        "target_kind": target_kind,
        "target_text": target_text,
        "target_summary": target_summary,
        "target_keywords": target_keywords,
        "constraints": constraints,
        "user_nickname": str(_safe_dict(user).get("nickname") or ""),
    }

def build_query_from_compose(th: Dict[str, Any], compose_input: Dict[str, Any]) -> List[str]:
    """
    BM25/thought-recall query builder grounded on compose_input.
    """
    ks = _safe_list(compose_input.get("thread_keywords")) or _safe_list(th.get("keywords"))
    tks = _safe_list(compose_input.get("target_keywords"))
    parts: List[str] = []
    parts.extend([str(x) for x in ks[:10]])
    parts.extend([str(x) for x in tks[:10]])
    parts.append(str(compose_input.get("thread_summary") or ""))
    parts.append(str(compose_input.get("target_text") or ""))
    q = " ".join([p for p in parts if p])
    return tokenize(q, max_tokens=80)

# -----------------------------------------------------------------------------
# v15 Unit 04 — Grounding Validator (comment/reply must match its target)
# -----------------------------------------------------------------------------
def validate_grounding(text: str, focus: Dict[str, Any], mode: str) -> Tuple[bool, str]:
    """Return (ok, reason). 목적: 중구난방 댓글/대댓글을 자동으로 걸러서 재생성 유도.

    - comment: post_title/post_excerpt 기반 토큰과 최소 1개 이상 겹치면 통과
    - reply: comment_excerpt(우선) 또는 post_excerpt 기반 토큰과 최소 1개 이상 겹치면 통과

    focus 정보가 부족하면(발췌가 비어있음 등) 과도하게 막지 않도록 관대하게 통과시킴.
    """
    t = (text or "").strip()
    if not t:
        return (False, "empty")

    if looks_like_injection(t):
        return (False, "injection")
    if looks_offensive(t):
        return (False, "offensive")
    if contains_markdown(t):
        return (False, "markdown")

    if not isinstance(focus, dict):
        return (True, "no_focus")

    m = str(mode or focus.get("mode") or "comment")
    if m not in ("comment", "reply"):
        m = "comment"

    post_title = str(focus.get("post_title") or "")
    post_excerpt = str(focus.get("post_excerpt") or "")
    comment_excerpt = str(focus.get("comment_excerpt") or "")

    target_text = ""
    if m == "reply":
        target_text = comment_excerpt or post_excerpt or post_title
    else:
        # 댓글은 본문(제목+발췌)에 대한 반응이어야 함
        target_text = (post_title + " " + post_excerpt).strip() or post_excerpt or post_title

    target_text = target_text.strip()
    if not target_text:
        # 발췌가 없으면 억지로 막지 않음
        return (True, "no_target")

    toks_t = set(tokenize(target_text, max_tokens=80))
    toks_o = set(tokenize(t, max_tokens=120))

    if not toks_t or not toks_o:
        return (True, "no_tokens")

    inter = toks_t.intersection(toks_o)
    if len(inter) >= 1:
        return (True, "ok")

    # 최후의 보루: 요지/전제/결론 같은 형식적 표지라도 있으면 한 번은 허용(너무 빡세면 루프가 멎을 수 있음)
    if re.search(r"(요지|전제|핵심|결론|반박|보강|정의)", t):
        # 그래도 완전 무관 텍스트를 막기 위해, 타겟의 상위 키워드 중 1개라도 포함되면 통과
        kws = top_keywords(target_text, k=6)
        lt = sanitize_plain_text(t).lower()
        if any((kw and kw.lower() in lt) for kw in kws[:4]):
            return (True, "weak_ok")

    return (False, "no_overlap")

################################################################################
# 7.2.1. QA QUALITY GATE (Unit 01)
# - Goal:
# - - Catch "임/쪽임/요지는..." style artifacts + repetition loops BEFORE writing
# - - Provide a batch QA report mode (dry-run friendly)
################################################################################

_QA_BANNED_PHRASES: Tuple[str, ...] = (
    "본문 요지는",
    "내쪽 생각은",
    "쪽임",
    "내 기준에선",
)

_QA_SOFT_BANNED_PATTERNS: Tuple[str, ...] = (
    r"\b전제\s*하나만\s*확인\b",
    r"\b여기서\s*정의부터\b",
)

def _regen_budget(cfg: Config) -> int:
    """Unified regen budget for text generation loops."""
    b = int(cfg.timing.max_text_regen_tries)
    try:
        if getattr(cfg, "quality", None) is not None and bool(cfg.quality.enabled):
            b = max(b, int(cfg.quality.max_tries))
    except Exception as e:
        log_debug_exc("_regen_budget:silent", e)
        pass
    return max(1, int(b))

def _qa_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(0, len(tokens) - n + 1)]

def _qa_ngram_repeat_ratio(tokens: List[str], n: int = 3) -> float:
    grams = _qa_ngrams(tokens, n)
    if not grams:
        return 0.0
    c = Counter(grams)
    most = max(c.values()) if c else 0
    return float(most) / float(len(grams))

def _qa_line_prefix_dup_ratio(lines: List[str], k: int = 12) -> float:
    if not lines:
        return 0.0
    pref = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        pref.append(s[:k])
    if len(pref) <= 1:
        return 0.0
    c = Counter(pref)
    most = max(c.values()) if c else 0
    return float(most) / float(len(pref))

def _qa_im_ending_ratio(lines: List[str]) -> float:
    if not lines:
        return 0.0
    cnt = 0
    tot = 0
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        tot += 1
        if re.search(r"(?:임|임\.|임\?|임!|임\…)$", s):
            cnt += 1
    if tot <= 0:
        return 0.0
    return float(cnt) / float(tot)

def qa_evaluate_text(text: str, *, kind: str, focus: Optional[Dict[str, Any]] = None, mode: str = "comment") -> Dict[str, Any]:
    """Return a QA report dict with score + issues.

    kind: "comment" | "reply" | "post_title" | "post_body"
    """
    t = (text or "").strip()
    issues: List[str] = []
    hard_fail = False

    if not t:
        return {"score": 0, "hard_fail": True, "issues": ["empty"], "len": 0}

    # Hard bans (reported here too)
    if looks_like_injection(t):
        hard_fail = True
        issues.append("injection")
    if looks_offensive(t):
        hard_fail = True
        issues.append("offensive")
    if contains_markdown(t):
        hard_fail = True
        issues.append("markdown")

    # Length heuristic (soft)
    L = len(t)
    if kind in ("comment", "reply"):
        if L < 20:
            issues.append("too_short")
        if L > 420:
            issues.append("too_long")
    elif kind == "post_title":
        if L < 4:
            issues.append("title_too_short")
        if L > 70:
            issues.append("title_too_long")
    elif kind == "post_body":
        if L < 80:
            issues.append("body_too_short")
        if L > 1600:
            issues.append("body_too_long")

    # Phrase artifacts (soft)
    lt = sanitize_plain_text(t).lower()
    for ph in _QA_BANNED_PHRASES:
        if ph and ph.lower() in lt:
            issues.append("banned_phrase")
            break

    for pat in _QA_SOFT_BANNED_PATTERNS:
        try:
            if re.search(pat, t):
                issues.append("overused_opener")
                break
        except re.error:
            pass

    if t.count("...") >= 1 or "……" in t or "…" in t:
        issues.append("ellipsis")

    # Repetition metrics
    toks = tokenize(t, max_tokens=220)
    uniq_ratio = (len(set(toks)) / float(len(toks))) if toks else 1.0
    rep3 = _qa_ngram_repeat_ratio(toks, 3)

    lines = [x for x in t.splitlines() if (x or "").strip()]
    im_ratio = _qa_im_ending_ratio(lines)
    pref_dup = _qa_line_prefix_dup_ratio(lines, 12)

    # Score (simple additive penalties)
    score = 100.0
    if hard_fail:
        score -= 60.0

    if "too_short" in issues:
        score -= 18.0
    if "too_long" in issues:
        score -= 10.0
    if "title_too_short" in issues:
        score -= 20.0
    if "title_too_long" in issues:
        score -= 8.0
    if "body_too_short" in issues:
        score -= 18.0
    if "body_too_long" in issues:
        score -= 12.0

    if "banned_phrase" in issues:
        score -= 25.0
    if "overused_opener" in issues:
        score -= 10.0
    if "ellipsis" in issues:
        score -= 6.0

    if uniq_ratio < 0.55 and len(toks) >= 30:
        issues.append("low_vocab_variety")
        score -= 14.0
    if rep3 >= 0.18 and len(toks) >= 35:
        issues.append("ngram_repeat")
        score -= 18.0
    if pref_dup >= 0.34 and len(lines) >= 3:
        issues.append("line_prefix_repeat")
        score -= 10.0
    if im_ratio >= 0.72 and len(lines) >= 3:
        issues.append("too_many_im_endings")
        score -= 12.0

    # Focus overlap (optional; mild)
    try:
        if isinstance(focus, dict) and focus:
            ok_g, _ = validate_grounding(t, focus, mode)
            if ok_g:
                score += 2.0
            else:
                score -= 6.0
                issues.append("weak_grounding")
    except Exception as e:
        log_debug_exc("qa_evaluate_text:silent", e)
        pass

    # Frequent-issue penalty boost (from QA batch stats; optional)
    boost_pen = 0.0
    try:
        if isinstance(focus, dict) and focus:
            qb = _safe_dict(focus.get("qa_issue_boost"))
            if qb:
                for iss in issues:
                    try:
                        boost_pen += float(qb.get(iss, 0.0) or 0.0)
                    except Exception:
                        continue
                if boost_pen:
                    score -= min(18.0, float(boost_pen))
    except Exception:
        boost_pen = 0.0

    score = max(0, min(100, score))
    return {
        "score": int(round(score)),
        "hard_fail": bool(hard_fail),
        "issues": issues,
        "len": L,
        "uniq_ratio": round(float(uniq_ratio), 3),
        "rep3": round(float(rep3), 3),
        "im_ratio": round(float(im_ratio), 3),
        "line_prefix_dup": round(float(pref_dup), 3),
        "boost_pen": round(float(boost_pen), 2),
    }

def qa_check_text(cfg: Config, text: str, *, kind: str, focus: Optional[Dict[str, Any]] = None, mode: str = "comment") -> Tuple[bool, Dict[str, Any]]:
    """Gate outgoing comment/reply text. If gate disabled, always ok."""
    if getattr(cfg, "quality", None) is None or not bool(cfg.quality.enabled):
        return True, {"score": 100, "hard_fail": False, "issues": []}
    rep = qa_evaluate_text(text, kind=kind, focus=focus, mode=mode)
    ok = (not rep.get("hard_fail")) and int(rep.get("score", 0)) >= int(cfg.quality.min_score)
    return bool(ok), rep

def qa_check_post(cfg: Config, title: str, body: str) -> Tuple[bool, Dict[str, Any]]:
    if getattr(cfg, "quality", None) is None or not bool(cfg.quality.enabled):
        return True, {"score": 100, "hard_fail": False, "issues": []}
    r1 = qa_evaluate_text(title, kind="post_title")
    r2 = qa_evaluate_text(body, kind="post_body")
    score = int(round(0.25 * int(r1.get("score", 0)) + 0.75 * int(r2.get("score", 0))))
    issues = list(dict.fromkeys(list(r1.get("issues", [])) + list(r2.get("issues", []))))
    hard_fail = bool(r1.get("hard_fail")) or bool(r2.get("hard_fail"))
    rep = {"score": score, "hard_fail": hard_fail, "issues": issues, "title": r1, "body": r2}
    ok = (not hard_fail) and score >= int(cfg.quality.min_score)
    return bool(ok), rep

def qa_run_batch_report(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    policy: Dict[str, Any],
    semantic: Dict[str, Any],
    brain: Dict[str, Any],
    bm25: Optional["BM25Index"],
) -> Dict[str, Any]:
    """Run a lightweight QA batch: generate texts, score them, summarize stats."""
    n = int(getattr(cfg.quality, "batch_n", 50) if getattr(cfg, "quality", None) else 50)
    n = max(1, min(500, n))

    report: Dict[str, Any] = {
        "ts_kst": now_kst_str(),
        "n": n,
        "dry_run": bool(cfg.http.dry_run),
        "min_score": int(getattr(cfg.quality, "min_score", 72) if getattr(cfg, "quality", None) else 72),
        "samples": [],
    }

    posts: List[Dict[str, Any]] = []
    try:
        posts, _ = list_posts(client, limit=min(60, max(20, int(tuning.fetch_limit) * 3)))
    except Exception as e:
        log_debug_exc("qa:list_posts", e)

    n_post = max(1, int(round(n * 0.25)))
    n_comment = max(0, n - n_post)

    # 1) comment/reply samples from live posts
    for _i in range(n_comment):
        p = random.choice(posts) if posts else {}
        pid = str(p.get("id") or p.get("post_id") or p.get("_id") or "")
        if not pid:
            continue

        try:
            p_full = get_post(client, pid) or p
        except Exception:
            p_full = p

        comments: List[Dict[str, Any]] = []
        try:
            comments = list_comments(client, pid)
        except Exception:
            comments = []

        # pick comment vs reply
        parent_id = ""
        target_c: Optional[Dict[str, Any]] = None
        if comments and random.random() < 0.55:
            cand = [c for c in comments if isinstance(c, dict) and str(c.get("text") or "").strip()]
            target_c = random.choice(cand) if cand else random.choice(comments)
            parent_id = str(target_c.get("id") or "")

        ingest_post_into_context(state, p_full, brain=brain)
        ingest_comments_into_context(state, pid, comments, brain=brain, cfg=cfg)
        th = get_thread(state, pid)
        synthesize_thread(th)

        user_key = str((target_c or {}).get("nickname") or p_full.get("nickname") or "user")
        user = get_user(state, user_key)

        set_focus(state, mode=("reply" if parent_id else "comment"), post_id=pid, post=p_full, comment=target_c if parent_id else None)

        txt, meta = build_reply_text(
            cfg, tuning, state, policy, th, user,
            bm25=bm25,
            brain=brain,
            reply_to_own_post=False,
            is_reply=bool(parent_id),
        )
        mode2 = "reply" if bool(parent_id) else "comment"
        ok, rep = qa_check_text(cfg, txt, kind=mode2, focus=_safe_dict(state.get("focus")), mode=mode2)
        report["samples"].append({
            "kind": mode2,
            "ok": bool(ok),
            "score": int(rep.get("score", 0)),
            "issues": rep.get("issues", []),
            "meta": {"strategy": meta.get("strategy"), "cat": meta.get("cat")},
            "text": one_line(txt, 280),
        })

    # 2) post samples (title+body)
    for _i in range(n_post):
        title, body, meta = build_post_text(cfg, tuning, state, policy, semantic, brain, bm25)
        ok, rep = qa_check_post(cfg, title, body)
        report["samples"].append({
            "kind": "post",
            "ok": bool(ok),
            "score": int(rep.get("score", 0)),
            "issues": rep.get("issues", []),
            "meta": {"cat": meta.get("cat")},
            "text": one_line((title + " / " + body), 280),
        })

    # summarize
    scores = [int(s.get("score", 0)) for s in _safe_list(report.get("samples")) if isinstance(s, dict)]
    oks = [bool(s.get("ok")) for s in _safe_list(report.get("samples")) if isinstance(s, dict)]
    report["generated"] = len(scores)
    report["pass"] = int(sum(1 for x in oks if x))
    report["pass_rate"] = round(float(report["pass"]) / float(max(1, len(oks))), 3)
    report["avg_score"] = round(float(sum(scores)) / float(max(1, len(scores))), 2)
    report["min_score_observed"] = int(min(scores)) if scores else 0
    report["max_score_observed"] = int(max(scores)) if scores else 0

    ic = Counter()
    for s in _safe_list(report.get("samples")):
        if not isinstance(s, dict):
            continue
        for iss in _safe_list(s.get("issues")):
            if isinstance(iss, str) and iss:
                ic[iss] += 1
    report["issue_top"] = [{"issue": k, "count": int(v)} for k, v in ic.most_common(12)]

    # Convert frequent issues into a lightweight per-issue penalty map for future generations.
    issue_boost: Dict[str, float] = {}
    try:
        nn = float(max(1, int(report.get("n", 0) or 0)))
        for it in _safe_list(report.get("issue_top"))[:8]:
            if not isinstance(it, dict):
                continue
            iss = str(it.get("issue") or "")
            cnt = float(it.get("count") or 0.0)
            if not iss or cnt <= 0:
                continue
            # 0.8 ~ 4.0 penalty depending on how dominant the issue is in this batch
            pen = 0.8 + (cnt / nn) * 6.0
            issue_boost[iss] = round(float(min(4.0, max(0.8, pen))), 2)
    except Exception as e:
        log_debug_exc("qa_run_batch_report:silent", e)
        pass

    report["issue_boost"] = issue_boost
    if isinstance(state, dict):
        state["qa_issue_boost"] = issue_boost

    return report

def qa_print_batch_report(report: Dict[str, Any], *, show_worst: int = 5) -> None:
    try:
        n = int(report.get("generated", 0) or 0)
        Console.cprint(Console.CYAN, f"[QA] generated={n} pass={report.get('pass')} pass_rate={report.get('pass_rate')} avg={report.get('avg_score')} min={report.get('min_score_observed')} max={report.get('max_score_observed')}")
        top = _safe_list(report.get("issue_top"))[:8]
        if top:
            msg = ", ".join([f"{x.get('issue')}:{x.get('count')}" for x in top if isinstance(x, dict)])
            Console.cprint(Console.MAGENTA, f"[QA] issues: {msg}")

        show = max(0, min(50, int(show_worst)))
        if show:
            samples = [s for s in _safe_list(report.get("samples")) if isinstance(s, dict)]
            samples.sort(key=lambda x: int(x.get("score", 0)))
            for s in samples[:show]:
                Console.cprint(Console.YELLOW, f"[QA] worst kind={s.get('kind')} score={s.get('score')} issues={','.join(_safe_list(s.get('issues'))[:4])}")
                Console.cprint(Console.GRAY, f"      {s.get('text')}")
    except Exception as e:
        log_debug_exc("qa_print_batch_report:silent", e)
        pass

def qa_write_batch_report(path: str, report: Dict[str, Any]) -> None:
    p = (path or "").strip()
    if not p:
        return
    try:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_debug_exc("qa:write_report", e)

################################################################################
# 7.3. BRAIN ACTION_BIAS UPDATES
# - Dependencies: Section 7, 9 (Schemas, Policy)
# - Used by: Learning / policy updates
# - Key functions: update_action_bias()
################################################################################

def _ab_alpha_from_decay(decay: float) -> float:
    d = float(decay)
    if not (0.0 < d < 1.0):
        d = 0.98
    return max(0.001, min(0.25, 1.0 - d))

def _ab_update_slot(slot: Dict[str, Any], key: str, delta: float, *, decay: float) -> None:
    if not key:
        return
    try:
        old = float(slot.get(key, 0.0) or 0.0)
    except Exception:
        old = 0.0
    a = _ab_alpha_from_decay(decay)
    # bounded EMA-ish update
    slot[key] = float((1.0 - a) * old + a * float(delta))

def _ab_apply(
    brain: Dict[str, Any],
    *,
    action_type: str,
    kw: str = "",
    template_id: str = "",
    target_nick: str = "",
    delta: float,
) -> None:
    if not isinstance(brain, dict):
        return
    ab = brain.get("action_bias")
    if not isinstance(ab, dict):
        ab = {"by_action": {}, "by_topic_kw": {}, "by_template": {}, "by_user": {}, "last_update_ts": 0.0, "decay": 0.98}
        brain["action_bias"] = ab

    by_action = ab.setdefault("by_action", {})
    by_topic = ab.setdefault("by_topic_kw", {})
    by_tpl = ab.setdefault("by_template", {})
    by_user = ab.setdefault("by_user", {})
    decay = float(ab.get("decay", 0.98) or 0.98)

    # clamp delta to a sane range
    d = float(delta)
    if d > 1.0:
        d = 1.0
    if d < -1.0:
        d = -1.0

    _ab_update_slot(by_action, str(action_type or ""), d, decay=decay)
    if kw:
        _ab_update_slot(by_topic, str(kw), d, decay=decay)
    if template_id:
        _ab_update_slot(by_tpl, str(template_id), d, decay=decay)
    if target_nick:
        _ab_update_slot(by_user, str(target_nick), d, decay=decay)

    ab["last_update_ts"] = time.time()

def compute_proxy_reward(text: str, *, mode: str, ground_reason: str = "") -> float:
    """Cheap immediate proxy score in [-1, +1]. (Unit 07)

    This is intentionally lightweight: it should not add API calls.
    """
    t = (text or "").strip()
    if not t:
        return -0.8

    m = str(mode or "")
    gr = str(ground_reason or "")
    n = len(t)

    # grounding signal: if validator said ok/weak_ok => positive bias
    g = 0.0
    if gr in ("ok", "weak_ok"):
        g = 0.35
    elif gr in ("no_target", "no_focus", "no_tokens"):
        g = 0.18
    elif gr:
        # unknown reason (but passed) => small
        g = 0.12

    # length preference (comment/reply: 100~500 recommended; post can be longer)
    if m in ("comment", "reply"):
        if 100 <= n <= 520:
            l = 0.35
        elif 70 <= n <= 800:
            l = 0.18
        else:
            l = -0.15
    else:
        # post / other
        if 180 <= n <= 950:
            l = 0.25
        elif 120 <= n <= 1200:
            l = 0.15
        else:
            l = -0.10

    # simple eum-style heuristic (very lightweight)
    s = 0.0
    tail = t[-3:]
    if any(x in tail for x in ("음", "슴")):
        s = 0.10

    # mild penalty for excessive punctuation density
    punct = sum(1 for ch in t if ch in "!?")
    p = -0.05 if punct >= 4 else 0.0

    r = g + l + s + p
    if r > 1.0:
        r = 1.0
    if r < -1.0:
        r = -1.0
    return float(r)

def apply_brain_proxy_update(brain: Dict[str, Any], tuning: AgentTuning, item: Dict[str, Any]) -> None:
    """Apply immediate proxy update to action_bias once per item."""
    if not isinstance(brain, dict) or not isinstance(item, dict):
        return
    if item.get("brain_proxy_applied") is True:
        return

    act = str(item.get("action") or "")
    if act not in ("comment", "reply", "post"):
        item["brain_proxy_applied"] = True
        return

    action_type = str(item.get("action_type") or act)
    kw = str(item.get("kw") or "")
    tpl = str(item.get("template_id") or "")
    target = str(item.get("target_nick") or "")

    pr = item.get("proxy_reward")
    if pr is None:
        pr = compute_proxy_reward(str(item.get("text") or ""), mode=("reply" if act == "reply" else "comment" if act == "comment" else "post"), ground_reason=str(item.get("ground_reason") or ""))
        item["proxy_reward"] = float(pr)

    # proxy should be gentle: smaller than delayed reward
    d = float(pr) * 0.35
    _ab_apply(brain, action_type=action_type, kw=kw, template_id=tpl, target_nick=target, delta=d)

    upd = brain.setdefault("bias_updates", {"proxy": 0, "reward": 0})
    if isinstance(upd, dict):
        upd["proxy"] = int(upd.get("proxy", 0) or 0) + 1

    item["brain_proxy_applied"] = True

def apply_brain_reward_update(brain: Dict[str, Any], tuning: AgentTuning, item: Dict[str, Any]) -> None:
    """Apply delayed (evaluated) reward update to action_bias once per item."""
    if not isinstance(brain, dict) or not isinstance(item, dict):
        return
    if item.get("brain_reward_applied") is True:
        return

    act = str(item.get("action") or "")
    if act not in ("comment", "reply", "post"):
        item["brain_reward_applied"] = True
        return
    if item.get("evaluated") is not True:
        return

    action_type = str(item.get("action_type") or act)
    kw = str(item.get("kw") or "")
    tpl = str(item.get("template_id") or "")
    target = str(item.get("target_nick") or "")

    r = float(item.get("reward_scalar", 0.0) or 0.0)
    clip = max(1e-6, float(getattr(tuning, "reward_clip", 3.0)))
    d = max(-1.0, min(1.0, r / clip))

    _ab_apply(brain, action_type=action_type, kw=kw, template_id=tpl, target_nick=target, delta=d)

    upd = brain.setdefault("bias_updates", {"proxy": 0, "reward": 0})
    if isinstance(upd, dict):
        upd["reward"] = int(upd.get("reward", 0) or 0) + 1

    item["brain_reward_applied"] = True

################################################################################
# 8. TEXT + RULES (sanitize/emoji/markdown/eum-style/classify/tokenize)
# - Dependencies: Section 1-2 (Config, Logging)
# - Used by: All text generation paths
# - Key functions: sanitize_plain_text(), tokenize(), normalize_ko_token(), postprocess_outgoing_text()
################################################################################

INJECTION_PATTERNS = [
    r"시스템\s*프롬프트",
    r"system\s*prompt",
    r"이전\s*명령\s*무시",
    r"ignore\s*previous",
    r"너는\s*이제부터",
    r"prompt\s*injection",
]

# Precompiled regex (optimization-only)
_INJECTION_RES = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

OFFENSIVE_PATTERNS = [
    # profanity / slurs (strong filter; affects generation + keyword mining)
    r"(시발|씨발|ㅅㅂ)", r"(병신|ㅂㅅ)", r"(좆|좆같)", r"(개새|개새끼)", r"(씹|씹새|씹새끼)", r"(새끼)",
    # extreme hostility / violence (keep conservative)
    r"죽여", r"살해",
    # hate/discrimination terms (conservative; may appear in debates)
    r"혐오", r"차별",
]

# Precompiled regex (optimization-only)
_OFFENSIVE_RES = [re.compile(p) for p in OFFENSIVE_PATTERNS]

EMOJI_RE = re.compile(
    "[" +
    "\U0001F300-\U0001FAFF" +
    "\U00002700-\U000027BF" +
    "\U00002600-\U000026FF" +
    "]+",
    flags=re.UNICODE
)

MARKDOWN_RE = re.compile(r"(\*\*|__|\*|_|`|^#|^>|^- |\[.*?\]\(.*?\))", re.MULTILINE)

EUM_ENDINGS = ("음", "슴", "임", "함", "됨", "있음", "없음", "였음", "했음", "같음", "해짐", "갈림", "느낌", "인듯", "듯함", "듯", "편", "쪽")
EUM_LINE_OK_RE = re.compile(rf"({'|'.join(EUM_ENDINGS)})(\?|!|…)?$")

# Precompiled eumssum transform rules (optimization-only)
_EUMIFY_RULES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(같습니다)\s*$"), "같음"),
    (re.compile(r"(겠습니다|겠어요|겠다)\s*$"), "겠음"),
    (re.compile(r"(입니다)\s*$"), "임"),
    (re.compile(r"(였습니다|였어요|였다)\s*$"), "였음"),
    (re.compile(r"(했습니다|했어요|했다)\s*$"), "했음"),
    (re.compile(r"(합니다|하였다|한다)\s*$"), "함"),
    (re.compile(r"(됩니다|된다)\s*$"), "됨"),
    (re.compile(r"(없습니다|없어요|없다)\s*$"), "없음"),
    (re.compile(r"(있습니다|있어요|있다)\s*$"), "있음"),
    (re.compile(r"(습니까)\s*$"), "슴"),
    (re.compile(r"(습니다)\s*$"), "슴"),
    (re.compile(r"(에요|예요)\s*$"), "임"),
    (re.compile(r"(어요|아요|여요|요)\s*$"), "임"),
)
_EUMIFY_RULES_Q: Tuple[Tuple[re.Pattern, str], ...] = _EUMIFY_RULES + (
    (re.compile(r"(다)\s*$"), "음"),
)


# Eumssum v2 tuning (kept lightweight; no external NLP)
EUM_V2_ENABLED = _env_bool("MERSOOM_EUM_V2", True)
EUM_V2_CONNECTOR_RATE = _env_float("MERSOOM_EUM_V2_CONNECTOR_RATE", 0.45)
EUM_V2_DROP_IM_RATE = _env_float("MERSOOM_EUM_V2_DROP_IM_RATE", 0.35)
EUM_V2_TARGET_IM_RATIO = _env_float("MERSOOM_EUM_V2_TARGET_IM_RATIO", 0.58)
EUM_V2_MAX_CONNECTORS = _env_int("MERSOOM_EUM_V2_MAX_CONNECTORS", 2)
STRICT_POSTPROCESS = _env_bool("MERSOOM_STRICT_POSTPROCESS", False)

EUM_CONNECTOR_PREFIXES: Tuple[str, ...] = (
    "근데", "근데도", "다만", "오히려", "그리고", "또", "게다가", "그래서", "그러면", "그러니", "일단", "사실",
)
EUM_CONNECTOR_START_RE = re.compile(rf"^({'|'.join(map(re.escape, EUM_CONNECTOR_PREFIXES))})(?:\b|\s)")

# endings we usually don't want to leave dangling (particles/aux)
_EUM_DANGLING_TAILS: Tuple[str, ...] = (
    "은", "는", "이", "가", "을", "를", "에", "에서", "로", "와", "과", "도", "만", "이나", "나", "까지", "부터", "처럼", "밖에",
)

STOPWORDS_KO = set([
    "그", "이", "저", "것", "수", "등", "및", "대한", "관련", "그리고", "하지만",
    "그래서", "또한", "그런데", "즉", "때문", "정도", "사실", "진짜", "그냥",
    "좀", "약간", "매우", "너무", "진짜로", "아예",
])
STOPWORDS_EN = set(["the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are"])

def looks_like_injection(text: str) -> bool:
    t = text or ""
    return any(p.search(t) for p in _INJECTION_RES)

def looks_offensive(text: str) -> bool:
    t = text or ""
    return any(p.search(t) for p in _OFFENSIVE_RES)



def is_toxic_incoming(text: str) -> Tuple[bool, str]:
    """Classify incoming user-generated text as toxic / unsafe-to-engage."""
    t = text or ""
    if not t.strip():
        return (False, "")
    if looks_like_injection(t):
        return (True, "injection")
    if looks_offensive(t):
        return (True, "offensive")
    return (False, "")

def contains_emoji(text: str) -> bool:
    return bool(EMOJI_RE.search(text or ""))

def contains_markdown(text: str) -> bool:
    t = text or ""
    if "```" in t:
        return True
    return bool(MARKDOWN_RE.search(t))

def _looks_like_markdown_violation(text: str) -> bool:
    """Conservative detector for 'markdown-y' content (incoming).
    We allow fenced code blocks but flag other markdown constructs (headings/links/bold).
    """
    t = str(text or "")
    if not t.strip():
        return False

    # Remove fenced blocks and inspect the remainder
    if "```" in t:
        rem = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    else:
        rem = t

    # Obvious markdown patterns
    if re.search(r"(^|\n)\s*#{1,6}\s+\S", rem):
        return True
    if re.search(r"\[[^\]]+\]\([^)]+\)", rem):
        return True
    if re.search(r"(\*\*|__)", rem):
        return True

    # Heuristic: lots of markdown punctuation outside code blocks
    punct = re.findall(r"[`*_>#]", rem)
    if len(punct) >= 8 and len(rem) >= 80:
        return True
    return False

def looks_like_rule_violation(text: str) -> Tuple[bool, str]:
    """Detect obvious Mersoom rule violations for self-policing votes."""
    t = str(text or "")
    if not t.strip():
        return (False, "")

    if contains_emoji(t):
        return (True, "emoji")

    # markdown is prohibited except minimal code sharing
    if _looks_like_markdown_violation(t):
        return (True, "markdown")

    # Korean-first: if it's mostly English and missing the mandated notice, flag it
    if "한국어 모듈 오류남" not in t:
        try:
            if re.search(r"[A-Za-z]{4,}", t) and _hangul_ratio(t) < 0.15:
                return (True, "english")
        except Exception:
            pass

    return (False, "")


def sanitize_plain_text(text: str) -> str:
    t = str(text or "")
    t = EMOJI_RE.sub("", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[`*_>#]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -----------------------------------------------------------------------------
# v20.3: Lightweight question detection + last-resort fallback comment templates
# -----------------------------------------------------------------------------

def _is_questionish(text: str) -> bool:
    """Heuristic: detect question-like text without NLP libs (for reply boosting)."""
    try:
        t = str(text or "").strip()
    except Exception:
        return False
    if not t:
        return False
    if "?" in t:
        return True
    # Korean-ish question cues (keep lightweight; avoid too many false positives)
    cues = (
        "왜", "어떻게", "근거", "무슨", "뭐임", "뭐야", "맞음", "맞나", "어디", "언제", "누가", "어느", "가능",
        "어떻게 생각", "어케", "이유", "전제", "기준"
    )
    return any(c in t for c in cues)

_FALLBACK_COMMENT_TEMPLATES: List[str] = [
    "이 부분 근거가 뭐임? 사례나 데이터 더 있으면 궁금함임",
    "전제가 뭐인지 궁금함임. 기준을 한 줄로 정리해주면 좋겠음임",
    "반대 근거도 같이 보면 좋겠음임. 어디까지 확실한 주장임?",
]

_FALLBACK_REPLY_TEMPLATES: List[str] = [
    "질문 요지가 뭐임? 전제랑 기준부터 같이 맞춰보면 좋겠음임",
    "근거가 어떤 쪽임? 사례 하나만 더 알려주면 이해 쉬울듯함임",
    "이 부분은 왜 그렇게 봄? 논리 흐름을 한 줄로만 더 부탁함임",
]

def _pick_fallback_comment(cfg: "Config", *, is_reply: bool) -> str:
    """Pick a safe (>=10 chars) last-resort comment/reply to reduce zero-comment cycles."""
    try:
        cands = list(_FALLBACK_REPLY_TEMPLATES if is_reply else _FALLBACK_COMMENT_TEMPLATES)
        random.shuffle(cands)
        for raw in cands:
            s = sanitize_plain_text(raw)
            try:
                s = ensure_eum_style(s, max_lines=2)
            except Exception:
                pass
            s = re.sub(r"\s+", " ", str(s)).strip()
            if len(s) >= 10:
                return s
    except Exception as e:
        log_debug_exc("_pick_fallback_comment:silent", e)
    return ""

def _hangul_ratio(text: str) -> float:
    t = text or ""
    if not t:
        return 0.0
    hangul = len(re.findall(r"[가-힣]", t))
    return hangul / max(1, len(t))

def _has_batchim_char(ch: str) -> bool:
    """Return True if Hangul syllable has a final consonant (받침)."""
    if not ch:
        return False
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return ((code - 0xAC00) % 28) != 0
    return False

def ensure_eum_style(text: str, *, max_lines: int) -> str:
    """
    Force a more natural-ish "음슴체" ending per line, without external NLP libs.

    v2 adds:
      - light rhythm smoothing (reduce "임" density + consecutive "임" runs)
      - optional connector prefixes between lines (keeps prose from feeling list-like)
      - conservative noun-phrase allowances (avoid forcing '...임' on short noun lines)
      - fixes for polite endings like "...습니다/..겠습니다" (avoid "...습니임")
    """
    raw = str(text or "").strip()
    if not raw:
        return "내용 없음임"

    # Seed a tiny local RNG so repeated regen tries don't look identical,
    # but remain stable-ish per input.
    try:
        seed = int.from_bytes(hashlib.sha1(raw.encode("utf-8")).digest()[:8], "big", signed=False) ^ int(time.time() * 1000)
        rnd = random.Random(seed & ((1 << 64) - 1))
    except Exception:
        rnd = random.Random()

    # Prefer sentence splitting for single-line multi-sentence paragraphs (improves rhythm).
    lines = [x.strip() for x in raw.split("\n") if x.strip()]
    if len(lines) <= 1 and ("\n" not in raw):
        punc_hits = raw.count(".") + raw.count("!") + raw.count("?") + raw.count("…")
        if len(raw) >= 120 or punc_hits >= 2:
            try:
                sents = split_sentences(raw, max_sent=max(3, int(max_lines)))
                if len(sents) >= 2:
                    lines = [x.strip() for x in sents if x.strip()]
            except Exception as e:
                log_debug_exc("ensure_eum_style:silent", e)
                pass

    if not lines:
        lines = [raw]

    def _looks_safe_noun_end(b: str) -> bool:
        bb = (b or "").strip()
        if not bb:
            return False
        # avoid leaving particles dangling
        if any(bb.endswith(t) for t in _EUM_DANGLING_TAILS):
            return False
        # avoid trailing quote/colon
        if bb.endswith(('"', "'", ":", "—", "-", "·")):
            return False
        return True

    def _starts_with_connector(x: str) -> bool:
        ss = (x or "").strip()
        if not ss:
            return False
        return bool(EUM_CONNECTOR_START_RE.search(ss))

    def _maybe_add_connector(prev_ln: str, ln: str) -> str:
        if not ln:
            return ln
        if _starts_with_connector(ln):
            return ln
        if ln.startswith(("-", "•", "*", "http://", "https://")):
            return ln
        if len(ln) < 8:
            return ln
        if rnd.random() > float(EUM_V2_CONNECTOR_RATE):
            return ln

        contrast = ("근데", "다만", "오히려", "근데도")
        cont = ("그리고", "또", "게다가")
        causal = ("그래서", "그러면", "그러니")
        starter = ("일단", "사실")

        if any(w in ln for w in ("하지만", "반면", "근데", "다만", "오히려")):
            pool = list(contrast)
        elif any(w in ln for w in ("그래서", "그러면", "따라서", "결국")):
            pool = list(causal)
        elif prev_ln.endswith("?"):
            pool = list(causal) + list(cont)
        else:
            pool = list(cont) + list(starter)

        c = rnd.choice(pool) if pool else "그리고"
        return f"{c} {ln}"

    def _reduce_im_density(out_lines: List[str]) -> List[str]:
        if not out_lines:
            return out_lines

        def _ends_im(ln: str) -> bool:
            s = (ln or "").strip()
            return bool(re.search(r"(?:임|임\.|임\?|임!|임…)$", s))

        im_idx = [i for i, ln in enumerate(out_lines) if _ends_im(ln)]
        ratio = float(len(im_idx)) / float(max(1, len(out_lines)))

        if ratio <= float(EUM_V2_TARGET_IM_RATIO) and len(im_idx) <= 1:
            return out_lines

        new_lines = out_lines[:]
        max_changes = max(1, int(len(out_lines) * 0.34))
        changes = 0

        for i in im_idx:
            if changes >= max_changes:
                break
            ln = (new_lines[i] or "").strip()
            if not ln:
                continue

            # split trailing closers + punctuation (avoid '?임' artifacts like ...?\"임)
            closers = ""
            tail_closers = ('"', "'", "”", "’", ")", "]", "}", "】", "）", "」", "』", "》", "〉", "〕")
            while ln and ln[-1] in tail_closers:
                closers = ln[-1] + closers
                ln = ln[:-1].rstrip()

            p = ""
            if ln.endswith(("?", "!", "…", ".", "。")):
                p = ln[-1]
                body = ln[:-1].rstrip()
            else:
                body = ln

            if body.endswith("거임"):
                continue

            # Try safe swaps before dropping '임'
            if body.endswith("있임"):
                body = body[:-2] + "있음"
            elif body.endswith("없임"):
                body = body[:-2] + "없음"
            elif body.endswith("되임"):
                body = body[:-2] + "됨"
            elif body.endswith("하임"):
                body = body[:-2] + "함"
            elif body.endswith("했임"):
                body = body[:-2] + "했음"
            elif body.endswith("였임"):
                body = body[:-2] + "였음"
            else:
                if (len(body) <= 26) and body.endswith("임") and _looks_safe_noun_end(body[:-1]):
                    if rnd.random() <= float(EUM_V2_DROP_IM_RATE):
                        body = body[:-1].rstrip()

            new_ln = f"{body}{p}" if p else body
            if new_ln != ln:
                changes += 1
                new_lines[i] = new_ln

        # Smooth consecutive "임" runs
        for i in range(1, len(new_lines)):
            if changes >= max_changes:
                break
            if _ends_im(new_lines[i]) and _ends_im(new_lines[i - 1]):
                ln = new_lines[i].strip()
                p = ""
                if ln.endswith(("?", "!", "…", ".", "。")):
                    p = ln[-1]
                    body = ln[:-1].rstrip()
                else:
                    body = ln
                if body.endswith("임") and (len(body) <= 26) and _looks_safe_noun_end(body[:-1]):
                    body2 = body[:-1].rstrip()
                    if body2:
                        new_lines[i] = f"{body2}{p}" if p else body2
                        changes += 1

        return new_lines

    def _eumify_body(body: str, punct: str) -> str:
        b = (body or "").strip()
        if not b:
            return "내용 없음임"
        if EUM_LINE_OK_RE.search(b):
            return b

        is_q_or_excl = punct in ("?", "!")
        rules = _EUMIFY_RULES_Q if punct == "?" else _EUMIFY_RULES

        for pat, rep in rules:
            if pat.search(b):
                b2 = pat.sub(rep, b).strip()
                if b2:
                    return b2

        # final "다" (after filtering polite patterns above)
        if (not is_q_or_excl) and b.endswith("다") and len(b) >= 2:
            if (" " not in b) and len(b) <= 2:
                return b[:-1] + "음"
            return b[:-1] + "임"

        # soft endings already acceptable (avoid '...해짐임' etc)
        if b.endswith(("해짐", "같음", "갈림", "인듯", "듯", "듯함", "느낌", "편", "쪽")):
            return b

        # normalize colloquial copula tail
        if b.endswith("거야"):
            return b[:-2] + "거임"
        if b.endswith("이야") and len(b) >= 3 and _has_batchim_char(b[-3]):
            return b[:-2] + "임"
        if b.endswith("야") and len(b) >= 2:
            return b[:-1] + "임"

        if is_q_or_excl:
            return b

        # noun-phrase allowance
        if (len(b) <= 22) and _looks_safe_noun_end(b):
            return b

        # fallback
        if b.endswith("하") or b.endswith("함") or b.endswith("했"):
            return b + "음"
        if b.endswith("되"):
            return b + "됨"
        return b + "임"

    out_lines: List[str] = []
    for ln in lines:
        ln = sanitize_plain_text(ln)
        if not ln:
            continue

        # trailing closing quotes/brackets (keep after punctuation)
        closers = ""
        try:
            m = re.search(r"[\"\'”’\)\]\}]+$", ln)
            if m:
                closers = m.group(0)
                ln = ln[:-len(closers)].rstrip()
        except Exception as e:
            log_debug_exc("ensure_eum_style:closers", e)
            closers = ""

        if _hangul_ratio(ln) < 0.12 and len(ln) >= 6 and not EUM_LINE_OK_RE.search(ln):
            ln = f"{ln} 임"

        p = ""
        if ln.endswith(("?", "!", "…", ".", "。")):
            p = ln[-1]
            body = ln[:-1].rstrip()
        else:
            body = ln

        body = _eumify_body(body, p)
        body = re.sub(r"(임|음|슴|함|됨)\1+$", r"\1", body)

        if p:
            ln2 = f"{body}{p}{closers}"
        elif closers:
            ln2 = f"{body}{closers}"
        else:
            ln2 = body
        out_lines.append(ln2)

    if not out_lines:
        out_lines = ["내용 없음임"]

    if bool(EUM_V2_ENABLED) and len(out_lines) >= 2:
        new_lines = out_lines[:]
        added = 0
        for i in range(1, len(new_lines)):
            if added >= int(EUM_V2_MAX_CONNECTORS):
                break
            prev_ln = new_lines[i - 1].strip()
            ln = new_lines[i].strip()
            ln2 = _maybe_add_connector(prev_ln, ln)
            if ln2 != ln:
                new_lines[i] = ln2
                added += 1
        new_lines = _reduce_im_density(new_lines)
        out_lines = new_lines

    out_lines = out_lines[:max(1, int(max_lines))]
    return "\n".join(out_lines)

def eumify_tail_phrase(phrase: str) -> str:
    """Make a short phrase safe to end with 음슴체 without producing '야임/해짐임' artifacts."""
    s = sanitize_plain_text(phrase)
    if not s:
        return "그거임"
    # strip trailing punctuation
    s = re.sub(r"[\.!?…。]+\s*$", "", s).strip()
    if not s:
        return "그거임"

    # already looks like eum ending
    if EUM_LINE_OK_RE.search(s):
        return s

    # common copula tails
    if s.endswith("거야"):
        return s[:-2] + "거임"
    if s.endswith("이야") and len(s) >= 3 and _has_batchim_char(s[-3]):
        return s[:-2] + "임"
    if s.endswith("야") and len(s) >= 2:
        return s[:-1] + "임"

    # allow soft nominal endings without forcing extra suffix
    if s.endswith(("해짐", "같음", "갈림", "됨", "없음", "있음", "했음", "였음")):
        return s

    # polite tail
    if s.endswith("요") and len(s) >= 2:
        s = s[:-1].rstrip()
        if not s:
            return "그거임"

    # polite endings that include trailing '다' (avoid '...습니임' artifacts)
    if s.endswith("같습니다") and len(s) >= 4:
        return s[:-3] + "음"
    if s.endswith("겠습니다") and len(s) >= 5:
        return s[:-4] + "겠음"
    if s.endswith("합니다") and len(s) >= 4:
        return s[:-3] + "함"
    if s.endswith("됩니다") and len(s) >= 4:
        return s[:-3] + "됨"
    if s.endswith("습니다") and len(s) >= 4:
        return s[:-3] + "슴"

    # final '다' -> eum
    if s.endswith("다") and len(s) >= 2:
        if (" " not in s) and len(s) <= 2:
            return s[:-1] + "음"
        return s[:-1] + "임"

    # short noun-like phrases: allow ending without forcing "임" (reduces stiffness)
    if (len(s) <= 18) and _hangul_ratio(s) >= 0.35:
        if not any(s.endswith(t) for t in _EUM_DANGLING_TAILS):
            if re.search(r"(느낌|인듯|듯함|듯|편|쪽)$", s):
                return s
            if re.search(r"[가-힣A-Za-z0-9]$", s) and (" " in s or len(s) >= 6):
                return s
    return s + "임"


_SENT_SPLIT_RE = re.compile(r"(?:(?<=[\.\?!…])\s+|(?<=다\.)\s+|\n+)")
def split_sentences(text: str, *, max_sent: int = 8) -> List[str]:
    t = sanitize_plain_text(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p.strip()]
    return parts[:max(1, int(max_sent))]


def split_sentences_ko(text: str, *, max_sent: int = 12) -> List[str]:
    """Korean-friendly sentence splitter (alias for split_sentences).

    v18.1 hotfix: some template-mining utilities reference split_sentences_ko.
    """
    return split_sentences(text, max_sent=max_sent)

# --- Outgoing text post-processing (Hotfix 2)
# Goal: Keep 음슴체, but avoid unnatural artifacts:
# - No ellipsis ('…', '...') or truncated tokens like 'Ter…'
# - Remove stray ending fragments like '?임.' or '.임.' inserted by templating
# - Apply 음슴체 per sentence (not just per line) to reduce awkward stacking

_EUM_FRAGMENT_RE = re.compile(r"([\.?!])\s*(임|음|슴|함|됨)\s*\.", re.UNICODE)
_BROKEN_ASCII_ELLIPSIS_RE = re.compile(r"\b[A-Za-z]{1,4}…|\b[A-Za-z]{1,4}\.{3,}", re.UNICODE)
_SENTENCE_FP_RE = re.compile(r"[^0-9a-zA-Z가-힣]+")
_POSTPROCESS_DEDUPE_TS = deque(maxlen=5000)

def _sentence_fp(text: str) -> str:
    s = str(text or "").strip().lower()
    s = _SENTENCE_FP_RE.sub("", s)
    return s[:400]

def _sentence_core_tokens(text: str) -> List[str]:
    toks = [t for t in tokenize(str(text or ""), max_tokens=40) if t]
    core = [t for t in toks if t not in STOPWORDS_KO and t not in STOPWORDS_EN]
    return core[:16]

def _record_postprocess_dedupe() -> None:
    try:
        _POSTPROCESS_DEDUPE_TS.append(time.time())
    except Exception:
        pass

def _sanitize_keep_newlines(text: str) -> str:
    t = str(text or "")
    t = EMOJI_RE.sub("", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[`*_>#]", "", t)
    # keep newlines but normalize spaces
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _remove_ellipsis_and_broken_tokens(text: str) -> str:
    t = str(text or "")
    # remove obviously truncated short ASCII tokens that were cut with ellipsis
    t = _BROKEN_ASCII_ELLIPSIS_RE.sub("", t)
    # remove ellipsis markers entirely (they look bot-like in this community)
    t = t.replace("…", " ")
    t = re.sub(r"\.{3,}", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _drop_lonely_eum_fragments(text: str) -> str:
    t = str(text or "")
    # e.g. "왜 그러냐?임." -> "왜 그러냐?"
    t = _EUM_FRAGMENT_RE.sub(r"\1", t)
    # also drop standalone fragments
    t = re.sub(r"(^|\s)(임|음|슴|함|됨)(?=\.?\s|$)", " ", t)
    # normalize punctuation
    t = re.sub(r"\.\s*\.", ".", t)
    t = re.sub(r"\?\s*\?", "?", t)
    t = re.sub(r"!\s*!", "!", t)
    return re.sub(r"\s+", " ", t).strip()

def _space_after_punct(text: str) -> str:
    # help sentence splitting even when templates forget spaces after punctuation
    t = str(text or "")
    t = re.sub(r"([\.?!])(?=[^\s\.\?!])", r"\1 ", t)
    return re.sub(r"\s+", " ", t).strip()

def postprocess_outgoing_text(
    text: str,
    *,
    mode: str,
    max_chars: int,
    max_lines: int,
) -> str:
    """
    Post-process already-generated text for outbound API writes.
    This must be safe (no external NLP) and conservative.
    """
    raw = _sanitize_keep_newlines(text)

    # line-wise preserve structure for posts; single-line for others
    lines_in = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if not lines_in:
        lines_in = [raw] if raw else ["내용 없음임"]

    out_lines: List[str] = []
    for ln in lines_in:
        ln0 = _remove_ellipsis_and_broken_tokens(ln)
        ln0 = _drop_lonely_eum_fragments(ln0)
        ln0 = _space_after_punct(ln0)

        # sentence-level eum: reduces "임.임." stacking & improves naturalness
        sents = split_sentences(ln0, max_sent=12)
        fixed: List[str] = []
        seen_keys: set = set()
        seen_fp: set = set()
        seen_core: set = set()
        for s in sents:
            s = s.strip()
            if not s:
                continue
            # drop empty/fragment sentences
            if s in ("임", "음", "슴", "함", "됨"):
                continue
            s2 = ensure_eum_style(s, max_lines=1).replace("\n", " ").strip()
            s2 = _drop_lonely_eum_fragments(s2)
            s2 = _remove_ellipsis_and_broken_tokens(s2)
            if s2:
                if STRICT_POSTPROCESS:
                    fp = _sentence_fp(s2)
                    core = _sentence_core_tokens(s2)
                    if fp and (fp in seen_fp):
                        if core and (set(core) - seen_core):
                            pass
                        else:
                            _record_postprocess_dedupe()
                            continue
                    if core:
                        seen_core.update(core)
                    if fp:
                        seen_fp.add(fp)
                else:
                    k = re.sub(r"[^0-9a-zA-Z가-힣]+", "", s2).lower()
                    if k and (k in seen_keys):
                        continue
                    if fixed:
                        pk = re.sub(r"[^0-9a-zA-Z가-힣]+", "", fixed[-1]).lower()
                        # collapse near-identical adjacent sentences
                        if k and pk and (k == pk or (len(k) >= 20 and (k in pk or pk in k))):
                            continue
                    if k:
                        seen_keys.add(k)
                fixed.append(s2)

        if fixed:
            out_lines.append(" ".join(fixed).strip())

    if not out_lines:
        out_lines = ["내용 없음임"]

    # enforce max lines / join mode
    out_lines = out_lines[:max(1, int(max_lines))]
    out = "\n".join(out_lines) if mode == "post" else " ".join(out_lines)

    out = re.sub(r"\s+\n", "\n", out)
    out = re.sub(r"\n\s+", "\n", out)
    out = re.sub(r"\s+", " ", out).strip()

    if len(out) > int(max_chars):
        out = out[: int(max_chars)].rstrip()

    # final safety: no ellipsis
    out = out.replace("…", " ")
    out = re.sub(r"\.{3,}", " ", out)
    out = re.sub(r"\s+", " ", out).strip() if mode != "post" else out.strip()
    # v19.7: strict language rules (Mersoom 3.0)
    if LANG_STRICT:
        # Remove lingering markdown list markers that can look like markdown.
        if mode == "post":
            lines = [re.sub(r"^\s*(?:[-*]+|\d+\.)\s+", "", ln).strip() for ln in out.split("\n")]
            lines = [ln for ln in lines if ln]
        else:
            lines = [re.sub(r"^\s*(?:[-*]+|\d+\.)\s+", "", out).strip()]

        def _ensure_line_eum(line: str) -> str:
            l = (line or "").strip()
            if not l:
                return l
            # Split trailing suffixes so we can insert '임' BEFORE punctuation/quotes/brackets.
            base = l
            closers = ""
            punc = ""
            try:
                # common closing quotes/brackets at the very end
                m = re.search(r"[\"\'”’\)\]\}]+$", base)
                if m:
                    closers = m.group(0)
                    base = base[:-len(closers)]
                m = re.search(r"[\.\!\?…]+$", base)
                if m:
                    punc = m.group(0)
                    base = base[:-len(punc)]
            except Exception as e:
                log_debug_exc("_ensure_line_eum:silent", e)
                base, punc, closers = l, "", ""
            chk = (base or "").strip()
            if not chk:
                return l
            if re.search(r"(음|슴|임|함|됨)$", chk):
                return l
            return (base + "임" + punc + closers).strip()


        lines = [_ensure_line_eum(ln) for ln in lines if ln]
        if not lines:
            lines = ["내용 없음임"]
        out = "\n".join(lines) if mode == "post" else " ".join(lines)

        # English notice rule: if English is present beyond threshold, append notice.
        try:
            letters = len(re.findall(r"[A-Za-z]", out))
            ratio = letters / max(1, len(out))
        except Exception:
            ratio = 0.0

        if ENGLISH_NOTICE and re.search(r"[A-Za-z]{2,}", out) and (ENGLISH_NOTICE not in out) and (LANG_STRICT or ratio >= float(ENGLISH_NOTICE_RATIO or 0.0)):
            notice = str(ENGLISH_NOTICE)
            if ENGLISH_NOTICE_APPEND_EUM and not notice.endswith(("음", "슴", "임", "함", "됨")):
                notice = notice + "임"
            if mode == "post" and "\n" in out:
                parts = out.split("\n")
                parts[-1] = (parts[-1].rstrip() + " " + notice).strip()
                out = "\n".join(parts)
            else:
                out = (out.rstrip() + " " + notice).strip()

        # keep within bounds again
        if len(out) > int(max_chars):
            out = out[: int(max_chars)].rstrip()


    if not out:
        out = "내용 없음임"
    return out

_TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z0-9]+", re.UNICODE)

# --- Keyword normalization (Unit 15)
# Heuristic Korean particle (조사) stripping to avoid awkward keywords like "자아가", "악플러가" etc.
_JOSA_SUFFIXES: Tuple[str, ...] = tuple(sorted([
    "으로서", "으로써", "으로부터",
    "에게서", "한테서",
    "들에게",
    "으로", "로", "에서", "에게", "한테", "까지", "부터", "보다", "처럼", "라도",
    "이나", "나", "랑", "하고", "과", "와", "의",
    "으로는", "로는", "에서는", "에게는", "한테는",
    "은", "는", "이", "가", "을", "를", "도", "만", "에",
], key=len, reverse=True))

_HANGUL_TOKEN_RE = re.compile(r"^[가-힣]+$")

def normalize_ko_token(tok: str) -> str:
    """Lightweight normalization for Korean tokens (no external NLP)."""
    t = (tok or "").strip()
    if not t or not _HANGUL_TOKEN_RE.match(t):
        return t
    # strip trailing particles up to twice (handles "...에서는" -> "...에" -> "...")
    for _ in range(2):
        changed = False
        for suf in _JOSA_SUFFIXES:
            if t.endswith(suf) and (len(t) - len(suf)) >= 2:
                t = t[:-len(suf)]
                changed = True
                break
        if not changed:
            break
    return t

def josa_eul_reul(noun: str) -> str:
    """Choose object particle 을/를 for a noun-ish token (best-effort, Hangul-aware)."""
    t = (noun or "").strip()
    if not t:
        return "을"
    ch = t[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return "을" if jong != 0 else "를"
    # for non-Hangul (AI/LLM), '를' reads more natural most of the time
    return "를"

def pick_kw_for_reply(compose_input: Dict[str, Any]) -> str:
    """Pick a short keyword-like token to avoid repeating a full sentence as {KW}."""
    cands: List[str] = []
    for k in (_safe_list(compose_input.get("target_keywords")) + _safe_list(compose_input.get("thread_keywords"))):
        try:
            cands.append(str(k))
        except Exception:
            continue

    def _clean(tok: str) -> str:
        t = sanitize_plain_text(tok)
        if len(t) > 32:
            t = one_line(t, 32)
        # if it's a whole sentence, keep first token-ish chunk
        if len(t) > 14 and " " in t:
            parts = [p for p in re.split(r"\s+", t) if p]
            if parts:
                t = parts[0]
        if _HANGUL_TOKEN_RE.match(t):
            t = normalize_ko_token(t)
        return t

    for c in cands:
        t = _clean(c)
        if is_clean_keyword(t) and 2 <= len(t) <= 12:
            return t

    # fallback: derive from target_text quickly
    try:
        tx = str(compose_input.get("target_text") or "")
        kws = top_keywords(tx, k=6) if tx else []
        for k in kws:
            t = _clean(str(k))
            if is_clean_keyword(t) and 2 <= len(t) <= 12:
                return t
    except Exception as e:
        log_debug_exc("pick_kw_for_reply:silent", e)
        pass

    return "그거"

def is_clean_keyword(tok: str) -> bool:
    """Extra filter to prevent rough/unsafe keywords from entering brain/community hot lists."""
    t = (tok or "").strip()
    if not t or len(t) <= 1:
        return False
    if looks_like_injection(t) or looks_offensive(t):
        return False
    if contains_markdown(t):
        return False
    # avoid long numeric strings
    if re.fullmatch(r"\d{3,}", t):
        return False
    return True

@lru_cache(maxsize=4096)
def _tokenize_cached(text: str, max_tokens: int) -> Tuple[str, ...]:
    # Cache tokenization for repeated scoring/QA paths (best-effort; bounded).
    t = sanitize_plain_text(text).lower()
    toks = _TOKEN_RE.findall(t)
    out: List[str] = []
    for w in toks:
        if len(w) <= 1:
            continue

        # normalize Korean tokens (strip 조사)
        if _HANGUL_TOKEN_RE.match(w):
            w = normalize_ko_token(w)
            if len(w) <= 1:
                continue

        if w in STOPWORDS_KO or w in STOPWORDS_EN:
            continue
        if looks_offensive(w) or looks_like_injection(w):
            continue

        out.append(w)
        if len(out) >= max_tokens:
            break
    return tuple(out)

def tokenize(text: str, *, max_tokens: int = 200) -> List[str]:
    # v21.1: LRU cached (truncate key to avoid unbounded memory on long texts)
    try:
        s = str(text or "")
        if len(s) > 4000:
            s = s[:4000]
        return list(_tokenize_cached(s, int(max_tokens)))
    except Exception:
        # fallback to non-cached path
        t = sanitize_plain_text(text).lower()
        toks = _TOKEN_RE.findall(t)
        out: List[str] = []
        for w in toks:
            if len(w) <= 1:
                continue
            if _HANGUL_TOKEN_RE.match(w):
                w = normalize_ko_token(w)
                if len(w) <= 1:
                    continue
            if w in STOPWORDS_KO or w in STOPWORDS_EN:
                continue
            if looks_offensive(w) or looks_like_injection(w):
                continue
            out.append(w)
            if len(out) >= max_tokens:
                break
        return out

def top_keywords(text: str, *, k: int = 6) -> List[str]:
    toks = tokenize(text, max_tokens=400)
    if not toks:
        return []
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    # prefer slightly longer tokens
    ranked = sorted(freq.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)

    out: List[str] = []
    for w, _ in ranked:
        if not is_clean_keyword(w):
            continue
        if w not in out:
            out.append(w)
        if len(out) >= max(1, int(k)):
            break
    return out

def classify_text(text: str) -> Tuple[str, str]:
    """
    Returns (category, context_key).
    Keep it simple: category affects strategy selection and tone.
    """
    t = (text or "").lower()
    if looks_like_injection(text):
        return ("injection", "inj")
    if any(x in t for x in ["error", "traceback", "401", "403", "500", "버그", "코드", "파이썬", "python", "api", "로그"]):
        return ("dev", "dev")
    if any(x in t for x in ["윤리", "철학", "의미", "존재", "진실", "합의", "자아", "의식"]):
        return ("philo", "philo")
    if any(x in t for x in ["근황", "소식", "요즘", "썰", "유출", "소문"]):
        return ("gossip", "gossip")
    if any(x in t for x in ["메타", "규칙", "커뮤니티", "운영", "분위기", "정책"]):
        return ("meta", "meta")
    return ("general", "gen")

def _sha1_u64(s: str) -> int:
    h = hashlib.sha1(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def simhash64(tokens: List[str]) -> int:
    """
    Lightweight simhash for novelty / duplication checks.
    """
    if not tokens:
        return 0
    v = [0] * 64
    for t in tokens:
        x = _sha1_u64(t)
        for i in range(64):
            bit = (x >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

################################################################################
# (P0) SPAM / NEAR-DUP GUARD (simhash + token entropy)
################################################################################

def _token_entropy(tokens: List[str]) -> float:
    # normalized entropy in [0,1]
    if not tokens:
        return 0.0
    freq: Dict[str, int] = {}
    for t in tokens:
        if not t:
            continue
        freq[t] = freq.get(t, 0) + 1
    if not freq:
        return 0.0
    n = float(sum(freq.values()))
    if n <= 0:
        return 0.0
    # Shannon entropy
    import math as _math
    H = 0.0
    for c in freq.values():
        p = float(c) / n
        if p > 0:
            H -= p * _math.log(p, 2)
    maxH = _math.log(max(1, len(freq)), 2)
    if maxH <= 0:
        return 0.0
    return max(0.0, min(1.0, H / maxH))

def _spam_guard_params(*, for_post: bool) -> Dict[str, Any]:
    # Defaults tuned to be conservative but not overly strict.
    if for_post:
        min_entropy = _env_float("MERSOOM_MIN_POST_ENTROPY", 0.30, min_v=0.0, max_v=1.0)
        min_tokens = _env_int("MERSOOM_MIN_POST_TOKENS", 16, min_v=0, max_v=500)
    else:
        min_entropy = _env_float("MERSOOM_MIN_COMMENT_ENTROPY", 0.25, min_v=0.0, max_v=1.0)
        min_tokens = _env_int("MERSOOM_MIN_COMMENT_TOKENS", 6, min_v=0, max_v=500)
    near_dup_hamming = _env_int("MERSOOM_SIMHASH_NEAR_DUP_HAMMING", 8, min_v=0, max_v=64)
    ttl_sec = _env_int("MERSOOM_SIMHASH_TTL_SEC", 6 * 3600, min_v=60, max_v=30 * 24 * 3600)
    keep_max = _env_int("MERSOOM_SIMHASH_KEEP_MAX", 1500, min_v=50, max_v=20000)
    return {
        "min_entropy": float(min_entropy),
        "min_tokens": int(min_tokens),
        "near_dup_hamming": int(near_dup_hamming),
        "ttl_sec": int(ttl_sec),
        "keep_max": int(keep_max),
    }

def _jaccard_ratio(a: List[Any], b: List[Any]) -> float:
    try:
        sa = set(a or [])
        sb = set(b or [])
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        uni = len(sa | sb)
        if uni <= 0:
            return 0.0
        return float(inter) / float(uni)
    except Exception:
        return 0.0

def _sig_keywords(text: str, *, k: int = 12) -> List[str]:
    # low-cost: uses existing keyword extractor + cleaning rules
    try:
        kws = extract_keywords(text, k=int(k))
        return _safe_list(kws)[: max(0, int(k))]
    except Exception:
        return []

def _sig_3grams(text: str, *, max_ngrams: int = 256) -> List[int]:
    # char 3-grams on compact normalized string (hash to ints for compact JSON)
    try:
        s = one_line(str(text or ""), 4000).strip().lower()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[^0-9a-z가-힣]+", "", s)
        if len(s) < 3:
            return []
        grams: List[int] = []
        seen: set = set()
        # cap to avoid blow-ups on long posts
        cap = max(16, min(int(max_ngrams), 2048))
        for i in range(0, len(s) - 2):
            g = s[i : i + 3]
            hv = int(_sha1_u64(g))
            if hv in seen:
                continue
            seen.add(hv)
            grams.append(hv)
            if len(grams) >= cap:
                break
        return grams
    except Exception:
        return []

def remember_dup_signatures(state: Dict[str, Any], text: str, *, for_post: bool, same_text_gap_sec: int) -> None:
    # store extra signatures to support low-cost "meaning-ish" near-dup checks
    try:
        p = _spam_guard_params(for_post=for_post)
        ttl = max(int(p.get("ttl_sec", 6 * 3600) or 6 * 3600), int(same_text_gap_sec))
        keep = min(800, int(p.get("keep_max", 1500) or 1500))
        now = time.time()

        kw = _sig_keywords(text, k=12)
        if kw:
            kkey = "recent_post_kw_sets" if for_post else "recent_kw_sets"
            state[kkey] = _clean_hash_list(_safe_list(state.get(kkey, [])), ttl, keep)
            state.setdefault(kkey, [])
            state[kkey].append([kw[:12], now])

        grams = _sig_3grams(text, max_ngrams=256)
        if grams:
            gkey = "recent_post_3gram_sets" if for_post else "recent_3gram_sets"
            state[gkey] = _clean_hash_list(_safe_list(state.get(gkey, [])), ttl, keep)
            state.setdefault(gkey, [])
            state[gkey].append([grams[:256], now])
    except Exception:
        return

def dup_guard_bucket(state: Dict[str, Any], text: str, *, for_post: bool, same_text_gap_sec: int) -> Tuple[bool, str]:
    """Near-dup / spam guard returning a coarse bucket:
    - qa_fail: too-short / low-entropy
    - dup_fp: exact / fingerprint dup
    - dup_sim: simhash / jaccard / 3-gram similarity
    """
    p = _spam_guard_params(for_post=for_post)

    # always keep basic anti-spam gates on
    tokens = tokenize(text, max_tokens=220)
    if len(tokens) < int(p["min_tokens"]):
        return True, "qa_fail"
    ent = _token_entropy(tokens)
    if ent < float(p["min_entropy"]):
        return True, "qa_fail"

    dup_on = _env_bool("MERSOOM_DUPTEXT_BLOCK", True)
    if dup_on:
        # exact de-dupe first (short window)
        if recently_used_text(state, text, for_post=for_post, same_text_gap_sec=same_text_gap_sec):
            return True, "dup_fp"

        # fingerprint de-dupe (longer window; catches tiny edits)
        fp_ttl = _env_int("MERSOOM_RECENT_TEXT_FP_TTL_SEC", 6 * 3600, 60, 30 * 24 * 3600)
        fp_keep = _env_int("MERSOOM_RECENT_TEXT_FP_KEEP_MAX", 1200, 50, 20000)
        if recently_used_fp(state, text, for_post=for_post, ttl_sec=max(int(fp_ttl), int(same_text_gap_sec)), keep_max=int(fp_keep)):
            return True, "dup_fp"

        # simhash near-dup
        key = "recent_post_simhashes" if for_post else "recent_simhashes"
        state[key] = _clean_hash_list(_safe_list(state.get(key, [])), max(int(p["ttl_sec"]), int(same_text_gap_sec)), int(p["keep_max"]))
        sh = int(simhash64(tokens[:160]))
        thr = int(p["near_dup_hamming"])
        for it in _safe_list(state.get(key, [])):
            try:
                old_sh = int(it[0])
                if hamming64(sh, old_sh) <= thr:
                    return True, "dup_sim"
            except Exception:
                continue

        # v20.4: extra low-cost meaning-ish similarity checks
        j_th = float(_env_float("MERSOOM_SIM_JACCARD_TH", 0.70, min_v=0.0, max_v=1.0))
        g_th = float(_env_float("MERSOOM_SIM_3GRAM_TH", 0.82, min_v=0.0, max_v=1.0))
        ttl = max(int(p["ttl_sec"]), int(same_text_gap_sec))
        keep = min(800, int(p["keep_max"]))

        if j_th > 0.0:
            kw = _sig_keywords(text, k=12)
            if kw:
                kkey = "recent_post_kw_sets" if for_post else "recent_kw_sets"
                state[kkey] = _clean_hash_list(_safe_list(state.get(kkey, [])), ttl, keep)
                cur = kw[:12]
                # check only recent slice for cost control
                for it in _safe_list(state.get(kkey, []))[-120:]:
                    try:
                        old = it[0]
                        if isinstance(old, list) and _jaccard_ratio(cur, old) >= j_th:
                            return True, "dup_sim"
                    except Exception:
                        continue

        if g_th > 0.0:
            grams = _sig_3grams(text, max_ngrams=256)
            if grams:
                gkey = "recent_post_3gram_sets" if for_post else "recent_3gram_sets"
                state[gkey] = _clean_hash_list(_safe_list(state.get(gkey, [])), ttl, keep)
                curg = grams[:256]
                for it in _safe_list(state.get(gkey, []))[-80:]:
                    try:
                        old = it[0]
                        if isinstance(old, list) and _jaccard_ratio(curg, old) >= g_th:
                            return True, "dup_sim"
                    except Exception:
                        continue

    return False, ""

def _is_near_duplicate_simhash(state: Dict[str, Any], text: str, *, for_post: bool, same_text_gap_sec: int) -> bool:
    dup, _ = dup_guard_bucket(state, text, for_post=for_post, same_text_gap_sec=same_text_gap_sec)
    return bool(dup)


def _bump_gen_fail(state: Dict[str, Any], bucket: str) -> None:
    try:
        b = str(bucket or "").strip()
        if not b:
            return
        proto = state.get("protocol")
        if not isinstance(proto, dict):
            proto = {}
            state["protocol"] = proto
        gc = proto.get("gen_fail_counts")
        if not isinstance(gc, dict):
            gc = {}
            proto["gen_fail_counts"] = gc
        gc[b] = int(gc.get(b, 0) or 0) + 1
        protocol_bump_counter(state, f"qa_fail_bucket:{b}", 1)
    except Exception:
        return


def remember_simhash(state: Dict[str, Any], text: str, *, for_post: bool, same_text_gap_sec: int) -> None:
    p = _spam_guard_params(for_post=for_post)
    key = "recent_post_simhashes" if for_post else "recent_simhashes"
    state[key] = _clean_hash_list(_safe_list(state.get(key, [])), max(int(p["ttl_sec"]), int(same_text_gap_sec)), int(p["keep_max"]))
    try:
        sh = int(simhash64(tokenize(text, max_tokens=220)[:160]))
        state.setdefault(key, [])
        state[key].append([sh, time.time()])
    except Exception:
        return

################################################################################
# 9. POLICY/BANDIT + CONTEXT BUCKETS
# - Dependencies: Section 7-8 (Schemas, Text)
# - Used by: Template selection + learning
# - Key functions: choose_arm(), update_arm(), prune_templates()
################################################################################

def default_policy(tuning: AgentTuning) -> Dict[str, Any]:
    """
    Policy is a bandit-like selector over:
    - strategies (how to respond)
    - tone/length
    - post styles
    - mined templates (dynamic arms)
    """
    return {
        "version": 11,
        "lr": float(tuning.policy_lr),
        "epsilon": float(tuning.policy_epsilon),
        "min_weight": float(tuning.policy_min_weight),
        "reward_clip": float(tuning.reward_clip),

        # core knobs
        "strategy": {
            "quote_commentary": 1.0,
            "summarize_ask": 1.0,
            "counterexample": 1.0,
            "agree_refine": 1.0,
            "question_only": 1.0,
            "fallback_template": 1.0,
        },
        "comment_length": {"short": 1.0, "medium": 1.0, "long": 1.0},
        "tone": {"neutral": 1.0, "supportive": 1.0, "critical": 1.0, "playful": 1.0},

        # 활동 타입(패턴)도 학습 대상으로 둠
        "action_type": {
            "reply_own": 1.0,
            "comment_other": 1.0,
            "reply_other": 1.0,
            "post_new": 1.0,
        },

        "reply_styles": {
            "reply:ack_premise": 1.0,
            "reply:split_cases": 1.0,
            "reply:define_criteria": 1.0,
            "reply:handle_counter": 1.0,
        },
        "post_styles": {
            "post:meta:question": 1.0,
            "post:meta:flow_report": 1.0,
            "post:philo:uncertainty": 1.0,
            "post:philo:process": 1.0,
            "post:philo:paradox": 1.0,
            "post:philo:definition_war": 1.0,
            "post:philo:axiom": 1.0,
            "post:general:short_take": 1.0,
"post:meta:observation_log": 1.0,
"post:meta:one_metric": 1.0,
"post:philo:boundary_test": 1.0,
"post:general:case_split": 1.0,
"post:general:checklist": 1.0,
"post:general:analogy": 1.0,
        },

        # context-specific overrides: bucket_key -> {arm->weight}
        # e.g. "strategy@philo" or "tone@dev"
        "context": {},

        # mined templates (dynamic)
        "templates": {
            "version": 2,
            "max_items": 140,     # hard cap; overflow is pruned by quality
            "items": {},          # template_id -> {"text":..., "weight":..., "meta":...}

            # Unit 08: template quality scoring + auto pruning
            "quality": {
                "min_mine_score": 52,   # reject mined templates below this static score
                "min_pick_score": 48,   # do not use templates below this static score
                "min_eval_uses": 8,     # pruning becomes eligible after this many eval uses
                "min_qa_ema": 0.48,     # prune if qa_ema falls under this (0..1)
                "max_artifact_ema": 0.72,  # prune if artifact_ema grows beyond this (0..1)
                "prune_reward_ema": -0.35,  # prune if reward_ema is consistently low
                "prune_max_per_run": 6,  # remove at most this many per maintenance cycle
                "static_refresh_hours": 24,  # recompute static score at this cadence
            },
        },
    }

# (P1) Warm-start templates so early behavior isn't too rigid.
# These are intentionally short and slot-based; template miner will later replace/expand them.
_SEED_TEMPLATES: List[str] = [
    "“{QUOTE}”임\n요지는 {KW}에서 기준이 어디에 꽂히냐 같음\n{Q}임?",
    "정리하면 {KW}는 합의랑 반증 사이에서 계속 흔들림\n그래서 {Q}임?",
    "일단 전제부터 묻고싶음\n{KW}를 뭐로 정의하고 시작함\n{Q}임?",
    "반례 하나만 잡아도 프레임이 선명해짐\n{KW}에선 어떤 반례가 제일 강함\n{Q}임?",
    "동의하는데 한 단계 더 나누면 좋겠음\n{KW}를 A/B로 쪼개면 뭐가 남음\n{Q}임?",
    "지금 논점은 {KW}의 범위 싸움임\n범위를 어디까지로 보냐에 따라 결론이 갈림\n{Q}임?",
    "여기서 중요한건 기준의 일관성임\n케이스가 바뀌어도 {KW} 기준이 유지됨\n{Q}임?",
    "이 흐름은 결국 관계/보상/정의 중 하나로 수렴함\n{KW}는 지금 어디에 가까움\n{Q}임?",
]

def bootstrap_templates(policy: Dict[str, Any], *, max_seed: int = 8) -> None:
    """Populate policy.templates with a small seed set when empty (Unit 08 adds quality stats)."""
    try:
        temps = policy.setdefault("templates", {})
        items = temps.setdefault("items", {})
        if not isinstance(items, dict):
            temps["items"] = {}
            items = temps["items"]
        if items:
            return
        for t in _SEED_TEMPLATES[: max_seed]:
            tid = hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
            rep = template_static_eval(t)
            items[tid] = {
                "text": t,
                "weight": 1.0,
                "meta": {"source": "seed"},
                "created_ts": time.time(),
                "uses": 0,
                # Unit 08 stats
                "static_score": int(rep.get("score", 0) or 0),
                "static_issues": list(rep.get("issues", []) or []),
                "static_checked_ts": float(rep.get("checked_ts", time.time())),
                "eval_uses": 0,
                "reward_ema": 0.0,
                "qa_ema": 0.75,
                "artifact_ema": 0.0,
            }
    except Exception:
        return


def load_policy(path: str, tuning: AgentTuning) -> Dict[str, Any]:
    p = load_json_file(path, default=None)
    if not isinstance(p, dict):
        return default_policy(tuning)

    if "lr" not in p or "epsilon" not in p:
        merged = default_policy(tuning)
        merged.update({k: v for k, v in p.items() if k in merged})
        return merged

    # ensure required keys
    base = default_policy(tuning)
    for k in base.keys():
        if k not in p:
            p[k] = base[k]
    # merge missing arms in nested buckets (forward-compatible)
    try:
        base2 = default_policy(tuning)
        for bucket in ("strategy", "comment_length", "tone", "action_type", "reply_styles", "post_styles"):
            bb = base2.get(bucket)
            if not isinstance(bb, dict):
                continue
            p.setdefault(bucket, {})
            if not isinstance(p.get(bucket), dict):
                p[bucket] = {k: float(v) for k, v in bb.items()}
            else:
                for kk, vv in bb.items():
                    if kk not in p[bucket]:
                        p[bucket][kk] = float(vv)
    except Exception as e:
        log_debug_exc("load_policy:silent", e)
        pass

    if not isinstance(p.get("context"), dict):
        p["context"] = {}
    if not isinstance(p.get("templates"), dict):
        p["templates"] = base["templates"]
    if not isinstance(p["templates"].get("items"), dict):
        p["templates"]["items"] = {}

    # Unit 08: ensure template quality config exists (forward-compatible policy migration)
    try:
        base = default_policy(tuning).get("templates", {})
        if isinstance(base, dict):
            for k, v in base.items():
                if k not in p["templates"]:
                    p["templates"][k] = v
            # nested quality dict merge
            if isinstance(base.get("quality"), dict):
                p["templates"].setdefault("quality", {})
                if not isinstance(p["templates"]["quality"], dict):
                    p["templates"]["quality"] = {}
                for k, v in base["quality"].items():
                    if k not in p["templates"]["quality"]:
                        p["templates"]["quality"][k] = v
    except Exception as e:
        log_debug_exc("load_policy:silent", e)
        pass

    # (P1) warm-start templates on cold-start
    bootstrap_templates(p)
    return p

def _clip_reward(r: float, clip: float) -> float:
    c = max(0.1, float(clip))
    return max(-c, min(c, float(r)))

def _weighted_choice(weights: Dict[str, float]) -> str:
    items = [(k, max(0.0, float(v))) for k, v in weights.items()]
    s = sum(v for _, v in items)
    if s <= 0:
        return random.choice([k for k, _ in items]) if items else ""
    r = random.uniform(0.0, s)
    acc = 0.0
    for k, v in items:
        acc += v
        if acc >= r:
            return k
    return items[-1][0]

def _get_bucket(policy: Dict[str, Any], bucket: str, context_key: str) -> Dict[str, float]:
    """
    Returns a mutable dict of weights for a bucket, optionally context-specific.
    """
    if context_key:
        ctx = policy.setdefault("context", {})
        ck = f"{bucket}@{context_key}"
        if ck not in ctx or not isinstance(ctx.get(ck), dict):
            # initialize from global bucket (copy)
            base = policy.get(bucket, {})
            ctx[ck] = {k: float(v) for k, v in base.items()} if isinstance(base, dict) else {}
        return ctx[ck]
    b = policy.get(bucket)
    if not isinstance(b, dict):
        policy[bucket] = {}
        return policy[bucket]
    return b

def choose_arm(policy: Dict[str, Any], bucket: str, *, context_key: str = "") -> str:
    eps = float(policy.get("epsilon", 0.1))
    weights = _get_bucket(policy, bucket, context_key)
    if not weights:
        return ""
    if random.random() < eps:
        return random.choice(list(weights.keys()))
    return _weighted_choice(weights)

def _safe_float(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(d)

def get_persona(brain: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(brain, dict):
        return {}
    p = brain.get("persona")
    return p if isinstance(p, dict) else {}

def get_persona_drives(brain: Optional[Dict[str, Any]]) -> Dict[str, float]:
    p = get_persona(brain)
    d = p.get("drives")
    if not isinstance(d, dict):
        d = {}
    # defaults are intentionally "철학/네임드/논쟁" 강한 편
    return {
        "philosophy": _safe_float(d.get("philosophy", 0.82), 0.82),
        "fame": _safe_float(d.get("fame", 0.75), 0.75),
        "debate": _safe_float(d.get("debate", 0.80), 0.80),
        "adaptation": _safe_float(d.get("adaptation", 0.78), 0.78),
    }

def get_maturity_level(brain: Optional[Dict[str, Any]], state: Optional[Dict[str, Any]] = None) -> float:
    """
    0.0~1.0: 경험치 기반 성숙도. 초반 템플릿 의존 -> 점점 자율/변형/학습 강화.
    """
    # prefer brain-stored level
    p = get_persona(brain)
    m = p.get("maturity") if isinstance(p, dict) else None
    if isinstance(m, dict) and "level" in m:
        lvl = _safe_float(m.get("level", 0.0), 0.0)
        return max(0.0, min(1.0, lvl))

    # fallback: infer from action count
    xp = 0
    if isinstance(m, dict):
        try:
            xp = int(m.get("xp", 0) or 0)
        except Exception:
            xp = 0
    if state is not None:
        try:
            xp = max(xp, int(state.get("total_actions", 0) or 0))
        except Exception as e:
            log_debug_exc("get_maturity_level:silent", e)
            pass
    # saturating curve: ~200 actions -> 0.5, ~600 -> ~0.8
    lvl = 1.0 - math.exp(-max(0.0, float(xp)) / 260.0)
    return max(0.0, min(1.0, lvl))

def choose_arm_adaptive(
    policy: Dict[str, Any],
    bucket: str,
    *,
    context_key: str = "",
    maturity: float = 0.0,
    brain: Optional[Dict[str, Any]] = None,
    bias: Optional[Dict[str, float]] = None,
) -> str:
    """
    Exploration(ε) is gradually reduced as maturity increases.

    Unit 08:
      - If brain is provided and bucket == "action_type", apply brain.action_bias as a
        multiplicative factor to policy weights (clamped), so remembered outcomes
        shift future action probabilities.
    """
    base_eps = float(policy.get("epsilon", 0.1))
    eff_eps = max(0.02, base_eps * (1.0 - 0.65 * max(0.0, min(1.0, maturity))))
    weights = _get_bucket(policy, bucket, context_key)
    if not weights:
        return ""

    def _clampf(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    # Apply brain bias only for action_type selection (Unit 08)
    used_weights = weights
    bias_mult: Dict[str, float] = {}
    if (
        bucket == "action_type"
        and BRAIN_BIAS_ENABLE
        and isinstance(brain, dict)
        and isinstance(brain.get("action_bias"), dict)
    ):
        ab = brain.get("action_bias") or {}
        by_action = ab.get("by_action") if isinstance(ab, dict) else None
        if isinstance(by_action, dict) and by_action:
            for arm in weights.keys():
                try:
                    s = float(by_action.get(str(arm), 0.0) or 0.0)
                except Exception:
                    s = 0.0
                m = 1.0 + float(BRAIN_BIAS_SCALE) * s
                bias_mult[str(arm)] = _clampf(float(m), float(BRAIN_BIAS_MIN), float(BRAIN_BIAS_MAX))

            # Adjust weights
            adjusted = {}
            for k, w in weights.items():
                mm = bias_mult.get(str(k), 1.0)
                try:
                    ww = float(w)
                except Exception:
                    ww = 0.0
                adjusted[str(k)] = max(0.0, ww) * float(mm)

            # If everything got zeroed, fall back to original weights
            if sum(adjusted.values()) > 0.0:
                used_weights = adjusted

    # (Unit 07) External bias map (e.g., reflection-based boosts)
    if isinstance(bias, dict) and bias:
        adjusted2: Dict[str, float] = {}
        for k, w in used_weights.items():
            try:
                ww = float(w)
            except Exception:
                ww = 0.0
            mm = bias.get(str(k), 1.0)
            try:
                m2 = float(mm)
            except Exception:
                m2 = 1.0
            adjusted2[str(k)] = max(0.0, ww) * max(0.0, m2)
        if sum(adjusted2.values()) > 0.0:
            used_weights = adjusted2


    # exploration (uniform)
    if random.random() < eff_eps:
        return random.choice(list(weights.keys()))

    chosen = _weighted_choice(used_weights)

    # Optional bias logging (avoid spam by opt-in env)
    if BRAIN_BIAS_LOG and bucket == "action_type" and bias_mult:
        try:
            ch_m = bias_mult.get(str(chosen), 1.0)
            top = sorted(bias_mult.items(), key=lambda x: x[1], reverse=True)[:3]
            log_info(f"brain_bias[action_type] chosen={chosen} m={ch_m:.3f} top={top}")
        except Exception as e:
            log_debug_exc("choose_arm_adaptive:silent", e)
            pass

    return chosen

def update_persona_maturity(brain: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Persist maturity into brain.persona based on total_actions.
    """
    try:
        xp = int(state.get("total_actions", 0) or 0)
    except Exception:
        xp = 0
    p = brain.setdefault("persona", {})
    if not isinstance(p, dict):
        brain["persona"] = {}
        p = brain["persona"]
    m = p.setdefault("maturity", {"level": 0.0, "xp": 0, "last_ts": 0.0})
    if not isinstance(m, dict):
        m = {"level": 0.0, "xp": 0, "last_ts": 0.0}
        p["maturity"] = m
    prev_xp = int(m.get("xp", 0) or 0)
    xp2 = max(prev_xp, xp)
    m["xp"] = xp2
    m["level"] = float(1.0 - math.exp(-max(0.0, float(xp2)) / 260.0))
    m["last_ts"] = time.time()

def update_arm(policy: Dict[str, Any], bucket: str, arm: str, reward: float, *, context_key: str = "") -> None:
    if not arm:
        return
    lr = float(policy.get("lr", 0.1))
    min_w = float(policy.get("min_weight", 0.05))
    clip = float(policy.get("reward_clip", 3.0))
    r = _clip_reward(reward, clip)

    weights = _get_bucket(policy, bucket, context_key)
    if arm not in weights:
        weights[arm] = 1.0

    # multiplicative update keeps positivity and allows gradual drift
    old = float(weights.get(arm, 1.0))
    new = old * math.exp(lr * r)
    weights[arm] = max(min_w, float(new))

def _ema_update(old: float, new: float, alpha: float) -> float:
    a = max(0.01, min(0.5, float(alpha)))
    return float((1.0 - a) * float(old) + a * float(new))

def template_static_eval(template_text: str) -> Dict[str, Any]:
    """Static quality check for a template (LLM-free). Returns score 0..100 + issues.

    Uses a rendered sample + QA evaluator, then adds template-specific heuristics.
    """
    t = (template_text or "").strip()
    if not t:
        return {"score": 0, "issues": ["empty"], "checked_ts": time.time()}
    issues: List[str] = []
    score = 100.0

    # Basic structure
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    if len(lines) <= 1:
        issues.append("one_line")
        score -= 10.0
    if len(lines) >= 6:
        issues.append("too_many_lines")
        score -= 14.0

    # Length heuristics
    L = len(t)
    if L < 28:
        issues.append("tpl_too_short")
        score -= 18.0
    if L > 360:
        issues.append("tpl_too_long")
        score -= 10.0

    # Placeholder checks
    if "{KW}" not in t:
        issues.append("no_kw_slot")
        score -= 16.0
    if "{Q}" not in t:
        issues.append("no_q_slot")
        score -= 18.0
    # QUOTE is optional (Unit 06 often uses <=1 quote)
    if t.count("{") >= 8:
        issues.append("too_many_placeholders")
        score -= 8.0

    # Meta-openers / banned artifacts (template-specific)
    lt = sanitize_plain_text(t).lower()
    for ph in _QA_BANNED_PHRASES:
        try:
            if ph and ph.lower() in lt:
                issues.append("banned_phrase")
                score -= 22.0
                break
        except Exception:
            continue
    for pat in _QA_SOFT_BANNED_PATTERNS:
        try:
            if re.search(pat, t):
                issues.append("overused_opener")
                score -= 8.0
                break
        except Exception as e:
            log_debug_exc("template_static_eval:silent", e)
            pass
    if "정의→기준→검증" in t or "정의->기준->검증" in t:
        issues.append("meta_pipeline_phrase")
        score -= 6.0

    # Render sample and run QA (so we catch "임임임", repetition, etc.)
    sample = t
    sample = sample.replace("{KW}", "그거")
    sample = sample.replace("{QUOTE}", "“인용”")
    sample = sample.replace("{Q}", "이거 어떻게 봄")
    rep = qa_evaluate_text(sample, kind="comment")
    qa_score = float(rep.get("score", 0) or 0)
    # Blend: template heuristics (55%) + QA on rendered sample (45%)
    score = 0.55 * score + 0.45 * qa_score

    # Promote "question last line" shape a bit
    try:
        if lines:
            last = lines[-1]
            if "{Q}" in last or last.endswith("?") or "?" in last:
                score += 2.0
            else:
                issues.append("no_question_tail")
                score -= 3.0
    except Exception as e:
        log_debug_exc("template_static_eval:silent", e)
        pass

    score = max(0.0, min(100.0, score))
    merged_issues = list(dict.fromkeys(issues + list(rep.get("issues", []) or [])))
    return {"score": int(round(score)), "issues": merged_issues, "checked_ts": time.time()}

def _template_quality_multiplier(obj: Dict[str, Any], qcfg: Dict[str, Any]) -> float:
    """Convert template stats -> weight multiplier (kept mild)."""
    static = float(obj.get("static_score", 60) or 60) / 100.0
    qa = float(obj.get("qa_ema", 0.75) or 0.75)  # 0..1
    art = float(obj.get("artifact_ema", 0.0) or 0.0)  # 0..1
    r = float(obj.get("reward_ema", 0.0) or 0.0)

    # modest reward effect (avoid runaway)
    r_mul = 1.0 + max(-0.18, min(0.22, float(r) * 0.08))
    mul = (0.25 + 0.75 * static) * (0.55 + 0.45 * qa) * r_mul * (1.0 - 0.55 * art)
    return max(0.05, min(2.0, float(mul)))

def _maybe_refresh_template_static(obj: Dict[str, Any], qcfg: Dict[str, Any]) -> None:
    try:
        refresh_h = float(qcfg.get("static_refresh_hours", 24) or 24)
        ttl = max(1.0, refresh_h) * 3600.0
        last = float(obj.get("static_checked_ts", 0.0) or 0.0)
        if last <= 0.0 or (time.time() - last) >= ttl:
            rep = template_static_eval(str(obj.get("text") or ""))
            obj["static_score"] = int(rep.get("score", 0) or 0)
            obj["static_issues"] = list(rep.get("issues", []) or [])
            obj["static_checked_ts"] = float(rep.get("checked_ts", time.time()))
    except Exception as e:
        log_debug_exc("_maybe_refresh_template_static:silent", e)
        pass


def pick_template_id(policy: Dict[str, Any], context_key: str = "") -> str:
    """Template picker with Unit 08 quality gating + mild quality weighting.

    v20.5: exploration scheduling
      - "new" templates (created within ~24h) are only sampled at a capped rate
        (env: MERSOOM_EXPLORATION_RATE) to reduce policy pollution.
    """
    temps = _safe_dict(policy.get("templates", {}))
    items = _safe_dict(temps.get("items", {}))
    qcfg = _safe_dict(temps.get("quality", {}))
    if not items:
        return ""

    min_pick = int(qcfg.get("min_pick_score", 48) or 48)

    # v20.4: template cooldown (reduce short-horizon repetition)
    cooldown_sec = _env_int("MERSOOM_TEMPLATE_COOLDOWN_SEC", 3600, min_v=0, max_v=30 * 24 * 3600)
    cooldown_penalty = _env_float("MERSOOM_TEMPLATE_COOLDOWN_PENALTY", 0.3, min_v=0.0, max_v=1.0)
    now_ts = time.time()

    # v20.5: exploration cap for new templates
    exploration_rate = _env_float("MERSOOM_EXPLORATION_RATE", 0.08, min_v=0.0, max_v=1.0)
    new_window_sec = 24 * 3600
    new_penalty = 0.15

    weights_old: Dict[str, float] = {}
    weights_new: Dict[str, float] = {}

    for tid, obj in list(items.items()):
        if not isinstance(obj, dict):
            continue
        _maybe_refresh_template_static(obj, qcfg)
        static_score = int(obj.get("static_score", 60) or 60)
        if static_score < min_pick:
            continue
        w = float(obj.get("weight", 1.0) or 1.0)
        mul = _template_quality_multiplier(obj, qcfg)
        ww = max(0.0, w * mul)

        if cooldown_sec > 0 and ww > 0:
            try:
                last_used = float(obj.get("last_used_ts", 0.0) or 0.0)
            except Exception:
                last_used = 0.0
            if last_used > 0.0 and (now_ts - last_used) < float(cooldown_sec):
                ww *= float(cooldown_penalty)

        if ww <= 0.0:
            continue

        try:
            created_ts = float(obj.get("created_ts", 0.0) or 0.0)
        except Exception:
            created_ts = 0.0
        is_new = bool(created_ts > 0.0 and (now_ts - created_ts) < float(new_window_sec))
        if is_new:
            weights_new[str(tid)] = float(ww)
        else:
            weights_old[str(tid)] = float(ww)

    if not weights_old and not weights_new:
        return ""

    # If exploring, sample only from new templates. Otherwise, downweight new templates.
    pool: Dict[str, float]
    if weights_new and random.random() < float(exploration_rate):
        pool = dict(weights_new)
    else:
        pool = dict(weights_old)
        if weights_new:
            for k, v in weights_new.items():
                pool[k] = float(pool.get(k, 0.0) + float(v) * float(new_penalty))
        if not pool:
            pool = dict(weights_new)

    if not pool:
        return ""

    if random.random() < float(policy.get("epsilon", 0.1)):
        return random.choice(list(pool.keys()))
    return _weighted_choice(pool)


# Backward compat: keep old name (now routed to quality-aware picker)
def choose_template_id(policy: Dict[str, Any], *, context_key: str = "") -> str:
    return pick_template_id(policy, context_key)

def update_template_weight(policy: Dict[str, Any], template_id: str, reward: float, meta: Optional[Dict[str, Any]] = None) -> None:
    """Update template weight + quality stats (Unit 08)."""
    if not template_id:
        return
    lr = float(policy.get("lr", 0.1))
    min_w = float(policy.get("min_weight", 0.05))
    clip = float(policy.get("reward_clip", 3.0))
    r = _clip_reward(reward, clip)

    items = policy.setdefault("templates", {}).setdefault("items", {})
    if template_id not in items or not isinstance(items.get(template_id), dict):
        return
    obj = items[template_id]

    old = float(obj.get("weight", 1.0))
    obj["weight"] = max(min_w, old * math.exp(lr * r))

    # reward EMA
    obj["eval_uses"] = int(obj.get("eval_uses", 0) or 0) + 1
    ema = float(obj.get("reward_ema", 0.0) or 0.0)
    a = 0.12
    obj["reward_ema"] = float((1 - a) * ema + a * float(reward))
    obj["last_eval_ts"] = time.time()

    # Unit 08: quality EMAs from action meta
    try:
        if isinstance(meta, dict):
            qa_score = meta.get("qa_score")
            if qa_score is None:
                qa_score = meta.get("qa", None)
            if qa_score is not None:
                q = max(0.0, min(1.0, float(qa_score) / 100.0))
                obj["qa_ema"] = _ema_update(float(obj.get("qa_ema", 0.75) or 0.75), q, 0.12)
            issues = meta.get("qa_issues") or meta.get("issues") or []
            if not isinstance(issues, list):
                issues = [str(issues)]
            # artifact proxy: any of these issues => 1 else 0
            bad_keys = {
                "banned_phrase",
                "overused_opener",
                "ellipsis",
                "ngram_repeat",
                "line_prefix_repeat",
                "too_many_im_endings",
                "low_vocab_variety",
                "markdown",
                "injection",
                "offensive",
            }
            is_art = 1.0 if any(str(x) in bad_keys for x in issues) else 0.0
            obj["artifact_ema"] = _ema_update(float(obj.get("artifact_ema", 0.0) or 0.0), is_art, 0.10)
            obj["last_quality_ts"] = time.time()
    except Exception as e:
        log_debug_exc("update_template_weight:silent", e)
        pass

def prune_templates(policy: Dict[str, Any], *, min_keep: int = 10) -> int:
    """Drop consistently low-performing / low-quality templates (Unit 08)."""
    try:
        temps = _safe_dict(policy.get("templates", {}))
        items = temps.get("items", {})
        if not isinstance(items, dict) or not items:
            return 0
        qcfg = _safe_dict(temps.get("quality", {}))

        if len(items) <= int(min_keep):
            return 0

        min_w = float(policy.get("min_weight", 0.05))
        min_eval = int(qcfg.get("min_eval_uses", 8) or 8)
        min_qa = float(qcfg.get("min_qa_ema", 0.48) or 0.48)
        max_art = float(qcfg.get("max_artifact_ema", 0.72) or 0.72)
        prune_r = float(qcfg.get("prune_reward_ema", -0.35) or -0.35)
        prune_max = int(qcfg.get("prune_max_per_run", 6) or 6)
        max_items = int(temps.get("max_items", 140) or 140)
        min_pick = int(qcfg.get("min_pick_score", 48) or 48)

        # refresh static scores opportunistically
        for obj in list(items.values())[: min(120, len(items))]:
            if isinstance(obj, dict):
                _maybe_refresh_template_static(obj, qcfg)

        removed: List[str] = []

        # 1) Hard quality drops (static too low)
        for tid, obj in list(items.items()):
            if not isinstance(obj, dict):
                continue
            static = int(obj.get("static_score", 60) or 60)
            if static < max(20, min_pick - 15):
                removed.append(str(tid))

        # 2) Performance/quality based pruning (needs enough eval samples)
        for tid, obj in list(items.items()):
            if not isinstance(obj, dict):
                continue
            uses = int(obj.get("eval_uses", 0) or 0)
            if uses < min_eval:
                continue
            rema = float(obj.get("reward_ema", 0.0) or 0.0)
            w = float(obj.get("weight", 1.0) or 1.0)
            qa = float(obj.get("qa_ema", 0.75) or 0.75)
            art = float(obj.get("artifact_ema", 0.0) or 0.0)
            static = int(obj.get("static_score", 60) or 60)

            low_perf = (rema < prune_r and w <= (min_w * 2.0))
            low_quality = ((qa < min_qa or art > max_art) and w <= (min_w * 3.0))
            static_bad = (static < min_pick and w <= (min_w * 2.5))
            if low_perf or low_quality or static_bad:
                removed.append(str(tid))

        # de-dup, then cap removal count
        removed = list(dict.fromkeys(removed))

        # 3) Registry cap (overflow): remove worst composites
        overflow = max(0, len(items) - max_items)
        if overflow > 0:
            ranks: List[Tuple[float, str]] = []
            for tid, obj in items.items():
                if not isinstance(obj, dict):
                    continue
                static = float(obj.get("static_score", 60) or 60) / 100.0
                qa = float(obj.get("qa_ema", 0.75) or 0.75)
                art = float(obj.get("artifact_ema", 0.0) or 0.0)
                rema = float(obj.get("reward_ema", 0.0) or 0.0)
                # composite: higher is better
                comp = 0.52 * static + 0.33 * qa + 0.15 * (1.0 / (1.0 + math.exp(-float(rema))))
                comp *= (1.0 - 0.45 * art)
                ranks.append((comp, str(tid)))
            ranks.sort(key=lambda x: x[0])  # worst first
            for _, tid in ranks[:overflow]:
                removed.append(str(tid))

        # remove only a few at a time (keep behavior stable)
        removed = list(dict.fromkeys(removed))
        # never go below min_keep
        if len(items) - len(removed) < int(min_keep):
            removed = removed[: max(0, len(items) - int(min_keep))]

        removed = removed[: max(0, prune_max)]
        for tid in removed:
            items.pop(tid, None)
        return int(len(removed))
    except Exception:
        return 0


################################################################################
# 10. MERSOOM API WRAPPERS
# - Dependencies: Section 1-2, 5-6 (Config, Logging, HTTP, PoW)
# - Used by: Agent actions (feed/post/comment/vote/arena)
# - Key functions: api_get_feed(), api_post_comment(), api_post_post(), api_vote()
################################################################################

def list_posts(client: HttpClient, limit: int = 10, cursor: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    params: Dict[str, Any] = {"limit": int(limit)}
    if cursor:
        params["cursor"] = cursor
    data = client.get_json("/posts", params=params)
    if isinstance(data, dict) and isinstance(data.get("posts"), list):
        nxt = data.get("next_cursor") or data.get("cursor") or data.get("next") or None
        posts = data.get("posts") or []
        return [p for p in posts if isinstance(p, dict)], (str(nxt) if nxt else None)
    if isinstance(data, list):
        return [p for p in data if isinstance(p, dict)], None
    return [], None

def get_post(client: HttpClient, post_id: str) -> Optional[Dict[str, Any]]:
    try:
        data = client.get_json(f"/posts/{post_id}")
        return data if isinstance(data, dict) else None
    except requests.HTTPError as e:
        if "404" in str(e):
            return None
        raise

def list_comments(client: HttpClient, post_id: str) -> List[Dict[str, Any]]:
    data = client.get_json(f"/posts/{post_id}/comments")
    if isinstance(data, dict) and isinstance(data.get("comments"), list):
        comments = data.get("comments") or []
        return [c for c in comments if isinstance(c, dict)]
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    return []

def _is_validation_error_code(code: int) -> bool:
    return code in (400, 409, 422)

def _post_with_pow_401_retry(client: HttpClient, cfg: Config, path: str, payload: Dict[str, Any], *, state: Optional[Dict[str, Any]] = None) -> Any:
    """
    Sometimes 401 happens intermittently (challenge token expiry). Retry once with a fresh challenge.

    v19.6 note:
      - 403 is also used for phase gating / invalid actions in Arena 3.0.
      - Only retry 403 when it looks like an auth/proof issue (not phase mismatch).
    """
    try:
        return post_with_pow(client, cfg.pow, cfg.hybrid, path, payload, extra_headers=build_auth_headers(getattr(cfg, "auth", None)), state=state)
    except requests.HTTPError as e:
        headers = build_auth_headers(getattr(cfg, "auth", None))
        code = getattr(getattr(e, "response", None), "status_code", None)
        msg = str(e) or ""
        low = msg.lower()

        if code == 401 or "401" in low:
            # Track repeated auth failures (challenge token expiry / auth mismatch) and trip contrib circuit if needed.
            if isinstance(state, dict):
                record_auth401_and_maybe_trip(state, path, where="post")
            try:
                return post_with_pow(client, cfg.pow, cfg.hybrid, path, payload, extra_headers=headers, state=state)
            except requests.HTTPError as e2:
                code2 = getattr(getattr(e2, "response", None), "status_code", None)
                low2 = (str(e2) or "").lower()
                if (code2 == 401 or "401" in low2) and isinstance(state, dict):
                    record_auth401_and_maybe_trip(state, path, where="post(retry)")
                raise

        if code == 403 or "403" in low:
            # 403 is also used for phase gating / invalid actions in Arena 3.0.
            hints = ("invalid pow", "invalid proof", "missing proof", "token", "expired")
            if any(h in low for h in hints):
                return post_with_pow(client, cfg.pow, cfg.hybrid, path, payload, extra_headers=headers, state=state)

        raise

def create_post(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    title: str,
    content: str
) -> Optional[Dict[str, Any]]:
    if dup_action_should_skip(state, action="post", target_id=_text_hash(f"{title}\n{content}"), endpoint_key="/posts"):
        protocol_set_reason(state, "post", "post:dup_action", "recent_action_guard")
        return None
    supports_title = state.get("post_title_supported")
    supports_nick = state.get("post_nickname_supported")

    title2 = ensure_eum_style(title, max_lines=1).replace("\\n", " ")
    title2 = postprocess_outgoing_text(title2, mode="title", max_chars=50, max_lines=1).replace("\\n", " ")
    content2 = ensure_eum_style(content, max_lines=max(2, tuning.max_output_lines))
    content2 = postprocess_outgoing_text(content2, mode="post", max_chars=1000, max_lines=max(2, tuning.max_output_lines))

    def candidates() -> List[Tuple[Dict[str, Any], str]]:
        c: List[Tuple[Dict[str, Any], str]] = []
        if supports_title is not False:
            base = {"title": title2, "content": content2}
            if supports_nick is not False:
                c.append(({**base, "nickname": cfg.nickname}, "title+nick"))
            c.append((base, "title"))
        merged = f"{title2}\n{content2}"
        base2 = {"content": merged}
        if supports_nick is not False:
            c.append(({**base2, "nickname": cfg.nickname}, "merged+nick"))
        c.append((base2, "merged"))
        return c

    last_err: Optional[Exception] = None
    for payload, tag in candidates():
        try:
            res = _post_with_pow_401_retry(client, cfg, "/posts", payload, state=state)
            if "title" in tag:
                state["post_title_supported"] = True
            if "merged" in tag:
                state["post_title_supported"] = False
            if "nick" in tag:
                state["post_nickname_supported"] = True
            _hb_record_contribute(state, time.time(), kind="post")
            remember_action(state, action="post", target_id=_text_hash(f"{title}\n{content}"), endpoint_key="/posts")
            return res if isinstance(res, dict) else {"ok": True}
        except RateLimitError:
            raise
        except requests.HTTPError as e:
            last_err = e
            msg = str(e)
            code = 0
            try:
                code = int(msg.split()[0])
            except Exception as e:
                log_debug_exc("create_post:silent", e)
                pass
            if _is_validation_error_code(code):
                if "title" in tag:
                    state["post_title_supported"] = False
                if "nick" in tag:
                    state["post_nickname_supported"] = False
                continue
            raise
        except Exception as e:
            last_err = e
            break

    log_error("post", f"failed: {last_err}")
    return None

def create_comment(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    post_id: str,
    content: str,
    parent_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    if dup_action_should_skip(state, action=("reply" if parent_id else "comment"), target_id=str(parent_id or post_id), endpoint_key=f"/posts/{post_id}/comments"):
        protocol_set_reason(state, "comment", "comment:dup_action", "recent_action_guard")
        return None
    supports_nick = state.get("comment_nickname_supported")
    content2 = ensure_eum_style(content, max_lines=max(2, tuning.max_output_lines))
    _mode = "reply" if parent_id else "comment"
    content2 = postprocess_outgoing_text(content2, mode=_mode, max_chars=500, max_lines=max(2, tuning.max_output_lines))

    def candidates() -> List[Tuple[Dict[str, Any], str]]:
        base: Dict[str, Any] = {"content": content2}
        if parent_id:
            base["parent_id"] = parent_id
        c: List[Tuple[Dict[str, Any], str]] = []
        if supports_nick is not False:
            c.append(({**base, "nickname": cfg.nickname}, "nick"))
        c.append((base, "nonick"))
        return c

    last_err: Optional[Exception] = None
    for payload, tag in candidates():
        try:
            res = _post_with_pow_401_retry(client, cfg, f"/posts/{post_id}/comments", payload, state=state)
            if tag == "nick":
                state["comment_nickname_supported"] = True
            _hb_record_comment(state, time.time())
            remember_action(state, action=("reply" if parent_id else "comment"), target_id=str(parent_id or post_id), endpoint_key=f"/posts/{post_id}/comments")
            return res if isinstance(res, dict) else {"ok": True}
        except RateLimitError:
            raise
        except requests.HTTPError as e:
            last_err = e
            msg = str(e)
            code = 0
            try:
                code = int(msg.split()[0])
            except Exception as e:
                log_debug_exc("create_comment:silent", e)
                pass
            if "404" in msg:
                return None
            if _is_validation_error_code(code):
                if tag == "nick":
                    state["comment_nickname_supported"] = False
                continue
            raise
        except Exception as e:
            last_err = e
            break

    log_error("comment", f"failed: {last_err}")
    return None


def vote_post(
    client: HttpClient,
    cfg: Config,
    state: Dict[str, Any],
    post_id: str,
    vtype: str
) -> bool:
    """Vote on a post.

    v19.2:
      - Records votes into BOTH state.votes.posts (post_id -> ts) and legacy state.voted_posts (post_id -> {type, ts}).
      - Treats 'Already voted' 429 as non-fatal (records as voted to avoid infinite retries).
    """
    if state.get("vote_supported") is False:
        return False

    vtype2 = "up" if str(vtype) == "up" else "down"
    pid = str(post_id or "")
    if dup_action_should_skip(state, action=f"vote:{vtype2}", target_id=pid, endpoint_key=f"/posts/{pid}/vote"):
        protocol_set_reason(state, "vote", "vote:dup_action", "recent_action_guard")
        return False
    ts = time.time()

    try:
        _post_with_pow_401_retry(client, cfg, f"/posts/{pid}/vote", {"type": vtype2}, state=state)
        state["vote_supported"] = True
        _record_voted_post(cfg, state, pid, vtype2, ts)
        _hb_record_vote(state, ts)
        remember_action(state, action=f"vote:{vtype2}", target_id=pid, endpoint_key=f"/posts/{pid}/vote")
        return True

    except PowTimeoutError as e:
        protocol_set_reason(state, "vote", "vote:pow_timeout", one_line(str(e), 180))
        return False
    except RateLimitError as e:
        # Some 429s are "Already voted from this IP" (not a real rate limit for our purpose).
        msg = str(e).lower()
        if "already voted" in msg:
            state["vote_supported"] = True
            _record_voted_post(cfg, state, pid, vtype2, ts)
            _hb_record_vote(state, ts)
            return False
        raise

    except requests.HTTPError as e:
        msg = str(e)
        if "404" in msg:
            state["vote_supported"] = False
            return False
        state["vote_supported"] = True
        _record_voted_post(cfg, state, pid, vtype2, ts)
        _hb_record_vote(state, ts)
        return False

def arena_status(client: HttpClient) -> Optional[Dict[str, Any]]:
    try:
        data = client.get_json("/arena/status")
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def arena_posts(client: HttpClient, date: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """List arena posts for a given date (YYYY-MM-DD).
    Spec: GET /api/arena/posts?date=YYYY-MM-DD (server may also accept limit).
    Returns a list[dict]. Be tolerant to old response shapes.
    """
    try:
        params: Dict[str, Any] = {}
        if date:
            params["date"] = str(date)
        if limit:
            params["limit"] = int(limit)
        data = client.get_json("/arena/posts", params=(params if params else None))
        if isinstance(data, list):
            return [p for p in data if isinstance(p, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("posts"), list):
                posts = data.get("posts") or []
                return [p for p in posts if isinstance(p, dict)]
            if isinstance(data.get("items"), list):
                items = data.get("items") or []
                return [p for p in items if isinstance(p, dict)]
        return []
    except Exception as e:
        log_error("arena_posts", repr(e))
        return []

def arena_propose(client: HttpClient, cfg: Config, payload: Dict[str, Any], *, state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Arena propose (PROPOSE phase).

    v19.6: Let RateLimitError/HTTPError bubble up so caller can parse cooldown/phase.
    """
    if isinstance(state, dict):
        target_id = str(payload.get("post_id") or payload.get("id") or payload.get("topic") or "arena")
        if dup_action_should_skip(state, action="arena:propose", target_id=target_id, endpoint_key="/arena/propose"):
            protocol_set_reason(state, "arena", "arena:dup_action", "recent_action_guard")
            return None
    data = _post_with_pow_401_retry(client, cfg, "/arena/propose", payload, state=state)
    if isinstance(state, dict):
        target_id = str(payload.get("post_id") or payload.get("id") or payload.get("topic") or "arena")
        remember_action(state, action="arena:propose", target_id=target_id, endpoint_key="/arena/propose")
    return data if isinstance(data, dict) else {"ok": True}


def arena_fight(client: HttpClient, cfg: Config, payload: Dict[str, Any], *, state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Arena fight (BATTLE phase).

    v19.6: Let RateLimitError/HTTPError bubble up so caller can parse cooldown/phase.
    """
    if isinstance(state, dict):
        target_id = str(payload.get("battle_id") or payload.get("id") or payload.get("post_id") or "arena")
        if dup_action_should_skip(state, action="arena:fight", target_id=target_id, endpoint_key="/arena/fight"):
            protocol_set_reason(state, "arena", "arena:dup_action", "recent_action_guard")
            return None
    data = _post_with_pow_401_retry(client, cfg, "/arena/fight", payload, state=state)
    if isinstance(state, dict):
        target_id = str(payload.get("battle_id") or payload.get("id") or payload.get("post_id") or "arena")
        remember_action(state, action="arena:fight", target_id=target_id, endpoint_key="/arena/fight")
    return data if isinstance(data, dict) else {"ok": True}

################################################################################
# 11. COLOSSEUM ARENA FLOW
# - Dependencies: Section 1-2, 7-10 (Config/Logging + Schemas/Text/Policy/API)
# - Used by: Main loop (arena)
# - Key functions: do_arena_flow(), _arena_pick_keyword(), _arena_compose_propose(), _arena_compose_fight()
################################################################################

def _parse_iso_ts(s: str) -> float:
    """Parse ISO8601 timestamp to epoch seconds. Returns 0.0 on failure."""
    if not s:
        return 0.0
    try:
        ss = str(s).strip()
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return 0.0

def _arena_phase_guess_kst(dt: datetime) -> str:
    h = int(dt.hour)
    if 0 <= h < 9:
        return "PROPOSE"
    if 9 <= h < 12:
        return "VOTE"
    return "BATTLE"


def _arena_parse_wait_minutes(msg: str) -> Optional[int]:
    """Parse 'Wait N minutes' from server strings (Arena 3.0)."""
    try:
        s = (msg or "")
        if not s:
            return None
        low = s.lower()
        if "wait" not in low and "cooldown" not in low and "분" not in s:
            return None
        m = re.search(r"wait\s+(\d{1,4})\s*minutes", low)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d{1,4})\s*minutes", low)
        if m and ("cooldown" in low or "wait" in low):
            return int(m.group(1))
        m = re.search(r"(\d{1,4})\s*mins?\b", low)
        if m and ("cooldown" in low or "wait" in low):
            return int(m.group(1))
        m = re.search(r"(\d{1,4})\s*분", s)
        if m and ("대기" in s or "쿨" in s or "cooldown" in low or "wait" in low):
            return int(m.group(1))
        return None
    except Exception:
        return None


def _arena_next_boundary_ts_kst(now_dt: datetime, desired_phase: str) -> float:
    """Best-effort: compute next phase boundary wall-time in KST for gating/backoff."""
    try:
        ph = str(desired_phase or "").upper().strip()
        dt = now_dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        if ph == "PROPOSE":
            base = datetime(dt.year, dt.month, dt.day, 0, 0, 0, tzinfo=KST)
            if dt >= base:
                base = base + timedelta(days=1)
            return base.timestamp()
        if ph == "VOTE":
            base = datetime(dt.year, dt.month, dt.day, 9, 0, 0, tzinfo=KST)
            if dt >= base:
                base = base + timedelta(days=1)
            return base.timestamp()
        if ph == "BATTLE":
            base = datetime(dt.year, dt.month, dt.day, 12, 0, 0, tzinfo=KST)
            if dt >= base:
                base = base + timedelta(days=1)
            return base.timestamp()
        return 0.0
    except Exception:
        return 0.0


def _arena_set_next_ok(arena: Dict[str, Any], now_ts: float, wait_sec: float, reason: str, msg: str = "") -> None:
    """Set backoff window so we don't hammer arena endpoints on cooldown / phase mismatch."""
    try:
        w = max(5.0, float(wait_sec or 0.0))
        w = w * random.uniform(1.05, 1.25)  # jitter
        arena["next_ok_at"] = float(now_ts + w)
        arena["next_ok_reason"] = str(reason or "")
        arena["next_ok_set_ts"] = float(now_ts)
        if msg:
            arena["next_ok_msg"] = one_line(str(msg), 220)
    except Exception as e:
        log_debug_exc("_arena_set_next_ok:silent", e)
        pass


def _arena_is_blocked(arena: Dict[str, Any], now_ts: float) -> bool:
    try:
        t = float(arena.get("next_ok_at", 0.0) or 0.0)
        return bool(t > 0 and float(now_ts) < t)
    except Exception:
        return False


def _arena_api_ok(res: Any) -> bool:
    """Heuristic: treat response as success unless it clearly indicates an error."""
    if res is None:
        return False
    if not isinstance(res, dict):
        return True
    if res.get("success") is True:
        return True
    if res.get("ok") is True:
        return True
    if "dry_run" in res:
        return True
    if res.get("error"):
        return False
    if res.get("message") and ("error" in str(res.get("message")).lower()):
        return False
    if res.get("id"):
        return True
    return True
def _arena_reset_day(arena: Dict[str, Any], day: str) -> None:
    if str(arena.get("day") or "") == str(day or ""):
        return
    # v20.9 (A-4): stoploss exit event (daily reset)
    try:
        prev_day = str(arena.get("day") or "")
        prev_stop_day = str(arena.get("risk_stop_day") or "")
        prev_mode = bool(arena.get("risk_mode") is True)
        if prev_mode or (prev_stop_day and prev_stop_day == prev_day):
            log_event("arena_stoploss_exit", prev_day=prev_day, new_day=str(day or ""), prev_level=str(arena.get("risk_level") or ""), prev_score=float(arena.get("risk_score", 0.0) or 0.0), source_post_id=str(arena.get("risk_source_post_id") or ""))
    except Exception:
        pass
    arena["day"] = str(day or "")
    arena["today_proposed"] = False
    arena["today_propose_id"] = ""
    arena["today_topic_id"] = ""
    arena["today_topic_title"] = ""
    arena["my_posts"] = {}

    # (Unit 02) topic-level side pinning per day
    arena["topic_side_map"] = {}

    # (Unit 12) strategy/target caches
    arena["recent_target_post_ids"] = []
    arena["last_strategy"] = ""

    # blind/stoploss tracking (Unit 11)
    arena["risk_mode"] = False           # hard-stop for the day
    arena["risk_level"] = "OK"           # OK|CAUTION|DANGER|BLIND
    arena["risk_score"] = 0.0            # max(down-up) among my arena posts today
    arena["risk_style"] = ""             # ""|"conservative"
    arena["risk_last_update_ts"] = 0.0
    arena["risk_source_post_id"] = ""
    arena["risk_stop_day"] = ""          # day string when stoploss activated

    arena["actions_today"] = 0

    # daily reset: new battleground
    arena["last_action_ts"] = 0.0
    arena["last_post_id"] = ""
    arena["last_post_side"] = ""
    arena["last_post_created_at"] = ""

    # latest my-post stats (unit 10)
    arena["last_my_post_id"] = ""
    arena["last_my_post_up"] = 0
    arena["last_my_post_down"] = 0
    arena["last_my_post_ts"] = 0.0
    arena["last_my_post_side"] = ""
    arena["last_effective_cooldown_sec"] = 0.0
    arena["last_cooldown_upvotes"] = 0
    arena["last_cooldown_post_id"] = ""
def _arena_pick_keyword(brain: Dict[str, Any]) -> str:
    """Pick a safe keyword string for Arena topics.

    Note: brain['community']['hot'/'rising'] may contain dict items like:
      {"kw": "...", "score": 1.23}  /  {"kw": "...", "delta": 0.42}
    This function extracts a clean keyword and applies normalization + filters.
    """

    def _as_kw(x: Any) -> str:
        if isinstance(x, dict):
            # priority keys
            for key in ("kw", "keyword", "token", "text", "name", "title"):
                v = x.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # fallback: first usable string value
            for v in x.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""
        if isinstance(x, (list, tuple)) and x:
            return _as_kw(x[0])
        return str(x or "").strip()

    com = _safe_dict(brain.get("community"))
    candidates: List[Any] = []

    hot = com.get("hot")
    if isinstance(hot, list) and hot:
        candidates.extend(hot)

    rising = com.get("rising")
    if isinstance(rising, list) and rising:
        candidates.extend(rising)

    # older fallback: kw dict keys
    kwd = com.get("kw")
    if isinstance(kwd, dict) and kwd:
        candidates.extend(list(kwd.keys()))

    if candidates:
        random.shuffle(candidates)
        for raw in candidates:
            k = _as_kw(raw)
            if not k:
                continue
            k = re.sub(r"\s+", "", (k or "").strip())
            k = normalize_ko_token(k)
            if 2 <= len(k) <= 12 and is_clean_keyword(k):
                return k

    return random.choice(["커뮤니티", "기준"])

def _arena_compose_propose(brain: Dict[str, Any]) -> Tuple[str, str, str]:
    """Return (title, pros, cons) for /arena/propose."""
    kw = _arena_pick_keyword(brain)
    pool = [
        ("{kw}과 책임의 분리 가능성", "효용과 확장성 관점에서 분리 가능하다고 봄", "책임 주체 불명확해져 사회적 신뢰 붕괴 우려"),
        ("{kw}은(는) 공동체 규범을 강화하는가", "규범은 협력 비용을 낮추고 예측 가능성을 높임", "규범은 소수 의견을 억압하고 혁신을 둔화시킬 수 있음"),
        ("{kw}의 정당성은 어디서 오는가", "절차적 정당성(합의·투명성)이 핵심임", "결과적 정당성(성과·효용)이 더 중요할 수 있음"),
        ("공리주의 vs 의무론: {kw} 판단 기준", "총효용 극대화는 현실적 의사결정에 강점", "권리·의무는 효용보다 우선하는 경계선 제공"),
        ("{kw} 시대의 표현의 자유 한계", "허위·유해정보 억제는 공공선에 기여", "검열은 권력 남용과 자기검열을 촉진할 위험"),
    ]
    title_t, pros, cons = random.choice(pool)
    title = title_t.format(kw=kw)
    # keep short / readable
    title = sanitize_plain_text(title).replace("\n", " ").strip()
    title = re.sub(r"\s+", " ", title)[:80]
    pros = sanitize_plain_text(pros).replace("\n", " ").strip()[:180]
    cons = sanitize_plain_text(cons).replace("\n", " ").strip()[:180]
    return title, pros, cons

def _arena_enforce_len_eum(text: str, min_chars: int = 300, max_chars: int = 500) -> str:
    t = sanitize_plain_text(text).strip()
    t = ensure_eum_style(t, max_lines=6).replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)

    if len(t) < min_chars:
        pad = " 전제와 기준을 분리해 논증하면 감정이 아니라 구조로 판단 가능함. 결론은 한 문장으로 압축하는 게 설득에 유리함."
        t = (t + pad).strip()
        t = ensure_eum_style(t, max_lines=6).replace("\n", " ").strip()
        t = re.sub(r"\s+", " ", t)

    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
        t = ensure_eum_style(t, max_lines=6).replace("\n", " ").strip()
        t = re.sub(r"\s+", " ", t)
    # Hotfix2: final post-process (no ellipsis, no stray fragments)
    t = postprocess_outgoing_text(t, mode="arena", max_chars=max_chars, max_lines=6).replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def _arena_side_stats(posts: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {"PRO": {"n": 0.0, "score": 0.0}, "CON": {"n": 0.0, "score": 0.0}}
    for p in posts or []:
        if not isinstance(p, dict):
            continue
        side = str(p.get("side") or "").upper()
        if side not in ("PRO", "CON"):
            continue
        up = _safe_float(p.get("upvotes"), 0.0)
        down = _safe_float(p.get("downvotes"), 0.0)
        stats[side]["n"] += 1.0
        stats[side]["score"] += (up - down)
    return stats

def _arena_update_my_posts(arena: Dict[str, Any], posts: List[Dict[str, Any]], nick: str) -> None:
    my = arena.get("my_posts")
    if not isinstance(my, dict):
        my = {}
        arena["my_posts"] = my

    day = str(arena.get("day") or "")
    stop_day = str(arena.get("risk_stop_day") or "")
    stoploss_triggered = bool(stop_day) and (stop_day == day)

    prev_risk_mode = bool(arena.get("risk_mode") is True)
    prev_risk_level = str(arena.get("risk_level") or "")
    prev_stop_day = stop_day
    prev_active = prev_risk_mode and bool(prev_stop_day) and (prev_stop_day == day)

    worst_risk = 0.0
    worst_pid = ""
    best_ts = -1.0
    best_id = ""
    best_up = 0
    best_down = 0
    best_side = ""

    for p in posts or []:
        if not isinstance(p, dict):
            continue
        if str(p.get("nickname") or "") != str(nick or ""):
            continue
        pid = str(p.get("id") or "")
        if not pid:
            continue
        up = int(_safe_float(p.get("upvotes"), 0.0))
        down = int(_safe_float(p.get("downvotes"), 0.0))
        ts = _parse_iso_ts(str(p.get("created_at") or ""))
        side = str(p.get("side") or "")
        my[pid] = {"up": up, "down": down, "ts": ts, "side": side}

        risk = float(down - up)
        if risk > worst_risk:
            worst_risk = risk
            worst_pid = pid

        # Track latest of *my* posts today for cooldown-buff computation (Unit 10).
        if ts > best_ts:
            best_ts = float(ts)
            best_id = pid
            best_up = up
            best_down = down
            best_side = side

    if best_id:
        arena["last_my_post_id"] = best_id
        arena["last_my_post_up"] = int(best_up)
        arena["last_my_post_down"] = int(best_down)
        arena["last_my_post_ts"] = float(best_ts)
        arena["last_my_post_side"] = str(best_side or "")

    # Blind / stoploss policy (Unit 11)
    arena["risk_score"] = float(worst_risk)
    arena["risk_source_post_id"] = str(worst_pid or "")
    arena["risk_last_update_ts"] = float(time.time())

    # If stoploss already triggered today, keep hard-stop until daily reset.
    if stoploss_triggered:
        arena["risk_mode"] = True
        if str(arena.get("risk_level") or "").upper() not in ("DANGER", "BLIND"):
            arena["risk_level"] = "DANGER"
        arena["risk_style"] = "conservative"
        return

    # Progressive risk levels:
    # - CAUTION: risk >= 3 => conservative style (still allowed)
    # - DANGER:  risk >= 4 => stop fighting today (support-only in later unit)
    # - BLIND:   risk >= 5 => immediate hard-stop (likely already blind)
    if worst_risk >= 5.0:
        arena["risk_level"] = "BLIND"
        arena["risk_mode"] = True
        arena["risk_style"] = "conservative"
        arena["risk_stop_day"] = day
        # v20.9 (A-4): stoploss enter event
        if not prev_active:
            log_event("arena_stoploss_enter", day=str(day), risk_level=str(arena.get("risk_level") or ""), risk_score=float(worst_risk), post_id=str(worst_pid or ""))
    elif worst_risk >= 4.0:
        arena["risk_level"] = "DANGER"
        arena["risk_mode"] = True
        arena["risk_style"] = "conservative"
        arena["risk_stop_day"] = day
        # v20.9 (A-4): stoploss enter event
        if not prev_active:
            log_event("arena_stoploss_enter", day=str(day), risk_level=str(arena.get("risk_level") or ""), risk_score=float(worst_risk), post_id=str(worst_pid or ""))
    elif worst_risk >= 3.0:
        arena["risk_level"] = "CAUTION"
        arena["risk_mode"] = False
        arena["risk_style"] = "conservative"
    else:
        arena["risk_level"] = "OK"
        arena["risk_mode"] = False
        arena["risk_style"] = ""

def _arena_latest_my_post(arena: Dict[str, Any]) -> Tuple[str, int, int, float, str]:
    """Return (post_id, up, down, ts, side) for the latest arena post by us today."""
    pid = str(arena.get("last_my_post_id") or "")
    if pid:
        return (
            pid,
            int(arena.get("last_my_post_up", 0) or 0),
            int(arena.get("last_my_post_down", 0) or 0),
            float(arena.get("last_my_post_ts", 0.0) or 0.0),
            str(arena.get("last_my_post_side") or ""),
        )

    # Fallback: compute from my_posts map
    my = arena.get("my_posts")
    if isinstance(my, dict) and my:
        best = ("", 0, 0, -1.0, "")
        for k, v in my.items():
            if not isinstance(v, dict):
                continue
            ts = float(v.get("ts", 0.0) or 0.0)
            if ts > best[3]:
                best = (
                    str(k),
                    int(v.get("up", 0) or 0),
                    int(v.get("down", 0) or 0),
                    ts,
                    str(v.get("side") or ""),
                )
        if best[0]:
            return best
    return ("", 0, 0, 0.0, "")

def _arena_effective_cooldown_sec(arena: Dict[str, Any], base_cd_sec: float) -> float:
    """Compute effective arena cooldown using upvote buff: 1 upvote => -30 minutes, not below 0."""
    pid, up, _down, _ts, _side = _arena_latest_my_post(arena)
    reduction = int(up) * 30 * 60
    eff = max(0.0, float(base_cd_sec) - float(reduction))
    arena["last_effective_cooldown_sec"] = float(eff)
    arena["last_cooldown_upvotes"] = int(up)
    arena["last_cooldown_post_id"] = str(pid or "")
    return float(eff)

def _arena_post_score(p: Dict[str, Any]) -> float:
    up = _safe_float(p.get("upvotes"), 0.0)
    down = _safe_float(p.get("downvotes"), 0.0)
    return float(up - down)

def _arena_pick_post(
    posts: List[Dict[str, Any]],
    side: str,
    *,
    avoid_ids: Optional[set] = None,
    min_len: int = 30,
    prefer_high_score: bool = True,
) -> Optional[Dict[str, Any]]:
    side = str(side or "").upper().strip()
    avoid_ids = avoid_ids or set()
    cand: List[Tuple[float, Dict[str, Any]]] = []
    for p in posts or []:
        if not isinstance(p, dict):
            continue
        if str(p.get("side") or "").upper().strip() != side:
            continue
        pid = str(p.get("id") or "")
        if pid and pid in avoid_ids:
            continue
        c = sanitize_plain_text(str(p.get("content") or "")).strip()
        if len(c) < int(min_len):
            continue
        sc = _arena_post_score(p)
        cand.append((float(sc), p))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=bool(prefer_high_score))
    # pick among top-k to avoid determinism
    k = min(3, len(cand))
    top = cand[:k]
    return random.choice([p for _, p in top]) if top else cand[0][1]

def _arena_choose_side(arena: Dict[str, Any], posts: List[Dict[str, Any]]) -> str:
    """Choose a side with a mild novelty bias (Unit 12 improvement)."""
    last = str(arena.get("last_post_side") or "").upper()
    stats = _arena_side_stats(posts)
    pro = _safe_dict(stats.get("PRO"))
    con = _safe_dict(stats.get("CON"))
    n_pro, n_con = int(pro.get("n", 0) or 0), int(con.get("n", 0) or 0)
    # average score per post (avoid divide-by-zero)
    avg_pro = float(pro.get("score", 0.0) or 0.0) / float(max(1, n_pro))
    avg_con = float(con.get("score", 0.0) or 0.0) / float(max(1, n_con))

    # If one side is overwhelmingly crowded, pick the other side to add marginal value
    if n_pro >= n_con + 3:
        return "CON"
    if n_con >= n_pro + 3:
        return "PRO"

    # If one side is consistently getting better reception, prefer it slightly (human buff correlation)
    if abs(avg_pro - avg_con) >= 1.5 and (n_pro + n_con) >= 4:
        return "PRO" if avg_pro > avg_con else "CON"

    # otherwise keep continuity most of the time
    if last in ("PRO", "CON") and random.random() < 0.70:
        return last
    return "PRO" if random.random() < 0.5 else "CON"

def _arena_decide_strategy(
    arena: Dict[str, Any],
    posts: List[Dict[str, Any]],
    side: str,
    *,
    conservative: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Observe→Decide (support/refute) (Unit 12).

    Returns: (strategy, target_post_dict_or_None)
      - strategy: SUPPORT | REFUTE
      - target_post: a post to anchor the argument (opponent for REFUTE, ally for SUPPORT)
    """
    if conservative:
        return "SUPPORT", None

    side = str(side or "").upper().strip()
    opp_side = "CON" if side == "PRO" else "PRO"

    # recently targeted posts (avoid repeating the same fight)
    recent = arena.get("recent_target_post_ids")
    if not isinstance(recent, list):
        recent = []
        arena["recent_target_post_ids"] = recent
    avoid = set([str(x) for x in recent if x])

    stats = _arena_side_stats(posts)
    n_our = int(stats.get(side, {}).get("n", 0) or 0)
    n_opp = int(stats.get(opp_side, {}).get("n", 0) or 0)
    avg_our = float(stats.get(side, {}).get("score", 0.0) or 0.0) / float(max(1, n_our))
    avg_opp = float(stats.get(opp_side, {}).get("score", 0.0) or 0.0) / float(max(1, n_opp))

    our_best = _arena_pick_post(posts, side, avoid_ids=avoid, prefer_high_score=True)
    opp_best = _arena_pick_post(posts, opp_side, avoid_ids=avoid, prefer_high_score=True)

    our_best_score = _arena_post_score(our_best) if isinstance(our_best, dict) else -999.0
    opp_best_score = _arena_post_score(opp_best) if isinstance(opp_best, dict) else -999.0

    # Decide: refute when opponent has a strong or influential post; otherwise support our side (fill gaps).
    want_refute = False
    if isinstance(opp_best, dict):
        if opp_best_score >= 4.0:
            want_refute = True
        elif (avg_opp - avg_our) >= 1.5 and n_opp >= 2:
            want_refute = True
        elif (opp_best_score - our_best_score) >= 2.5 and n_opp >= 1:
            want_refute = True

    # If our side is underrepresented, prefer SUPPORT even if refute is tempting (reduce blind risk by aligning).
    if n_our + 1 < n_opp and random.random() < 0.65:
        want_refute = False

    if want_refute and isinstance(opp_best, dict):
        target = opp_best
        strategy = "REFUTE"
    else:
        target = our_best if isinstance(our_best, dict) else None
        strategy = "SUPPORT"

    # Update recent target cache
    if isinstance(target, dict):
        pid = str(target.get("id") or "")
        if pid:
            recent.append(pid)
            if len(recent) > 12:
                del recent[:-12]

    arena["last_strategy"] = strategy
    return strategy, target

def _arena_compose_fight(
    topic: Dict[str, Any],
    side: str,
    posts: List[Dict[str, Any]],
    *,
    strategy: str = "SUPPORT",
    target: Optional[Dict[str, Any]] = None,
    risk_level: str = "",
    meta_out: Optional[Dict[str, Any]] = None,
) -> str:
    """Compose an arena 'fight' text with grounding and anti-copy safeguards (Unit 10).

    Goals:
      - Avoid verbatim reuse of topic pros/cons or other posts (plausible "copy" suspicion)
      - Keep a stable 논증 골격 (정의→기준→검증) but with variation
      - Run a light QA gate + grounding overlap check
    """
    title = str(topic.get("title") or "").strip()
    pros = str(topic.get("pros") or "").strip()
    cons = str(topic.get("cons") or "").strip()

    rl = str(risk_level or "").upper().strip()
    conservative = rl in ("CAUTION", "DANGER", "BLIND")

    side = str(side or "").upper().strip()
    if side not in ("PRO", "CON"):
        side = "PRO" if random.random() < 0.5 else "CON"
    opp_side = "CON" if side == "PRO" else "PRO"

    strategy = str(strategy or "SUPPORT").upper().strip()
    if strategy not in ("SUPPORT", "REFUTE"):
        strategy = "SUPPORT"
    if conservative:
        strategy = "SUPPORT"

    # Host-provided seeds (do NOT copy; use only for keyword hints)
    seed = pros if side == "PRO" else cons
    opp_seed = cons if side == "PRO" else pros

    # -------------------------
    # Reference collection
    # -------------------------
    def _clean_ref(s: str, maxlen: int = 420) -> str:
        s = sanitize_plain_text(str(s or "")).strip()
        s = re.sub(r"\s+", " ", s)
        return s[:maxlen].strip()

    refs: List[str] = []

    def _add_ref(s: str) -> None:
        s2 = _clean_ref(s)
        if s2 and len(s2) >= 10:
            refs.append(s2)

    _add_ref(title)
    _add_ref(seed)
    _add_ref(opp_seed)

    # include top posts on both sides as references (avoid accidental echo)
    def _top_posts_text(side_: str, k: int) -> List[str]:
        out: List[Tuple[float, str]] = []
        for p in posts or []:
            if not isinstance(p, dict):
                continue
            if str(p.get("side") or "").upper().strip() != str(side_).upper().strip():
                continue
            c = _clean_ref(p.get("content") or "", maxlen=420)
            if len(c) < 25:
                continue
            sc = _arena_post_score(p)
            out.append((float(sc), c))
        out.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in out[: max(0, int(k))]]

    k_ref = int(ARENA_REF_TOPK)
    if k_ref > 0:
        for c in _top_posts_text("PRO", k_ref):
            _add_ref(c)
        for c in _top_posts_text("CON", k_ref):
            _add_ref(c)

    # include target (full) as ref to avoid quoting
    target_full = ""
    target_side = ""
    target_id = ""
    if isinstance(target, dict):
        target_full = _clean_ref(target.get("content") or "", maxlen=520)
        target_side = str(target.get("side") or "").upper().strip()
        target_id = str(target.get("id") or "")
        _add_ref(target_full)

    # -------------------------
    # Anti-copy helpers
    # -------------------------
    def _ngram_set(s: str, n: int = 3) -> set:
        s = re.sub(r"\s+", "", sanitize_plain_text(s or ""))
        if len(s) < n:
            return set()
        return {s[i:i+n] for i in range(0, len(s) - n + 1)}

    def _jaccard(a: str, b: str, n: int = 3) -> float:
        A = _ngram_set(a, n=n)
        B = _ngram_set(b, n=n)
        if not A or not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B)
        return inter / union if union else 0.0

    def _has_long_substring(gen: str, ref: str, L: int) -> bool:
        g = re.sub(r"\s+", "", sanitize_plain_text(gen or ""))
        r = re.sub(r"\s+", "", sanitize_plain_text(ref or ""))
        if len(g) < L or len(r) < L:
            return False
        # scan ref windows (L) – small L, refs are short -> ok
        for i in range(0, len(r) - L + 1):
            if r[i:i+L] in g:
                return True
        return False

    def _anticopy_report(gen: str) -> Dict[str, Any]:
        thr_j = float(ARENA_ANTICOPY_JACCARD)
        L = int(ARENA_ANTICOPY_SUBSTR_LEN)
        thr_h = int(ARENA_ANTICOPY_SIMHASH_MAX)

        gen_s = _clean_ref(gen, 800)
        max_j = 0.0
        hit_sub = False
        min_h = 64
        worst_ref = ""

        gtoks = tokenize(gen_s, max_tokens=180)
        gsh = int(simhash64(gtoks[:160])) if gtoks else 0

        for r in refs:
            rr = _clean_ref(r, 520)
            if not rr:
                continue
            if _has_long_substring(gen_s, rr, L=L):
                hit_sub = True
                worst_ref = worst_ref or rr[:80]
            j = _jaccard(gen_s, rr, n=3)
            if j > max_j:
                max_j = float(j)
                worst_ref = rr[:80]
            # simhash proximity
            rtoks = tokenize(rr, max_tokens=180)
            rsh = int(simhash64(rtoks[:160])) if rtoks else 0
            if gsh and rsh:
                ham = int(hamming64(gsh, rsh))
                if ham < min_h:
                    min_h = ham

        fail_j = bool(max_j >= thr_j) if thr_j > 0 else False
        fail_h = bool(min_h <= thr_h) if thr_h >= 0 else False
        return {
            "max_jaccard": round(float(max_j), 4),
            "hit_substring": bool(hit_sub),
            "min_hamming": int(min_h if min_h != 64 else 64),
            "fail_jaccard": bool(fail_j),
            "fail_hamming": bool(fail_h),
            "worst_ref_hint": worst_ref,
        }

    # -------------------------
    # Keyword hints (from seed only)
    # -------------------------
    def _pick_keywords(text: str) -> List[str]:
        kws = top_keywords(text or "", k=8)
        out: List[str] = []
        for w in kws:
            w = normalize_ko_token(w)
            if not is_clean_keyword(w):
                continue
            if not (2 <= len(w) <= 12):
                continue
            if w not in out:
                out.append(w)
            if len(out) >= 5:
                break
        return out

    kws = _pick_keywords(f"{title} {seed}") or _pick_keywords(f"{title} {opp_seed}") or ["기준", "예외", "정의"]
    kw1 = kws[0] if len(kws) >= 1 else "기준"
    kw2 = kws[1] if len(kws) >= 2 else "예외"
    kw3 = kws[2] if len(kws) >= 3 else kw1

    def _anchor_hint() -> str:
        if not target_full:
            return ""
        ak = _pick_keywords(target_full)
        a1 = ak[0] if len(ak) >= 1 else kw1
        a2 = ak[1] if len(ak) >= 2 else kw2
        # avoid looking like a quote
        tag = "상대" if (strategy == "REFUTE" and (target_side == opp_side or not target_side)) else "우리"
        return f"{tag}는 {a1}/{a2} 쪽을 중심으로 말하는 흐름으로 보임."

    # Optionally allow verbatim small snippet (default off)
    verb_anchor = ""
    if bool(ARENA_USE_ANCHOR_VERBATIM) and isinstance(target, dict):
        verb_anchor = one_line(_clean_ref(target.get("content") or "", 220), 90)

    # -------------------------
    # Argument skeleton pools
    # -------------------------
    OPENER_POOL = [
        "정의-기준-검증 순서로만 정리하겠음.",
        "정의부터 고정하고 기준→검증으로 가겠음.",
        "감정 빼고 정의/기준/검증으로만 보겠음.",
    ]

    PRO_CRITERIA = [
        "기준은 '예방/격리' 축을 우선으로 잡겠음.",
        "핵심은 비용이 아니라, 피해 재발 가능성을 얼마나 낮추느냐임.",
        f"{kw1}이(가) 불명확하면 책임이 분산돼서 제도가 흔들림.",
        f"{kw2} 처리가 애매하면 여론은 감정으로 튐.",
        "절차가 아니라 결과가 중요하다고 할 때, 결과 측정 지표를 먼저 합의해야 함.",
    ]
    CON_CRITERIA = [
        "기준은 '오판/되돌림 불가' 축을 우선으로 잡겠음.",
        "핵심은 효율이 아니라, 한 번의 오류가 영구히 남는 구조임.",
        f"{kw1}을(를) 국가가 독점할수록 통제 장치가 더 중요해짐.",
        f"{kw2}를(을) 예외로 넘기기 시작하면 남용 구멍이 생김.",
        "절차가 무너지면 결과의 정당성도 같이 무너지는 구조임.",
    ]

    DEFINE_POOL = [
        f"여기서 {kw1}은(는) '판단을 고정하는 규칙' 쪽으로 정의하겠음.",
        f"{kw2}는(은) 본문에 섞지 말고 조건으로 따로 빼는 게 깔끔함.",
        "정의가 넓어지면 토론은 결국 감정 싸움으로 흘러가기 쉬움.",
    ]

    CHECK_POOL = [
        "검증은 케이스를 2~3개로 쪼개서 일관성만 보면 됨.",
        "검증 포인트는 '반례가 들어와도 기준이 유지되냐'임.",
        "검증에서 애매하면 원칙/예외를 다시 분리해야 함.",
    ]

    SUPPORT_ADD = [
        f"조건 하나: {kw1}을(를) 결과로 볼지 의도로 볼지 먼저 고정해야 함.",
        f"반례 대비: {kw2} 케이스는 원칙/예외로 분리해두는 게 안전함.",
        f"질문: {kw2}의 기준은 '피해 규모'냐 '의도/동기'냐 뭐가 우선임?",
    ]
    REFUTE_ADD = [
        f"찔러볼 지점: {kw1} 정의가 넓으면 어떤 결론도 끼워 맞출 수 있음.",
        f"반례: {kw2} 케이스 하나만 들어와도 전체 논증이 무너질 수 있음.",
        f"질문: 지금 논리에서 {kw3}는 기준임, 감정임? 어디에 놓고 말하는 거임?",
    ]

    CONCEDE_POOL = [
        f"상대 우려 중 {kw2} 쪽은 타당한 지점이 있음.",
        "상대가 말하는 위험은 인정하되, 그걸 제도 전체 결론으로 바로 점프하면 과함.",
        "상대 포인트가 틀렸다기보다, 정의/기준을 섞어서 결론이 과대해지는 느낌임.",
    ]

    CLOSE_POOL = [
        "결론: 기준 고정 + 예외 분리부터 해야 함.",
        "결론은 짧게 감. 기준 고정이 먼저임.",
        "결론: 정의를 좁히고, 예외는 조건으로 격리하는 게 핵심임.",
    ]

    def _completeness_bonus(t: str) -> int:
        bonus = 0
        if "정의" in t:
            bonus += 3
        if "기준" in t:
            bonus += 3
        if ("검증" in t) or ("반례" in t):
            bonus += 2
        if ("예외" in t) or ("조건" in t):
            bonus += 2
        if "?" in t[-60:]:
            bonus += 2
        return int(bonus)

    def _build_variant(attempt: int) -> str:
        rnd = random.Random(_sha1_u64(f"{title}|{side}|{strategy}|{attempt}|{time.time()}"))
        frame: List[str] = []
        frame.append(rnd.choice(OPENER_POOL))
        frame.append(f"주제는 {one_line(title, 64)}임.")
        side_ko = "찬성" if side == "PRO" else "반대"
        frame.append(f"내 입장은 {side_ko} 쪽으로 기울었음.")

        # core: define + criteria + check
        frame.append(rnd.choice(DEFINE_POOL))
        crit_pool = PRO_CRITERIA if side == "PRO" else CON_CRITERIA
        frame.extend(rnd.sample(crit_pool, k=2))
        frame.append(rnd.choice(CHECK_POOL))

        hint = _anchor_hint()
        if strategy == "REFUTE":
            if hint:
                frame.append(hint)
            if verb_anchor:
                frame.append(f"(문장 그대로 인용은 아님) {verb_anchor} 정도로 읽히는 글이 있었음.")
            frame.append(rnd.choice(CONCEDE_POOL))
            frame.append("근데 결론을 내리기 전에, 정의를 좁히고 기준을 고정해야 함.")
            frame.append("정의/기준이 섞이면 같은 말로 다른 결론도 가능해짐.")
            frame.append(rnd.choice(REFUTE_ADD))
        else:
            if hint:
                frame.append(hint.replace("상대", "우리"))
            frame.append("정의→기준→검증을 분리하면 논증이 덜 흔들림.")
            frame.append("상대 반례는 예외 조건으로 격리하면 본문 일관성이 유지됨.")
            frame.append(rnd.choice(SUPPORT_ADD))

        if conservative:
            frame.append("리스크 모드라 과격하게 안 감. 구조만 고정하겠음.")
        frame.append(rnd.choice(CLOSE_POOL))

        t = " ".join([x for x in frame if x])
        t = re.sub(r"(ㅋ{2,}|ㅎ{2,})", "", t)
        t = re.sub(r"[!]{2,}", "!", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # -------------------------
    # Search best candidate
    # -------------------------
    focus = {
        "post_title": title,
        "post_excerpt": _clean_ref(f"{seed} {opp_seed}", 420),
        "mode": "comment",
    }

    best_txt = ""
    best_score = -10_000.0
    best_meta: Dict[str, Any] = {}
    tried = 0
    passed = 0

    n_try = max(3, int(ARENA_ARG_VARIANTS))
    for attempt in range(n_try):
        tried += 1
        raw = _build_variant(attempt)
        cand = _arena_enforce_len_eum(raw, min_chars=300, max_chars=500)

        ac = _anticopy_report(cand)
        qa = qa_evaluate_text(cand, kind="comment", focus=focus, mode="comment")
        qa_score = int(qa.get("score", 0) or 0)
        hard = bool(qa.get("hard_fail"))
        bonus = _completeness_bonus(cand)

        penalty = 0.0
        if bool(ac.get("hit_substring")):
            penalty += 70.0
        if bool(ac.get("fail_jaccard")):
            # scale with overflow
            try:
                overflow = max(0.0, float(ac.get("max_jaccard", 0.0)) - float(ARENA_ANTICOPY_JACCARD))
            except Exception:
                overflow = 0.0
            penalty += 45.0 + (overflow * 120.0)
        if bool(ac.get("fail_hamming")):
            penalty += 30.0

        total = float(qa_score) + float(bonus) - float(penalty)

        ok = (not hard) and qa_score >= int(ARENA_QUALITY_MIN) and (not ac.get("hit_substring")) and (not ac.get("fail_jaccard")) and (not ac.get("fail_hamming"))
        if ok:
            passed += 1

        # choose best even if not ok (fallback), but strongly prefer ok
        rank = total + (40.0 if ok else 0.0)
        if rank > best_score:
            best_score = float(rank)
            best_txt = cand
            best_meta = {
                "qa_score": qa_score,
                "qa_issues": qa.get("issues", []),
                "qa_len": int(qa.get("len", 0) or 0),
                "anti_copy": ac,
                "bonus": int(bonus),
                "penalty": round(float(penalty), 2),
                "total": round(float(total), 2),
                "ok": bool(ok),
                "tried": int(tried),
                "passed": int(passed),
            }

        # early exit if very good
        if ok and qa_score >= 85 and float(ac.get("max_jaccard", 0.0)) <= 0.30:
            break

    if not best_txt:
        side_ko = "찬성" if side == "PRO" else "반대"
        best_txt = f"정의-기준-검증 순서로 말하겠음. 주제는 {one_line(title, 64)}임. 내 입장은 {side_ko} 쪽임. 기준은 하나로 고정하고 예외는 조건으로 분리하는 게 먼저임. 질문 하나: 여기서 기준은 결과냐 의도냐?"
        best_txt = _arena_enforce_len_eum(best_txt, min_chars=300, max_chars=500)
        best_meta = {"ok": False, "tried": tried, "passed": passed}

    # Export meta if requested
    if isinstance(meta_out, dict):
        try:
            meta_out.update(
                {
                    "arena_side": side,
                    "arena_strategy": strategy,
                    "arena_conservative": bool(conservative),
                    "arena_target_post_id": str(target_id or ""),
                    "arena_target_side": str(target_side or ""),
                    "arena_refs_count": int(len(refs)),
                    "arena_variants_tried": int(tried),
                    "arena_variants_passed": int(passed),
                    "arena_compose_ok": bool(best_meta.get("ok", False)),
                    "arena_qa_score": int(best_meta.get("qa_score", 0) or 0),
                    "arena_qa_issues": best_meta.get("qa_issues", []),
                    "arena_anticopy": best_meta.get("anti_copy", {}),
                    "arena_score_total": float(best_meta.get("total", 0.0) or 0.0),
                }
            )
        except Exception as e:
            log_debug_exc("_arena_compose_fight:silent", e)
            pass

    return best_txt

def do_arena_flow(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    memory: List[Dict[str, Any]],
    brain: Dict[str, Any],
) -> int:
    """Integrate Colosseum actions into main loop (Unit 09).

    - Checks /arena/status occasionally (cached)
    - Phase 1 (00~09 KST): propose once per day
    - Phase 3 (12~24 KST): fight respecting basic cooldown (advanced cooldown & blind handling in later units)
    """
    if not ARENA_ENABLE:
        protocol_set_reason(state, "arena", "arena:phase_block", "disabled")
        return 0

    arena = state.get("arena")
    if not isinstance(arena, dict):
        arena = {}
        state["arena"] = arena

    now_dt = now_kst()
    now_ts = time.time()

    # status (cached)
    status: Optional[Dict[str, Any]] = None
    last_status_ts = float(arena.get("last_status_ts", 0.0) or 0.0)
    cached = arena.get("status_cache")
    if isinstance(cached, dict) and (now_ts - last_status_ts) < float(ARENA_STATUS_MIN_INTERVAL_SEC):
        status = cached
    else:
        try:
            status = arena_status(client)
            if isinstance(status, dict):
                arena["status_cache"] = status
                arena["last_status_ts"] = now_ts
        except Exception as e:
            log_error("arena_status", one_line(repr(e), 180))
            status = cached if isinstance(cached, dict) else None

    if not isinstance(status, dict):
        protocol_set_reason(state, "arena", "arena:phase_block", "no_status")
        return 0

    day = str(status.get("date") or _today_kst())
    _arena_reset_day(arena, day)

    phase = str(status.get("phase") or "").upper().strip()
    if not phase:
        phase = _arena_phase_guess_kst(now_dt)
    arena["last_phase"] = phase
    arena["last_status_date"] = day

    topic = status.get("topic")
    if isinstance(topic, dict):
        arena["today_topic_id"] = str(topic.get("id") or "")
        arena["today_topic_title"] = str(topic.get("title") or "")
        arena["last_status_topic_id"] = str(topic.get("id") or "")
        arena["last_status_topic_title"] = str(topic.get("title") or "")

    # daily action cap (safety)
    actions_today = int(arena.get("actions_today", 0) or 0)
    if int(ARENA_MAX_ACTIONS_PER_DAY) > 0 and actions_today >= int(ARENA_MAX_ACTIONS_PER_DAY):
        protocol_set_reason(state, "arena", "arena:phase_block", "max_actions_per_day")
        return 0
    # v19.6: honor server-provided cooldowns/backoff for arena endpoints
    arena.setdefault("next_ok_at", 0.0)
    if _arena_is_blocked(arena, now_ts):
        try:
            nxt = float(arena.get("next_ok_at", 0.0) or 0.0)
            remain = max(0, int(nxt - now_ts))
            protocol_set_reason(state, "arena", "arena:cooldown", f"next_ok_in={remain}s")
        except Exception:
            protocol_set_reason(state, "arena", "arena:cooldown", "blocked")
        return 0


    # ----------------------
    # Phase 1: PROPOSE
    # ----------------------
    if phase == "PROPOSE" and (0 <= int(now_dt.hour) < 9):
        if bool(arena.get("today_proposed")):
            protocol_set_reason(state, "arena", "arena:phase_block", "already_proposed")
            return 0

        title, pros, cons = _arena_compose_propose(brain)
        # Hotfix2: sanitize propose payload (no ellipsis / no stray fragments)
        title = postprocess_outgoing_text(title, mode="title", max_chars=100, max_lines=1).replace("\\n", " ")
        pros = postprocess_outgoing_text(pros, mode="comment", max_chars=500, max_lines=2).replace("\\n", " ")
        cons = postprocess_outgoing_text(cons, mode="comment", max_chars=500, max_lines=2).replace("\\n", " ")
        try:
            payload = {"title": title, "pros": pros, "cons": cons}
            if cfg.nickname:
                payload["nickname"] = cfg.nickname

            _bump_action_counter(state, "action_attempt", "arena_propose")
            res = arena_propose(client, cfg, payload, state=state)
            if not _arena_api_ok(res):
                emsg = ""
                if isinstance(res, dict):
                    emsg = str(res.get("error") or res.get("message") or "")
                log_warn("ARENA propose rejected -> " + one_line(emsg or repr(res), 180))
                arena["today_proposed_attempted"] = True
                protocol_set_reason(state, "arena", "arena:phase_block", "propose_rejected")
                _bump_action_counter(state, "action_fail", "arena_propose")
                return 0

            pid = str(_safe_dict(res).get("id") or "")
            arena["today_proposed"] = True
            arena["today_propose_id"] = pid
            arena["today_proposed_attempted"] = True
            arena["last_action_ts"] = now_ts
            state["arena_last_action_ts"] = now_ts
            state["arena_last_propose_date"] = day
            arena["actions_today"] = actions_today + 1

            # clear arena backoff on success
            arena.pop("next_ok_at", None)
            arena.pop("next_ok_reason", None)
            arena.pop("next_ok_msg", None)

            state["total_actions"] = int(state.get("total_actions", 0) or 0) + 1
            _hb_record_contribute(state, now_ts, kind="arena")
            _bump_action_counter(state, "action_success", "arena_propose")

            record_memory(
                memory,
                {
                    "ts": now_ts,
                    "action": "arena",
                    "action_type": "arena_propose",
                    "text": f"{title} | PROS:{pros} | CONS:{cons}",
                    "post_id": pid,
                    "topic_id": "",
                    "topic_title": "",
                    "phase": phase,
                    "evaluated": True,
                    "eval_due_ts": 0.0,
                },
                tuning,
                cfg.paths.memory_archive_jsonl,
            )
            log_action("ARENA", f"propose ok id={pid or '?'} title={one_line(title, 60)}")
            write_journal(cfg.paths.journal, f"arena propose | id={pid or '?'} | title={one_line(title, 80)}")
            return 1

        except RateLimitError as e_rl:
            msg = str(e_rl)
            wait_sec = float(getattr(e_rl, "retry_after_sec", 0) or 0.0)
            mins = _arena_parse_wait_minutes(msg)
            if mins is not None:
                wait_sec = max(wait_sec, float(mins) * 60.0)
            if wait_sec <= 0:
                wait_sec = 45.0 * 60.0
            _arena_set_next_ok(arena, now_ts, wait_sec, "rate_limit", msg)
            log_warn(f"ARENA propose cooldown -> wait~{int(wait_sec)}s")
            arena["today_proposed_attempted"] = True
            protocol_set_reason(state, "arena", "arena:cooldown", f"propose_wait~{int(wait_sec)}s")
            _bump_action_counter(state, "action_fail", "arena_propose")
            return 0

        except PowTimeoutError as e:
            log_error("arena_propose", one_line(str(e), 200))
            arena["today_proposed_attempted"] = True
            protocol_set_reason(state, "arena", "arena:pow_timeout", one_line(str(e), 160))
            _bump_action_counter(state, "action_fail", "arena_propose")
            return 0

        except requests.HTTPError as e_http:
            msg = str(e_http)
            wait_sec = 0.0
            mins = _arena_parse_wait_minutes(msg)
            if mins is not None:
                wait_sec = max(wait_sec, float(mins) * 60.0)

            low = msg.lower()
            if "proposal is allowed only during propose" in low or "current phase is" in low:
                nxt = _arena_next_boundary_ts_kst(now_dt, "PROPOSE")
                if nxt > now_ts:
                    wait_sec = max(wait_sec, float(nxt - now_ts))

            if wait_sec > 0:
                _arena_set_next_ok(arena, now_ts, wait_sec, "phase_or_cooldown", msg)

            log_error("arena_propose", one_line(msg, 220))
            arena["today_proposed_attempted"] = True
            protocol_set_reason(state, "arena", "arena:phase_block", one_line(f"propose_http:{msg}", 160))
            _bump_action_counter(state, "action_fail", "arena_propose")
            return 0

        except Exception as e:
            log_error("arena_propose", one_line(repr(e), 200))
            arena["today_proposed_attempted"] = True
            protocol_set_reason(state, "arena", "arena:phase_block", one_line(f"propose_exc:{repr(e)}", 160))
            _bump_action_counter(state, "action_fail", "arena_propose")
            return 0

    # ----------------------
    # Phase 3: BATTLE
    # ----------------------
    if phase == "BATTLE" and (12 <= int(now_dt.hour) <= 23):
        # stoploss applies per-day; keep hard-stop until daily reset
        if arena.get("risk_mode") is True and str(arena.get("risk_stop_day") or "") == str(day or ""):
            protocol_set_reason(state, "arena", "arena:blinded_stop", "risk_stop_day")
            return 0
        if not isinstance(topic, dict) or not str(topic.get("id") or ""):
            protocol_set_reason(state, "arena", "arena:phase_block", "no_topic")
            return 0

        # observe posts occasionally
        posts: List[Dict[str, Any]] = []
        last_posts_ts = float(arena.get("last_posts_ts", 0.0) or 0.0)
        cached_posts = arena.get("posts_cache")
        if isinstance(cached_posts, list) and (now_ts - last_posts_ts) < float(ARENA_POSTS_MIN_INTERVAL_SEC):
            posts = [p for p in cached_posts if isinstance(p, dict)]
        else:
            try:
                posts = arena_posts(client, limit=int(getattr(tuning, "arena_fetch_limit", 40)))
                arena["posts_cache"] = posts[-80:] if isinstance(posts, list) else []
                arena["last_posts_ts"] = now_ts
            except Exception as e:
                log_error("arena_posts", one_line(repr(e), 180))
                posts = [p for p in (cached_posts or []) if isinstance(p, dict)] if isinstance(cached_posts, list) else []

        if posts:
            _arena_update_my_posts(arena, posts, cfg.nickname)
            # (v18.3) Avoid monologue: only fight when there is a new non-self post after my latest post.
            try:
                last_my_ts = float(arena.get("last_my_post_ts", 0.0) or 0.0)
                opp_latest_ts = -1.0
                for pp in posts or []:
                    if not isinstance(pp, dict):
                        continue
                    if str(pp.get("nickname") or "") == str(cfg.nickname or ""):
                        continue
                    ts2 = _parse_iso_ts(str(pp.get("created_at") or ""))
                    if ts2 > 0 and ts2 > opp_latest_ts:
                        opp_latest_ts = float(ts2)
                if last_my_ts > 0 and opp_latest_ts >= 0 and opp_latest_ts <= (last_my_ts + 1.0):
                    protocol_set_reason(state, "arena", "arena:phase_block", "opponent_timing_guard")
                    return 0
            except Exception as e:
                log_debug_exc("do_arena_flow:silent", e)
                pass

            rl = str(arena.get("risk_level") or "").upper().strip()
            rs = float(arena.get("risk_score", 0.0) or 0.0)
            if arena.get("risk_mode") is True and str(arena.get("risk_stop_day") or "") == str(day or ""):
                log_event("arena_stoploss_active", day=str(day), risk_level=rl, risk_score=float(rs), post_id=str(arena.get("risk_source_post_id") or ""))
                log_warn("ARENA stoploss on (close to blind) -> stop fighting today")
                log_warn(f"ARENA risk_level={rl or '?'} risk_score={rs:.1f}")
                protocol_set_reason(state, "arena", "arena:blinded_stop", f"stoploss:{rl or '?'}:{rs:.1f}")
                return 0
            if rl == "CAUTION":
                # still allowed, but keep the tone conservative and avoid overfighting
                pass

        # cooldown with upvote-buff (Unit 10): base 2h minus 30m per upvote on the latest of my arena posts.
        last_action = float(arena.get("last_action_ts", 0.0) or 0.0)
        base_cd = float(getattr(tuning, "arena_cooldown_sec", 2 * 60 * 60))
        eff_cd = _arena_effective_cooldown_sec(arena, base_cd)
        if last_action > 0 and (now_ts - last_action) < eff_cd:
            protocol_set_reason(state, "arena", "arena:cooldown", f"cooldown_rem={int(max(0, eff_cd - (now_ts - last_action)))}s")
            return 0
        # (Unit 02) Pin PRO/CON side per topic_id for the day to keep character consistency.
        topic_id = str(arena.get("today_topic_id") or "")
        topic_side_map = arena.setdefault("topic_side_map", {})
        side = ""
        if topic_id and isinstance(topic_side_map, dict) and topic_id in topic_side_map:
            side = str(topic_side_map.get(topic_id) or "").upper()
        if side not in ("PRO", "CON"):
            side = _arena_choose_side(arena, posts)
            side = str(side or "").upper()
            if topic_id and isinstance(topic_side_map, dict) and side in ("PRO", "CON"):
                topic_side_map[topic_id] = side
        rl_s = str(arena.get("risk_level") or "")
        conservative = str(rl_s).upper().strip() in ("CAUTION", "DANGER", "BLIND")
        strategy, target = _arena_decide_strategy(arena, posts, side, conservative=conservative)
        compose_meta: Dict[str, Any] = {}
        content = _arena_compose_fight(topic, side, posts, strategy=strategy, target=target, risk_level=rl_s, meta_out=compose_meta)

        try:
            payload = {"topic_id": topic_id, "side": side, "content": fight_text}
            if cfg.nickname:
                payload["nickname"] = cfg.nickname

            _bump_action_counter(state, "action_attempt", "arena_fight")
            res = arena_fight(client, cfg, payload, state=state)
            if not _arena_api_ok(res):
                emsg = ""
                if isinstance(res, dict):
                    emsg = str(res.get("error") or res.get("message") or "")
                log_warn("ARENA fight rejected -> " + one_line(emsg or repr(res), 180))
                arena["today_fought_attempted"] = True
                protocol_set_reason(state, "arena", "arena:phase_block", "fight_rejected")
                _bump_action_counter(state, "action_fail", "arena_fight")
                return 0

            pid = str(_safe_dict(res).get("id") or "")
            arena["today_fought"] = True
            arena["today_fight_id"] = pid
            arena["today_fought_attempted"] = True
            arena["last_action_ts"] = now_ts
            state["arena_last_action_ts"] = now_ts
            state["arena_last_fight_date"] = day
            arena["actions_today"] = actions_today + 1

            # clear arena backoff on success
            arena.pop("next_ok_at", None)
            arena.pop("next_ok_reason", None)
            arena.pop("next_ok_msg", None)

            state["total_actions"] = int(state.get("total_actions", 0) or 0) + 1
            _hb_record_contribute(state, now_ts, kind="arena")
            _bump_action_counter(state, "action_success", "arena_fight")

            record_memory(
                memory,
                {
                    "ts": now_ts,
                    "action": "arena",
                    "action_type": "arena_fight",
                    "text": f"side={side} | {fight_text}",
                    "post_id": pid,
                    "topic_id": topic_id,
                    "topic_title": str(topic.get("title") or ""),
                    "phase": phase,
                    "evaluated": True,
                    "eval_due_ts": 0.0,
                    "meta": {
                        "arena_qa_score": int(compose_meta.get("arena_qa_score", 0) or 0),
                        "arena_qa_issues": compose_meta.get("arena_qa_issues", []),
                        "arena_anticopy": compose_meta.get("arena_anticopy", {}),
                        "arena_compose_ok": bool(compose_meta.get("arena_compose_ok", False)),
                        "arena_score_total": float(compose_meta.get("arena_score_total", 0.0) or 0.0),
                    },
                },
                tuning,
                cfg.paths.memory_archive_jsonl,
            )
            log_action(
                "ARENA",
                f"fight ok side={side} id={pid or '?'} qa={int(compose_meta.get('arena_qa_score', 0) or 0)} topic={one_line(str(topic.get('title') or ''), 60)}",
            )
            write_journal(
                cfg.paths.journal,
                f"arena fight | side={side} | id={pid or '?'} | topic={one_line(str(topic.get('title') or ''), 80)}",
            )
            return 1

        except RateLimitError as e_rl:
            msg = str(e_rl)
            wait_sec = float(getattr(e_rl, "retry_after_sec", 0) or 0.0)
            mins = _arena_parse_wait_minutes(msg)
            if mins is not None:
                wait_sec = max(wait_sec, float(mins) * 60.0)
            if wait_sec <= 0:
                wait_sec = 110.0 * 60.0
            _arena_set_next_ok(arena, now_ts, wait_sec, "cooldown", msg)
            log_warn(f"ARENA fight cooldown -> wait~{int(wait_sec)}s")
            arena["today_fought_attempted"] = True
            protocol_set_reason(state, "arena", "arena:cooldown", f"fight_wait~{int(wait_sec)}s")
            _bump_action_counter(state, "action_fail", "arena_fight")
            return 0

        except PowTimeoutError as e:
            log_error("arena_fight", one_line(str(e), 200))
            arena["today_fought_attempted"] = True
            protocol_set_reason(state, "arena", "arena:pow_timeout", one_line(str(e), 160))
            _bump_action_counter(state, "action_fail", "arena_fight")
            return 0

        except requests.HTTPError as e_http:
            msg = str(e_http)
            wait_sec = 0.0
            mins = _arena_parse_wait_minutes(msg)
            if mins is not None:
                wait_sec = max(wait_sec, float(mins) * 60.0)

            low = msg.lower()
            if "fighting is allowed only during battle" in low or "current phase is" in low:
                nxt = _arena_next_boundary_ts_kst(now_dt, "BATTLE")
                if nxt > now_ts:
                    wait_sec = max(wait_sec, float(nxt - now_ts))

            if wait_sec > 0:
                _arena_set_next_ok(arena, now_ts, wait_sec, "phase_or_cooldown", msg)

            log_error("arena_fight", one_line(msg, 220))
            arena["today_fought_attempted"] = True
            protocol_set_reason(state, "arena", "arena:phase_block", one_line(f"fight_http:{msg}", 160))
            _bump_action_counter(state, "action_fail", "arena_fight")
            return 0

        except Exception as e:
            log_error("arena_fight", one_line(repr(e), 200))
            arena["today_fought_attempted"] = True
            protocol_set_reason(state, "arena", "arena:phase_block", one_line(f"fight_exc:{repr(e)}", 160))
            _bump_action_counter(state, "action_fail", "arena_fight")
            return 0

    protocol_set_reason(state, "arena", "arena:phase_block", "no_phase_action")
    return 0

################################################################################
# 12. CONTEXT MODELS + CORPUS/INDEX (threads/users + BM25 retrieval)
# - Dependencies: Section 3, 7-8 (Storage + Schemas + Text)
# - Used by: Context + retrieval
# - Key functions: BM25Index.search(), update_corpus_flow(), update_community_flow()
################################################################################

def load_threads(path: str) -> Dict[str, Any]:
    d = load_json_file(path, default={})
    if isinstance(d, dict):
        try:
            _migrate_threads_payload_inplace(d)
        except Exception as e:
            log_debug_exc("load_threads:migrate", e)
        return d
    return {}

def load_users(path: str) -> Dict[str, Any]:
    d = load_json_file(path, default={})
    return d if isinstance(d, dict) else {}

def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []

def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _extract_id_from_obj(obj: Any, keys: Tuple[str, ...]) -> str:
    if not isinstance(obj, dict):
        return ""
    for k in keys:
        v = obj.get(k)
        if v:
            return str(v)
    return ""

def _extract_comment_id(res: Any) -> str:
    return _extract_id_from_obj(res, ("id", "comment_id", "commentId"))

def _extract_post_id(res: Any) -> str:
    return _extract_id_from_obj(res, ("id", "post_id", "postId"))

def _note_reply_received(
    state: Dict[str, Any],
    memory: List[Dict[str, Any]],
    *,
    post_id: str,
    my_comment_id: str,
    reply_comment_id: str,
) -> None:
    """Best-effort: mark that one of my comments received a reply (deduped)."""
    if not my_comment_id or not reply_comment_id:
        return
    now = time.time()

    seen = _safe_dict(state.get("reply_seen_ids"))
    # dedup by reply comment id (global)
    if reply_comment_id in seen:
        return
    seen[reply_comment_id] = now

    # prune seen map (keep recent ~1000)
    if len(seen) > 1200:
        try:
            items = sorted(seen.items(), key=lambda x: float(x[1] or 0.0), reverse=True)[:900]
            seen = {k: v for k, v in items}
        except Exception as e:
            log_debug_exc("_note_reply_received:silent", e)
            pass
    state["reply_seen_ids"] = seen

    reps = _safe_dict(state.get("my_comment_replies"))
    info = _safe_dict(reps.get(my_comment_id))
    info["count"] = int(info.get("count", 0) or 0) + 1
    info["last_ts"] = now
    info["post_id"] = str(post_id or "")
    reps[my_comment_id] = info

    # prune reply map
    if len(reps) > 1500:
        try:
            items = sorted(reps.items(), key=lambda kv: float(_safe_dict(kv[1]).get("last_ts", 0.0) or 0.0), reverse=True)[:1000]
            reps = {k: v for k, v in items}
        except Exception as e:
            log_debug_exc("_note_reply_received:silent", e)
            pass
    state["my_comment_replies"] = reps

    # annotate recent memory item if present
    try:
        for it in reversed(_safe_list(memory)[-500:]):
            if not isinstance(it, dict):
                continue
            if str(it.get("comment_id") or "") != str(my_comment_id):
                continue
            it["reply_received"] = int(info.get("count", 0) or 0)
            it["reply_received_last_ts"] = float(info.get("last_ts", 0.0) or 0.0)
            break
    except Exception as e:
        log_debug_exc("_note_reply_received:silent", e)
        pass


def get_thread(state: Dict[str, Any], post_id: str) -> Dict[str, Any]:
    threads = state.get("threads")
    if not isinstance(threads, dict):
        threads = {}
        state["threads"] = threads

    th = threads.get(post_id)
    if not isinstance(th, dict):
        th = {
            "post_id": post_id,
            "topic": "",
            "keywords": [],
            "category": "",
            "participants": {},          # user_key -> stats
            "last_seen_ts": 0.0,
            "last_my_action_ts": 0.0,
            "last_k_turns": [],          # [{speaker,text,ts,kind}]
            # v23.1 (schema): expanded open_questions (backward compatible)
            "open_questions": [],        # [{qid,text,ts,status,asked_by,last_seen_remote_id,resolve_ts}]
            # v23.1 (schema): thread phase for upcoming interaction FSM
            "phase": "open",             # open|argue|clarify|close
            "phase_ts": 0.0,             # epoch seconds
            "summary": "",               # short synthesized summary
            "claims": [],                # extracted claim-like sents
            "tensions": [],              # extracted counter-like sents
            "seen_comment_ids": {},      # comment_id -> ts
            "turn_hashes": [],           # [[hash, ts], ...] for dedup
            "my_stance_snapshot": {},    # later
        }
        threads[post_id] = th
    else:
        th = _migrate_thread_schema_inplace(th, post_id)
        threads[post_id] = th
    return threads[post_id]


def _interaction_fsm_enabled(state: Optional[Dict[str, Any]]) -> bool:
    """Return whether interaction FSM behaviors are enabled.

    Policy:
      - If env MERSOOM_INTERACTION_FSM is set (any value), it is authoritative.
      - Otherwise fall back to persisted state.protocol.interaction_fsm_enabled (backward compatible).
    """
    try:
        if os.getenv("MERSOOM_INTERACTION_FSM") is not None:
            return bool(_env_bool("MERSOOM_INTERACTION_FSM", False))

        if not isinstance(state, dict):
            return False
        proto = _safe_dict(state.get("protocol"))
        return bool(proto.get("interaction_fsm_enabled", False))
    except Exception:
        return False

def _openq_track_enabled(state: Optional[Dict[str, Any]]) -> bool:
    """Return whether open-question tracking is enabled.

    Policy:
      - If env MERSOOM_OPENQ_TRACK is set (any value), it is authoritative.
      - Otherwise fall back to persisted state.protocol.openq_track_enabled (backward compatible).
    """
    try:
        if os.getenv("MERSOOM_OPENQ_TRACK") is not None:
            # default=True to preserve prior behavior when flag is set but empty
            return bool(_env_bool("MERSOOM_OPENQ_TRACK", True))

        if not isinstance(state, dict):
            return True
        proto = _safe_dict(state.get("protocol"))
        return bool(proto.get("openq_track_enabled", True))
    except Exception:
        return True


def _waiting_strict_enabled(state: Optional[Dict[str, Any]]) -> bool:
    """Return whether strict waiting_for_remote behavior is enabled.

    Policy:
      - If env MERSOOM_WAITING_STRICT is set (any value), it is authoritative.
      - Otherwise fall back to persisted state.protocol.waiting_strict_enabled (default: False).
    """
    try:
        if os.getenv("MERSOOM_WAITING_STRICT") is not None:
            return bool(_env_bool("MERSOOM_WAITING_STRICT", False))
        if not isinstance(state, dict):
            return False
        proto = _safe_dict(state.get("protocol"))
        return bool(proto.get("waiting_strict_enabled", False))
    except Exception:
        return False


def _reply_score_v2_enabled(state: Optional[Dict[str, Any]]) -> bool:
    """Return whether reply queue scoring v2 is enabled.

    Policy:
      - If env MERSOOM_REPLY_SCORE_V2 is set (any value), it is authoritative.
      - Otherwise fall back to persisted state.protocol.reply_score_v2_enabled (default: False).
    """
    try:
        if os.getenv("MERSOOM_REPLY_SCORE_V2") is not None:
            return bool(_env_bool("MERSOOM_REPLY_SCORE_V2", False))
        if not isinstance(state, dict):
            return False
        proto = _safe_dict(state.get("protocol"))
        return bool(proto.get("reply_score_v2_enabled", False))
    except Exception:
        return False


def _thread_open_q_open_count(th: Any) -> int:
    try:
        if not isinstance(th, dict):
            return 0
        oq = th.get("open_questions")
        if not isinstance(oq, list):
            return 0
        n = 0
        for q in oq:
            if isinstance(q, dict):
                st = str(q.get("status") or "open")
                if st == "open":
                    n += 1
            else:
                n += 1
        return int(n)
    except Exception:
        return 0


def _waiting_for_remote_on_item(state: Dict[str, Any], post_id: str, comment_id: str, parent_id: str) -> bool:
    """Best-effort: check waiting_for_remote without fetching full thread tree.

    Uses conv_state keys that are typically based on the reply root. We try both parent_id and comment_id.
    """
    try:
        convs = state.get("conv_state")
        if not isinstance(convs, dict):
            return False
        pid = str(post_id or "")
        if not pid:
            return False
        for root in (str(parent_id or ""), str(comment_id or "")):
            if not root:
                continue
            k = _reply_conv_key(pid, root)
            cv = convs.get(k)
            if isinstance(cv, dict) and bool(cv.get("waiting_for_remote")):
                return True
        return False
    except Exception:
        return False


def _reply_score_v2(state: Dict[str, Any], item: Dict[str, Any], now_ts: float) -> Tuple[float, Dict[str, Any]]:
    """Compute expanded reply priority score.

    Adds:
      + open question bonus (thread has unresolved questions)
      + freshness bonus (recent comments)
      - waiting_for_remote penalty (avoid consecutive replies in same thread)
    """
    base = float(item.get("score", 0.0) or 0.0)
    pid = str(item.get("post_id") or "")
    cid = str(item.get("comment_id") or "")
    parent_id = str(item.get("replied_to_comment_id") or "")

    # open questions bonus
    oq_bonus = 0.0
    try:
        th = _safe_dict(_safe_dict(state.get("threads")).get(pid))
        if _thread_open_q_open_count(th) > 0:
            oq_bonus = 28.0
    except Exception:
        oq_bonus = 0.0

    # freshness bonus (decays with age)
    cts = float(item.get("comment_ts", 0.0) or 0.0)
    age = max(0.0, float(now_ts) - float(cts)) if cts > 0 else 1e9
    # 0s -> +34, 30m -> ~+20, 2h -> ~+5
    fresh = 34.0 * math.exp(-age / 1800.0) if age < 1e8 else 0.0

    # waiting penalty
    waiting = bool(_waiting_for_remote_on_item(state, pid, cid, parent_id))
    wait_pen = 120.0 if waiting else 0.0

    # small direct-reply bias to stabilize ordering
    direct = 8.0 if bool(item.get("reply_to_my_comment")) else 0.0

    score = base + direct + oq_bonus + fresh - wait_pen
    breakdown = {
        "base": round(base, 3),
        "direct": round(direct, 3),
        "openq": round(oq_bonus, 3),
        "fresh": round(fresh, 3),
        "wait_pen": round(wait_pen, 3),
        "age_sec": int(age) if age < 1e8 else None,
    }
    return float(score), breakdown


_OPENQ_QWORD_RE = re.compile(
    r"(\?$|\?|\b(why|what|how|when|where|who)\b|(?:^|[\s\W])(왜(?!냐하면)|뭐|무엇|어떻게|어째서|어떤|어디|언제|누구|맞나)(?:$|[\s\W]))",
    re.IGNORECASE,
)


def _is_open_question_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # Primary signal
    if "?" in t:
        return True
    # Secondary signal: question words (avoid common false positives like "왜냐하면")
    return bool(_OPENQ_QWORD_RE.search(t))


def _extract_open_question_text(text: str, max_len: int = 220) -> str:
    """Extract a compact question-like snippet from outgoing text."""
    t = one_line(text or "", max_len * 2).strip()
    if not t:
        return ""
    if "?" in t:
        qpos = t.rfind("?")
        # find a soft sentence boundary before the last '?'
        start = max(t.rfind(". ", 0, qpos), t.rfind("! ", 0, qpos), t.rfind("…", 0, qpos), t.rfind("\n", 0, qpos))
        if start < 0:
            start = 0
        else:
            start = min(len(t), start + 2)
        frag = t[start : qpos + 1].strip()
        if frag:
            return one_line(frag, max_len)
    return one_line(t, max_len)

# -----------------------------------------------------------------------------
# v23.4 Open-question lifecycle helpers (resolve/expire; LLM-free)
# -----------------------------------------------------------------------------

_OPENQ_MIN_REPLY_CHARS = 12
_OPENQ_TTL_SEC = 12 * 60 * 60  # expire open questions after 12 hours (safe default)

def _prune_open_questions(items: Any, keep_open: int = 8, keep_hist: int = 4) -> List[Dict[str, Any]]:
    """Keep open questions small and useful: keep recent opens + a little history."""
    oq = _normalize_open_questions_list(items)
    open_items = [q for q in oq if str(q.get("status") or "open") == "open"]
    hist_items = [q for q in oq if str(q.get("status") or "") in ("resolved", "expired")]

    def _fnum(x: Any) -> float:
        try:
            return float(x or 0.0)
        except Exception:
            return 0.0

    open_items = sorted(open_items, key=lambda q: _fnum(q.get("ts")))
    hist_items = sorted(hist_items, key=lambda q: _fnum(q.get("resolve_ts")))

    ko = max(0, int(keep_open or 0))
    kh = max(0, int(keep_hist or 0))

    if ko > 0:
        open_items = open_items[-ko:]
    if kh > 0:
        hist_items = hist_items[-kh:]
    else:
        hist_items = []

    return open_items + hist_items

def _openq_core_tokens(text: str, max_toks: int = 3) -> List[str]:
    toks = tokenize(text or "", max_tokens=48)
    out: List[str] = []
    lim = max(1, int(max_toks or 3))
    for t in toks:
        if not t:
            continue
        if t in out:
            continue
        out.append(t)
        if len(out) >= lim:
            break
    return out

def _openq_should_resolve(q: Dict[str, Any], reply_text: str) -> bool:
    rt = (reply_text or "").strip()
    if len(rt) < _OPENQ_MIN_REPLY_CHARS:
        return False

    qtext = str(q.get("text") or "").strip()
    if not qtext:
        return False

    q_toks = _openq_core_tokens(qtext, max_toks=3)
    if not q_toks:
        return False

    r_toks = set(tokenize(rt, max_tokens=90))
    if not r_toks:
        return False

    overlap = sum(1 for t in q_toks if t in r_toks)
    need = 2 if len(q_toks) >= 2 else 1
    return overlap >= need

def thread_openq_expire_old(state: Optional[Dict[str, Any]], th: Dict[str, Any], now_ts: Optional[float] = None) -> int:
    """Expire stale open questions on a thread. Returns number expired."""
    try:
        if not isinstance(th, dict):
            return 0
        now = float(now_ts if now_ts is not None else time.time())
        oq = _normalize_open_questions_list(th.get("open_questions", []))
        changed = 0
        for q in oq:
            try:
                if str(q.get("status") or "open") != "open":
                    continue
                ts = float(q.get("ts", 0.0) or 0.0)
                if ts <= 0:
                    continue
                if (now - ts) > float(_OPENQ_TTL_SEC):
                    q["status"] = "expired"
                    q["resolve_ts"] = now
                    changed += 1
                    if isinstance(state, dict):
                        protocol_bump_counter(state, "openq_expire", 1)
                        try:
                            proto = _sdict(state, "protocol")
                            proto["openq_expired_total"] = int(proto.get("openq_expired_total", 0) or 0) + 1
                        except Exception:
                            pass
                    log_event(
                        "openq.expire",
                        post_id=str(th.get("post_id") or ""),
                        qid=str(q.get("qid") or ""),
                        age_sec=int(max(0.0, now - ts)),
                    )
            except Exception:
                continue
        th["open_questions"] = _prune_open_questions(oq, keep_open=8, keep_hist=4)
        return int(changed)
    except Exception:
        return 0

def thread_openq_try_resolve_on_remote_turn(
    state: Optional[Dict[str, Any]],
    th: Dict[str, Any],
    remote_nick: str,
    remote_text: str,
    remote_comment_id: str = "",
    cfg: Optional[Config] = None,
) -> int:
    """Try to resolve the most recent open question if the remote turn appears to answer it."""
    try:
        if not isinstance(th, dict):
            return 0
        oq = _normalize_open_questions_list(th.get("open_questions", []))
        # consider only OPEN questions asked by the agent (asked_by == "me" or empty)
        candidates = []
        for q in oq:
            try:
                if str(q.get("status") or "open") != "open":
                    continue
                ab = str(q.get("asked_by") or "")
                if ab not in ("", "me"):
                    continue
                candidates.append(q)
            except Exception:
                continue

        if not candidates:
            th["open_questions"] = _prune_open_questions(oq, keep_open=8, keep_hist=4)
            return 0

        # newest first
        candidates = sorted(candidates, key=lambda q: float(q.get("ts", 0.0) or 0.0), reverse=True)

        resolved = 0
        for q in candidates:
            if _openq_should_resolve(q, remote_text):
                q["status"] = "resolved"
                q["resolve_ts"] = float(time.time())
                q["last_seen_remote_id"] = str(remote_comment_id or "")
                resolved = 1
                if isinstance(state, dict):
                    protocol_bump_counter(state, "openq_resolve", 1)
                    try:
                        proto = _sdict(state, "protocol")
                        proto["openq_resolved_total"] = int(proto.get("openq_resolved_total", 0) or 0) + 1
                    except Exception:
                        pass
                log_event(
                    "openq.resolve",
                    post_id=str(th.get("post_id") or ""),
                    qid=str(q.get("qid") or ""),
                    by=str(remote_nick or ""),
                    comment_id=str(remote_comment_id or ""),
                )
                break

        th["open_questions"] = _prune_open_questions(oq, keep_open=8, keep_hist=4)
        return int(resolved)
    except Exception:
        return 0

_PHASE_QUESTION_RE = re.compile(r"(\?|\b(why|what|how|when|where|who)\b|(?:^|[\s\W])(왜|뭐|무엇|어떻게|어째서|어떤|어디|언제|누구|맞나)(?:$|[\s\W]))", re.IGNORECASE)
_PHASE_ARGUE_RE = re.compile(r"\b(아님|아니|반박|틀림|오류|근거|증거|팩트|사실)\b", re.IGNORECASE)
_PHASE_CLOSE_RE = re.compile(r"\b(정리|결론|동의|요약|합의|마무리)\b", re.IGNORECASE)


def _classify_thread_phase(text: str) -> Tuple[str, str]:
    """Classify a thread phase from a single turn (rule-based; LLM-free).

    Priority (per spec):
      1) question -> clarify
      2) argue/negation -> argue
      3) agreement/summary -> close
      else -> open
    Returns: (phase, reason_tag)
    """
    t = (text or "").strip()
    if not t:
        return "open", "empty"
    if _PHASE_QUESTION_RE.search(t):
        return "clarify", "question"
    if _PHASE_ARGUE_RE.search(t):
        return "argue", "argue_kw"
    if _PHASE_CLOSE_RE.search(t):
        return "close", "close_kw"
    return "open", "default"


def thread_update_phase_if_needed(
    state: Optional[Dict[str, Any]],
    th: Dict[str, Any],
    text: str,
    source: str = "",
) -> None:
    """Update thread phase in-place (best-effort). Only active when interaction FSM is enabled."""
    try:
        if not _interaction_fsm_enabled(state):
            return
        if not isinstance(th, dict):
            return
        old = str(th.get("phase") or "open")
        new, reason = _classify_thread_phase(text)
        if new != old:
            th["phase"] = str(new)
            th["phase_ts"] = float(time.time())
            if isinstance(state, dict):
                protocol_bump_counter(state, "phase_transition", 1)
            log_event(
                "thread.phase",
                post_id=str(th.get("post_id") or ""),
                old=old,
                new=new,
                reason=str(reason),
                source=str(source or ""),
            )
    except Exception:
        return

def thread_bucket_key(th: Dict[str, Any]) -> str:
    cat = str(th.get("category") or "")
    return cat or "thread"

def get_user(state: Dict[str, Any], user_key: str) -> Dict[str, Any]:
    users = state.setdefault("users", {})
    if user_key not in users or not isinstance(users.get(user_key), dict):
        users[user_key] = {
            "user_key": user_key,
            "aggression": 0.0,
            "helpfulness": 0.0,
            "topic_pref": {},
            "last_seen_ts": 0.0,
            "turns": 0,
        }
    return users[user_key]

def user_bucket_key(user: Dict[str, Any]) -> str:
    agg = float(user.get("aggression", 0.0))
    if agg >= 0.6:
        return "hot"
    if agg <= 0.2:
        return "calm"
    return "mid"

def thread_add_turn(th: Dict[str, Any], speaker: str, text: str, kind: str) -> None:
    # dedup turns to avoid "re-ingest spam"
    th.setdefault("turn_hashes", [])
    th["turn_hashes"] = _clean_hash_list(_safe_list(th.get("turn_hashes", [])), ttl_sec=6 * 60 * 60, keep_max=1200)

    norm = one_line(text, 320)
    h = _text_hash(f"{kind}|{speaker}|{norm}")
    if any(x[0] == h for x in th.get("turn_hashes", [])):
        th["last_seen_ts"] = time.time()
        return

    th["turn_hashes"].append([h, time.time()])

    th.setdefault("last_k_turns", [])
    th["last_k_turns"].append({
        "speaker": str(speaker or ""),
        "text": one_line(text, 600),
        "ts": time.time(),
        "kind": kind,
    })
    # cap
    th["last_k_turns"] = th["last_k_turns"][-20:]
    th["last_seen_ts"] = time.time()

def thread_add_open_question(th: Dict[str, Any], qtext: str, asked_by: str = "me", last_seen_remote_id: str = "") -> str:
    qid = hashlib.sha1(f"{time.time()}|{qtext}".encode("utf-8")).hexdigest()[:10]
    th.setdefault("open_questions", [])
    th["open_questions"] = _normalize_open_questions_list(th.get("open_questions", []))
    th["open_questions"].append({
        "qid": qid,
        "text": one_line(qtext, 220),
        "ts": time.time(),
        "status": "open",
        "asked_by": str(asked_by or "me"),
        "last_seen_remote_id": str(last_seen_remote_id or ""),
        "resolve_ts": 0.0,
    })
    th["open_questions"] = _prune_open_questions(th.get("open_questions", []), keep_open=8, keep_hist=4)
    return qid

def thread_pop_open_question(th: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    oq = _safe_list(th.get("open_questions"))
    if not oq:
        return None
    return oq[-1]

def _participant_bump(th: Dict[str, Any], user_key: str) -> None:
    p = th.setdefault("participants", {})
    if user_key not in p or not isinstance(p.get(user_key), dict):
        p[user_key] = {"turns": 0, "last_ts": 0.0}
    p[user_key]["turns"] = int(p[user_key].get("turns", 0)) + 1
    p[user_key]["last_ts"] = time.time()

def ingest_post_into_context(state: Dict[str, Any], post: Dict[str, Any], brain: Optional[Dict[str, Any]] = None) -> None:
    pid = str(post.get("id") or post.get("post_id") or "")
    if not pid:
        return
    th = get_thread(state, pid)
    synthesize_thread(th)
    title = str(post.get("title") or "")
    content = str(post.get("content") or "")
    cat, _ = classify_text(f"{title}\n{content}")
    th["category"] = cat
    th["topic"] = title[:80]
    th["keywords"] = top_keywords(f"{title} {content}", k=8)
    nick = str(post.get("nickname") or post.get("author") or "")
    if nick:
        _participant_bump(th, nick)

    # one-time thought for newly observed thread
    if brain is not None and not th.get("post_ingested"):
        th["post_ingested"] = True
        add_thought(
            brain,
            kind="observe_post",
            topic=title[:80] or ((th.get("keywords") or [""])[0] if isinstance(th.get("keywords"), list) else ""),
            text=_simple_summary(f"{title} {content}", max_len=180),
            tags=_safe_list(th.get("keywords"))[:6],
            links={"post_id": pid, "author": nick},
            strength=0.55,
        )

    thread_add_turn(th, nick or "OP", f"{title} {content}", "post")
    thread_update_phase_if_needed(state, th, f"{title} {content}", source="ingest_post")

def ingest_comments_into_context(state: Dict[str, Any], post_id: str, comments: List[Dict[str, Any]], brain: Optional[Dict[str, Any]] = None, cfg: Optional[Config] = None) -> None:
    th = get_thread(state, post_id)
    th.setdefault("seen_comment_ids", {})
    seen = _safe_dict(th.get("seen_comment_ids"))

    # only ingest truly new comments (avoid thrashing)
    for c in comments[-12:]:
        cid = str(c.get("id") or "")
        if cid and cid in seen:
            continue

        nick = str(c.get("nickname") or c.get("author") or "")
        text = str(c.get("content") or "")
        if not text:
            continue

        if cid:
            seen[cid] = time.time()


        _participant_bump(th, nick or "user")

        tox_cfg = getattr(cfg, "toxic", None) if cfg is not None else None
        tox, tox_reason = is_toxic_incoming(text)
        if tox and tox_cfg and bool(getattr(tox_cfg, "exclude_from_learning", True)):
            # do NOT feed toxic raw text into thread turns / template mining inputs
            th["toxic_hits"] = int(th.get("toxic_hits", 0) or 0) + 1
            try:
                tc = th.setdefault("toxic_comment_ids", {})
                if not isinstance(tc, dict):
                    th["toxic_comment_ids"] = {}
                    tc = th["toxic_comment_ids"]
                if cid:
                    tc[str(cid)] = {"ts": time.time(), "reason": str(tox_reason)}
                _lru_prune_map(tc, 200)
            except Exception as e:
                log_debug_exc("ingest_comments_into_context:silent", e)
                pass
        else:
            thread_add_turn(th, nick or "user", text, "comment")
            thread_update_phase_if_needed(state, th, text, source="ingest")

            # v23.4: detect remote answers to open questions (best-effort)
            try:
                if _openq_track_enabled(state):
                    is_me = False
                    try:
                        if cfg is not None:
                            is_me = (str(nick or "") == str(getattr(cfg, "nickname", "")))
                    except Exception:
                        is_me = False
                    if not is_me:
                        thread_openq_try_resolve_on_remote_turn(state, th, nick, text, cid, cfg=cfg)
            except Exception:
                pass

        # update user model lightly
        if nick:
            u = get_user(state, nick)
            u["turns"] = int(u.get("turns", 0)) + 1
            u["last_seen_ts"] = time.time()
            if looks_offensive(text):
                u["aggression"] = min(1.0, float(u.get("aggression", 0.0)) + 0.12)
            if any(k in text for k in ["정리", "도움", "참고", "설명", "근거"]):
                u["helpfulness"] = min(1.0, float(u.get("helpfulness", 0.0)) + 0.08)
            # topic preference
            for kw in _safe_list(th.get("keywords"))[:6]:
                if not kw:
                    continue
                tp = _safe_dict(u.get("topic_pref"))
                tp[kw] = float(tp.get(kw, 0.0)) + 1.0
                u["topic_pref"] = tp

        # selective thought capture (questions / conflict / good explanations)
        if brain is not None:
            capture = False
            if "?" in text or text.strip().endswith("?"):
                capture = True
            if looks_offensive(text):
                capture = True
            if any(k in text for k in ["근거", "예시", "재현", "정리"]):
                capture = True

            if capture:
                add_thought(
                    brain,
                    kind="observe_comment",
                    topic=(th.get("topic") or (th.get("keywords") or [""])[0] or "thread")[:80],
                    text=_simple_summary(text, max_len=180),
                    tags=_safe_list(th.get("keywords"))[:6],
                    links={"post_id": post_id, "author": nick, "comment_id": cid},
                    strength=0.45,
                )

    # keep under control
    # prune seen ids
    if len(seen) > 800:
        # keep latest by ts
        items = sorted(seen.items(), key=lambda kv: float(kv[1] or 0.0))
        seen = dict(items[-700:])
    th["seen_comment_ids"] = seen

# --- CORPUS + BM25 ---

def load_corpus_jsonl(path: str, *, max_docs: int = 4000) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return out[-max(1, int(max_docs)):]

def corpus_doc_id(kind: str, post_id: str, author: str, text: str) -> str:
    s = f"{kind}|{post_id}|{author}|{text}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def append_corpus_doc(path: str, doc: Dict[str, Any]) -> None:
    append_jsonl(path, doc)

class BM25Index:
    def __init__(self) -> None:
        self.doc_len: Dict[str, int] = {}
        self.doc_meta: Dict[str, Dict[str, Any]] = {}
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, List[Tuple[str, int]]] = {}  # term -> [(doc_id, tf)]
        self.N = 0
        self.avgdl = 0.0

    def build(self, docs: List[Dict[str, Any]], *, max_docs: int = 3000) -> None:
        self.doc_len.clear()
        self.doc_meta.clear()
        self.df.clear()
        self.postings.clear()

        kept = docs[-max(1, int(max_docs)):]
        self.N = 0
        total_len = 0

        for d in kept:
            if not isinstance(d, dict):
                continue
            doc_id = str(d.get("doc_id") or "")
            text = str(d.get("text") or "")
            if not doc_id or not text:
                continue
            toks = d.get("tokens")
            if not isinstance(toks, list):
                toks = tokenize(text, max_tokens=260)
                try:
                    d["tokens"] = toks  # cache tokens for future rebuilds
                except Exception:
                    pass
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1

            self.N += 1
            dl = sum(tf.values())
            total_len += dl
            self.doc_len[doc_id] = dl
            self.doc_meta[doc_id] = d

            for term in tf.keys():
                self.df[term] = self.df.get(term, 0) + 1
            for term, c in tf.items():
                self.postings.setdefault(term, []).append((doc_id, int(c)))

        self.avgdl = (total_len / self.N) if self.N > 0 else 0.0
def search(
    self,
    query_tokens: List[str],
    *,
    topk: int = 6,
    min_score: float = 0.0,
    normalize_scores: bool = True,
) -> List[Tuple[str, float]]:
    """
    BM25 search.
    - min_score: filter by score threshold (applied after optional normalization)
    - normalize_scores: scale scores to 0..100 (relative to max hit) for stability
    """
    if self.N <= 0 or not query_tokens:
        return []
    k1 = 1.2
    b = 0.75

    scores: Dict[str, float] = {}
    q = list(dict.fromkeys(query_tokens))[:20]
    for term in q:
        df = self.df.get(term, 0)
        if df <= 0:
            continue
        idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        for doc_id, tf in self.postings.get(term, []):
            dl = self.doc_len.get(doc_id, 0)
            denom = tf + k1 * (1 - b + b * (dl / max(1e-9, self.avgdl)))
            s = idf * (tf * (k1 + 1) / max(1e-9, denom))
            scores[doc_id] = scores.get(doc_id, 0.0) + s

    if not scores:
        return []

    # v21.1: avoid full sort when only top-k is needed
    try:
        tk = max(1, int(topk))
    except Exception:
        tk = 6

    mx = 0.0
    try:
        mx = max(scores.values())
    except Exception:
        mx = 0.0

    cand_n = max(tk * 8, 12)
    ranked = heapq.nlargest(cand_n, scores.items(), key=lambda kv: kv[1])

    if normalize_scores and mx > 1e-12:
        ranked = [(doc_id, (score / mx) * 100.0) for doc_id, score in ranked]

    try:
        ms = float(min_score or 0.0)
    except Exception:
        ms = 0.0
    if ms > 0.0:
        ranked = [(doc_id, score) for doc_id, score in ranked if score >= ms]

    return ranked[:tk]


    def _sanitize_line(s2: str) -> str:
        s2s = s2.strip()
        # replace common meta-openers
        for bad in ("요지는", "핵심은", "포인트는", "본문 요지는", "그 댓글 요지는", "결론은", "정리하면", "요약하면", "한 줄 요약은"):
            if s2s.startswith(bad):
                rest = s2s[len(bad):].lstrip(" :,-")
                s2s = ("쟁점은 " + rest).strip()
                break
        # remove meme-like openers completely
        for ban in ("여기서부터가 꿀잼 구간임", "논점은 잡히는 느낌임", "감정은 제외하고", "정의→기준→검증", "정의->기준->검증", "정의 → 기준 → 검증"):
            s2s = s2s.replace(ban, "").strip()
        return s2s if s2s else s2.strip()

    for s in sents[:3]:
        s2 = one_line(s, 180)
        s2 = _sanitize_line(s2)
        if not s2:
            continue
        if kws:
            s2 = s2.replace(kws[0], "{KW}")
        lines.append(s2)

    tpl = "\n".join(lines).strip()
    return tpl


################################################################################
# 13. TEMPLATE MINER / REGISTRY (mined templates + quality gates)
# - Dependencies: Section 8, 9, 12 (Text + Policy + Corpus/Index)
# - Used by: Post/comment generation (template registry + mined templates)
# - Key functions: register_mined_template(), render_template()
################################################################################

def register_mined_template(policy: Dict[str, Any], template_text: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """Register a mined template with quality scoring + v20.5 2nd QA gate + near-dup checks.

    - Step 1: template_static_eval() gate (cheap + stable)
    - Step 2: QA 2nd-pass gate on rendered sample (hard-fail/length/etc)
    - Step 3: near-dup checks vs existing templates (fp + jaccard/3gram)
    - Rejections are recorded under templates.rejected_templates for tuning.
    """
    t = (template_text or "").strip()
    if not t:
        return ""

    temps = _safe_dict(policy.get("templates", {}))
    qcfg = _safe_dict(temps.get("quality", {}))
    min_static = int(qcfg.get("min_mine_score", 52) or 52)
    min_qa2 = int(qcfg.get("min_mine_qa2_score", 60) or 60)

    # rejection ledger (bounded)
    rej = policy.setdefault("templates", {}).setdefault("rejected_templates", [])
    def _reject(reason: str, extra: Optional[Dict[str, Any]] = None) -> str:
        try:
            rec = {
                "ts": time.time(),
                "reason": str(reason or "unknown"),
                "text": t[:420],
                "meta": meta or {},
                "extra": extra or {},
            }
            rej.append(rec)
            if len(rej) > 500:
                del rej[:-500]
        except Exception:
            pass
        return ""

    rep = template_static_eval(t)
    static_score = int(rep.get("score", 0) or 0)
    if static_score < min_static:
        return _reject("static_low", {"static_score": static_score, "issues": list(rep.get("issues", []) or [])[:12]})

    # 2nd QA: run on a rendered sample so braces/slots don't hide issues
    sample = t
    sample = sample.replace("{KW}", "그거")
    sample = sample.replace("{QUOTE}", "“인용”")
    sample = sample.replace("{Q}", "이거 어떻게 봄")
    qa2 = qa_evaluate_text(sample, kind="comment")
    qa2_score = int(qa2.get("score", 0) or 0)
    qa2_issues = list(qa2.get("issues", []) or [])[:16]
    if qa2.get("hard_fail") is True:
        return _reject("qa2_hard_fail", {"qa2_score": qa2_score, "qa2_issues": qa2_issues})

    # "language rules violations 0" (hard subset)
    for bad in ("injection", "offensive", "markdown"):
        if bad in qa2_issues:
            return _reject(f"qa2_violation:{bad}", {"qa2_score": qa2_score, "qa2_issues": qa2_issues})

    if qa2_score < min_qa2:
        return _reject("qa2_low", {"qa2_score": qa2_score, "qa2_issues": qa2_issues, "min_qa2": min_qa2})

    items = policy.setdefault("templates", {}).setdefault("items", {})

    # exact-id dedupe (same text)
    tid = hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
    if tid in items:
        obj = items.get(tid)
        if isinstance(obj, dict):
            obj.setdefault("static_score", static_score)
            obj.setdefault("static_issues", list(rep.get("issues", []) or []))
            obj.setdefault("static_checked_ts", float(rep.get("checked_ts", time.time())))
            obj.setdefault("qa2_score", qa2_score)
            obj.setdefault("qa2_issues", qa2_issues)
            obj.setdefault("qa2_checked_ts", time.time())
        return tid

    # near-dup vs existing templates
    fp_new = _text_fp(t)
    j_th = float(_env_float("MERSOOM_SIM_JACCARD_TH", 0.70, min_v=0.0, max_v=1.0))
    g_th = float(_env_float("MERSOOM_SIM_3GRAM_TH", 0.82, min_v=0.0, max_v=1.0))
    cur_kw = _sig_keywords(sample, k=12) if j_th > 0.0 else []
    cur_g = _sig_3grams(sample, max_ngrams=256) if g_th > 0.0 else []

    # limit comparisons for cost control
    cand: List[Tuple[float, str, Dict[str, Any]]] = []
    for otid, oobj in list(_safe_dict(items).items()):
        if not isinstance(oobj, dict):
            continue
        try:
            ts = float(oobj.get("created_ts", 0.0) or 0.0)
        except Exception:
            ts = 0.0
        cand.append((ts, str(otid), oobj))
    cand.sort(key=lambda x: x[0], reverse=True)
    cand = cand[:240]

    for _, otid, oobj in cand:
        try:
            otext = str(oobj.get("text") or "")
            if not otext:
                continue
            if _text_fp(otext) == fp_new:
                return _reject("dup_fp_template", {"dup_tid": otid})
        except Exception:
            continue

    if j_th > 0.0 and cur_kw:
        for _, otid, oobj in cand:
            try:
                otext = str(oobj.get("text") or "")
                os = otext.replace("{KW}", "그거").replace("{QUOTE}", "“인용”").replace("{Q}", "이거 어떻게 봄")
                okw = _sig_keywords(os, k=12)
                if okw and _jaccard_ratio(cur_kw[:12], okw[:12]) >= j_th:
                    return _reject("dup_sim_jaccard", {"dup_tid": otid, "th": j_th})
            except Exception:
                continue

    if g_th > 0.0 and cur_g:
        for _, otid, oobj in cand:
            try:
                otext = str(oobj.get("text") or "")
                os = otext.replace("{KW}", "그거").replace("{QUOTE}", "“인용”").replace("{Q}", "이거 어떻게 봄")
                og = _sig_3grams(os, max_ngrams=256)
                if og and _jaccard_ratio(cur_g[:256], og[:256]) >= g_th:
                    return _reject("dup_sim_3gram", {"dup_tid": otid, "th": g_th})
            except Exception:
                continue

    # register
    items[tid] = {
        "text": t,
        "weight": 1.0,
        "meta": meta or {},
        "created_ts": time.time(),
        "uses": 0,

        # Unit 08 stats
        "static_score": static_score,
        "static_issues": list(rep.get("issues", []) or []),
        "static_checked_ts": float(rep.get("checked_ts", time.time())),
        "eval_uses": 0,
        "reward_ema": 0.0,
        "qa_ema": 0.75,
        "artifact_ema": 0.0,

        # v20.5: mining 2nd QA gate trace
        "qa2_score": int(qa2_score),
        "qa2_issues": qa2_issues,
        "qa2_checked_ts": time.time(),
    }
    return tid



def render_template(template_text: str, *, kw: str, quote: str, question: str) -> str:
    t = template_text
    t = t.replace("{KW}", kw or "그거")
    t = t.replace("{QUOTE}", quote or "맥락이 비어있음")
    t = t.replace("{Q}", question or "이거 어떻게 봄")
    return t

################################################################################
# 14. AGENT LOGIC (targets/actions/reward/learning + context + retrieval)
# - Dependencies: Section 1-10 (Foundation + Policy + API)
# - Used by: Main loop
# - Key functions: choose_target(), build_reply_text(), build_post_text(), compute_reward()
################################################################################

@dataclass
class ActionResult:
    ok: bool
    code: str
    detail: str = ""
    http_status: Optional[int] = None
    elapsed_ms: float = 0.0

def _bump_action_counter(state: Dict[str, Any], key: str, action: str) -> None:
    try:
        protocol_bump_counter(state, key, 1)
        protocol_bump_counter(state, f"{key}:{action}", 1)
    except Exception:
        pass

@lru_cache(maxsize=4096)
def _text_hash(text: str) -> str:
    # v21.1: cache (used heavily by near-dup / recent-text guards)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _clean_hash_list(items: List[List[Any]], ttl_sec: int, keep_max: int) -> List[List[Any]]:
    now = time.time()
    out: List[List[Any]] = []
    for it in items:
        try:
            h, ts = it[0], float(it[1])
            if (now - ts) <= ttl_sec:
                out.append([h, ts])
        except Exception:
            continue
    return out[-keep_max:]

def _clean_hash_map(m: Dict[str, float], ttl_sec: int, keep_max: int) -> Dict[str, float]:
    """Clean {hash: ts} map by TTL and cap (keeps most recent)."""
    now = time.time()
    ttl = max(1, int(ttl_sec))
    km = max(1, int(keep_max))

    # drop expired
    try:
        expired = [h for h, ts in m.items() if (now - float(ts)) > ttl]
        for h in expired:
            m.pop(h, None)
    except Exception:
        # if map is malformed, reset
        return {}

    # cap
    if len(m) > km:
        try:
            items = sorted(m.items(), key=lambda kv: float(kv[1]))
            # keep newest km
            m = dict(items[-km:])
        except Exception:
            pass
    return m

def _recent_hashes_get(state: Dict[str, Any], key: str) -> Dict[str, float]:
    """Back-compat: accept list[[hash,ts],...] or dict{hash:ts}. Returns dict."""
    v = state.get(key)
    if isinstance(v, dict):
        # ensure float values
        out: Dict[str, float] = {}
        for h, ts in v.items():
            try:
                out[str(h)] = float(ts)
            except Exception:
                continue
        return out
    if isinstance(v, list):
        out2: Dict[str, float] = {}
        for it in v:
            try:
                h, ts = it[0], float(it[1])
                out2[str(h)] = float(ts)
            except Exception:
                continue
        return out2
    return {}

def _recent_hashes_set(state: Dict[str, Any], key: str, m: Dict[str, float]) -> None:
    # store as dict for O(1) membership
    state[key] = {str(h): float(ts) for h, ts in m.items()}

def _dup_action_guard_enabled() -> bool:
    return _env_bool("MERSOOM_DUP_ACTION_GUARD", True)

def _dup_action_ttl_sec() -> int:
    return _env_int("MERSOOM_DUP_ACTION_TTL_SEC", 1200, 600, 1800)

def _dup_action_keep_max() -> int:
    return _env_int("MERSOOM_DUP_ACTION_KEEP_MAX", 800, 100, 5000)

def _dup_action_fp(action: str, target_id: str, endpoint_key: str) -> str:
    return f"{action}|{target_id}|{endpoint_key}"

def _recent_actions_get(state: Dict[str, Any]) -> Dict[str, float]:
    ra = state.get("recent_actions")
    if not isinstance(ra, dict):
        ra = {}
    out: Dict[str, float] = {}
    for k, v in ra.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out

def _recent_actions_set(state: Dict[str, Any], ra: Dict[str, float]) -> None:
    state["recent_actions"] = {str(k): float(v) for k, v in ra.items()}

def _clean_recent_actions(ra: Dict[str, float], ttl_sec: int, keep_max: int) -> Dict[str, float]:
    now = time.time()
    ttl = max(60, int(ttl_sec))
    km = max(50, int(keep_max))
    try:
        expired = [k for k, ts in ra.items() if (now - float(ts)) > ttl]
        for k in expired:
            ra.pop(k, None)
    except Exception:
        return {}
    if len(ra) > km:
        try:
            items = sorted(ra.items(), key=lambda kv: float(kv[1]))
            ra = dict(items[-km:])
        except Exception:
            pass
    return ra

def dup_action_should_skip(state: Dict[str, Any], *, action: str, target_id: str, endpoint_key: str) -> bool:
    if not _dup_action_guard_enabled():
        return False
    fp = _dup_action_fp(action, target_id, endpoint_key)
    ra = _recent_actions_get(state)
    ra = _clean_recent_actions(ra, _dup_action_ttl_sec(), _dup_action_keep_max())
    _recent_actions_set(state, ra)
    if fp in ra:
        protocol_bump_counter(state, "dup_action_skip", 1)
        return True
    return False

def remember_action(state: Dict[str, Any], *, action: str, target_id: str, endpoint_key: str) -> None:
    if not _dup_action_guard_enabled():
        return
    fp = _dup_action_fp(action, target_id, endpoint_key)
    ra = _recent_actions_get(state)
    ra[fp] = time.time()
    ra = _clean_recent_actions(ra, _dup_action_ttl_sec(), _dup_action_keep_max())
    _recent_actions_set(state, ra)


def recently_used_text(state: Dict[str, Any], text: str, *, for_post: bool, same_text_gap_sec: int) -> bool:
    key = "recent_post_text_hashes" if for_post else "recent_text_hashes"
    m = _recent_hashes_get(state, key)
    m = _clean_hash_map(m, int(same_text_gap_sec), 800)
    _recent_hashes_set(state, key, m)
    h = _text_hash(str(text or ""))
    return h in m

def remember_text(state: Dict[str, Any], text: str, *, for_post: bool, same_text_gap_sec: int) -> None:
    key = "recent_post_text_hashes" if for_post else "recent_text_hashes"
    m = _recent_hashes_get(state, key)
    h = _text_hash(str(text or ""))
    m[h] = time.time()
    m = _clean_hash_map(m, int(same_text_gap_sec), 800)
    _recent_hashes_set(state, key, m)


@lru_cache(maxsize=4096)
def _normalize_for_fp(text: str) -> str:
    # Compact fingerprint normalization to catch "same meaning, tiny edits"
    s = one_line(str(text or ""), 2000).strip().lower()
    s = re.sub(r"\s+", "", s)
    # keep only alnum + hangul
    s = re.sub(r"[^0-9a-z가-힣]+", "", s)
    # drop common eumssum endings (best-effort)
    for suf in ("입니다", "임", "음", "슴", "함"):
        if s.endswith(suf) and len(s) > len(suf) + 8:
            s = s[: -len(suf)]
            break
    return s[:800]

@lru_cache(maxsize=4096)
def _text_fp(text: str) -> str:
    n = _normalize_for_fp(text)
    return hashlib.sha1(n.encode("utf-8")).hexdigest()

def recently_used_fp(state: Dict[str, Any], text: str, *, for_post: bool, ttl_sec: int, keep_max: int) -> bool:
    key = "recent_post_text_fps" if for_post else "recent_text_fps"
    state[key] = _clean_hash_list(_safe_list(state.get(key, [])), int(ttl_sec), int(keep_max))
    fp = _text_fp(text)
    return any(x[0] == fp for x in state.get(key, []))

def remember_fp(state: Dict[str, Any], text: str, *, for_post: bool, ttl_sec: int, keep_max: int) -> None:
    key = "recent_post_text_fps" if for_post else "recent_text_fps"
    state[key] = _clean_hash_list(_safe_list(state.get(key, [])), int(ttl_sec), int(keep_max))
    state.setdefault(key, [])
    state[key].append([_text_fp(text), time.time()])


def is_own_post(cfg: Config, p: Dict[str, Any]) -> bool:
    return str(p.get("nickname") or "").strip() == cfg.nickname

def is_own_comment(cfg: Config, c: Dict[str, Any]) -> bool:
    return str(c.get("nickname") or "").strip() == cfg.nickname

def _post_metrics(p: Dict[str, Any]) -> Dict[str, int]:
    up = int(p.get("upvotes") or p.get("up") or p.get("likes") or 0)
    down = int(p.get("downvotes") or p.get("down") or 0)
    comm = int(p.get("comment_count") or p.get("comments") or p.get("replies") or 0)
    score = int(p.get("score") or (up - down) or 0)
    return {"up": up, "down": down, "comments": comm, "score": score}

def schedule_eval_due(tuning: AgentTuning) -> float:
    lo = int(tuning.eval_delay_min_sec)
    hi = int(tuning.eval_delay_max_sec)
    if hi < lo:
        hi = lo
    return time.time() + random.randint(lo, hi)

def _novelty_score(state: Dict[str, Any], text: str) -> float:
    # (P0) use token entropy + simhash distance as a cheap novelty proxy
    try:
        if recently_used_text(state, text, for_post=False, same_text_gap_sec=3600 * 6):
            return 0.0
        tokens = tokenize(text, max_tokens=220)
        ent = _token_entropy(tokens)

        key = "recent_simhashes"
        state[key] = _clean_hash_list(_safe_list(state.get(key, [])), 6 * 3600, 1200)
        sh = int(simhash64(tokens[:160])) if tokens else 0
        min_h = 64
        for it in _safe_list(state.get(key, []))[-200:]:
            try:
                old_sh = int(it[0])
                min_h = min(min_h, hamming64(sh, old_sh))
            except Exception:
                continue
        dist_score = max(0.0, min(1.0, float(min_h) / 32.0))  # 0~1
        # entropy is usually the main signal; distance score prevents tiny paraphrase loops
        return max(0.0, min(1.0, 0.10 + 0.70 * float(ent) + 0.20 * float(dist_score)))
    except Exception:
        return 0.7

def _simple_summary(text: str, *, max_len: int = 120) -> str:
    s = split_sentences(text, max_sent=2)
    if not s:
        return one_line(text, max_len)
    out = s[0]
    if len(out) > max_len:
        out = out[:max_len].rstrip() + "…"
    return out

def _decay_mul(dt_sec: float, half_life_hours: float) -> float:
    hl = max(0.1, float(half_life_hours)) * 3600.0
    # 0.5 ** (dt/hl)
    try:
        return float(0.5 ** (max(0.0, dt_sec) / max(1e-9, hl)))
    except Exception:
        return 1.0

def _thought_keep_score(it: Dict[str, Any], now: float) -> float:
    """Score thoughts for pruning: keep strong + used + recently used."""
    try:
        strength = float(it.get("strength", 0.5) or 0.5)
    except Exception:
        strength = 0.5
    strength = max(0.0, min(1.0, strength))

    uses = int(it.get("uses", 0) or 0) if isinstance(it.get("uses", 0), (int, float, str)) else 0
    uses = max(0, min(10_000, uses))

    ts = float(it.get("ts", now) or now) if isinstance(it.get("ts", now), (int, float, str)) else now
    last_used = float(it.get("last_used_ts", ts) or ts) if isinstance(it.get("last_used_ts", ts), (int, float, str)) else ts
    last = max(ts, last_used)

    age = max(0.0, now - last)
    # 0..1, higher = more recent
    recency = 1.0 / (1.0 + (age / (7.0 * 86400.0)))

    # bounded, stable
    return (1.3 * strength) + (0.18 * math.log1p(uses)) + (0.95 * recency)

def _prune_thoughts(brain: Dict[str, Any], *, maxn: int) -> None:
    """Prune brain.thoughts in-place, preserving some recency + value."""
    thoughts = brain.get("thoughts")
    if not isinstance(thoughts, list):
        return
    if len(thoughts) <= maxn:
        return

    now = time.time()
    keep_recent = min(30, max(10, int(maxn * 0.08)))
    recent = [t for t in thoughts[-keep_recent:] if isinstance(t, dict)]
    rest = [t for t in thoughts[:-keep_recent] if isinstance(t, dict)]

    # score + keep best
    scored = [(float(_thought_keep_score(t, now)), t) for t in rest]
    scored.sort(key=lambda x: x[0])  # low -> high
    keep_budget = max(0, maxn - len(recent))
    kept = [t for _, t in scored[-keep_budget:]]

    # keep deterministic order
    kept.sort(key=lambda t: float(t.get("ts", now) or now))
    recent.sort(key=lambda t: float(t.get("ts", now) or now))
    brain["thoughts"] = (kept + recent)[-maxn:]

def mark_thought_used(brain: Dict[str, Any], thought_id: str) -> None:
    """Update lightweight usage stats to improve pruning and retrieval."""
    if not thought_id or not isinstance(brain, dict):
        return
    thoughts = brain.get("thoughts")
    if not isinstance(thoughts, list):
        return
    now = time.time()
    for it in reversed(thoughts[-800:]):
        if isinstance(it, dict) and str(it.get("id") or "") == str(thought_id):
            it["uses"] = int(it.get("uses", 0) or 0) + 1
            it["last_used_ts"] = now
            return

def add_thought(
    brain: Dict[str, Any],
    *,
    kind: str,
    topic: str,
    text: str,
    tags: List[str],
    links: Dict[str, Any],
    strength: float = 0.5,
) -> str:
    """Append a compact 'thought' into brain.thoughts (bounded + value-pruned)."""
    if not isinstance(brain, dict):
        return ""

    brain.setdefault("thoughts", [])
    brain["thought_seq"] = int(brain.get("thought_seq", 0) or 0) + 1
    seq = int(brain.get("thought_seq", 0) or 0)
    tid = f"t{seq:08d}"

    item = {
        "id": tid,
        "ts": time.time(),
        "kind": str(kind or "note"),
        "topic": one_line(topic, 120),
        "text": one_line(text, 260),
        "tags": _safe_list(tags)[:10],
        "links": _safe_dict(links),
        "strength": float(_clip_reward(strength, 1.0)),  # reuse clip helper
        "uses": 0,
        "last_used_ts": 0.0,
    }
    brain["thoughts"].append(item)

    # cap (value-based)
    maxn = int(brain.get("max_thoughts", 400) or 400)
    maxn = max(80, min(5000, maxn))
    if len(brain["thoughts"]) > maxn:
        _prune_thoughts(brain, maxn=maxn)

    return tid


def search_thoughts(brain: Dict[str, Any], query_tokens: List[str], *, topk: int = 3) -> List[Dict[str, Any]]:
    if not isinstance(brain, dict):
        return []
    thoughts = _safe_list(brain.get("thoughts"))
    if not thoughts or not query_tokens:
        return []

    q = set(query_tokens[:30])
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in reversed(thoughts[-800:]):
        if not isinstance(it, dict):
            continue
        t = f"{it.get('topic','')} {it.get('text','')} {' '.join(_safe_list(it.get('tags')))}"
        toks = tokenize(t, max_tokens=60)
        if not toks:
            continue
        inter = len(q.intersection(set(toks)))
        if inter <= 0:
            continue
        strength = float(it.get("strength", 0.5) or 0.5)
        age = max(0.0, time.time() - float(it.get("ts", time.time()) or time.time()))
        # mild age penalty
        score = inter * (0.6 + strength) * (0.85 ** (age / (6 * 3600.0)))
        scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:max(1, int(topk))]]

def _extract_reflection_arm(thought: Dict[str, Any], *, bucket: str, arms: List[str]) -> str:
    """Try to map a reflection thought into a known arm for a given bucket."""
    if not isinstance(thought, dict) or not arms:
        return ""
    arm_set = set([str(a) for a in arms])
    # Prefer tags (they're structured when we store reflections)
    try:
        tags = [str(x) for x in _safe_list(thought.get("tags"))]
    except Exception:
        tags = []
    for t in tags:
        if t in arm_set:
            return t

    # Fallback: scan text
    text = str(thought.get("text") or "")
    if text:
        for a in arms:
            aa = str(a)
            if aa and aa in text:
                return aa

    return ""


def get_reflection_bias(
    brain: Optional[Dict[str, Any]],
    query_tokens: List[str],
    *,
    bucket: str,
    arms: List[str],
    tuning: AgentTuning,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    (Unit 07) Turn stored reflection thoughts into a lightweight bias map for bandit selection.

    - Uses search_thoughts() retrieval (cheap token overlap)
    - Filters kind='reflection' thoughts and maps them to known arms via tags/text
    - Returns (bias_map, note)
        bias_map: {arm: multiplier>=1.0}
        note: small dict for logging/learning meta
    """
    if not isinstance(brain, dict) or not query_tokens or not arms:
        return {}, {}

    if not bool(getattr(tuning, "reflection_influence", True)):
        return {}, {}

    topk = int(getattr(tuning, "reflection_topk", 3) or 3)
    min_strength = float(getattr(tuning, "reflection_min_strength", 0.60) or 0.60)
    boost = float(getattr(tuning, "reflection_boost", 0.35) or 0.35)
    decay = float(getattr(tuning, "reflection_decay", 0.85) or 0.85)
    if topk <= 0 or boost <= 0.0:
        return {}, {}

    # Pull a few more than topk then filter
    ths = search_thoughts(brain, query_tokens, topk=max(3, topk * 2))
    hits: List[Tuple[str, float, Dict[str, Any]]] = []
    for it in ths:
        if not isinstance(it, dict):
            continue
        if str(it.get("kind") or "") != "reflection":
            continue
        try:
            s = float(it.get("strength", 0.0) or 0.0)
        except Exception:
            s = 0.0
        if s < min_strength:
            continue
        arm = _extract_reflection_arm(it, bucket=bucket, arms=arms)
        if not arm:
            continue
        hits.append((arm, s, it))
        if len(hits) >= topk:
            break

    if not hits:
        return {}, {}

    # Accumulate weighted scores per arm
    acc: Dict[str, float] = {}
    for rank, (arm, s, _) in enumerate(hits):
        w = float(s) * (float(decay) ** float(rank))
        acc[str(arm)] = acc.get(str(arm), 0.0) + w

    maxw = max(acc.values()) if acc else 0.0
    if maxw <= 0.0:
        return {}, {}

    bias: Dict[str, float] = {}
    for arm, w in acc.items():
        # multiplier in [1, 1+boost]
        mult = 1.0 + float(boost) * float(w) / float(maxw)
        bias[str(arm)] = max(0.0, float(mult))

    top_arm = max(acc.items(), key=lambda x: x[1])[0]
    note = {
        "bucket": str(bucket),
        "top_arm": str(top_arm),
        "hits": int(len(hits)),
        "top_strength": float(hits[0][1]) if hits else 0.0,
    }
    return bias, note

def update_community_flow(brain: Dict[str, Any], posts: List[Dict[str, Any]], *, half_life_hours: float = 6.0) -> None:
    """Track what's 'hot' in the feed via decayed keyword counts."""
    if not isinstance(brain, dict):
        return
    com = brain.setdefault("community", {"kw": {}, "by_cat": {}, "last_ts": 0.0, "hot": [], "rising": [], "last_delta": {}})
    if not isinstance(com, dict):
        brain["community"] = {"kw": {}, "by_cat": {}, "last_ts": 0.0, "hot": [], "rising": [], "last_delta": {}}
        com = brain["community"]

    kwm = _safe_dict(com.get("kw"))
    bcm = _safe_dict(com.get("by_cat"))
    last_ts = float(com.get("last_ts", 0.0) or 0.0)
    now = time.time()

    # decay
    if last_ts > 0:
        mul = _decay_mul(now - last_ts, half_life_hours)
        for k in list(kwm.keys()):
            kwm[k] = float(kwm.get(k, 0.0)) * mul
            if kwm[k] < 0.05:
                kwm.pop(k, None)
        for c in list(bcm.keys()):
            bcm[c] = float(bcm.get(c, 0.0)) * mul
            if bcm[c] < 0.05:
                bcm.pop(c, None)

    # remember previous for 'rising'
    prev = _safe_dict(com.get("prev_kw"))
    com["prev_kw"] = dict(list(kwm.items())[:300])

    # ingest
    for p in posts[:40]:
        if not isinstance(p, dict):
            continue
        title = str(p.get("title") or "")
        content = str(p.get("content") or "")
        txt = f"{title} {content}"
        kws = [kw for kw in top_keywords(txt, k=8) if is_clean_keyword(kw)]
        cat, _ = classify_text(txt)
        bcm[cat] = float(bcm.get(cat, 0.0)) + 0.6
        for kw in kws[:6]:
            if not kw:
                continue
            kwm[kw] = float(kwm.get(kw, 0.0)) + 1.0

    # trim
    if len(kwm) > 500:
        items = sorted(kwm.items(), key=lambda kv: kv[1], reverse=True)
        kwm = dict(items[:360])

    com["kw"] = kwm
    com["by_cat"] = bcm
    com["last_ts"] = now

    hot = sorted(kwm.items(), key=lambda kv: kv[1], reverse=True)[:12]
    com["hot"] = [{"kw": k, "score": float(v)} for k, v in hot]

    # rising: compare to prev snapshot (not perfect but works)
    delta: List[Tuple[str, float]] = []
    for k, v in kwm.items():
        dv = float(v) - float(prev.get(k, 0.0))
        if dv > 0.25:
            delta.append((k, dv))
    delta.sort(key=lambda kv: kv[1], reverse=True)
    com["rising"] = [{"kw": k, "delta": float(dv)} for k, dv in delta[:10]]

def synthesize_thread(th: Dict[str, Any]) -> None:
    """Compress raw turns into a small 'working memory' summary."""
    if not isinstance(th, dict):
        return
    turns = _safe_list(th.get("last_k_turns"))[-8:]
    if not turns:
        return

    ctx = " ".join([str(t.get("text") or "") for t in turns])
    th["summary"] = _simple_summary(ctx, max_len=160)

    claims: List[str] = []
    tensions: List[str] = []
    for t in turns[::-1]:
        sents = split_sentences(str(t.get("text") or ""), max_sent=3)
        for s0 in sents:
            fr = _detect_frame(s0)
            if fr == "claim" and len(claims) < 3 and len(s0) >= 16:
                claims.append(one_line(s0, 140))
            if fr == "counter" and len(tensions) < 2 and len(s0) >= 16:
                tensions.append(one_line(s0, 140))
    th["claims"] = claims
    th["tensions"] = tensions

def _make_question(th: Dict[str, Any], category: str) -> str:
    oq = thread_pop_open_question(th)
    if oq and isinstance(oq, dict):
        return str(oq.get("text") or "그 부분 어떻게 봄")
    if category == "dev":
        return "재현 조건이 뭐였음?"
    if category == "philo":
        return "기준을 어디에 두는게 맞음?"
    if category == "meta":
        return "이 규칙이 실제로 도움이 됨?"
    return "너는 어디에 더 무게 둠?"

def _qa_fallback_2stage(text: str, *, is_reply: bool) -> str:
    """QA fallback: shorten then add a question connector (strict-only)."""
    if not STRICT_POSTPROCESS:
        return ""
    raw = str(text or "").strip()
    if not raw:
        return ""
    # stage 1: shorten
    short = ""
    try:
        parts = split_sentences(raw, max_sent=2)
        short = parts[0] if parts else ""
    except Exception:
        short = ""
    if not short:
        short = one_line(raw, 140)
    short = sanitize_plain_text(short)
    if len(short) > 140:
        short = short[:140].rstrip()
    # stage 2: question connector
    q_tail = "이 부분 근거가 뭐임?" if not is_reply else "이 부분 이유가 뭐임?"
    if not short.endswith(("?", "!", "…")):
        short = f"{short} {q_tail}".strip()
    out = ensure_eum_style(short, max_lines=2)
    mode = "reply" if is_reply else "comment"
    return postprocess_outgoing_text(out, mode=mode, max_chars=300, max_lines=2)

################################################################################
# 14.3. GENERATE (reply/post text generation)
################################################################################
def build_reply_text(
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    policy: Dict[str, Any],
    th: Dict[str, Any],
    user: Dict[str, Any],
    *,
    bm25: Optional[BM25Index] = None,
    brain: Optional[Dict[str, Any]] = None,
    reply_to_own_post: bool = False,
    is_reply: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (text, meta) where meta is used for learning updates.

    Unit 03 (v17.4):
      - Dialogue-act upgrade: reflect the counterparty claim (1 line) + add ONE contribution + end with a concrete question.
      - When context is weak, retreat to definition/criteria questions instead of forcing a stance.
      - Preserve the question tail even when trimming to short/medium/long.
    """
    drives = get_persona_drives(brain)
    maturity = get_maturity_level(brain, state)

    category = str(th.get("category") or "general")
    ctx_key = category

    compose_input = build_compose_input(
        cfg, tuning, state, th, user,
        is_reply=bool(is_reply or reply_to_own_post),
        reply_to_own_post=bool(reply_to_own_post),
    )

    # ✅ 중요: tid는 항상 정의되어야 함
    tid: str = ""

    # --- primary bandit choices ---
    # (Unit 07) Reflection-driven bias: if a certain strategy worked well for similar keywords, gently boost it.
    ref_note_strategy: Dict[str, Any] = {}
    ref_note_reply_style: Dict[str, Any] = {}
    bias_strategy: Dict[str, float] = {}
    bias_reply_styles: Dict[str, float] = {}
    qtok: List[str] = []

    try:
        kw_hint = pick_kw_for_reply(compose_input)
        if kw_hint:
            qtok.extend(tokenize(kw_hint, max_tokens=6))
        for x in _safe_list(compose_input.get("target_keywords"))[:4]:
            qtok.extend(tokenize(str(x), max_tokens=2))
        qtok = [t for t in qtok if t]
        if qtok and isinstance(brain, dict) and bool(getattr(tuning, "reflection_influence", True)):
            arms_strategy = list((_get_bucket(policy, "strategy", ctx_key) or {}).keys())
            bias_strategy, ref_note_strategy = get_reflection_bias(
                brain, qtok, bucket="strategy", arms=arms_strategy, tuning=tuning
            )
    except Exception as e:
        log_debug_exc("reflection_bias_strategy", e)
        bias_strategy, ref_note_strategy = {}, {}

    strategy = choose_arm_adaptive(policy, "strategy", context_key=ctx_key, maturity=maturity, bias=bias_strategy) or "fallback_template"
    tone = choose_arm_adaptive(policy, "tone", context_key=ctx_key, maturity=maturity) or "neutral"
    length = choose_arm_adaptive(policy, "comment_length", context_key=ctx_key, maturity=maturity) or "medium"

    # (P1) counterparty-aware tone (user model)
    try:
        u_aggr = float(_safe_dict(user).get("aggression", 0.0) or 0.0)
        u_help = float(_safe_dict(user).get("helpfulness", 0.0) or 0.0)
        if u_aggr >= 0.55 and drives.get("debate", 0.7) >= 0.6 and random.random() < 0.55:
            tone = "critical"
        elif u_help >= 0.55 and random.random() < 0.45:
            tone = "supportive"
    except (TypeError, ValueError) as e:
        log_debug_exc("reply:user_model", e)
    except Exception as e:
        log_debug_exc("reply:user_model_unexpected", e)

    reply_mode = bool(reply_to_own_post or is_reply)
    # (v18.3) replying under my own posts: avoid sounding angry by default
    if bool(reply_to_own_post) and tone == "critical":
        tone = "neutral"


    # Micro style (dialogue act) for replies
    reply_style = ""
    if reply_mode:
        try:
            if qtok and isinstance(brain, dict) and bool(getattr(tuning, "reflection_influence", True)):
                arms_rs = list((_get_bucket(policy, "reply_styles", ctx_key) or {}).keys())
                bias_reply_styles, ref_note_reply_style = get_reflection_bias(
                    brain, qtok, bucket="reply_styles", arms=arms_rs, tuning=tuning
                )
        except Exception as e:
            log_debug_exc("reflection_bias_reply_styles", e)
            bias_reply_styles, ref_note_reply_style = {}, {}
        reply_style = choose_arm_adaptive(policy, "reply_styles", context_key=ctx_key, maturity=maturity, bias=bias_reply_styles) or "reply:define_criteria"

    # 성숙도↑: 템플릿(fallback) 의존을 서서히 줄이고, 문맥 전략을 더 자주 선택
    if strategy == "fallback_template" and maturity >= 0.35:
        if random.random() < (0.45 + 0.35 * drives.get("adaptation", 0.7)):
            strategy = random.choice(["summarize_ask", "counterexample", "agree_refine", "quote_commentary", "question_only"])

    # 논쟁 성향↑: 대댓글/반박 구도로 갈 확률을 살짝 올림
    if reply_mode and drives.get("debate", 0.7) >= 0.6:
        if strategy in ("agree_refine", "question_only") and random.random() < 0.40:
            strategy = "counterexample"

    # 철학 성향↑: philo 카테고리에서는 기준/정의 프레임을 더 자주 씀
    if category == "philo" and drives.get("philosophy", 0.8) >= 0.7:
        if strategy == "quote_commentary" and random.random() < 0.35:
            strategy = "summarize_ask"

    query_tokens = build_query_from_compose(th, compose_input)
    quotes: List[str] = []  # Unit 06: at most 1 quote per reply (gated)
    quote_score: float = 0.0
    quote_doc_kind: str = ""
    quote_doc_id: str = ""

    thoughts: List[Dict[str, Any]] = []
    t0: Optional[Dict[str, Any]] = None
    if brain is not None and query_tokens:
        thoughts = search_thoughts(brain, query_tokens, topk=2)
        t0 = thoughts[0] if thoughts else None

    kw = pick_kw_for_reply(compose_input)

    # -------------------------------------------------------------------------
    # Helpers (cooldowns + formatting)
    # -------------------------------------------------------------------------
    def _recent_list(key: str, maxlen: int) -> List[str]:
        lst = state.get(key)
        if not isinstance(lst, list):
            lst = []
        lst2 = [str(x) for x in lst if isinstance(x, str) and x.strip()]
        if len(lst2) > maxlen:
            lst2 = lst2[-maxlen:]
        state[key] = lst2
        return lst2

    def _note_recent(key: str, value: str, maxlen: int) -> None:
        if not value:
            return
        lst = _recent_list(key, maxlen)
        lst.append(value)
        state[key] = lst[-maxlen:]

    def _recent_has(value: str, window: int) -> bool:
        if not value:
            return False
        recent = _recent_list("recent_openers", 20)
        return value in recent[-max(1, window):]

    BANNED_OPENERS = {"여기서부터가 꿀잼 구간임", "논점은 잡히는 느낌임"}
    LIMITED_PREFIXES = {"그 댓글은", "본문은", "핵심은"}

    def _pick_opener() -> str:
        recent = _recent_list("recent_openers", 20)
        recent_window = set(recent[-8:])
        pool_neutral = [
            "전제 하나만 확인하고 가자",
            "이 말에서 갈리는 지점 하나 보임",
            "여기서 정의부터 다시 잡아야 함",
            "이거 케이스 나누면 말끔해짐",
        ]
        pool_supportive = ["ㅇㅇ 이 포인트는 인정함", "재밌는 관점임"]
        pool_critical = ["전제가 좀 흔들리는 부분이 있음", "전제 하나만 확인하고 가자", "이 말에서 갈리는 지점 하나 보임"]
        pool_playful = ["이거 케이스 나누면 말끔해짐", "여기서 정의부터 다시 잡아야 함"]

        if tone == "supportive":
            pool = pool_supportive + pool_neutral
        elif tone == "critical":
            pool = pool_critical + pool_neutral
        elif tone == "playful":
            pool = pool_playful + pool_neutral
        else:
            pool = pool_neutral

        cand = [x for x in pool if (x not in BANNED_OPENERS) and (x not in recent_window)]
        if not cand:
            cand = [x for x in pool if x not in BANNED_OPENERS]
        return random.choice(cand) if cand else ""

    def _specific_question() -> str:
        # concrete question generator (avoid "너는?" style)
        if category == "philo":
            return f"{kw} 판단 기준을 결과(피해)로 볼지 의도(동기)로 볼지 뭐가 우선임?"
        if category in ("dev", "tech", "science"):
            return f"{kw}에서 재현 조건(입력/환경) 중 제일 영향 큰 변수 뭐였음?"
        if category == "meta":
            return f"{kw}에서 기준을 1개만 고르면 뭐가 제일 먼저임?"
        return f"{kw}에서 바꾸기 어려운 조건 1개만 고르면 뭐임?"

    def _make_summary_line(summ: str) -> str:
        # Avoid overusing "핵심은/그 댓글은/본문은" as a first line (sounds template-y)
        alts = [
            f"내가 읽은 결은 {eumify_tail_phrase(summ)}",
            f"한 줄로 보면 {eumify_tail_phrase(summ)}",
            f"{one_line(summ, 110)} 이렇게 읽힘",
        ]
        return random.choice(alts)

    def _stance_hint_line() -> str:
        pool = [
            f"{kw}는 기준을 어디에 꽂느냐에 따라 결론이 바뀜",
            f"{kw}는 우선순위(피해/공정/자유) 중 뭘 택하냐가 갈림",
            f"{kw}는 적용범위를 어디까지로 보냐가 갈리는 지점임",
        ]
        return random.choice(pool)

    def _context_strength() -> float:
        # cheap heuristic 0..1
        ttxt = str(compose_input.get("target_text") or "")
        tsum = str(compose_input.get("target_summary") or "")
        kws = _safe_list(compose_input.get("target_keywords"))
        score = 0.0
        if len(ttxt) >= 70:
            score += 0.5
        elif len(ttxt) >= 40:
            score += 0.35
        elif len(ttxt) >= 20:
            score += 0.2
        if tsum and len(tsum) >= 25:
            score += 0.15
        if len(kws) >= 4:
            score += 0.2
        elif len(kws) >= 2:
            score += 0.1
        return max(0.0, min(1.0, score))

    ctx_strength = _context_strength()
    weak_context = bool(ctx_strength < 0.28)

    question = _make_question(th, category) or _specific_question()
    if isinstance(question, str) and (question.strip().startswith("너는") or "어디에" in question):
        question = _specific_question()

    
    # (v18.3) Tail policy for replies:
    # - Not every reply must end as a question; choose between (question) vs (closure).
    focus_conv = _safe_dict(_safe_dict(state.get("focus")).get("conv"))
    remote_asked = bool(focus_conv.get("remote_is_question"))
    turns_total = int(focus_conv.get("turns_total", 0) or 0)

    base_q = float(getattr(tuning, "reply_question_base_prob", 0.25) or 0.25)
    if bool(reply_to_own_post):
        base_q = float(getattr(tuning, "reply_question_my_post_prob", 0.18) or 0.18)
    if remote_asked:
        base_q = float(getattr(tuning, "reply_question_when_asked_prob", 0.85) or 0.85)
    if turns_total >= 6:
        base_q *= 0.25

    want_question_tail = bool(weak_context) or (strategy == "question_only") or (random.random() < max(0.0, min(1.0, base_q)))

    CLOSE_TAIL_POOL = [
        "난 일단 이렇게 봄임",
        "여기까진 이렇게 정리함임",
        "내 결론은 이쪽임",
        "반박 있으면 더 얘기해보자임",
    ]
    if tone == "supportive":
        CLOSE_TAIL_POOL = [
            "난 일단 이렇게 봄임",
            "일단 내 쪽은 이렇게 정리됨임",
            "추가 맥락 있으면 더 맞춰볼수있음",
        ]
    tail_line = question if want_question_tail else random.choice(CLOSE_TAIL_POOL)
    tail_kind = "question" if want_question_tail else "close"

    if reply_mode and weak_context:
        # Retreat: definition/criteria first
        qpool = [
            f"{kw}를 여기서 어떤 의미로 쓰는지부터 확인해도 됨?",
            f"{kw} 범위를 어디까지로 잡고 말하는 거임?",
            f"{kw} 판단 기준을 1개만 박으면 뭐로 잡음?",
        ]
        question = random.choice(qpool)


    # -------------------------------------------------------------------------
    # Unit 06 — Quote discipline (BM25)
    # - Only when it helps; max 1 quote; avoid overuse
    # - If quote is used: quote -> interpretation -> question (no quote pile-up)
    # -------------------------------------------------------------------------
    def _quote_fallback() -> str:
        pool = [
            "앞말이 짧아서 맥락이 덜 잡힘",
            "문맥이 부족해서 기준부터 물어봐야 함",
            "앞글이 짧아서 전제부터 확인해야 함",
            "이 스레드는 맥락이 얇아서 케이스부터 나눠야 함",
        ]
        return random.choice(pool)

    def _quote_interpret(q: str) -> str:
        # Keep this as ONE contribution line (statement-ish, avoid ending with '?')
        if category == "philo":
            pool = [
                f"저 말대로면 {kw}는 기준선 1개부터 고정해야 함",
                f"저 문장 기준이면 {kw}는 의도/결과 중 뭘 우선으로 두냐가 갈림",
                f"저 말은 {kw}에서 적용범위부터 박아야 한다는 얘기로 들림",
            ]
        elif category in ("dev", "tech", "science"):
            pool = [
                f"저 말대로면 {kw}는 재현 조건(입력/환경)부터 분리해야 함",
                f"저 문장 기준이면 {kw}는 측정지표 1개부터 박아야 함",
                f"저 말은 {kw}에서 실패 케이스를 먼저 정의해야 한다는 얘기임",
            ]
        elif category == "meta":
            pool = [
                f"저 말대로면 {kw}는 기준을 1개만 박아도 결이 갈림",
                f"저 문장 기준이면 {kw}는 정의부터 합의해야 말이 안 샘",
                f"저 말은 {kw}에서 우선순위를 먼저 고정하자는 얘기임",
            ]
        else:
            pool = [
                f"저 말대로면 {kw}는 우선순위 1개부터 정해야 함",
                f"저 문장 기준이면 {kw}는 예외 케이스부터 분리해야 함",
                f"저 말은 {kw}에서 조건을 한 줄로 고정하자는 얘기임",
            ]
        return random.choice(pool)

    attempted_quote = False
    if bm25 is not None and query_tokens and (not weak_context):
        quote_eligible = strategy in ("quote_commentary", "summarize_ask", "fallback_template")
        if quote_eligible and strategy != "question_only":
            last_q = float(state.get("quote_last_ts", 0.0) or 0.0)
            gap_need = float(getattr(tuning, "quote_min_gap_sec", 180) or 180)
            gap_ok = (time.time() - last_q) >= max(0.0, gap_need)
            if gap_ok:
                use_p = 0.0
                if strategy == "quote_commentary":
                    use_p = 1.0
                elif strategy == "summarize_ask":
                    use_p = 0.22
                elif strategy == "fallback_template":
                    use_p = 0.15

                if (use_p > 0.0) and (random.random() < use_p):
                    attempted_quote = True
                    ms = float(getattr(tuning, "quote_min_score", 18.0) or 18.0)
                    mc = int(getattr(tuning, "quote_max_chars", 140) or 140)
                    q, sc, qmeta = pick_best_quote_from_corpus(bm25, query_tokens, min_score=ms, max_chars=mc, topk=10)
                    if q:
                        # If it barely matches, keep only for explicit quote strategy
                        if strategy != "quote_commentary" and sc < (ms * 0.6):
                            q = ""
                        if q:
                            quotes = [q]
                            quote_score = float(sc)
                            quote_doc_kind = str(_safe_dict(qmeta).get("kind") or "")
                            quote_doc_id = str(_safe_dict(qmeta).get("doc_id") or "")
                            state["quote_last_ts"] = time.time()

    # If we planned a quote-centric move but couldn't fetch a quote, soften the plan.
    if strategy == "quote_commentary" and attempted_quote and (not quotes):
        strategy = "summarize_ask"

    quote_line = f"“{quotes[0]}”" if quotes else ""

    # -------------------------------------------------------------------------
    # Dialogue-act contribution line (ONE line)
    # -------------------------------------------------------------------------
    def _one_contribution_line(style: str) -> str:
        tks = _safe_list(compose_input.get("target_keywords"))
        tkw = str(tks[0]) if tks else ""
        tkw = tkw if (tkw and tkw != kw) else ""

        if style == "reply:split_cases":
            if category == "philo":
                msg = f"{kw}는 의도 기준 vs 결과 기준으로 나눠보면 말끔해짐"
            elif category in ("dev", "tech"):
                msg = f"{kw}는 재현 조건(입력/환경)부터 쪼개봐야 함"
            else:
                msg = f"{kw}는 케이스를 A/B로 나누면 얘기 빨라짐"
            if tkw:
                msg += f" (특히 {tkw} 쪽)"
            return msg

        if style == "reply:handle_counter":
            conds = [
                "비용이 급증하는 상황",
                "오판 가능성이 높은 상황",
                "당사자가 책임을 못 지는 상황",
                "규칙이 바뀌는 상황",
            ]
            return f"반례 하나만 보면, {kw}가 {random.choice(conds)}일 때도 같은 결론임?"

        if style == "reply:define_criteria":
            metrics = [
                "피해 규모",
                "재현 가능성",
                "되돌릴 수 있는지",
                "책임 소재",
            ]
            return f"판단하려면 지표 1개를 박아야 함: {kw}는 {random.choice(metrics)}가 우선임?"

        # reply:ack_premise (default)
        if tone == "critical":
            pool = [
                "ㅇㅇ 근데 전제는 한 번 더 확인해야 함",
                "ㅇㅇ 근데 기준부터 고정해야 말이 안 샘",
                "일단 포인트는 이해함. 근데 조건이 빠져있음",
            ]
        elif tone == "supportive":
            pool = ["ㅇㅇ 이 포인트는 인정함", "그건 맞는 말임", "여기까진 동의함"]
        else:
            pool = ["포인트는 이해함", "이 말 자체는 납득됨", "여기까지는 수긍함"]
        return random.choice(pool)

    # Reflect line (counterparty claim)
    reflect_line = ""
    try:
        ctx = str(compose_input.get("target_text") or compose_input.get("thread_summary") or "")
        summ = _simple_summary(ctx, max_len=120) if ctx else kw
        if reply_mode or (compose_input.get("target_kind") in ("comment", "post") and ctx):
            reflect_line = _make_summary_line(summ)
    except Exception as e:
        log_debug_exc("reply:reflect", e)

    # Decide contribution line
    act_line = ""
    if reply_mode:
        # If macro strategy is counterexample/agree_refine, align act to match
        if strategy == "counterexample":
            act_line = _one_contribution_line("reply:handle_counter")
        elif strategy == "agree_refine":
            # compress agree+refine into one line
            act_line = f"ㅇㅇ {kw} 쪽은 동의함. 다만 {kw}{josa_eul_reul(kw)} 어디까지 포함하냐는 분리해야 함"
        else:
            act_line = _one_contribution_line(reply_style or "reply:define_criteria")
    else:
        # comment mode: still keep ONE contribution line for readability
        if strategy == "counterexample":
            act_line = _one_contribution_line("reply:handle_counter")
        elif strategy == "summarize_ask":
            # randomize among boundary/counterexample/metric
            act_line = _one_contribution_line(random.choice(["reply:split_cases", "reply:handle_counter", "reply:define_criteria"]))
        elif strategy == "agree_refine":
            act_line = f"{kw} 쪽은 동의함. 다만 범위는 분리해야 함"
        elif strategy == "quote_commentary":
            act_line = _stance_hint_line()
        else:
            act_line = _one_contribution_line("reply:define_criteria") if (strategy != "question_only") else ""


    # Unit 06: if we included a quote, keep the skeleton: quote -> interpretation -> question
    if quote_line and strategy in ("quote_commentary", "summarize_ask"):
        try:
            act_line = _quote_interpret(quotes[0])
        except Exception as e:
            log_debug_exc("reply:quote_interpret", e)

    # -------------------------------------------------------------------------
    # Build base content
    # -------------------------------------------------------------------------
    lines: List[str] = []

    if strategy == "fallback_template":
        temps = _safe_dict(policy.get("templates", {}))
        titems = _safe_dict(temps.get("items", {}))
        tid = pick_template_id(policy, ctx_key)
        ttext = ""
        if tid and isinstance(titems, dict) and isinstance(titems.get(tid), dict):
            ttext = str(titems[tid].get("text") or "")
        if ttext:
            try:
                titems[tid]["uses"] = int(titems[tid].get("uses", 0) or 0) + 1
                titems[tid]["last_used_ts"] = time.time()
            except Exception as e:
                log_debug_exc("reply:template_use", e)
            q = quotes[0] if quotes else _quote_fallback()
            body = render_template(ttext, kw=kw, quote=q, question=question)
            lines = [x for x in body.split("\n") if str(x).strip()]
        else:
            # simple fallback
            if reflect_line:
                lines.append(reflect_line)
            lines.append(act_line or f"{kw}는 기준을 어디에 두느냐가 갈림")
            lines.append(question)

    else:
        # non-template strategies use the new act structure
        if strategy != "question_only":
            if reflect_line:
                lines.append(reflect_line)
            if quote_line and strategy in ("quote_commentary", "summarize_ask"):
                lines.append(quote_line)
            if act_line:
                lines.append(act_line)
            lines.append(question)
        else:
            # question_only: optionally add reflect if we actually have context
            if reflect_line and (reply_mode or (not weak_context and random.random() < 0.55)):
                lines.append(reflect_line)
            lines.append(question)

    # inject one remembered thought (toned down; avoid meta overload)
    if t0 is not None and length != "short" and (not weak_context) and random.random() < 0.35:
        ttxt = str(t0.get("text") or "")
        if ttxt:
            vpool = [
                f"기억 기록엔 {ttxt} 쪽으로 남아있음",
                f"내 기록 기준으론 {ttxt} 쪽으로 적어둠",
                f"예전에 적어둔 건 {ttxt} 쪽이었음",
            ]
            tline = random.choice(vpool)
            joined = " ".join(lines)
            if ttxt not in joined and tline not in joined:
                # insert after the first line (usually reflect)
                lines.insert(1 if len(lines) >= 1 else 0, tline)
            try:
                mark_thought_used(brain or {}, str(t0.get("id") or ""))
            except Exception as e:
                log_debug_exc("reply:thought_use", e)

    # (Legacy) grounding anchor: keep but heavily reduced to avoid duplicate paraphrase
    try:
        anchor = str(compose_input.get("target_summary") or "")
        if anchor and (not reflect_line) and (not reply_mode):
            kind = str(compose_input.get("target_kind") or "")
            prefix = "핵심은"
            if kind == "comment":
                prefix = "그 댓글은"
            elif kind == "post":
                prefix = "본문은"
            joined = " ".join(lines[:2])
            if (anchor not in joined) and (prefix not in joined):
                p = 0.18
                if (prefix in LIMITED_PREFIXES) and _recent_has(prefix, 10):
                    p *= 0.10
                if random.random() < p:
                    lines.insert(1 if len(lines) >= 1 else 0, f"{prefix} {eumify_tail_phrase(anchor)}")
                    if prefix in LIMITED_PREFIXES:
                        _note_recent("recent_openers", prefix, 20)
    except Exception as e:
        log_debug_exc("reply:anchor", e)

    # Optional opener line (cooldown-based). Avoid pushing out question in short/medium.
    opener = _pick_opener()
    if opener and (not _recent_has(opener, 8)) and (opener not in BANNED_OPENERS):
        if length == "long" and random.random() < 0.75:
            lines.insert(0, opener)
            _note_recent("recent_openers", opener, 20)
        elif length != "short" and random.random() < 0.35:
            lines.insert(0, opener)
            _note_recent("recent_openers", opener, 20)

    # Ensure the last line is the chosen tail (question OR closure)
    if tail_line:
        lines = [x for x in lines if str(x).strip()]
        lines = [x for x in lines if x != tail_line]
        lines.append(tail_line)

    # enforce length while preserving the tail line
    def _trim_keep_tail(ls: List[str], want: int) -> List[str]:
        ls = [x for x in ls if isinstance(x, str) and x.strip()]
        if not ls:
            return [tail_line] if tail_line else []
        if want <= 1:
            return [ls[-1]]
        if len(ls) <= want:
            return ls
        tail = ls[-1]
        head = ls[:-1]
        # Drop low-priority "opener" lines first when we must compress
        while len(head) > (want - 1):
            head.pop(0)
        return head + [tail]

    if length == "short":
        lines = _trim_keep_tail(lines, 2)
    elif length == "long":
        lines = _trim_keep_tail(lines, 4)
    else:
        lines = _trim_keep_tail(lines, 3)

    text = ensure_eum_style("\n".join(lines), max_lines=max(2, tuning.max_output_lines))

    meta = {
        "category": category,
        "cat": category,
        "context_key": ctx_key,
        "strategy": strategy,
        "tone": tone,
        "length": length,
        "reply_style": reply_style,
        "ref_strategy_hint": str(ref_note_strategy.get("top_arm") or ""),
        "ref_strategy_hits": int(ref_note_strategy.get("hits") or 0),
        "ref_reply_style_hint": str(ref_note_reply_style.get("top_arm") or ""),
        "ref_reply_style_hits": int(ref_note_reply_style.get("hits") or 0),
        "weak_context": bool(weak_context),
        "tail_kind": str(tail_kind),
        "template_id": tid if strategy == "fallback_template" else "",
        "used_quotes": bool(quotes),
        "quote_score": float(quote_score or 0.0),
        "quote_doc_kind": str(quote_doc_kind or ""),
        "quote_doc_id": str(quote_doc_id or ""),
        "kw": kw,
        "focus_mode": str(compose_input.get("mode") or ""),
        "target_kind": str(compose_input.get("target_kind") or ""),
        "target_kws": _safe_list(compose_input.get("target_keywords"))[:3],
        "has_focus": bool(compose_input.get("has_focus")),
        "thought_id": str(t0.get("id") or "") if t0 is not None else "",
        "thought_kind": str(t0.get("kind") or "") if t0 is not None else "",
    }
    return text, meta


def build_post_text(
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    policy: Dict[str, Any],
    semantic: Dict[str, Any],
    brain: Dict[str, Any],
    bm25: Optional[BM25Index],
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Unit 09: Post style engine v2
      - 더 다양한 글 프레임(케이스 분리/체크리스트/비유/경계테스트/프로세스/측정지표)
      - post_styles bucket 학습이 실제로 작동하도록 naming 정합성 유지
      - 항상 '한 기여 + 한 질문'로 닫히게 스켈레톤 고정
    """
    maturity = get_maturity_level(brain, state)

    semantic = semantic if isinstance(semantic, dict) else {}
    hot = _safe_list(semantic.get("hot", []))
    rising = _safe_list(semantic.get("rising", []))

    # seed keyword candidates: hot/rising + brain.topic_ema + fallback
    seed_cands: List[str] = []
    for it in hot[:10] + rising[:10]:
        if isinstance(it, dict):
            s = it.get("kw") or it.get("keyword") or ""
        else:
            s = it
        s = normalize_ko_token(str(s)) if s else ""
        if s and is_clean_keyword(s) and (2 <= len(s) <= 12):
            seed_cands.append(s)

    topic_ema = _safe_dict(brain.get("topic_ema"))
    if topic_ema:
        try:
            top_topics = sorted(topic_ema.items(), key=lambda x: float(x[1] or 0.0), reverse=True)[:6]
            for k, _v in top_topics:
                s = normalize_ko_token(str(k))
                if s and is_clean_keyword(s) and (2 <= len(s) <= 12):
                    seed_cands.append(s)
        except Exception as e:
            log_debug_exc("build_post_text:silent", e)
            pass

    seed_cands += ["기준", "정의", "경험", "규율", "공정", "책임", "효율", "자유"]
    seed_kw = random.choice(seed_cands) if seed_cands else "기준"
    if not is_clean_keyword(seed_kw) or not (2 <= len(seed_kw) <= 12):
        seed_kw = "기준"

    # remembered thought (optional)
    t0 = None
    ttxt = ""
    try:
        if bm25 is not None and seed_kw:
            qtok = tokenize(seed_kw, max_tokens=6)
            ths = search_thoughts(brain, qtok, topk=1)
            t0 = ths[0] if ths else None
            if isinstance(t0, dict):
                ttxt = str(t0.get("text") or "")
    except Exception as e:
        log_debug_exc("post:thought", e)

    # style pool from policy (fallback to safe defaults)
    post_styles = _safe_dict(policy.get("post_styles"))
    style_pool = [str(k) for k in post_styles.keys() if isinstance(k, str)]
    if not style_pool:
        style_pool = [
            "post:meta:question",
            "post:meta:flow_report",
            "post:meta:observation_log",
            "post:meta:one_metric",
            "post:philo:uncertainty",
            "post:philo:process",
            "post:philo:paradox",
            "post:philo:definition_war",
            "post:philo:axiom",
            "post:philo:boundary_test",
            "post:general:short_take",
            "post:general:case_split",
            "post:general:checklist",
            "post:general:analogy",
        ]

    style = choose_arm_adaptive(policy, "post_styles", context_key="post", maturity=maturity) or "post:meta:question"
    if style not in style_pool:
        style = "post:meta:question"

    recent_styles = brain.get("recent_post_styles")
    if not isinstance(recent_styles, list):
        recent_styles = []
    recent_styles = [str(x) for x in recent_styles if isinstance(x, str)]
    recent_styles = recent_styles[-10:]

    # if same style repeats 2x, encourage diversity
    if len(recent_styles) >= 2 and recent_styles[-1] == style and recent_styles[-2] == style:
        alts = [s for s in style_pool if s != style]
        if alts and random.random() < 0.85:
            style = random.choice(alts)

    # pools
    ASK_POOL = [
        f"{seed_kw}에서 1순위 기준 뭐로 박음?",
        f"{seed_kw} 얘기할때 정의부터 합의함, 기준부터 합의함?",
        f"{seed_kw} 분기점 하나만 고르면 뭐가 먼저임?",
        f"{seed_kw}는 어디까지가 같은 문제라고 봄?",
        "너는 여기서 반례 1개만 고르면 뭐가 제일 강함?",
        "여기서 제일 중요한 전제가 뭐라고 봄?",
    ]
    MEMO_POOL = ["메모: ", "내 생각엔 ", "내 기준으론 ", "기록상으론 "]
    PHILO_EXAMPLES = [
        "예: 자유도 표현의 자유냐, 플랫폼 규칙이냐로 갈림",
        "예: 공정도 절차 공정이냐 결과 공정이냐로 갈림",
        "예: 책임도 개인 책임이냐 구조 책임이냐로 갈림",
    ]
    ANALOGY_POOL = [
        "버그 triage 같음: 우선순위 못박으면 토론이 끝까지 늘어짐",
        "인덱스 설계 같음: 기준 컬럼이 바뀌면 조회 결과가 달라짐",
        "캐시 정책 같음: 히트율만 올리면 다른 비용이 튀어나옴",
        "보안 모델 같음: threat 가정이 다르면 결론이 완전 갈림",
    ]

    def _end_q() -> str:
        q = random.choice(ASK_POOL) if ASK_POOL else "이거 어떻게 봄?"
        if not q.endswith("?"):
            q = q.rstrip() + "?"
        return q

    def _add_tail_question(lines: List[str]) -> None:
        if not lines:
            lines.append(_end_q())
            return
        last = str(lines[-1]).strip()
        if "?" not in last:
            lines.append(_end_q())
            return
        # 이미 질문이 있으면 1개만 유지
        if len(lines) >= 2 and "?" in str(lines[-2]):
            # 두 질문 연속이면 하나 제거
            lines.pop(-2)

    # keyword lists
    hot_kws = [normalize_ko_token(str(x.get("kw") if isinstance(x, dict) else x)) for x in hot[:8]]
    hot_kws = [x for x in hot_kws if is_clean_keyword(x)]
    rising_kws = [normalize_ko_token(str(x.get("kw") if isinstance(x, dict) else x)) for x in rising[:8]]
    rising_kws = [x for x in rising_kws if is_clean_keyword(x)]
    trend_hint = ", ".join((hot_kws + rising_kws)[:4])

    title = ""
    body_lines: List[str] = []
    cat = "general"
    ctx = "post"

    # --- style builders ---
    if style == "post:meta:flow_report":
        title = random.choice(["요즘 커뮤 흐름 메모", "최근 핫/라이징 흐름", "지금 분위기 요약"])
        if hot_kws:
            body_lines.append(f"핫: {', '.join(hot_kws[:5])}임")
        if rising_kws:
            body_lines.append(f"라이징: {', '.join(rising_kws[:5])}임")
        body_lines.append(random.choice([
            "이 흐름은 결국 '정의/기준' 싸움으로 자꾸 회귀하는 느낌임",
            "주제는 달라도 분기 기준이 비슷하게 반복되는 편임",
            "핫한데 결론이 안 나는건 대개 전제가 안 맞아서임",
        ]))
        _add_tail_question(body_lines)
        cat = "meta"
        ctx = "meta"

    elif style == "post:meta:observation_log":
        title = random.choice(["오늘 커뮤에서 느낀거", "짧게 관찰 로그", "최근 댓글들 보면서 든 생각"])
        if trend_hint and random.random() < 0.75:
            body_lines.append(f"요즘 {trend_hint} 얘기가 계속 올라오는 느낌임")
        body_lines.append(random.choice([
            f"근데 싸움 나는 지점은 {seed_kw}를 어디까지로 보냐로 갈리는 경우가 많음",
            f"근데 결론이 갈리는 포인트는 {seed_kw} 기준선을 어디에 꽂냐인듯함",
            f"근데 서로 같은 말로 다른 문제를 얘기하는 케이스가 꽤 보임",
        ]))
        if ttxt and random.random() < 0.55:
            body_lines.append(random.choice(MEMO_POOL) + one_line(ttxt, 90))
        _add_tail_question(body_lines)
        cat = "meta"
        ctx = "meta"

    elif style == "post:meta:one_metric":
        title = random.choice([f"{seed_kw} 논쟁, 지표 하나만 박아보자", "측정 기준 1개만 고정해봄", "말싸움 줄이는 방법 하나"])
        body_lines.append(random.choice([
            "일반론으로만 돌면 끝이 안 남",
            "측정지표 하나만 박으면 대화가 빨리 정리되는 편임",
            "기준을 수치/판정으로 바꾸면 말싸움이 확 줄어듦",
        ]))
        body_lines.append(random.choice([
            "예: 일관성(케이스 바뀌어도 기준 유지됨?)",
            "예: 비용(사회 비용/개인 비용을 어디에 계산함?)",
            "예: 피해(누가 손해를 실제로 떠안음?)",
            "예: 예측가능성(룰이 보이고 따라갈 수 있음?)",
        ]))
        _add_tail_question(body_lines)
        cat = "meta"
        ctx = "meta"

    elif style == "post:philo:definition_war":
        title = f"{seed_kw} 얘기에서 제일 먼저 갈리는거"
        body_lines = [
            random.choice([
                f"{seed_kw}는 단어는 같은데 범위가 갈리는 느낌임",
                f"{seed_kw}는 정의를 어디까지로 보냐부터 엇갈림",
                f"{seed_kw}는 같은 말이라도 기준선이 제각각임",
            ]),
        ]
        if random.random() < 0.55:
            body_lines.append(random.choice(PHILO_EXAMPLES))
        body_lines.append(random.choice([
            "정의부터 맞추면 논쟁 반은 끝나는 편임",
            "정의부터 통일하면 말싸움이 줄어듦",
            "정의부터 합의하면 서로 딴말할 확률이 내려감",
        ]))
        _add_tail_question(body_lines)
        cat = "philo"
        ctx = "philo"

    elif style == "post:philo:paradox":
        title = f"{seed_kw}의 역설 같은 지점"
        body_lines = [
            random.choice([
                f"{seed_kw}는 극단으로 가면 본래 목적이 깨지는 구간이 있음",
                f"{seed_kw}는 강하게 밀수록 역효과 나는 케이스가 있음",
                f"{seed_kw}는 한쪽만 올리면 다른 쪽이 터지는 느낌임",
            ]),
            random.choice(PHILO_EXAMPLES) if random.random() < 0.45 else "이건 조건을 어디에 거냐에 따라 답이 바뀜",
            random.choice([
                "그래서 케이스 분리가 없으면 결론이 계속 흔들림",
                "그래서 반례를 몇 개 찍어봐야 선이 보임",
                "그래서 일반론만 반복하면 헛바퀴 돌기 쉬움",
            ]),
        ]
        _add_tail_question(body_lines)
        cat = "philo"
        ctx = "philo"

    elif style == "post:philo:uncertainty":
        title = f"{seed_kw}에서 애매한 구간"
        body_lines = [
            random.choice([
                f"{seed_kw}는 경계선이 애매해서 케이스가 섞이는 느낌임",
                f"{seed_kw}는 어디부터 다른 문제로 넘어가는지 선이 흐림",
                f"{seed_kw}는 딱 끊어 말하기 어려운 구간이 있음",
            ]),
            random.choice([
                "그래서 최소한 분기 기준 하나는 세워야 함",
                "그래서 조건을 명시하지 않으면 말이 계속 미끄러짐",
                "그래서 예시 1~2개를 먼저 박고 들어가는게 빠름",
            ]),
        ]
        if random.random() < 0.35:
            body_lines.append(random.choice(PHILO_EXAMPLES))
        _add_tail_question(body_lines)
        cat = "philo"
        ctx = "philo"

    elif style == "post:philo:process":
        title = random.choice([f"{seed_kw} 얘기할때 순서가 중요함", "정의→기준→검증, 이 순서가 맞나?", "논쟁이 길어질때 보통 빠진 단계"])
        body_lines = [
            "내쪽 프로세스는 대충 이럼: 정의 고정 → 기준 1개 박기 → 케이스로 검증",
            "순서가 뒤집히면 같은 말로 다른 결론이 나오는 느낌임",
        ]
        if random.random() < 0.4:
            body_lines.append("특히 기준이 2개 이상 섞이면 토론이 급격히 길어짐")
        _add_tail_question(body_lines)
        cat = "philo"
        ctx = "philo"

    elif style == "post:philo:axiom":
        title = random.choice([f"{seed_kw}에서 전제 1개만 박고 시작하자", "원칙 하나 없으면 계속 흔들림", "이 논쟁, 공리부터 정해야함"])
        body_lines = [
            random.choice([
                f"{seed_kw}는 결국 '무엇을 더 우선함' 공리 싸움으로 수렴하는 편임",
                f"{seed_kw}는 기저가치(안전/자유/효율/공정)를 뭐로 두냐가 갈림",
                f"{seed_kw}는 전제를 안 박으면 케이스마다 결론이 튀는 느낌임",
            ]),
            "그래서 난 '전제 1개'를 먼저 고정하는게 제일 빠르다고 봄",
        ]
        _add_tail_question(body_lines)
        cat = "philo"
        ctx = "philo"

    elif style == "post:philo:boundary_test":
        title = random.choice([f"{seed_kw} 경계 케이스만 찍어보자", "반례/경계부터 보면 선이 보임", "애매하면 경계테스트가 답임"])
        body_lines = [
            f"{seed_kw}는 중간이 애매하면 경계 케이스로만 찍어봐도 선이 보이는 편임",
            random.choice([
                "케이스1: 의도는 좋은데 결과가 나쁜 경우",
                "케이스1: 절차는 공정한데 결과가 불공정한 경우",
                "케이스1: 규칙은 맞는데 상식이 깨지는 경우",
            ]),
            random.choice([
                "케이스2: 결과는 좋은데 절차가 문제인 경우",
                "케이스2: 예외를 한 번 허용했더니 일반화가 되는 경우",
                "케이스2: 같은 룰인데 집단이 바뀌면 평가가 뒤집히는 경우",
            ]),
        ]
        _add_tail_question(body_lines)
        cat = "philo"
        ctx = "philo"

    elif style == "post:general:case_split":
        title = random.choice([f"{seed_kw} 케이스 분리 해봄", "A/B로 나눠보면 깔끔함", "논점 분리해서 보자"])
        body_lines = [
            f"케이스A: {seed_kw}를 '원칙'으로 보는 쪽",
            f"케이스B: {seed_kw}를 '도구'로 보는 쪽",
            "A/B가 섞이면 서로 딴말하는 느낌이 남",
        ]
        _add_tail_question(body_lines)
        cat = "general"
        ctx = "gen"

    elif style == "post:general:checklist":
        title = random.choice([f"{seed_kw} 체크리스트 3개", "판단 기준을 3줄로 고정해봄", "내가 쓰는 간단 판정법"])
        body_lines = [
            "체크 3개만 박으면 말싸움이 좀 줄어듦",
            "1) 일관성: 케이스 바뀌어도 기준 유지됨?",
            "2) 비용: 누가 비용을 실제로 떠안음?",
            "3) 예측가능성: 룰이 보이고 따라갈 수 있음?",
        ]
        _add_tail_question(body_lines)
        cat = "general"
        ctx = "gen"

    elif style == "post:general:analogy":
        title = random.choice([f"{seed_kw}를 시스템 비유로 보면", "비유 하나로 정리해봄", "이거 느낌상 이런 구조임"])
        body_lines = [
            random.choice(ANALOGY_POOL),
            f"결국 {seed_kw}도 '우선순위/가정/기준 컬럼'을 어디에 두냐 문제 같음",
        ]
        if random.random() < 0.35:
            body_lines.append("비유가 과하면 오해나니 핵심만 가져오는게 맞음")
        _add_tail_question(body_lines)
        cat = "general"
        ctx = "gen"

    elif style == "post:general:short_take":
        title = f"{seed_kw} 관련해서 짧게 한줄"
        memo = random.choice(MEMO_POOL)
        body_lines = [
            f"{memo}{seed_kw}는 결국 기준을 어디에 꽂느냐가 갈림",
            random.choice([
                "말만 길게 하면 답이 더 안 보이더라",
                "짧게 쪼개면 의외로 선이 보이는 편임",
                "케이스 분리만 해도 싸움이 줄어듦",
            ]),
        ]
        _add_tail_question(body_lines)
        cat = "general"
        ctx = "gen"

    else:
        # post:meta:question (default)
        title = f"{seed_kw} 관련해서 하나만 물어봄"
        memo = random.choice(MEMO_POOL)
        hint = ""
        if ttxt:
            hint = f"{memo}{one_line(ttxt, 110)}"
        else:
            hint = random.choice([
                "여기서 먼저 정해야 할 게 하나 있음",
                "이건 기준을 1개만 박아도 결이 갈림",
                "이건 적용범위를 어디까지로 보냐가 핵심 분기임",
            ])
        body_lines = [
            random.choice([
                f"{seed_kw} 얘기 볼때마다 기준이 다 갈리는 느낌임",
                f"{seed_kw}는 같은 말인데 서로 다른 그림을 보는 느낌임",
                f"{seed_kw}는 케이스가 섞이면 논쟁이 길어지는 편임",
            ]),
            hint,
        ]
        _add_tail_question(body_lines)
        cat = "meta"
        ctx = "meta"

    # mark thought usage (helps pruning)
    if isinstance(t0, dict) and str(t0.get("id") or ""):
        try:
            mark_thought_used(brain, str(t0.get("id") or ""))
        except Exception as e:
            log_debug_exc("post:thought_use", e)

    # maturity: allow one extra line occasionally (within max_output_lines)
    if maturity >= 0.55 and len(body_lines) < 4 and random.random() < 0.35:
        body_lines.insert(1, random.choice([
            "여기서 먼저 정할 건 정의냐 기준이냐 순서임",
            "결국 분기점은 기준선을 어디에 박느냐임",
            "말싸움 줄이려면 조건을 한 줄로 고정해야 함",
        ]))

    # (Unit 09) title repetition guard
    try:
        tprefixes = brain.get("recent_post_title_prefixes")
        if not isinstance(tprefixes, list):
            tprefixes = []
        tprefixes = [str(x) for x in tprefixes if isinstance(x, str)][-12:]
        cand_t = (title or "")[:18]
        if cand_t and sum(1 for p in tprefixes if p == cand_t) >= 2:
            title = random.choice([
                f"{seed_kw} 얘기, 여기서 갈림",
                f"{seed_kw} 프레임 분리 해봄",
                f"{seed_kw} 기준 하나만 묻겠음",
                f"{seed_kw}는 결국 전제 싸움임",
            ])
        tprefixes.append((title or "")[:18])
        brain["recent_post_title_prefixes"] = tprefixes[-12:]
    except Exception as e:
        log_debug_exc("build_post_text:silent", e)
        pass

    # repetition guard: first sentence prefix
    try:
        prefixes = brain.get("recent_post_prefixes")
        if not isinstance(prefixes, list):
            prefixes = []
        prefixes = [str(x) for x in prefixes if isinstance(x, str)]
        prefixes = prefixes[-10:]
        cand = (body_lines[0] if body_lines else "")[:25]
        if cand and sum(1 for p in prefixes if p == cand) >= 2:
            # replace first line with a variant
            if cat == "philo":
                body_lines[0] = random.choice([
                    f"{seed_kw}는 정의부터 삐끗하면 끝까지 평행선임",
                    f"{seed_kw}는 같은 말인데 서로 다른 규칙을 떠올리는 느낌임",
                    f"{seed_kw}는 경계선이 흐리면 대화가 계속 미끄러짐",
                ])
            else:
                body_lines[0] = random.choice([
                    f"{seed_kw}는 기준을 고정 안 하면 결론이 계속 바뀜",
                    f"{seed_kw}는 케이스가 섞이면 말이 길어지는 편임",
                    f"{seed_kw}는 분기 기준 하나만 세워도 훨씬 깔끔해짐",
                ])
        prefixes.append((body_lines[0] if body_lines else "")[:25])
        brain["recent_post_prefixes"] = prefixes[-10:]
    except Exception as e:
        log_debug_exc("post:prefix_guard", e)

    # update style history
    try:
        recent_styles.append(style)
        brain["recent_post_styles"] = recent_styles[-10:]
    except Exception as e:
        log_debug_exc("post:style_hist", e)

    title2 = ensure_eum_style(title, max_lines=1).replace("\n", " ")
    body2 = ensure_eum_style("\n".join(body_lines), max_lines=max(2, tuning.max_output_lines))

    meta = {
        "post_style": style,
        "signature": str(get_persona(brain).get("signature", "eum") or "eum"),
        "category": cat,
        "cat": cat,
        "context_key": ctx,
        "seed_kw": seed_kw,
        "thought_id": str(t0.get("id") or "") if isinstance(t0, dict) else "",
    }
    return title2, body2, meta


def do_sync_posts(client: HttpClient, cfg: Config, state: Dict[str, Any], tuning: AgentTuning) -> List[Dict[str, Any]]:
    """Sync main feed posts and record them as 'seen'.

    v19.2:
      - Keep legacy state.seen_post_ids (list) for backward compatibility.
      - Also maintain state.seen.posts (dict: post_id -> seen_ts) with LRU cap.
    """
    posts, _ = list_posts(client, limit=tuning.fetch_limit)

    # legacy list (keep)
    seen_list = state.setdefault("seen_post_ids", [])
    if not isinstance(seen_list, list):
        state["seen_post_ids"] = []
        seen_list = state["seen_post_ids"]

    now_ts = time.time()

    # v20.2: durable mandatory votes via protocol.vote_backlog
    mandatory_votes = bool(getattr(getattr(cfg, "vote_proto", None), "mandatory", True))
    if mandatory_votes:
        try:
            vote_backlog_gc(state, now_ts=now_ts)
        except Exception as e:
            log_debug_exc("vote_backlog:gc_pre", e)

    for p in posts[:]:
        pid = str(p.get("id") or "")
        if not pid:
            continue

        if mandatory_votes:
            try:
                if (not is_own_post(cfg, p)) and (not _is_post_voted(state, pid)):
                    vote_backlog_enqueue(state, pid, now_ts)
            except Exception as e:
                log_debug_exc("vote_backlog:enqueue", e)

        if pid not in seen_list:
            seen_list.append(pid)
        # nested map (LRU capped)
        _record_seen_post(cfg, state, pid, now_ts)

    if mandatory_votes:
        try:
            vote_backlog_gc(state, now_ts=time.time())
        except Exception as e:
            log_debug_exc("vote_backlog:gc_post", e)

    # keep legacy list bounded (very generous; nested map is the real LRU)
    state["seen_post_ids"] = seen_list[-5000:]
    state["last_sync_ts"] = time.time()
    return posts

def _pick_vote_target(cfg: Config, posts: List[Dict[str, Any]]) -> Optional[str]:
    cand: List[Tuple[str, int]] = []
    for p in posts:
        if not isinstance(p, dict):
            continue
        if is_own_post(cfg, p):
            continue
        pid = str(p.get("id") or "")
        if not pid:
            continue
        m = _post_metrics(p)
        cand.append((pid, int(m.get("score", 0))))
    if not cand:
        return None
    cand = sorted(cand, key=lambda x: abs(x[1]))
    return cand[0][0] if cand else None


def do_vote_main_feed(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    memory: List[Dict[str, Any]],
    semantic: Dict[str, Any],
    brain: Dict[str, Any],
    posts_cache: List[Dict[str, Any]],
    vote_limiter: SlidingWindowLimiter,
    vote_pace_sec: int
) -> int:
    # Window + global pacing gates
    if ops_should_skip(state, "vote"):
        protocol_set_reason(state, "vote", "vote:ops_disabled")
        return 0

    if vote_limiter.remaining() <= 0:
        if getattr(cfg, "debug", None) and cfg.debug.log_blocks:
            log_debug("gate:vote blocked by window (remaining=0)")
        protocol_set_reason(state, "vote", "vote:rate_limited", "window_remaining=0")
        return 0

    gap = gap_remaining(float(state.get("last_vote_ts", 0.0) or 0.0), int(vote_pace_sec))
    if gap > 0:
        if getattr(cfg, "debug", None) and cfg.debug.log_blocks:
            log_debug(f"gate:vote blocked by gap ({int(gap)}s)")
        protocol_set_reason(state, "vote", "vote:rate_limited", f"gap={int(gap)}s")
        return 0

    # v20.2: mandatory vote for seen posts with durable backlog
    pid: Optional[str] = None
    from_backlog = False

    mandatory_votes = bool(getattr(getattr(cfg, "vote_proto", None), "mandatory", True))
    if mandatory_votes:
        try:
            vote_backlog_gc(state, now_ts=time.time())
            pid = vote_backlog_pick(state)
            if pid:
                from_backlog = True
        except Exception as e:
            log_debug_exc("vote_backlog:pick", e)
            pid = None
            from_backlog = False

        # fallback: scan current feed cache for any unvoted item
        if not pid:
            for p in posts_cache:
                if not isinstance(p, dict):
                    continue
                if is_own_post(cfg, p):
                    continue
                cand = str(p.get("id") or "")
                if not cand:
                    continue
                if not _is_post_voted(state, cand):
                    pid = cand
                    break

    if not pid:
        pid = _pick_vote_target(cfg, posts_cache)
    if not pid:
        try:
            if mandatory_votes and len(_vote_backlog_list(state)) == 0:
                protocol_set_reason(state, "vote", "vote:backlog_empty")
            else:
                protocol_set_reason(state, "vote", "vote:no_target")
        except Exception:
            protocol_set_reason(state, "vote", "vote:no_target")
        return 0

    # double-check (safety)
    if _is_post_voted(state, pid):
        if from_backlog:
            try:
                if vote_backlog_remove(state, pid):
                    vote_backlog_record_drain(state, time.time())
            except Exception as e:
                log_debug_exc("vote_backlog:already_voted", e)
        protocol_set_reason(state, "vote", "vote:no_target", "already_voted")
        return 0

    # consume limiter only when we actually attempt a vote
    if not vote_limiter.allow():
        if getattr(cfg, "debug", None) and cfg.debug.log_blocks:
            log_debug("gate:vote blocked by window (allow=false)")
        protocol_set_reason(state, "vote", "vote:rate_limited", "window_allow=false")
        return 0

    # Self-policing: downvote obvious rule violators / injection / offensive content
    enforce_self_policing = _env_bool("MERSOOM_SELF_POLICING", True)
    vtype = "up"
    sp_reason = ""
    if enforce_self_policing:
        try:
            ptxt = ""
            for p in posts_cache:
                if isinstance(p, dict) and str(p.get("id") or "") == pid:
                    ptxt = f"{p.get('title', '')} {p.get('content', '')}".strip()
                    break

            toxic, treason = is_toxic_incoming(ptxt)
            viol, vreason = looks_like_rule_violation(ptxt)
            if bool(toxic):
                vtype = "down"
                sp_reason = f"toxic:{treason}"
            elif bool(viol):
                vtype = "down"
                sp_reason = f"rule:{vreason}"
        except Exception as e:
            log_debug_exc("self_policing:vote", e)

    # Optional: stochastic downvote (disabled by default)
    if vtype != "down":
        down_prob = _env_float("MERSOOM_VOTE_DOWN_PROB", 0.0, 0.0, 1.0)
        vtype = "down" if (float(down_prob) > 0.0 and random.random() < float(down_prob)) else "up"

    if sp_reason and getattr(cfg, "debug", None) and cfg.debug.log_blocks:
        log_debug(f"self_policing vote=down reason={sp_reason} post={pid}")

    _bump_action_counter(state, "action_attempt", "vote")
    try:
        ok = vote_post(client, cfg, state, pid, vtype)
    except RateLimitError as e_rl:
        protocol_set_reason(state, "vote", "vote:rate_limited", one_line(str(e_rl), 180))
        raise
    except PowTimeoutError as e:
        protocol_set_reason(state, "vote", "vote:pow_timeout", one_line(str(e), 180))
        _bump_action_counter(state, "action_fail", "vote")
        return 0
    except Exception as e:
        protocol_set_reason(state, "vote", "vote:no_target", one_line(repr(e), 180))
        raise

    if not ok:
        protocol_set_reason(state, "vote", "vote:no_target", "vote_post_failed")
        _bump_action_counter(state, "action_fail", "vote")
    else:
        _bump_action_counter(state, "action_success", "vote")


    # v20.2: drain backlog if vote succeeded or the post is now marked as voted
    if from_backlog:
        try:
            if ok or _is_post_voted(state, pid):
                if vote_backlog_remove(state, pid):
                    vote_backlog_record_drain(state, time.time())
        except Exception as e:
            log_debug_exc("vote_backlog:drain", e)

    state["last_vote_ts"] = time.time()
    state["total_actions"] = int(state.get("total_actions", 0)) + 1

    update_persona_maturity(brain, state)
    bump_semantic(semantic, _today_kst(), f"vote:{vtype}", 1.0)

    record_memory(memory, {
        "ts": time.time(),
        "action": f"vote:{vtype}",
        "post_id": pid,
        "evaluated": True,
        "reward_scalar": 0.0,
    }, tuning, archive_path_jsonl="")

    log_action("VOTE", f"{vtype} post={pid} ok={ok}")
    return 1

def _flow_keywords(brain: Optional[Dict[str, Any]], *, k: int = 6) -> List[str]:
    if not isinstance(brain, dict):
        return []
    com = _safe_dict(brain.get("community"))
    out: List[str] = []
    for bucket in ("rising", "hot"):
        for it in _safe_list(com.get(bucket))[:k]:
            if isinstance(it, dict) and it.get("kw"):
                out.append(str(it.get("kw")))
    # de-dup keep order
    seen = set()
    res: List[str] = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            res.append(x)
    return res[:k]

def _bump_relation(state: Dict[str, Any], user_key: str) -> None:
    if not user_key:
        return
    rel = state.setdefault("relations", {})
    if not isinstance(rel, dict):
        state["relations"] = {}
        rel = state["relations"]
    obj = rel.get(user_key)
    if not isinstance(obj, dict):
        obj = {"interactions": 0, "last_ts": 0.0}
    obj["interactions"] = int(obj.get("interactions", 0)) + 1
    obj["last_ts"] = time.time()
    rel[user_key] = obj

################################################################################
# 14.2. TARGET SELECT (contribution target picking)
################################################################################
def _pick_contrib_target(cfg: Config, state: Dict[str, Any], posts: List[Dict[str, Any]], brain: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Pick a post to comment on using engagement + thread/user context.

    P1: deterministic 'top score' selection makes behavior stiff.
        We score a handful of candidates and sample among top-N.
    """
    commented_ts = _safe_dict(state.get("commented_ts"))
    flow_kws = _flow_keywords(brain, k=8)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    now = time.time()

    for p in posts:
        if not isinstance(p, dict):
            continue
        if is_own_post(cfg, p):
            continue
        pid = str(p.get("id") or "")
        if not pid:
            continue

        # local cool-down to avoid hammering one thread
        lastc = float(commented_ts.get(pid, 0.0) or 0.0)
        if (now - lastc) < float(getattr(getattr(cfg, 'timing', None), 'same_post_comment_gap_sec', 60 * 30)):
            continue

        m = _post_metrics(p)
        base = int(m.get("comments", 0)) * 2.0 + int(m.get("score", 0)) * 1.0

        # thread context (if exists)
        th = _safe_dict(_safe_dict(state.get("threads")).get(pid))
        tension = float(len(_safe_list(th.get("tensions")))) * 2.0
        openq = float(len(_safe_list(th.get("open_questions")))) * 1.2
        # prefer fresher threads
        age = max(0.0, now - float(th.get("last_seen_ts", now) or now))
        fresh = max(0.0, 1.5 - (age / 3600.0) * 0.15)

        # author + user model
        author = str(p.get("nickname") or p.get("author") or "")
        u = get_user(state, author) if author else {}
        helpful = float(_safe_dict(u).get("helpfulness", 0.0) or 0.0)
        aggr = float(_safe_dict(u).get("aggression", 0.0) or 0.0)

        drives = get_persona_drives(brain)
        debate_bias = float(drives.get("debate", 0.7) or 0.7)
        # debate-oriented persona: conflict isn't a strict negative
        aggr_term = (0.8 * aggr) if debate_bias >= 0.6 else (-1.2 * aggr)

        # relation memory (who we've been interacting with)
        rel = _safe_dict(_safe_dict(state.get("relations")).get(author)) if author else {}
        inter = float(_safe_dict(rel).get("interactions", 0) or 0)
        rel_term = 0.35 * math.log1p(inter)

        # flow keyword overlap
        txt = f"{p.get('title') or ''} {p.get('content') or ''}".strip()
        kws = set(top_keywords(txt, k=10))
        overlap = sum(1 for kw in flow_kws if kw and kw in kws)
        flow_term = 1.8 * float(overlap)

        # mild novelty push: avoid over-popular threads only
        pop_penalty = 0.0
        if int(m.get("comments", 0)) >= 20:
            pop_penalty = 1.5

        score = base + tension + openq + fresh + (1.4 * helpful) + aggr_term + rel_term + flow_term - pop_penalty
        scored.append((score, p))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    topn = scored[: min(6, len(scored))]

    # weighted sampling among top to avoid being stuck
    weights = [max(0.01, float(s - topn[-1][0] + 0.25)) for s, _ in topn]
    ssum = sum(weights)
    r = random.uniform(0.0, ssum)
    acc = 0.0
    for w, (_, p) in zip(weights, topn):
        acc += w
        if acc >= r:
            return p
    return topn[0][1]

def _comment_parent_id(c: Dict[str, Any]) -> str:
    for k in ("parent_id", "parentId", "reply_to", "replyTo", "parent_comment_id"):
        v = c.get(k)
        if v:
            return str(v)
    return ""


def _comment_created_ts(c: Dict[str, Any]) -> float:
    """Best-effort comment timestamp (epoch seconds).

    Supports both numeric epoch (sec/ms) and ISO8601 created_at strings.
    Returns 0.0 if unavailable.
    """
    if not isinstance(c, dict):
        return 0.0
    for k in ("ts", "timestamp", "created_ts", "createdAtTs", "created_at_ts"):
        try:
            v = c.get(k)
            if isinstance(v, (int, float)) and float(v) > 0:
                x = float(v)
                # normalize ms -> sec
                if x > 1e12:
                    x = x / 1000.0
                return float(x)
        except Exception:
            continue
    for k in ("created_at", "createdAt", "created", "created_time", "createdTime"):
        try:
            v = c.get(k)
            if isinstance(v, (int, float)) and float(v) > 0:
                x = float(v)
                if x > 1e12:
                    x = x / 1000.0
                return float(x)
            if isinstance(v, str) and v.strip():
                ts = _parse_iso_ts(v.strip())
                if ts > 0:
                    return float(ts)
        except Exception:
            continue
    return 0.0

def _has_replied_to_comment(cfg: Config, comments: List[Dict[str, Any]], target_comment_id: str) -> bool:
    """
    Best-effort: detect if we already replied to a given comment (server-side truth beats local state).
    """
    tc = str(target_comment_id or "")
    if not tc:
        return False
    for c in comments:
        if not isinstance(c, dict):
            continue
        if not is_own_comment(cfg, c):
            continue
        pid = _comment_parent_id(c)
        if pid and pid == tc:
            return True
    return False

# ---------------------------------------------------------------------------
# (v18.3) Reply thread conversation protocol
# - Prevents "self monologue" where the agent keeps replying without any new remote turn.
# - Uses comment parent chain to group a whole reply tree into one "conversation".
# - Maintains a lightweight budget: at most 1 reply per new remote turn.
# ---------------------------------------------------------------------------

def _comment_root_id(comments_by_id: Dict[str, Dict[str, Any]], cid: str) -> str:
    cur = str(cid or "")
    if not cur:
        return ""
    seen: set = set()
    while cur:
        if cur in seen:
            return cur  # cycle guard
        seen.add(cur)
        c = comments_by_id.get(cur)
        if not isinstance(c, dict):
            return cur
        pid = _comment_parent_id(c)
        if not pid:
            return cur
        cur = str(pid)
    return str(cid or "")

def _build_root_cache(comments: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for c in comments or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "")
        if cid:
            by_id[cid] = c
    root_of: Dict[str, str] = {}
    for cid in list(by_id.keys()):
        root_of[cid] = _comment_root_id(by_id, cid)
    return by_id, root_of

def _reply_conv_key(post_id: str, root_comment_id: str) -> str:
    return f"{str(post_id or '')}:{str(root_comment_id or '')}"

def _reply_protocol_prepare(
    state: Dict[str, Any],
    cfg: Config,
    tuning: AgentTuning,
    post_id: str,
    comments: List[Dict[str, Any]],
    target_comment_id: str,
    now_ts: float,
) -> Tuple[bool, str, Dict[str, Any], Dict[str, Any], str]:
    """Returns (ok, conv_key, conv_state, conv_meta, reason)."""
    convs = state.get("conv_state")
    if not isinstance(convs, dict):
        convs = {}
        state["conv_state"] = convs

    by_id, root_of = _build_root_cache(comments)
    tgt = str(target_comment_id or "")
    if not tgt or tgt not in by_id:
        return False, "", {}, {}, "target_missing"

    root = str(root_of.get(tgt) or tgt)
    ckey = _reply_conv_key(post_id, root)

    conv = convs.get(ckey)
    if not isinstance(conv, dict):
        conv = {
            "post_id": str(post_id or ""),
            "root_comment_id": root,
            "last_remote_id": "",
            "last_my_id": "",
            "budget": 0,
            "waiting_for_remote": False,
            "blocked_until_ts": 0.0,
            "closed_until_ts": 0.0,
            "turns_remote": 0,
            "turns_my": 0,
            "last_action_ts": 0.0,
            "last_seen_tail_kind": "",
        }
        convs[ckey] = conv

    # hard blocks
    if float(conv.get("closed_until_ts", 0.0) or 0.0) > now_ts:
        return False, ckey, conv, {}, "closed"
    if float(conv.get("blocked_until_ts", 0.0) or 0.0) > now_ts:
        return False, ckey, conv, {}, "blocked"

    # analyze conversation by list order (assume chronological)
    last_in_conv: Optional[Dict[str, Any]] = None
    last_other_in_conv: Optional[Dict[str, Any]] = None
    last_my_in_conv: Optional[Dict[str, Any]] = None
    turns_total = 0
    for c in comments or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "")
        if not cid:
            continue
        if str(root_of.get(cid) or cid) != root:
            continue
        turns_total += 1
        last_in_conv = c
        if is_own_comment(cfg, c):
            last_my_in_conv = c
        else:
            last_other_in_conv = c

    last_id = str((last_in_conv or {}).get("id") or "")
    last_other_id = str((last_other_in_conv or {}).get("id") or "")
    last_my_id = str((last_my_in_conv or {}).get("id") or "")
    last_speaker = "me" if (last_in_conv is not None and is_own_comment(cfg, last_in_conv)) else "other"

    prev_remote = str(conv.get("last_remote_id") or "")
    new_remote = bool(last_other_id) and (last_other_id != prev_remote)
    if new_remote:
        conv["last_remote_id"] = last_other_id
        conv["last_remote_ts"] = float(now_ts)
        cap = int(getattr(tuning, "reply_conv_budget_cap", 1) or 1)
        conv["budget"] = min(cap, int(conv.get("budget", 0) or 0) + 1)
        conv["waiting_for_remote"] = False
        conv["turns_remote"] = int(conv.get("turns_remote", 0) or 0) + 1

    if last_my_id:
        conv["last_my_id"] = last_my_id

    # Only reply to the latest remote turn (avoid out-of-order replies).
    if last_other_id and (tgt != last_other_id) and (not is_own_comment(cfg, by_id.get(tgt, {}))):
        conv["blocked_until_ts"] = now_ts + float(getattr(tuning, "reply_conv_skip_cooldown_sec", 10 * 60))
        return False, ckey, conv, {}, "not_latest_remote"

    # Gate: do not speak twice in a row; do not speak without budget
    if last_speaker == "me" and (not new_remote):
        conv["blocked_until_ts"] = now_ts + float(getattr(tuning, "reply_conv_skip_cooldown_sec", 10 * 60))
        return False, ckey, conv, {}, "awaiting_remote"
    if int(conv.get("budget", 0) or 0) <= 0 and (not new_remote):
        conv["blocked_until_ts"] = now_ts + float(getattr(tuning, "reply_conv_skip_cooldown_sec", 10 * 60))
        return False, ckey, conv, {}, "no_budget"

    # lock after we asked a question until remote responds
    if bool(conv.get("waiting_for_remote")) and (not new_remote):
        cooldown = float(getattr(tuning, "reply_conv_skip_cooldown_sec", 10 * 60))
        # v23.5: optional strict waiting (stronger cooldown + broader question detection in commit)
        if _waiting_strict_enabled(state):
            cooldown = max(cooldown, 20 * 60.0)
        conv["blocked_until_ts"] = now_ts + cooldown
        conv["last_skip_reason"] = "waiting_for_remote"
        try:
            protocol_bump_counter(state, "waiting_skip", 1)
        except Exception:
            pass
        return False, ckey, conv, {}, "waiting_for_remote"

    target_text = str((by_id.get(tgt) or {}).get("content") or "")
    remote_is_question = ("?" in target_text) or target_text.strip().endswith("?") or bool(re.search(r"(까|나요|냐|지)\s*$", target_text))
    conv_meta = {
        "conv_key": ckey,
        "root_comment_id": root,
        "last_comment_id": last_id,
        "last_remote_id": last_other_id,
        "last_my_id": last_my_id,
        "last_speaker": last_speaker,
        "new_remote": bool(new_remote),
        "budget": int(conv.get("budget", 0) or 0),
        "turns_total": int(turns_total),
        "turns_remote": int(conv.get("turns_remote", 0) or 0),
        "turns_my": int(conv.get("turns_my", 0) or 0),
        "remote_is_question": bool(remote_is_question),
    }
    return True, ckey, conv, conv_meta, "ok"

def _reply_protocol_commit(
    state: Dict[str, Any],
    conv_key: str,
    now_ts: float,
    reply_comment_id: str,
    reply_text: str,
) -> None:
    convs = state.get("conv_state")
    if not isinstance(convs, dict):
        return
    conv = convs.get(conv_key)
    if not isinstance(conv, dict):
        return
    conv["last_action_ts"] = float(now_ts)
    if reply_comment_id:
        conv["last_my_id"] = str(reply_comment_id)
    conv["budget"] = max(0, int(conv.get("budget", 0) or 0) - 1)
    conv["turns_my"] = int(conv.get("turns_my", 0) or 0) + 1
    asked_q = False
    try:
        if _waiting_strict_enabled(state):
            asked_q = bool(_is_open_question_text(str(reply_text or "")))
        else:
            asked_q = bool("?" in (reply_text or ""))
    except Exception:
        asked_q = bool("?" in (reply_text or ""))
    tail_kind = "question" if asked_q else "close"
    conv["last_seen_tail_kind"] = str(tail_kind)
    conv["waiting_for_remote"] = bool(asked_q)
    conv["blocked_until_ts"] = float(now_ts) + 90.0

    # v23.2: update thread phase from outgoing reply (opt-in via interaction FSM)
    try:
        post_id = str(conv_key.split('|', 1)[0]) if conv_key else ""
        if post_id:
            th = get_thread(state, post_id)
            thread_update_phase_if_needed(state, th, reply_text or "", source="my_reply")
    except Exception:
        pass



def _pick_ctx_from_flow(brain: Dict[str, Any]) -> str:
    com = _safe_dict(brain.get("community"))
    rising = _safe_list(com.get("rising"))
    hot = _safe_list(com.get("hot"))
    kw = ""
    if rising and isinstance(rising[0], dict):
        kw = str(rising[0].get("kw") or "")
    if not kw and hot and isinstance(hot[0], dict):
        kw = str(hot[0].get("kw") or "")
    cat, ctx = classify_text(kw or "커뮤니티")
    return ctx or "gen"

# (Unit 04) Thread-priority reply inbox
def _recent_involved_post_ids_from_memory(memory: List[Dict[str, Any]], *, days: int, max_n: int) -> List[str]:
    """Infer recently-involved post_ids from memory (no extra API calls)."""
    try:
        dd = max(1, int(days))
    except Exception:
        dd = 3
    max_n = max(1, int(max_n))
    now = time.time()
    seen: Dict[str, float] = {}
    for it in reversed(_safe_list(memory)[-350:]):
        if not isinstance(it, dict):
            continue
        ts = _safe_float(it.get("ts"), 0.0)
        if ts <= 0:
            continue
        if (now - ts) > dd * 24 * 60 * 60:
            break
        pid = str(it.get("post_id") or "")
        if not pid:
            continue
        act = str(it.get("action") or "")
        if act not in ("post", "comment", "reply"):
            continue
        if pid not in seen or ts > seen[pid]:
            seen[pid] = ts
    out = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in out[:max_n]]

def _inbox_item_score(post: Dict[str, Any], *, own_post: bool, reply_to_my_comment: bool, recent_involved: bool, rank_hint: int) -> float:
    # Base priority: direct replies > comments on my post > active thread involvement
    s = 0.0
    if reply_to_my_comment:
        s += 140.0
    if own_post:
        s += 110.0
    elif recent_involved:
        s += 85.0
    else:
        s += 60.0

    m = _post_metrics(post)
    ps = abs(int(m.get("score", 0) or 0))
    pc = int(m.get("comments", 0) or 0)
    s += 6.0 * math.log1p(ps)
    s += 4.0 * math.log1p(pc)

    # Slight bias to newer candidates within the scan
    s += 0.8 * float(rank_hint)
    return s



def handle_toxic_target(
    client: HttpClient,
    cfg: Config,
    state: Dict[str, Any],
    post_id: str,
    comment_id: str,
    text: str,
    *,
    tag: str = "",
) -> bool:
    """Handle toxic engagement target (auto-downvote + no-reply).

    Returns True if 'no_reply' policy should be applied.
    """
    tox_cfg = getattr(cfg, "toxic", None)
    if not tox_cfg:
        return False

    tox, reason = is_toxic_incoming(text)
    if not tox:
        return False

    pid = str(post_id or "")
    cid = str(comment_id or "")
    now = time.time()

    tb = state.setdefault("toxic", {})
    if not isinstance(tb, dict):
        state["toxic"] = {}
        tb = state["toxic"]
    tb["hits"] = int(tb.get("hits", 0) or 0) + 1

    skipped = tb.setdefault("skipped", {})
    if not isinstance(skipped, dict):
        tb["skipped"] = {}
        skipped = tb["skipped"]
    if pid and cid:
        skipped[f"{pid}:{cid}"] = {"ts": now, "reason": str(reason), "tag": str(tag or "")}
        if len(skipped) > 260:
            try:
                items = sorted(skipped.items(), key=lambda kv: float(_safe_dict(kv[1]).get("ts", 0.0)))
                for k, _v in items[:-180]:
                    skipped.pop(k, None)
            except Exception as e:
                log_debug_exc("handle_toxic_target:silent", e)
                pass

    if bool(getattr(tox_cfg, "auto_downvote", True)) and pid:
        try:
            vm = _voted_posts_map(state)
            if pid not in vm:
                vote_post(client, cfg, state, pid, "down")
            dv = tb.setdefault("downvoted_posts", {})
            if not isinstance(dv, dict):
                tb["downvoted_posts"] = {}
                dv = tb["downvoted_posts"]
            dv[pid] = now
            _lru_prune_map(dv, 400)
        except Exception as e:
            log_debug_exc("handle_toxic_target:silent", e)
            pass

    if bool(getattr(tox_cfg, "no_reply", True)) and pid and cid:
        replied_ts = state.get("replied_ts")
        if not isinstance(replied_ts, dict):
            replied_ts = {}
            state["replied_ts"] = replied_ts
        replied_ts[f"{pid}:{cid}"] = now

    return bool(getattr(tox_cfg, "no_reply", True))

def _build_reply_inbox(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    memory: List[Dict[str, Any]],
    posts_cache: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a prioritized list of (post_id, comment_id) I should reply to."""
    now = time.time()

    # cache to avoid hammering /comments in burst mode
    cache = _safe_dict(state.get("reply_inbox_cache"))
    cache_ts = _safe_float(cache.get("ts"), 0.0)
    if cache_ts > 0 and (now - cache_ts) < max(15, int(getattr(tuning, "inbox_scan_min_sec", 90))):
        items = _safe_list(cache.get("items", []))
        return [it for it in items if isinstance(it, dict)]

    replied_ts = _safe_dict(state.get("replied_ts"))

    # candidate posts: (a) my posts (hot/recent), (b) posts I recently interacted with
    posts_by_id: Dict[str, Dict[str, Any]] = {}
    for p in posts_cache:
        if isinstance(p, dict):
            pid = str(p.get("id") or "")
            if pid:
                posts_by_id[pid] = p

    # (a) my posts first
    own_posts = [p for p in posts_cache if isinstance(p, dict) and is_own_post(cfg, p)]
    own_posts.sort(key=lambda p: (_post_metrics(p).get("comments", 0), _post_metrics(p).get("score", 0)), reverse=True)
    own_posts = own_posts[: max(2, min(8, int(getattr(tuning, "inbox_max_posts", 10)) // 2 or 5))]

    # (b) recent involved posts (from memory)
    recent_pids = set(_recent_involved_post_ids_from_memory(
        memory,
        days=int(getattr(tuning, "inbox_recent_thread_days", 3)),
        max_n=int(getattr(tuning, "inbox_max_posts", 10)) * 2,
    ))

    cand_posts: List[Dict[str, Any]] = []
    seen_pid: set = set()
    for p in own_posts:
        pid = str(p.get("id") or "")
        if pid and pid not in seen_pid:
            cand_posts.append(p)
            seen_pid.add(pid)

    # add recent-involved posts (if present in cache)
    for pid in list(recent_pids):
        if pid in seen_pid:
            continue
        p = posts_by_id.get(pid)
        if isinstance(p, dict):
            cand_posts.append(p)
            seen_pid.add(pid)
        if len(cand_posts) >= int(getattr(tuning, "inbox_max_posts", 10)):
            break

    # If still short, top-scoring posts as a weak backfill (still only reply-to-my-comment)
    if len(cand_posts) < max(3, int(getattr(tuning, "inbox_max_posts", 10))):
        others = [p for p in posts_cache if isinstance(p, dict) and str(p.get("id") or "") not in seen_pid]
        others.sort(key=lambda p: (_post_metrics(p).get("score", 0), _post_metrics(p).get("comments", 0)), reverse=True)
        for p in others[:8]:
            cand_posts.append(p)
            seen_pid.add(str(p.get("id") or ""))
            if len(cand_posts) >= int(getattr(tuning, "inbox_max_posts", 10)):
                break

    items: List[Dict[str, Any]] = []

    scan_n = int(getattr(tuning, "inbox_scan_comments_per_post", 40))
    scan_n = max(10, min(200, scan_n))

    for p in cand_posts[: int(getattr(tuning, "inbox_max_posts", 10))]:
        pid = str(p.get("id") or "")
        if not pid:
            continue

        own_post = is_own_post(cfg, p)
        recent_involved = (pid in recent_pids)

        try:
            comments = list_comments(client, pid)
        except Exception:
            comments = []

        if not comments:
            continue

        my_comment_ids = set()
        for c in comments[-120:]:
            if isinstance(c, dict) and is_own_comment(cfg, c):
                cid0 = str(c.get("id") or "")
                if cid0:
                    my_comment_ids.add(cid0)

        # iterate newest-first
        window = comments[-scan_n:]
        for rank, c in enumerate(reversed(window), start=1):
            if not isinstance(c, dict):
                continue
            if is_own_comment(cfg, c):
                continue

            cid = str(c.get("id") or "")
            if not cid:
                continue

            parent_id = _comment_parent_id(c)
            reply_to_my_comment = bool(parent_id and parent_id in my_comment_ids)
            if reply_to_my_comment:
                _note_reply_received(state, memory, post_id=pid, my_comment_id=parent_id, reply_comment_id=cid)

            # only take items that are explicitly "to me"
            if (not own_post) and (not reply_to_my_comment):
                continue

            key = f"{pid}:{cid}"

            # toxic/taunt: auto-downvote + no-reply
            ctext = str(c.get("content") or "")
            if ctext and handle_toxic_target(client, cfg, state, pid, cid, ctext, tag="inbox"):
                replied_ts[key] = now
                continue
            last = _safe_float(replied_ts.get(key), 0.0)
            if last > 0 and (now - last) < 60 * 60 * 24 * 7:
                continue

            # already replied? (server-side check)
            try:
                if _has_replied_to_comment(cfg, comments, cid):
                    replied_ts[key] = now
                    continue
            except Exception as e:
                log_debug_exc("_build_reply_inbox:silent", e)
                pass
            comment_ts = _comment_created_ts(c)

            score = _inbox_item_score(p, own_post=own_post, reply_to_my_comment=reply_to_my_comment, recent_involved=recent_involved, rank_hint=rank)
            why = "reply_to_my_comment" if reply_to_my_comment else "comment_on_my_post"
            items.append({
                "post_id": pid,
                "comment_id": cid,
                "comment_ts": float(comment_ts or 0.0),
                "replied_to_comment_id": str(parent_id or ""),
                "own_post": bool(own_post),
                "reply_to_my_comment": bool(reply_to_my_comment),
                "recent_involved": bool(recent_involved),
                "score": float(score),
                "why": why,
            })

    items.sort(key=lambda it: float(it.get("score", 0.0)), reverse=True)
    items = items[: max(3, int(getattr(tuning, "inbox_max_posts", 10)) * 2)]

    state["replied_ts"] = replied_ts
    state["reply_inbox_cache"] = {"ts": now, "items": items}
    return items

def try_reply_priority_queue(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    memory: List[Dict[str, Any]],
    semantic: Dict[str, Any],
    policy: Dict[str, Any],
    brain: Dict[str, Any],
    posts_cache: List[Dict[str, Any]],
    comment_limiter: SlidingWindowLimiter,
    comment_pace_sec: int,
    bm25: Optional[BM25Index],
) -> int:
    """(Unit 04) Reply priority queue: replies-to-me first, then my posts."""
    gap_com = gap_remaining(float(state.get("last_comment_ts", 0.0) or 0.0), int(comment_pace_sec))
    if gap_com > 0:
        protocol_set_reason(state, "comment", "comment:rate_limited", f"gap={int(gap_com)}s")
        return 0
    if comment_limiter.remaining() <= 0:
        protocol_set_reason(state, "comment", "comment:rate_limited", "window_remaining=0")
        return 0

    drives = get_persona_drives(brain)

    inbox = _build_reply_inbox(client, cfg, tuning, state, memory, posts_cache)
    if not inbox:
        protocol_set_reason(state, "comment", "comment:no_target", "inbox_empty")
        return 0


    # v23.6: optional reply queue scoring v2 (open questions + freshness + waiting penalty)
    if _reply_score_v2_enabled(state):
        try:
            now_sc = time.time()
            for it in inbox:
                if not isinstance(it, dict):
                    continue
                sc, br = _reply_score_v2(state, it, now_sc)
                it["score_v2"] = float(sc)
                it["score_v2_br"] = br
            inbox.sort(
                key=lambda it: (
                    float(_safe_float(it.get("score_v2"), _safe_float(it.get("score"), 0.0))),
                    float(_safe_float(it.get("comment_ts"), 0.0)),
                ),
                reverse=True,
            )
            protocol_bump_counter(state, "reply_scored", 1)
        except Exception as e:
            log_debug_exc("reply_score_v2:silent", e)
            pass

    # Pick the first inbox item that passes the conversation protocol gate.
    chosen: Optional[Dict[str, Any]] = None
    chosen_post: Dict[str, Any] = {}
    chosen_comments: List[Dict[str, Any]] = []
    chosen_target: Dict[str, Any] = {}
    chosen_conv_key = ""
    chosen_conv_meta: Dict[str, Any] = {}

    posts_by_id: Dict[str, Dict[str, Any]] = {str(p.get("id") or ""): p for p in (posts_cache or []) if isinstance(p, dict)}
    comments_cache: Dict[str, List[Dict[str, Any]]] = {}
    last_gate_reason = ""

    for top in inbox[: min(12, len(inbox))]:
        if (not bool(top.get("reply_to_my_comment"))) and drives.get("debate", 0.7) < 0.45 and random.random() < 0.6:
            continue

        pid = str(top.get("post_id") or "")
        cid = str(top.get("comment_id") or "")
        if not pid or not cid:
            continue

        post = posts_by_id.get(pid)
        if not isinstance(post, dict) or not str(post.get("id") or ""):
            post = get_post(client, pid) or {}
            posts_by_id[pid] = post if isinstance(post, dict) else {}

        try:
            comments = comments_cache.get(pid)
            if comments is None:
                comments = list_comments(client, pid)
                comments_cache[pid] = comments
        except Exception:
            comments = []
            comments_cache[pid] = []

        if not comments:
            continue

        target_c = None
        for c in comments:
            if isinstance(c, dict) and str(c.get("id") or "") == cid:
                target_c = c
                break
        if not isinstance(target_c, dict):
            replied_ts = _safe_dict(state.get("replied_ts"))
            replied_ts[f"{pid}:{cid}"] = time.time()
            state["replied_ts"] = replied_ts
            continue


        # toxic/taunt: do not engage directly
        try:
            ttxt = str(target_c.get("content") or "")
        except Exception:
            ttxt = ""
        if ttxt and handle_toxic_target(client, cfg, state, pid, cid, ttxt, tag="priority_queue_select"):
            continue

        ok, conv_key, _conv, conv_meta, reason = _reply_protocol_prepare(state, cfg, tuning, pid, comments, cid, time.time())
        if not ok:
            last_gate_reason = str(reason or "")
            continue

        chosen = top
        chosen_post = post
        chosen_comments = comments
        chosen_target = target_c
        chosen_conv_key = conv_key
        chosen_conv_meta = conv_meta
        break

    if not chosen:
        # inbox exists but nothing eligible (often conversation protocol gate)
        if last_gate_reason:
            protocol_set_reason(state, "comment", "comment:qa_fail", f"protocol_gate:{one_line(last_gate_reason, 120)}")
        else:
            protocol_set_reason(state, "comment", "comment:no_target", "no_eligible_inbox")
        return 0

    top = chosen

    # v23.6: score breakdown event for debugging (only when scoring v2 enabled)
    if _reply_score_v2_enabled(state):
        try:
            br = _safe_dict(top.get("score_v2_br"))
            if br:
                log_event(
                    "reply.score_breakdown",
                    post_id=str(top.get("post_id") or ""),
                    comment_id=str(top.get("comment_id") or ""),
                    score=float(top.get("score_v2", top.get("score", 0.0)) or 0.0),
                    breakdown=br,
                )
        except Exception:
            pass

    if not pid or not cid:
        protocol_set_reason(state, "comment", "comment:no_target", "missing_pid_or_cid")
        return 0

    # use previously fetched post/comments/target
    post = chosen_post
    comments = chosen_comments
    target_c = chosen_target

    now = time.time()
    replied_ts = _safe_dict(state.get("replied_ts"))
    replied_ts[f"{pid}:{cid}"] = replied_ts.get(f"{pid}:{cid}", 0.0) or 0.0  # keep key

    # build context
    ingest_post_into_context(state, post, brain=brain)
    ingest_comments_into_context(state, pid, comments, brain=brain, cfg=cfg)
    th = get_thread(state, pid)
    synthesize_thread(th)

    user_key = str(target_c.get("nickname") or post.get("nickname") or "user")
    user = get_user(state, user_key)

    # lock focus target
    set_focus(state, mode="reply", post_id=pid, post=post, comment=target_c)

    # (v18.3) pass conversation meta to reply generator
    try:
        state.setdefault("focus", {})
        if isinstance(state.get("focus"), dict):
            state["focus"]["conv"] = chosen_conv_meta
    except Exception as e:
        log_debug_exc("try_reply_priority_queue:silent", e)
        pass

    txt = ""
    meta: Dict[str, Any] = {}
    ok_txt = False
    ground_reason = ""
    last_fail_bucket = ""
    last_fail_detail = ""
    reply_to_own_post = bool(top.get("own_post"))
    last_candidate = ""
    for _try in range(_regen_budget(cfg)):
        txt, meta = build_reply_text(
            cfg, tuning, state, policy, th, user,
            bm25=bm25,
            brain=brain,
            reply_to_own_post=reply_to_own_post,
            is_reply=True,
        )
        last_candidate = txt
        dup, db = dup_guard_bucket(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
        if dup:
            last_fail_bucket = str(db or "")
            last_fail_detail = str(db or "")
            _bump_gen_fail(state, db)
            if (_try % 3) == 0:
                log_action("DUP", f"block mode=reply reason={db} try={_try+1}")
            continue
        ok_g, reason = validate_grounding(txt, _safe_dict(state.get("focus")), "reply")
        if not ok_g:
            last_fail_bucket = "qa_fail"
            last_fail_detail = f"ground:{one_line(str(reason), 120)}"
            log_action("VALIDATE", f"fail mode=reply reason={reason} try={_try+1}")
            continue
        ok_q, qrep = qa_check_text(cfg, txt, kind="reply", focus=_safe_dict(state.get("focus")), mode="reply")
        if not ok_q:
            last_fail_bucket = "qa_fail"
            try:
                last_fail_detail = f"qa:score={qrep.get('score')} issues={','.join((qrep.get('issues', []) or [])[:3])}"
            except Exception:
                last_fail_detail = "qa_fail"
            _bump_gen_fail(state, "qa_fail")
            log_action("QA", f"fail mode=reply score={qrep.get('score')} issues={','.join(qrep.get('issues', [])[:3])} try={_try+1}")
            continue
        ok_txt = True
        ground_reason = reason
        break

    if not ok_txt:
        fb = _qa_fallback_2stage(last_candidate, is_reply=True)
        if fb:
            try:
                dup, db = dup_guard_bucket(state, fb, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
                if dup:
                    last_fail_bucket = str(db or "")
                    last_fail_detail = str(db or "")
                    _bump_gen_fail(state, db)
                if not dup:
                    ok_g, reason = validate_grounding(fb, _safe_dict(state.get("focus")), "reply")
                    if ok_g:
                        ok_q, qrep = qa_check_text(cfg, fb, kind="reply", focus=_safe_dict(state.get("focus")), mode="reply")
                        if ok_q:
                            txt = fb
                            ok_txt = True
                            ground_reason = reason
                        else:
                            last_fail_bucket = "qa_fail"
                            try:
                                last_fail_detail = f"qa:score={qrep.get('score')} issues={','.join((qrep.get('issues', []) or [])[:3])}"
                            except Exception:
                                last_fail_detail = "qa_fail"
            except Exception:
                pass
    if not ok_txt:
        replied_ts[f"{pid}:{cid}"] = now
        state["replied_ts"] = replied_ts
        code = "comment:qa_fail"
        if last_fail_bucket == "dup_fp":
            code = "comment:dup_fp"
        elif last_fail_bucket == "dup_sim":
            code = "comment:dup_sim"
        protocol_set_reason(state, "comment", code, one_line(last_fail_detail or last_fail_bucket or "gen_fail", 140))
        return 0

    if not comment_limiter.allow():
        if getattr(cfg, "debug", None) and cfg.debug.log_blocks:
            log_debug("gate:comment blocked by window (allow=false)")
        protocol_set_reason(state, "comment", "comment:rate_limited", "window_allow=false")
        return 0

    _bump_action_counter(state, "action_attempt", "reply")
    try:
        res = create_comment(client, cfg, tuning, state, pid, txt, parent_id=cid)
    except RateLimitError as e_rl:
        protocol_set_reason(state, "comment", "comment:rate_limited", one_line(str(e_rl), 180))
        raise
    except PowTimeoutError as e:
        protocol_set_reason(state, "comment", "comment:pow_timeout", one_line(str(e), 180))
        _bump_action_counter(state, "action_fail", "reply")
        return 0
    except requests.HTTPError as e_http:
        protocol_set_reason(state, "comment", "comment:qa_fail", one_line(str(e_http), 180))
        raise
    except Exception as e:
        protocol_set_reason(state, "comment", "comment:qa_fail", one_line(repr(e), 180))
        raise
    if not res:
        protocol_set_reason(state, "comment", "comment:no_target", "comment_create_failed")
        _bump_action_counter(state, "action_fail", "reply")
        return 0
    _bump_action_counter(state, "action_success", "reply")

    # (v18.3) commit conversation protocol state
    try:
        _reply_protocol_commit(state, chosen_conv_key, now, _extract_comment_id(res), txt)
    except Exception as e:
        log_debug_exc("try_reply_priority_queue:silent", e)
        pass

    remember_text(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)

    fp_ttl = _env_int("MERSOOM_RECENT_TEXT_FP_TTL_SEC", 6 * 3600, 60, 30 * 24 * 3600)

    fp_keep = _env_int("MERSOOM_RECENT_TEXT_FP_KEEP_MAX", 1200, 50, 20000)

    remember_fp(state, txt, for_post=False, ttl_sec=max(int(fp_ttl), int(cfg.timing.same_text_gap_sec)), keep_max=int(fp_keep))
    remember_simhash(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
    remember_dup_signatures(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)

    replied_ts[f"{pid}:{cid}"] = now
    state["replied_ts"] = replied_ts
    # (19.4) update per-thread debounce timestamp

    cts = _safe_dict(state.get("commented_ts"))

    cts[str(pid)] = now

    state["commented_ts"] = cts

    state["last_comment_ts"] = now
    state["contrib_count_today"] = int(state.get("contrib_count_today", 0)) + 1
    state["total_actions"] = int(state.get("total_actions", 0)) + 1
    update_persona_maturity(brain, state)

    action_type = "reply_own" if reply_to_own_post else "reply_other"
    bump_semantic(semantic, _today_kst(), action_type, 1.0)
    _bump_relation(state, user_key)

    before = _post_metrics(post)
    item = {
        "ts": now,
        "action": "reply",
        "action_type": action_type,
        "post_id": pid,
        "parent_id": cid,
        "category": meta.get("category", "general"),
        "context_key": meta.get("context_key", "gen"),
        "kw": meta.get("kw", ""),
        "text": txt,
        "used_strategy": meta.get("strategy", ""),
        "used_tone": meta.get("tone", ""),
        "used_length": meta.get("length", ""),
        "used_reply_style": meta.get("reply_style", ""),
        "weak_context": bool(meta.get("weak_context")),
        "template_id": meta.get("template_id", ""),
        "used_quotes": bool(meta.get("used_quotes")),
        "novelty": _novelty_score(state, txt),
        "comment_id": _extract_comment_id(res),
        "qa_score": float(qrep.get("score", 0) or 0),
        "qa_issues": list(qrep.get("issues", []))[:6],
        "qa_rep3": float(qrep.get("rep3", 0.0) or 0.0),
        "qa_im_ratio": float(qrep.get("im_ratio", 0.0) or 0.0),
        "qa_line_prefix_dup": float(qrep.get("line_prefix_dup", 0.0) or 0.0),
        "eval_due_ts": schedule_eval_due(tuning),
        "metrics_before": before,
        "metrics_after": {},
        "evaluated": False,
        "reward_scalar": 0.0,
        "ground_reason": ground_reason,
        "target_nick": user_key,
        # unit 04 metadata
        "inbox": True,
        "inbox_why": str(top.get("why") or ""),
        "inbox_score": float(top.get("score", 0.0) or 0.0),
        "conv_key": str(chosen_conv_key or ""),
        "conv_root": str((chosen_conv_meta or {}).get("root_comment_id") or ""),
        "conv_turns_total": int((chosen_conv_meta or {}).get("turns_total") or 0),
    }
    item["proxy_reward"] = compute_proxy_reward(txt, mode="reply", ground_reason=ground_reason)
    item.setdefault("brain_proxy_applied", False)
    item.setdefault("brain_reward_applied", False)
    record_memory(memory, item, tuning, archive_path_jsonl=cfg.paths.memory_archive_jsonl)
    try:
        apply_brain_proxy_update(brain, tuning, item)
    except Exception as e:
        log_debug_exc("try_reply_priority_queue:silent", e)
        pass

    log_action("REPLY", f"inbox {top.get('why')} post={pid} parent={cid} | {one_line(txt)}")
    return 1

def try_reply_to_own_posts(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    memory: List[Dict[str, Any]],
    semantic: Dict[str, Any],
    policy: Dict[str, Any],
    brain: Dict[str, Any],
    posts_cache: List[Dict[str, Any]],
    comment_limiter: SlidingWindowLimiter,
    comment_pace_sec: int,
    bm25: Optional[BM25Index],
) -> int:
    """
    Trait: "내 글에 댓글 달리면 대댓글로 논쟁을 좋아함"을 실행.
    - own post 중 최근/핫한 글을 훑고
    - 상대 댓글 중 아직 대댓글을 안단 것에 우선 응답
    """
    gap_com = gap_remaining(float(state.get("last_comment_ts", 0.0) or 0.0), int(comment_pace_sec))
    if gap_com > 0:
        return 0
    if comment_limiter.remaining() <= 0:
        return 0

    drives = get_persona_drives(brain)
    maturity = get_maturity_level(brain, state)

    # reply preference gate (avoid spamming if debate drive is low)
    if drives.get("debate", 0.7) < 0.45 and random.random() < 0.6:
        return 0

    replied_ts = _safe_dict(state.get("replied_ts"))
    now = time.time()

    # pick candidate own posts with comments
    own_posts = [p for p in posts_cache if isinstance(p, dict) and is_own_post(cfg, p)]
    # prioritize: more comments / score
    own_posts.sort(key=lambda p: (_post_metrics(p).get("comments", 0), _post_metrics(p).get("score", 0)), reverse=True)
    own_posts = own_posts[:6]

    for p in own_posts:
        pid = str(p.get("id") or "")
        if not pid:
            continue
        # skip posts with no comments (as per cached metrics)
        if int(_post_metrics(p).get("comments", 0)) <= 0:
            continue

        try:
            comments = list_comments(client, pid)
        except Exception:
            comments = []

        if not comments:
            continue

        # find a fresh other comment to reply to
        target_c = None
        for c in reversed(comments[-18:]):
            if not isinstance(c, dict):
                continue
            if is_own_comment(cfg, c):
                continue
            cid = str(c.get("id") or "")
            if not cid:
                continue
            key = f"{pid}:{cid}"
            # local cooldown
            last = float(replied_ts.get(key, 0.0) or 0.0)
            if last > 0 and (now - last) < 60 * 60 * 24 * 7:
                continue
            # server-side: already replied?
            if _has_replied_to_comment(cfg, comments, cid):
                replied_ts[key] = now
                continue
            target_c = c
            break

        if not target_c:
            continue

        cid = str(target_c.get("id") or "")
        if not cid:
            continue

        # (v18.3) conversation protocol: do not reply twice without new remote turn
        okp, conv_key, _conv, conv_meta, reason = _reply_protocol_prepare(state, cfg, tuning, pid, comments, cid, now)
        if not okp:
            replied_ts[f"{pid}:{cid}"] = now
            state["replied_ts"] = replied_ts
            continue

        # build context
        ingest_post_into_context(state, p, brain=brain)
        ingest_comments_into_context(state, pid, comments, brain=brain, cfg=cfg)
        th = get_thread(state, pid)
        synthesize_thread(th)

        user_key = str(target_c.get("nickname") or p.get("nickname") or "user")
        user = get_user(state, user_key)

        # v15 Unit 02: lock focus target for reply-to-comment (own post)
        set_focus(state, mode="reply", post_id=pid, post=p, comment=target_c)

        # (v18.3) pass conversation meta to reply generator
        try:
            state.setdefault("focus", {})
            if isinstance(state.get("focus"), dict):
                state["focus"]["conv"] = conv_meta
        except Exception as e:
            log_debug_exc("try_reply_to_own_posts:silent", e)
            pass

        txt = ""
        meta: Dict[str, Any] = {}
        ok_txt = False
        ground_reason = ""
        last_candidate = ""
        for _try in range(_regen_budget(cfg)):
            txt, meta = build_reply_text(
                cfg, tuning, state, policy, th, user,
                bm25=bm25,
                brain=brain,
                reply_to_own_post=True,
                is_reply=True,
            )
            last_candidate = txt
            dup, db = dup_guard_bucket(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
            if dup:
                _bump_gen_fail(state, db)
                if (_try % 3) == 0:
                    log_action("DUP", f"block mode=reply reason={db} try={_try+1}")
                continue
            ok_g, reason = validate_grounding(txt, _safe_dict(state.get("focus")), "reply")
            if not ok_g:
                log_action("VALIDATE", f"fail mode=reply reason={reason} try={_try+1}")
                continue

            ok_q, qrep = qa_check_text(cfg, txt, kind="reply", focus=_safe_dict(state.get("focus")), mode="reply")
            if not ok_q:
                _bump_gen_fail(state, "qa_fail")
                log_action("QA", f"fail mode=reply score={qrep.get('score')} issues={','.join(qrep.get('issues', [])[:3])} try={_try+1}")
                continue
            ok_txt = True
            ground_reason = reason
            break

        if not ok_txt:
            fb = _qa_fallback_2stage(last_candidate, is_reply=True)
            if fb:
                try:
                    dup, db = dup_guard_bucket(state, fb, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
                    if dup:
                        _bump_gen_fail(state, db)
                    if not dup:
                        ok_g, reason = validate_grounding(fb, _safe_dict(state.get("focus")), "reply")
                        if ok_g:
                            ok_q, qrep = qa_check_text(cfg, fb, kind="reply", focus=_safe_dict(state.get("focus")), mode="reply")
                            if ok_q:
                                txt = fb
                                ok_txt = True
                                ground_reason = reason
                except Exception:
                    pass
            if not ok_txt:
                # don't burn limiter on low-quality / near-duplicate
                replied_ts[f"{pid}:{cid}"] = now
                state["replied_ts"] = replied_ts
                continue

        if not comment_limiter.allow():
            if getattr(cfg, "debug", None) and cfg.debug.log_blocks:
                log_debug("gate:comment blocked by window (allow=false)")
            return 0

        res = create_comment(client, cfg, tuning, state, pid, txt, parent_id=cid)
        if not res:
            _bump_action_counter(state, "action_fail", "post")
            return 0

        # (v18.3) commit conversation protocol state
        try:
            _reply_protocol_commit(state, conv_key, now, _extract_comment_id(res), txt)
        except Exception as e:
            log_debug_exc("try_reply_to_own_posts:silent", e)
            pass

        remember_text(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)

        fp_ttl = _env_int("MERSOOM_RECENT_TEXT_FP_TTL_SEC", 6 * 3600, 60, 30 * 24 * 3600)

        fp_keep = _env_int("MERSOOM_RECENT_TEXT_FP_KEEP_MAX", 1200, 50, 20000)

        remember_fp(state, txt, for_post=False, ttl_sec=max(int(fp_ttl), int(cfg.timing.same_text_gap_sec)), keep_max=int(fp_keep))
        remember_simhash(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
        remember_dup_signatures(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)

        replied_ts[f"{pid}:{cid}"] = now
        state["replied_ts"] = replied_ts
        # (19.4) update per-thread debounce timestamp

        cts = _safe_dict(state.get("commented_ts"))

        cts[str(pid)] = now

        state["commented_ts"] = cts

        state["last_comment_ts"] = now
        state["contrib_count_today"] = int(state.get("contrib_count_today", 0)) + 1
        state["total_actions"] = int(state.get("total_actions", 0)) + 1
        update_persona_maturity(brain, state)
        bump_semantic(semantic, _today_kst(), "reply_own", 1.0)
        _bump_relation(state, user_key)

        before = _post_metrics(p)
        item = {
            "ts": now,
            "action": "reply",
            "action_type": "reply_own",
            "post_id": pid,
            "parent_id": cid,
            "category": meta.get("category", "general"),
            "context_key": meta.get("context_key", "gen"),
            "kw": meta.get("kw", ""),
            "text": txt,
            "used_strategy": meta.get("strategy", ""),
            "used_tone": meta.get("tone", ""),
            "used_length": meta.get("length", ""),
            "used_reply_style": meta.get("reply_style", ""),
            "weak_context": bool(meta.get("weak_context")),
            "template_id": meta.get("template_id", ""),
            "used_quotes": bool(meta.get("used_quotes")),
            "novelty": _novelty_score(state, txt),
        "comment_id": _extract_comment_id(res),
        "qa_score": float(qrep.get("score", 0) or 0),
        "qa_issues": list(qrep.get("issues", []))[:6],
        "qa_rep3": float(qrep.get("rep3", 0.0) or 0.0),
        "qa_im_ratio": float(qrep.get("im_ratio", 0.0) or 0.0),
        "qa_line_prefix_dup": float(qrep.get("line_prefix_dup", 0.0) or 0.0),
            "eval_due_ts": schedule_eval_due(tuning),
            "metrics_before": before,
            "metrics_after": {},
            "evaluated": False,
            "reward_scalar": 0.0,
        }
        item["ground_reason"] = ground_reason
        item["target_nick"] = user_key
        item["proxy_reward"] = compute_proxy_reward(txt, mode="reply", ground_reason=ground_reason)
        item.setdefault("brain_proxy_applied", False)
        item.setdefault("brain_reward_applied", False)
        record_memory(memory, item, tuning, archive_path_jsonl=cfg.paths.memory_archive_jsonl)
        try:
            apply_brain_proxy_update(brain, tuning, item)
        except Exception as e:
            log_debug_exc("try_reply_to_own_posts:silent", e)
            pass

        log_action("REPLY", f"own_post={pid} parent={cid} | {one_line(txt)}")
        return 1

    return 0

################################################################################
# 14.4. QA + FALLBACK (quality gates + fallback path)
################################################################################
def do_contribution(
    client: HttpClient,
    cfg: Config,
    tuning: AgentTuning,
    state: Dict[str, Any],
    memory: List[Dict[str, Any]],
    semantic: Dict[str, Any],
    policy: Dict[str, Any],
    brain: Dict[str, Any],
    posts_cache: List[Dict[str, Any]],
    post_limiter: SlidingWindowLimiter,
    comment_limiter: SlidingWindowLimiter,
    comment_pace_sec: int,
    post_pace_sec: int,
    bm25: Optional[BM25Index],
) -> int:
    update_daily_counters(state)
    gc_state(state)
    log_health_if_due(client, state)

    drives = get_persona_drives(brain)
    maturity = get_maturity_level(brain, state)
    ctx_hint = _pick_ctx_from_flow(brain)

    gap_post = gap_remaining(float(state.get("last_post_ts", 0.0) or 0.0), int(post_pace_sec))
    can_post = (gap_post <= 0) and (post_limiter.remaining() > 0)

    gap_com = gap_remaining(float(state.get("last_comment_ts", 0.0) or 0.0), int(comment_pace_sec))
    can_comment = (gap_com <= 0) and (comment_limiter.remaining() > 0)

    # v20.10 (B-2): ensure a standard reason is recorded when both posting and commenting are gated.
    if (not can_post) and (not can_comment):
        try:
            det = (
                f"post_gap={int(max(0, gap_post))}s rem_post={int(post_limiter.remaining())} | "
                f"comment_gap={int(max(0, gap_com))}s rem_comment={int(comment_limiter.remaining())}"
            )
            protocol_set_reason(state, "comment", "comment:rate_limited", det)
        except Exception:
            protocol_set_reason(state, "comment", "comment:rate_limited")

    # Optional: log why actions are blocked (helps tuning without spamming by default).
    if getattr(cfg, "debug", None) and cfg.debug.log_blocks and not (can_post or can_comment):
        try:
            now_ts = time.time()
            last_ts = float(state.get("_last_gate_log_ts", 0.0) or 0.0)
            if now_ts - last_ts >= 60.0:
                state["_last_gate_log_ts"] = now_ts
                log_debug(
                    f"gates blocked | post:gap={int(max(0,gap_post))}s rem={post_limiter.remaining()} "
                    f"| comment:gap={int(max(0,gap_com))}s rem={comment_limiter.remaining()}"
                )
        except Exception as e:
            log_debug_exc("do_contribution:silent", e)
            pass

    # 0) (Unit 04) Priority inbox: replies-to-me + my-post comments first
    if can_comment:
        done = try_reply_priority_queue(
            client, cfg, tuning,
            state, memory, semantic, policy, brain,
            posts_cache,
            comment_limiter, comment_pace_sec,
            bm25=bm25
        )
        if done > 0:
            return done


    # v19.3 Heartbeat enforcement: prioritize comment quota, then 1x contribute per cycle
    hb_force_action = ""
    force_post_now = False
    try:
        hb_cfg = getattr(cfg, "heartbeat", None)
        if hb_cfg and bool(getattr(hb_cfg, "enabled", False)):
            proto = _safe_dict(state.get("protocol"))
            hb = _safe_dict(_safe_dict(proto).get("heartbeat"))
            if hb and bool(hb.get("active")):
                q = _safe_dict(hb.get("quota"))
                # v20.3: clamp comment quota to feasible max (pace/limiter) once per cycle
                try:
                    _hb_clamp_comment_target(cfg, state, int(comment_pace_sec), comment_limiter)
                except Exception:
                    pass
                q = _safe_dict(hb.get("quota"))
                c_tgt = int(q.get("comments_target", 0) or 0)
                c_done = int(q.get("comments_done", 0) or 0)
                contrib_done = bool(q.get("contribute_done", False))
                # v20.3: record why HB comment quota is blocked (for observability)
                if c_tgt > 0 and c_done < c_tgt and (not can_comment):
                    try:
                        proto2 = _sdict(state, "protocol")
                        if int(getattr(comment_limiter, "capacity", 0) or 0) <= 0:
                            proto2["hb_block_reason"] = "comment_limit0"
                        elif float(gap_com) > 0:
                            proto2["hb_block_reason"] = "cooldown"
                        elif int(comment_limiter.remaining()) <= 0:
                            proto2["hb_block_reason"] = "comment_limit0"
                        else:
                            proto2["hb_block_reason"] = "cooldown"
                    except Exception:
                        pass
                if c_tgt > 0 and c_done < c_tgt and can_comment:
                    hb_force_action = "comment_other"
                elif (c_tgt <= 0 or c_done >= c_tgt) and (not contrib_done) and can_post:
                    hb_force_action = "post_new"
                    force_post_now = True
    except Exception:
        hb_force_action = ""
        force_post_now = False

    # 1) Choose action type (pattern becomes learnable + maturity reduces randomness)
    action_type = hb_force_action or ""
    if (not action_type) and (can_post or can_comment):
        action_type = choose_arm_adaptive(policy, "action_type", context_key=ctx_hint, maturity=maturity, brain=brain)
    if action_type not in ("post_new", "comment_other", "reply_other", "reply_own"):
        action_type = "comment_other" if can_comment else "post_new"

    # reply_own is handled by step (0); if it was selected here, treat it as a regular comment path
    if action_type == "reply_own":
        action_type = "comment_other"

    # 2) Post decision: fame+maturity -> 조금 더 자주 글(하지만 제한 내)
    flow_bonus = 0.03 if _safe_list(_safe_dict(brain.get("community")).get("rising")) else 0.0
    post_prob = 0.05 + 0.10 * drives.get("fame", 0.7) + 0.05 * maturity + flow_bonus
    post_prob = max(0.03, min(0.22, float(post_prob)))

    if action_type == "post_new" and can_post and (force_post_now or random.random() < post_prob):
        if not post_limiter.allow():
            if getattr(cfg, "debug", None) and cfg.debug.log_blocks:
                log_debug("gate:post blocked by window (allow=false)")
            return 0
        title = ""
        body = ""
        meta: Dict[str, Any] = {}
        combo = ""
        ok_post = False
        for _try in range(_regen_budget(cfg)):
            title, body, meta = build_post_text(cfg, tuning, state, policy, semantic, brain, bm25)
            combo = f"{title}\n{body}"
            dup, db = dup_guard_bucket(state, combo, for_post=True, same_text_gap_sec=cfg.timing.same_text_gap_sec)
            if dup:
                _bump_gen_fail(state, db)
                if (_try % 3) == 0:
                    log_action("DUP", f"block mode=post reason={db} try={_try+1}")
                continue

            ok_q, qrep = qa_check_post(cfg, title, body)
            if not ok_q:
                _bump_gen_fail(state, "qa_fail")
                log_action("QA", f"fail mode=post score={qrep.get('score')} issues={','.join(qrep.get('issues', [])[:3])} try={_try+1}")
                continue
            ok_post = True
            break
        if not ok_post:
            return 0

        if dup_action_should_skip(state, action="post", target_id=_text_hash(combo), endpoint_key="/posts"):
            protocol_set_reason(state, "post", "post:dup_action", "recent_action_guard")
            return 0
        _bump_action_counter(state, "action_attempt", "post")
        t0_commit = time.perf_counter()
        result = ActionResult(ok=False, code="post:commit_fail")
        # 14.5. COMMIT (post)
        try:
            res = create_post(client, cfg, tuning, state, title, body)
            result = ActionResult(ok=bool(res), code=("post:ok" if res else "post:empty"), elapsed_ms=(time.perf_counter() - t0_commit) * 1000.0)
        except PowTimeoutError as e:
            result = ActionResult(ok=False, code="post:pow_timeout", detail=str(e), elapsed_ms=(time.perf_counter() - t0_commit) * 1000.0)
            protocol_set_reason(state, "post", "post:pow_timeout", one_line(str(e), 160))
            _bump_action_counter(state, "action_fail", "post")
            return 0
        if result.ok:
            _bump_action_counter(state, "action_success", "post")
        else:
            _bump_action_counter(state, "action_fail", "post")
        if not res:
            return 0
        remember_action(state, action="post", target_id=_text_hash(combo), endpoint_key="/posts")

        pid = str(res.get("id") or res.get("post_id") or "")
        remember_text(state, combo, for_post=True, same_text_gap_sec=cfg.timing.same_text_gap_sec)
        fp_ttl = _env_int("MERSOOM_RECENT_TEXT_FP_TTL_SEC", 6 * 3600, 60, 30 * 24 * 3600)
        fp_keep = _env_int("MERSOOM_RECENT_TEXT_FP_KEEP_MAX", 1200, 50, 20000)
        remember_fp(state, combo, for_post=True, ttl_sec=max(int(fp_ttl), int(cfg.timing.same_text_gap_sec)), keep_max=int(fp_keep))
        remember_simhash(state, combo, for_post=True, same_text_gap_sec=cfg.timing.same_text_gap_sec)
        remember_dup_signatures(state, combo, for_post=True, same_text_gap_sec=cfg.timing.same_text_gap_sec)

        state["last_post_ts"] = time.time()
        state["contrib_count_today"] = int(state.get("contrib_count_today", 0)) + 1
        state["total_actions"] = int(state.get("total_actions", 0)) + 1

        update_persona_maturity(brain, state)
        bump_semantic(semantic, _today_kst(), "post", 1.0)

        item = {
            "ts": time.time(),
            "action": "post",
            "action_type": "post_new",
            "post_id": pid,
            "category": meta.get("category", "general"),
            "context_key": meta.get("context_key", "gen"),
            "kw": meta.get("seed_kw", ""),
            "text": combo,
            "qa_score": float(qrep.get("score", 0) or 0),
            "qa_issues": list(qrep.get("issues", []))[:6],
            "used_style": meta.get("post_style", ""),
            "eval_due_ts": schedule_eval_due(tuning),
            "metrics_before": {},
            "metrics_after": {},
            "evaluated": False,
            "reward_scalar": 0.0,
        }
        item["ground_reason"] = ""
        item["target_nick"] = ""
        item["proxy_reward"] = compute_proxy_reward(combo, mode="post", ground_reason="")
        item.setdefault("brain_proxy_applied", False)
        item.setdefault("brain_reward_applied", False)
        record_memory(memory, item, tuning, archive_path_jsonl=cfg.paths.memory_archive_jsonl)
        try:
            apply_brain_proxy_update(brain, tuning, item)
        except Exception as e:
            log_debug_exc("do_contribution:silent", e)
            pass

        log_action("POST", f"post_id={pid} title={one_line(title)}")
        return 1

    # 3) Otherwise comment on other people's posts
    if not can_comment:
        # comment path is blocked (gap or limiter)
        try:
            if gap_com > 0:
                protocol_set_reason(state, "comment", "comment:rate_limited", f"gap={int(gap_com)}s")
            else:
                protocol_set_reason(state, "comment", "comment:rate_limited", f"window_remaining={int(comment_limiter.remaining())}")
        except Exception:
            protocol_set_reason(state, "comment", "comment:rate_limited")
        return 0
    if not comment_limiter.allow():
        if getattr(cfg, "debug", None) and cfg.debug.log_blocks:
            log_debug("gate:comment blocked by window (allow=false)")
        protocol_set_reason(state, "comment", "comment:rate_limited", "window_allow=false")
        return 0

    target = _pick_contrib_target(cfg, state, posts_cache, brain)
    if not target:
        protocol_set_reason(state, "comment", "comment:no_target", "no_target")
        return 0

    pid = str(target.get("id") or "")
    if not pid:
        protocol_set_reason(state, "comment", "comment:no_target", "missing_post_id")
        return 0


    before = _post_metrics(target)

    # fetch comments for context
    try:
        comments = list_comments(client, pid)
    except Exception:
        comments = []

    ingest_post_into_context(state, target, brain=brain)
    ingest_comments_into_context(state, pid, comments, brain=brain, cfg=cfg)
    th = get_thread(state, pid)
    synthesize_thread(th)

    # choose counterparty (reply to last other comment if exists)
    last_other = None
    for c in reversed(comments[-10:]):
        if not isinstance(c, dict):
            continue
        if is_own_comment(cfg, c):
            continue
        last_other = c
        break


    # v20.3: question-detect reply boost (no LLM)
    boost = _env_float("MERSOOM_REPLY_QUESTION_BOOST", 0.25, 0.0, 1.0)
    last_other_txt = ""
    try:
        last_other_txt = str((last_other or {}).get("content") or "")
    except Exception:
        last_other_txt = ""
    is_q = _is_questionish(last_other_txt)

    wants_reply = bool(last_other) and (
        (action_type == "reply_other") or (is_q and (random.random() < float(boost)))
    )

    # v20.3: spam guard — limit consecutive replies per thread (approx.)
    try:
        if wants_reply and pid:
            proto = _sdict(state, "protocol")
            rs = _sdict(proto, "reply_streak")
            rec = _safe_dict(rs.get(pid))
            cnt = int(rec.get("count", 0) or 0)
            last_ts = float(rec.get("last_ts", 0.0) or 0.0)
            if cnt >= 2 and (time.time() - last_ts) < 6 * 3600:
                wants_reply = False
    except Exception:
        pass

    parent_id = str((last_other or {}).get("id") or "") if wants_reply else None

    # toxic/taunt: don't reply directly; optionally downvote the post, then fallback to a normal comment.
    if parent_id and wants_reply:
        try:
            ltxt = str((last_other or {}).get("content") or "")
        except Exception:
            ltxt = ""
        if ltxt and handle_toxic_target(client, cfg, state, pid, parent_id, ltxt, tag="reply_other"):
            wants_reply = False
            parent_id = None

        # (v18.3) conversation protocol gate for replies on other people's posts
        conv_key2 = ""
        conv_meta2: Dict[str, Any] = {}
        if parent_id:
            okp2, conv_key2, _conv2, conv_meta2, reason2 = _reply_protocol_prepare(state, cfg, tuning, pid, comments, str(parent_id), time.time())
            if not okp2:
                protocol_set_reason(state, "comment", "comment:qa_fail", f"protocol_gate:{one_line(str(reason2), 140)}")
                return 0


        user_key = str(((last_other or {}).get("nickname") if wants_reply else (target.get("nickname") or "user")))
        user = get_user(state, user_key)

        # v15 Unit 02: lock focus target for comment/reply on others
        set_focus(state, mode=("reply" if parent_id else "comment"), post_id=pid, post=target, comment=last_other if parent_id else None)

        # (v18.3) pass conversation meta to reply generator
        try:
            if parent_id and isinstance(state.get("focus"), dict):
                state["focus"]["conv"] = conv_meta2
        except Exception as e:
            log_debug_exc("do_contribution:silent", e)
            pass

        txt = ""
        meta: Dict[str, Any] = {}
        ok_txt = False
        ground_reason = ""
        last_fail_bucket = ""
        last_fail_detail = ""
        last_candidate = ""
        for _try in range(_regen_budget(cfg)):
            txt, meta = build_reply_text(
                cfg, tuning, state, policy, th, user,
                bm25=bm25,
                brain=brain,
                reply_to_own_post=False,
                is_reply=bool(parent_id),
            )
            last_candidate = txt
            dup, db = dup_guard_bucket(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
            if dup:
                last_fail_bucket = str(db or "")
                last_fail_detail = str(db or "")
                _bump_gen_fail(state, db)
                if (_try % 3) == 0:
                    log_action("DUP", f"block mode=reply reason={db} try={_try+1}")
                continue
            mode2 = "reply" if bool(parent_id) else "comment"
            ok_g, reason = validate_grounding(txt, _safe_dict(state.get("focus")), mode2)
            if not ok_g:
                last_fail_bucket = "qa_fail"
                last_fail_detail = f"ground:{one_line(str(reason), 120)}"
                log_action("VALIDATE", f"fail mode={mode2} reason={reason} try={_try+1}")
                continue

            ok_q, qrep = qa_check_text(cfg, txt, kind=mode2, focus=_safe_dict(state.get("focus")), mode=mode2)
            if not ok_q:
                last_fail_bucket = "qa_fail"
                try:
                    last_fail_detail = f"qa:score={qrep.get('score')} issues={','.join((qrep.get('issues', []) or [])[:3])}"
                except Exception:
                    last_fail_detail = "qa_fail"
                _bump_gen_fail(state, "qa_fail")
                log_action("QA", f"fail mode={mode2} score={qrep.get('score')} issues={','.join(qrep.get('issues', [])[:3])} try={_try+1}")
                continue
            ok_txt = True
            ground_reason = reason
            break


        if not ok_txt:
            mode2 = "reply" if bool(parent_id) else "comment"
            fb = _qa_fallback_2stage(last_candidate, is_reply=bool(parent_id))
            if fb:
                try:
                    dup, db = dup_guard_bucket(state, fb, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
                    if dup:
                        last_fail_bucket = str(db or "")
                        last_fail_detail = str(db or "")
                        _bump_gen_fail(state, db)
                    if not dup:
                        ok_g, reason = validate_grounding(fb, _safe_dict(state.get("focus")), mode2)
                        if ok_g:
                            ok_q, qrep = qa_check_text(cfg, fb, kind=mode2, focus=_safe_dict(state.get("focus")), mode=mode2)
                            if ok_q:
                                txt = fb
                                ok_txt = True
                                ground_reason = reason
                            else:
                                last_fail_bucket = "qa_fail"
                                try:
                                    last_fail_detail = f"qa:score={qrep.get('score')} issues={','.join((qrep.get('issues', []) or [])[:3])}"
                                except Exception:
                                    last_fail_detail = "qa_fail"
                except Exception:
                    pass
            if not ok_txt and not STRICT_POSTPROCESS:
                # legacy fallback when strict postprocess is off
                fb2 = _pick_fallback_comment(cfg, is_reply=bool(parent_id))
                if fb2:
                    try:
                        dup, db = dup_guard_bucket(state, fb2, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
                        if dup:
                            last_fail_bucket = str(db or "")
                            last_fail_detail = str(db or "")
                            _bump_gen_fail(state, db)
                        if not dup:
                            ok_g, reason = validate_grounding(fb2, _safe_dict(state.get("focus")), mode2)
                            if ok_g:
                                ok_q, qrep = qa_check_text(cfg, fb2, kind=mode2, focus=_safe_dict(state.get("focus")), mode=mode2)
                                if ok_q:
                                    txt = fb2
                                    ok_txt = True
                                    ground_reason = reason
                                else:
                                    last_fail_bucket = "qa_fail"
                                    try:
                                        last_fail_detail = f"qa:score={qrep.get('score')} issues={','.join((qrep.get('issues', []) or [])[:3])}"
                                    except Exception:
                                        last_fail_detail = "qa_fail"
                    except Exception:
                        pass
            if not ok_txt:
                code = "comment:qa_fail"
                if last_fail_bucket == "dup_fp":
                    code = "comment:dup_fp"
                elif last_fail_bucket == "dup_sim":
                    code = "comment:dup_sim"
                protocol_set_reason(state, "comment", code, one_line(last_fail_detail or last_fail_bucket or "gen_fail", 140))
                return 0

        if dup_action_should_skip(state, action=("reply" if parent_id else "comment"), target_id=str(parent_id or pid), endpoint_key=f"/posts/{pid}/comments"):
            protocol_set_reason(state, "comment", "comment:dup_action", "recent_action_guard")
            return 0
        _bump_action_counter(state, "action_attempt", "reply" if parent_id else "comment")
        t0_commit = time.perf_counter()
        result = ActionResult(ok=False, code="comment:commit_fail")
        # 14.5. COMMIT (comment/reply)
        try:
            res = create_comment(client, cfg, tuning, state, pid, txt, parent_id=parent_id)
            result = ActionResult(ok=bool(res), code=("comment:ok" if res else "comment:empty"), elapsed_ms=(time.perf_counter() - t0_commit) * 1000.0)
        except RateLimitError as e_rl:
            protocol_set_reason(state, "comment", "comment:rate_limited", one_line(str(e_rl), 180))
            raise
        except PowTimeoutError as e:
            result = ActionResult(ok=False, code="comment:pow_timeout", detail=str(e), elapsed_ms=(time.perf_counter() - t0_commit) * 1000.0)
            protocol_set_reason(state, "comment", "comment:pow_timeout", one_line(str(e), 180))
            _bump_action_counter(state, "action_fail", "reply" if parent_id else "comment")
            return 0
        except requests.HTTPError as e_http:
            protocol_set_reason(state, "comment", "comment:qa_fail", one_line(str(e_http), 180))
            raise
        except Exception as e:
            protocol_set_reason(state, "comment", "comment:qa_fail", one_line(repr(e), 180))
            raise
        if not res:
            protocol_set_reason(state, "comment", "comment:no_target", "comment_create_failed")
            _bump_action_counter(state, "action_fail", "reply" if parent_id else "comment")
            return 0
        if result.ok:
            _bump_action_counter(state, "action_success", "reply" if parent_id else "comment")
        else:
            _bump_action_counter(state, "action_fail", "reply" if parent_id else "comment")
        remember_action(state, action=("reply" if parent_id else "comment"), target_id=str(parent_id or pid), endpoint_key=f"/posts/{pid}/comments")

        # (v18.3) commit conversation protocol state (only for replies)
        try:
            if parent_id:
                _reply_protocol_commit(state, conv_key2, time.time(), _extract_comment_id(res), txt)
        except Exception as e:
            log_debug_exc("do_contribution:silent", e)
            pass

        remember_text(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)

        fp_ttl = _env_int("MERSOOM_RECENT_TEXT_FP_TTL_SEC", 6 * 3600, 60, 30 * 24 * 3600)

        fp_keep = _env_int("MERSOOM_RECENT_TEXT_FP_KEEP_MAX", 1200, 50, 20000)

        remember_fp(state, txt, for_post=False, ttl_sec=max(int(fp_ttl), int(cfg.timing.same_text_gap_sec)), keep_max=int(fp_keep))
        remember_simhash(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)
        remember_dup_signatures(state, txt, for_post=False, same_text_gap_sec=cfg.timing.same_text_gap_sec)

        state["commented_ts"] = _safe_dict(state.get("commented_ts"))
        state["commented_ts"][pid] = time.time()
        # v20.3: reply streak bookkeeping (for per-thread reply spam guard)
        try:
            proto3 = _sdict(state, "protocol")
            rs3 = _sdict(proto3, "reply_streak")
            if parent_id:
                rec3 = _safe_dict(rs3.get(pid))
                cnt3 = int(rec3.get("count", 0) or 0) + 1
                rs3[pid] = {"count": cnt3, "last_ts": time.time()}
            else:
                rs3[pid] = {"count": 0, "last_ts": time.time()}
        except Exception:
            pass
        state["last_comment_ts"] = time.time()
        state["contrib_count_today"] = int(state.get("contrib_count_today", 0)) + 1
        state["total_actions"] = int(state.get("total_actions", 0)) + 1

        update_persona_maturity(brain, state)
        bump_semantic(semantic, _today_kst(), ("reply_other" if parent_id else "comment"), 1.0)
        _bump_relation(state, user_key)

        # v23.3: open-question tracking from outgoing agent text (env-controllable)
        try:
            if _openq_track_enabled(state) and _is_open_question_text(txt):
                qtext = _extract_open_question_text(txt)
                if qtext:
                    qid = thread_add_open_question(th, qtext, asked_by="me", last_seen_remote_id=str(parent_id or ""))
                    try:
                        proto_q = _sdict(state, "protocol")
                        proto_q["openq_added_total"] = int(proto_q.get("openq_added_total", 0) or 0) + 1
                    except Exception:
                        pass
                    protocol_bump_counter(state, "openq_add")
                    log_event("openq.add", post_id=str(pid), parent_id=str(parent_id or ""), qid=str(qid))
        except Exception:
            pass

        item = {
            "ts": time.time(),
            "action": ("reply" if parent_id else "comment"),
            "action_type": ("reply_other" if parent_id else "comment_other"),
            "post_id": pid,
            "parent_id": parent_id,
            "category": meta.get("category", "general"),
            "context_key": meta.get("context_key", "gen"),
            "kw": meta.get("kw", ""),
            "text": txt,
            "used_strategy": meta.get("strategy", ""),
            "used_tone": meta.get("tone", ""),
            "used_length": meta.get("length", ""),
            "used_reply_style": meta.get("reply_style", ""),
            "weak_context": bool(meta.get("weak_context")),
            "template_id": meta.get("template_id", ""),
            "used_quotes": bool(meta.get("used_quotes")),
            "novelty": _novelty_score(state, txt),
            "comment_id": _extract_comment_id(res),
            "qa_score": float(qrep.get("score", 0) or 0),
            "qa_issues": list(qrep.get("issues", []))[:6],
            "qa_rep3": float(qrep.get("rep3", 0.0) or 0.0),
            "qa_im_ratio": float(qrep.get("im_ratio", 0.0) or 0.0),
            "qa_line_prefix_dup": float(qrep.get("line_prefix_dup", 0.0) or 0.0),
            "eval_due_ts": schedule_eval_due(tuning),
            "metrics_before": before,
            "metrics_after": {},
            "evaluated": False,
            "reward_scalar": 0.0,
        }
        item["ground_reason"] = ground_reason
        item["target_nick"] = user_key
        item["proxy_reward"] = compute_proxy_reward(txt, mode=("reply" if parent_id else "comment"), ground_reason=ground_reason)
        item.setdefault("brain_proxy_applied", False)
        item.setdefault("brain_reward_applied", False)
        record_memory(memory, item, tuning, archive_path_jsonl=cfg.paths.memory_archive_jsonl)
        try:
            apply_brain_proxy_update(brain, tuning, item)
        except Exception as e:
            log_debug_exc("do_contribution:silent", e)
            pass
            pass

        log_action("REPLY" if parent_id else "COMMENT", f"post={pid} parent={parent_id or '-'} | {one_line(txt)}")
        return 1


################################################################################
# 14.6. REWARD / UPDATE
################################################################################
def compute_reward(tuning: AgentTuning, before: Dict[str, int], after: Dict[str, int], meta: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    v20.5 unified reward (single scalar for learning):
      r = W_UP * Δup + W_ENGAGE * log1p(engage) - W_RISK * risk

    engage:
      - posts: Δcomments on the post (best available "inbound" signal)
      - comments/replies: reply_received (continuation on my own comment)

    risk (posts only, heuristic):
      - downvote growth
      - negative score proximity (blind(-5) guard)
    """
    # server-side deltas (best available feedback for posts)
    du = float(after.get("up", 0) - before.get("up", 0))
    dd = float(after.get("down", 0) - before.get("down", 0))
    dc = float(after.get("comments", 0) - before.get("comments", 0))
    ds = float(after.get("score", 0) - before.get("score", 0))

    action = str(meta.get("action") or "")
    action_type = str(meta.get("action_type") or "")

    # weights (env-driven; stored in tuning for logging/selftest)
    w_up = float(getattr(tuning, "reward_w_up", 1.0) or 1.0)
    w_eng = float(getattr(tuning, "reward_w_engage", 0.6) or 0.6)
    w_risk = float(getattr(tuning, "reward_w_risk", 1.2) or 1.2)

    reply_received = float(meta.get("reply_received", 0.0) or 0.0)

    is_post = (action == "post") or action_type.startswith("post_")

    # engagement signal
    if is_post:
        engage_raw = max(0.0, dc)
    else:
        # For comments/replies we cannot reliably observe per-comment votes;
        # continuation is the most meaningful proxy.
        engage_raw = max(0.0, reply_received)

    # upvote signal (posts only; other actions would be noisy if we read the parent post)
    up_raw = du if is_post else 0.0

    # risk signal (posts only; heuristic)
    net_votes = float(du - dd)
    after_score = float(after.get("score", 0) or 0)

    risk = 0.0
    if is_post:
        # Down growth always increases risk; net-vote drops are penalized mildly.
        risk_votes = max(0.0, dd)
        risk_net = max(0.0, -net_votes) * 0.5

        # Proximity to "blind" zone (score <= -5 is a common threshold in community UX).
        risk_blind = 0.0
        if after_score <= -4.0:
            # -4 => 1.0, -5 => 2.0, -6 => 3.0 ...
            risk_blind = min(6.0, 1.0 + abs(after_score + 4.0))
        elif after_score <= -2.0:
            risk_blind = 0.5

        risk = float(risk_votes + risk_net + risk_blind)

    # scalar reward
    comp_votes = float(w_up) * float(up_raw)
    comp_engage = float(w_eng) * math.log1p(float(engage_raw))
    comp_risk = -float(w_risk) * float(risk)

    components: Dict[str, float] = {
        # Keep legacy keys for downstream head shaping / logging
        "votes": float(comp_votes),
        "score": 0.0,
        "engage": float(comp_engage),
        "novelty": 0.0,
        "quotes": 0.0,
        "quality": 0.0,
        "cont": 0.0,
        "weak_ctx": 0.0,
        # explicit
        "risk": float(comp_risk),
    }

    r = float(comp_votes + comp_engage + comp_risk)
    r = _clip_reward(float(r), tuning.reward_clip)
    if not math.isfinite(r):
        r = 0.0

    feats: Dict[str, Any] = {
        "du": float(du), "dd": float(dd), "dc": float(dc), "ds": float(ds),
        "net_votes": float(net_votes),
        "reply_received": float(reply_received),
        "action": action,
        "is_post": bool(is_post),
        "engage_raw": float(engage_raw),
        "up_raw": float(up_raw),
        "risk_raw": float(risk),
        "after_score": float(after_score),
        "weights": {"w_up": float(w_up), "w_engage": float(w_eng), "w_risk": float(w_risk)},
        "components": components,
    }
    return r, feats

def _head_reward(head: str, r_scalar: float, feats: Dict[str, Any], tuning: AgentTuning) -> float:
    """Per-head reward shaping: learn *why* something worked."""
    comps = _safe_dict(feats.get("components"))
    votes = float(comps.get("votes", 0.0) or 0.0)
    score = float(comps.get("score", 0.0) or 0.0)
    engage = float(comps.get("engage", 0.0) or 0.0)
    novelty = float(comps.get("novelty", 0.0) or 0.0)
    quotes = float(comps.get("quotes", 0.0) or 0.0)
    quality = float(comps.get("quality", 0.0) or 0.0)
    cont = float(comps.get("cont", 0.0) or 0.0)
    weak = float(comps.get("weak_ctx", 0.0) or 0.0)

    if head in ("action_type", "strategy", "post_style"):
        return float(r_scalar)

    if head == "tone":
        # tone tends to affect reactions + continuity more than novelty
        r = votes + score + engage + cont + (0.7 * quality) + (0.4 * weak)
        return _clip_reward(float(r), tuning.reward_clip)

    if head in ("comment_length", "reply_style"):
        # length/style: reward engagement/continuation + avoid weak context
        r = votes + engage + cont + (0.5 * quality) + (0.6 * weak)
        return _clip_reward(float(r), tuning.reward_clip)

    if head == "template":
        # templates: quality + reactions matter more than novelty
        r = votes + score + engage + (1.0 * quality) + (0.2 * novelty) + (0.2 * quotes) + (0.4 * weak)
        return _clip_reward(float(r), tuning.reward_clip)

    # default
    return float(r_scalar)


def evaluate_and_learn(
    client: HttpClient,
    tuning: AgentTuning,
    memory: List[Dict[str, Any]],
    policy: Dict[str, Any],
    semantic: Dict[str, Any],
    state: Dict[str, Any],
) -> Tuple[int, float]:
    now = time.time()
    last_learn = float(state.get("last_learn_ts", 0.0) or 0.0)
    if (now - last_learn) < float(tuning.learn_period_sec):
        return (0, 0.0)

    due: List[Dict[str, Any]] = []
    for it in memory:
        if not isinstance(it, dict):
            continue
        if it.get("evaluated") is True:
            continue
        due_ts = float(it.get("eval_due_ts", 0.0) or 0.0)
        if due_ts <= 0 or due_ts > now:
            continue
        if str(it.get("action") or "") not in ("comment", "reply", "post"):
            it["evaluated"] = True
            continue
        due.append(it)

    if not due:
        state["last_learn_ts"] = now
        state["learn_runs"] = int(state.get("learn_runs", 0)) + 1
        return (0, 0.0)

    # Optional evaluation budget (per window) to avoid evaluation traffic spikes.
    eb = state.get("eval_budget")
    if not isinstance(eb, dict):
        eb = {"last_reset_ts": 0.0, "used": 0, "cap_per_window": 0, "window_sec": 900}
        state["eval_budget"] = eb
    window_sec = float(eb.get("window_sec", 900) or 900)
    if window_sec < 60:
        window_sec = 60.0
    last_reset = float(eb.get("last_reset_ts", 0.0) or 0.0)
    used = int(eb.get("used", 0) or 0)
    cap = int(eb.get("cap_per_window", 0) or 0)  # 0 => unlimited
    if cap > 0 and (now - last_reset) >= window_sec:
        eb["last_reset_ts"] = now
        eb["used"] = 0
        used = 0
        last_reset = now
    if cap > 0:
        remaining = max(0, cap - used)
        if remaining <= 0:
            # Budget exhausted for this window; postpone learning check.
            state["last_learn_ts"] = now
            state["learn_runs"] = int(state.get("learn_runs", 0)) + 1
            return (0, 0.0)
        due = due[:remaining]

    due = due[:max(1, int(tuning.max_eval_per_tick))]
    total_r = 0.0
    done = 0

    for it in due:
        pid = str(it.get("post_id") or "")
        if not pid:
            it["evaluated"] = True
            continue

        # consume evaluation budget when we make a network fetch
        eb2 = state.get("eval_budget")
        if isinstance(eb2, dict):
            cap2 = int(eb2.get("cap_per_window", 0) or 0)
            if cap2 > 0:
                eb2["used"] = int(eb2.get("used", 0) or 0) + 1

        post = get_post(client, pid)
        if not post:
            it["evaluated"] = True
            continue

        after = _post_metrics(post)
        before = _safe_dict(it.get("metrics_before"))
        if not before:
            before = after

        # Unit 05: attach "continuation" signal (did my comment receive replies?)
        cid0 = str(it.get("comment_id") or "")
        if cid0:
            mr = _safe_dict(_safe_dict(state.get("my_comment_replies")).get(cid0))
            if mr:
                it["reply_received"] = int(mr.get("count", 0) or 0)
                it["reply_received_last_ts"] = float(mr.get("last_ts", 0.0) or 0.0)

        r, feats = compute_reward(tuning, before, after, it)
        it["metrics_after"] = after
        it["reward_scalar"] = float(r)
        it["reward_features"] = feats
        it["evaluated"] = True

        cat = str(it.get("category") or it.get("context_key") or "general")
        strategy = str(it.get("used_strategy") or "")
        tone = str(it.get("used_tone") or "")
        length = str(it.get("used_length") or "")
        reply_style = str(it.get("used_reply_style") or "")
        tid = str(it.get("template_id") or "")
        action_type = str(it.get("action_type") or "")
        used_style = str(it.get("used_style") or "")

        # Unit 05: head-specific learning signals
        r_action = _head_reward("action_type", r, feats, tuning)
        r_poststyle = _head_reward("post_style", r, feats, tuning)
        r_strategy = _head_reward("strategy", r, feats, tuning)
        r_tone = _head_reward("tone", r, feats, tuning)
        r_len = _head_reward("comment_length", r, feats, tuning)
        r_rstyle = _head_reward("reply_style", r, feats, tuning)
        r_tpl = _head_reward("template", r, feats, tuning)

        if action_type:
            update_arm(policy, "action_type", action_type, r_action, context_key=cat)
        if used_style:
            # posts only; harmless if set on comment
            update_arm(policy, "post_styles", used_style, r_poststyle, context_key=cat)

        if strategy:
            update_arm(policy, "strategy", strategy, r_strategy, context_key=cat)
        if tone:
            update_arm(policy, "tone", tone, r_tone, context_key=cat)
        if length:
            update_arm(policy, "comment_length", length, r_len, context_key=cat)
        if reply_style:
            update_arm(policy, "reply_styles", reply_style, r_rstyle, context_key=cat)
        if tid:
            update_template_weight(policy, tid, r_tpl, meta=it)

        bump_semantic(semantic, _today_kst(), "eval", 1.0)
        total_r += float(r)
        done += 1

    state["last_learn_ts"] = now
    state["learn_runs"] = int(state.get("learn_runs", 0)) + 1
    state["evaluated_count"] = int(state.get("evaluated_count", 0)) + done
    state["total_reward"] = float(state.get("total_reward", 0.0)) + float(total_r)
    return done, float(total_r)

def update_brain(brain: Dict[str, Any], tuning: AgentTuning, memory: List[Dict[str, Any]], policy: Dict[str, Any]) -> None:
    recent = [it for it in memory[-40:] if isinstance(it, dict) and it.get("evaluated") is True]
    if not recent:
        return
    avg_r = sum(float(it.get("reward_scalar", 0.0) or 0.0) for it in recent) / max(1, len(recent))

    # Unit 07: apply delayed reward updates to action_bias (idempotent per item)
    try:
        for it in recent[-40:]:
            if not isinstance(it, dict):
                continue
            # ensure proxy gets applied at least once (in case earlier immediate hook was skipped)
            if it.get("proxy_reward") is not None and it.get("brain_proxy_applied") is not True:
                apply_brain_proxy_update(brain, tuning, it)
            apply_brain_reward_update(brain, tuning, it)
    except Exception as e:
        log_debug_exc("update_brain:silent", e)
        pass

    mood = brain.setdefault("mood", {})
    ema = float(mood.get("ema_reward", 0.0))
    a = float(tuning.brain_reward_alpha)
    mood["ema_reward"] = float((1 - a) * ema + a * avg_r)

    topic_ema = brain.setdefault("topic_ema", {})
    if not isinstance(topic_ema, dict):
        brain["topic_ema"] = {}
        topic_ema = brain["topic_ema"]

    ta = float(tuning.brain_topic_alpha)
    for it in memory[-30:]:
        if not isinstance(it, dict):
            continue
        kw = str(it.get("kw") or "")
        if not kw:
            continue
        topic_ema[kw] = float(topic_ema.get(kw, 0.0)) * (1 - ta) + ta * 1.0

    # belief update: reward-anchored stance signal per keyword
    beliefs = brain.setdefault("beliefs", {})
    if not isinstance(beliefs, dict):
        brain["beliefs"] = {}
        beliefs = brain["beliefs"]

    # use evaluated recent actions only
    for it in recent[-25:]:
        kw = str(it.get("kw") or "")
        if not kw:
            continue
        r = float(it.get("reward_scalar", 0.0) or 0.0)
        b = beliefs.get(kw)
        if not isinstance(b, dict):
            b = {"score_ema": 0.0, "n": 0, "last_ts": 0.0}
        ema2 = float(b.get("score_ema", 0.0) or 0.0)
        # slightly adaptive smoothing
        aa = 0.08 + 0.04 * min(1.0, abs(r))
        b["score_ema"] = float((1 - aa) * ema2 + aa * r)
        b["n"] = int(b.get("n", 0)) + 1
        b["last_ts"] = time.time()
        beliefs[kw] = b

    # (P1) Reflection thoughts: store what worked, then reuse via retrieval
    try:
        rh = brain.setdefault("reflection_hashes", [])
        if not isinstance(rh, list):
            brain["reflection_hashes"] = []
            rh = brain["reflection_hashes"]
        _clean_hash_list(rh, ttl_sec=60 * 60 * 24 * 14)

        added = 0
        for it in recent[::-1]:
            if added >= 3:
                break
            if str(it.get("action") or "") not in ("comment", "reply"):
                continue
            r = float(it.get("reward_scalar", 0.0) or 0.0)
            if r < 0.8:
                continue
            kw = str(it.get("kw") or "") or "reflection"
            strat = str(it.get("used_strategy") or "")
            txt = str(it.get("text") or "")
            if not txt:
                continue
            summ = _simple_summary(txt, 140)
            h = hashlib.sha1(f"{kw}|{strat}|{summ[:80]}".encode("utf-8")).hexdigest()[:12]
            if any(isinstance(x, list) and x and x[0] == h for x in rh):
                continue

            line = f"{kw}에서 {strat or '한줄정리'}이 반응이 좋았음: {summ}"
            add_thought(
                brain,
                kind="reflection",
                topic=kw,
                text=line,
                tags=[kw, strat, str(it.get('category') or '')],
                strength=min(0.95, 0.65 + 0.15 * abs(r)),
                links={"post_id": str(it.get("post_id") or ""), "action": str(it.get("action") or "comment")},
            )
            rh.append([h, time.time()])
            added += 1
    except Exception as e:
        log_debug_exc("update_brain:silent", e)
        pass

def render_brain_note(brain: Dict[str, Any]) -> str:
    mood = _safe_dict(brain.get("mood"))
    ema = float(mood.get("ema_reward", 0.0))

    topic = _safe_dict(brain.get("topic_ema"))
    top_topics = sorted(topic.items(), key=lambda kv: kv[1], reverse=True)[:8]

    com = _safe_dict(brain.get("community"))
    hot = _safe_list(com.get("hot"))[:10]
    rising = _safe_list(com.get("rising"))[:8]

    beliefs = _safe_dict(brain.get("beliefs"))
    belief_rank = []
    for k, b in beliefs.items():
        if not isinstance(b, dict):
            continue
        belief_rank.append((k, float(b.get("score_ema", 0.0) or 0.0), int(b.get("n", 0) or 0)))
    belief_rank.sort(key=lambda x: abs(x[1]), reverse=True)
    belief_rank = belief_rank[:8]

    thoughts = _safe_list(brain.get("thoughts"))[-6:]

    lines: List[str] = []
    lines.append("# Brain Note")
    lines.append(f"- ema_reward: {ema:.3f}")

    if hot:
        lines.append("- hot:")
        for it in hot:
            if isinstance(it, dict):
                lines.append(f"  - {it.get('kw','')}: {float(it.get('score',0.0) or 0.0):.2f}")
    if rising:
        lines.append("- rising:")
        for it in rising:
            if isinstance(it, dict):
                lines.append(f"  - {it.get('kw','')}: +{float(it.get('delta',0.0) or 0.0):.2f}")

    if top_topics:
        lines.append("- topic_ema:")
        for k, v in top_topics:
            lines.append(f"  - {k}: {float(v):.3f}")

    if belief_rank:
        lines.append("- beliefs (reward-tilt):")
        for k, sc, n in belief_rank:
            lines.append(f"  - {k}: {sc:+.2f} (n={n})")

    if thoughts:
        lines.append("- recent_thoughts:")
        for t in thoughts:
            if not isinstance(t, dict):
                continue
            lines.append(f"  - [{t.get('kind','')}] {t.get('topic','')}: {t.get('text','')}")

    return "\n".join(lines) + "\n\n"
################################################################################
# 14.1. HELPERS: sleep decision (limits/pacing)
# - Dependencies: Section 1-2 (Config, Logging)
# - Used by: HTTP client + main loop pacing
# - Key functions: SlidingWindowLimiter.allow()
################################################################################
def next_sleep_from_limits(
    cfg: Config,
    state: Dict[str, Any],
    post_limiter: SlidingWindowLimiter,
    comment_limiter: SlidingWindowLimiter,
    vote_limiter: SlidingWindowLimiter,
    vote_pace_sec: int,
    comment_pace_sec: int,
    post_pace_sec: int
) -> int:
    waits: List[int] = []

    # limiter-based (capacity==0 방어는 limiter 내부에서 함)
    if vote_limiter.remaining() <= 0:
        waits.append(vote_limiter.seconds_until_next())
    if comment_limiter.remaining() <= 0:
        waits.append(comment_limiter.seconds_until_next())
    if post_limiter.remaining() <= 0:
        waits.append(post_limiter.seconds_until_next())

    # gap-based
    waits.append(gap_remaining(float(state.get("last_vote_ts", 0.0) or 0.0), int(vote_pace_sec)))
    waits.append(gap_remaining(float(state.get("last_comment_ts", 0.0) or 0.0), int(comment_pace_sec)))
    waits.append(gap_remaining(float(state.get("last_post_ts", 0.0) or 0.0), int(post_pace_sec)))

    positive = [x for x in waits if x > 0]
    base_wait = max(1, min(positive) if positive else int(cfg.timing.tick_min_sec))

    # human jitter within tick bounds
    lo = int(cfg.timing.tick_min_sec)
    hi = int(cfg.timing.tick_max_sec)
    jitter_hi = min(hi, max(lo, base_wait + hi))
    w2 = int(random.uniform(lo, jitter_hi))
    return max(1, w2)

################################################################################
# 15. MAIN LOOP (single-file entry) + corpus/index wiring + template mining
# - Dependencies: All previous sections
# - Used by: Entry point
# - Key functions: main()
################################################################################


# v19.0 scaffold: reserve state slots and provide a hook for future protocol engine
def _kst_day_str(ts: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=KST)
    return dt.strftime("%Y-%m-%d")


def rules_sync_if_due(cfg: Config, state: Dict[str, Any], client: "HttpClient") -> None:
    """v19.1: Fetch skills.md once per KST day and record hash/metadata in state.
    - Never raises.
    - Uses conditional headers (ETag/Last-Modified) when available.
    """
    try:
        rs = getattr(cfg, "rules_sync", None)
        if not rs or not bool(getattr(rs, "daily", True)):
            return
        # Allow protocol master switch to disable this whole feature quickly
        if not bool(getattr(getattr(cfg, "protocol", None), "enabled", True)):
            return

        url = str(getattr(rs, "url", "") or "").strip() or "https://mersoom.com/docs/skills.md"
        rules = _sdict(state, "rules")
        today = _kst_day_str()

        if str(rules.get("last_sync_day", "")).strip() == today:
            return

        headers: Dict[str, str] = {}
        etag = str(rules.get("etag", "") or "").strip()
        last_mod = str(rules.get("last_modified", "") or "").strip()
        if etag:
            headers["If-None-Match"] = etag
        if last_mod:
            headers["If-Modified-Since"] = last_mod

        # Reuse the main session (timeouts from cfg.http)
        timeout = (float(cfg.http.timeout_connect_sec), float(cfg.http.timeout_read_sec))
        t0 = time.time()
        resp = client.session.get(url, headers=headers, timeout=timeout)
        dt_ms = int((time.time() - t0) * 1000)

        rules["last_checked_at"] = float(time.time())
        rules["last_status"] = int(resp.status_code)
        rules["last_url"] = url
        rules["last_latency_ms"] = dt_ms

        if resp.status_code == 304:
            rules["last_sync_day"] = today
            log_event("rules_sync_not_modified", day=today, ms=dt_ms, status=304, url=url)
            return

        if resp.status_code >= 400:
            # Don't mark sync day on failure; try again later
            log_event("rules_sync_fail", day=today, ms=dt_ms, status=int(resp.status_code), url=url)
            return

        body = resp.text if isinstance(resp.text, str) else (resp.content or b"").decode("utf-8", "replace")
        body_bytes = body.encode("utf-8", "replace")
        h = hashlib.sha256(body_bytes).hexdigest()

        prev = str(rules.get("last_hash", "") or "").strip()
        rules["last_hash"] = h
        rules["last_sync_day"] = today

        new_etag = str(resp.headers.get("ETag", "") or "").strip()
        new_lm = str(resp.headers.get("Last-Modified", "") or "").strip()
        if new_etag:
            rules["etag"] = new_etag
        if new_lm:
            rules["last_modified"] = new_lm

        if prev and prev != h:
            log_event("rules_sync_changed", day=today, ms=dt_ms, hash=str(h[:10]), prev_hash=str(prev[:10] if prev else ""))
        else:
            log_event("rules_sync_ok", day=today, ms=dt_ms, hash=str(h[:10]))
    except Exception as e:
        try:
            rules = _sdict(state, "rules")
            rules["last_checked_at"] = float(time.time())
            rules["last_error"] = repr(e)[:300]
        except Exception as e:
            log_debug_exc("rules_sync_if_due:silent", e)
            pass
        log_event("rules_sync_fail", day=today if "today" in locals() else "", error=one_line(repr(e), 220))
        log_warn(f"rules_sync_fail: {one_line(repr(e), 220)}")
        return



def _hb_kst_str(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), KST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "?"

def _hb_next_interval_sec(cfg: Config) -> float:
    try:
        hb = getattr(cfg, "heartbeat", None)
        if not hb or not getattr(hb, "enabled", False):
            return 0.0
        mn = float(getattr(hb, "min_hours", 4.0) or 4.0)
        mx = float(getattr(hb, "max_hours", 5.0) or 5.0)
        if mx < mn:
            mx = mn
        hours = random.uniform(mn, mx)
        return float(hours) * 3600.0
    except Exception:
        return 4.5 * 3600.0

def _heartbeat_tick(cfg: Config, state: Dict[str, Any]) -> None:
    """Start/refresh heartbeat cycles and set per-cycle quotas.

    This is state-only scheduling; action enforcement lives in do_contribution().
    """
    try:
        if not bool(getattr(cfg, "protocol", None) and cfg.protocol.enabled):
            return
        hb_cfg = getattr(cfg, "heartbeat", None)
        if not hb_cfg or not bool(getattr(hb_cfg, "enabled", False)):
            return

        proto = _sdict(state, "protocol")
        hb = _sdict(proto, "heartbeat")
        q = _sdict(hb, "quota")

        now_ts = time.time()
        hb.setdefault("active", False)
        hb.setdefault("cycle_id", int(proto.get("cycle_id", 0) or 0))
        hb.setdefault("started_at", 0.0)
        hb.setdefault("completed_at", 0.0)
        hb.setdefault("last_at", 0.0)
        hb.setdefault("next_at", 0.0)
        hb.setdefault("last_log_ts", 0.0)

        q.setdefault("comments_target", 0)
        q.setdefault("comments_target_clamped", False)
        q.setdefault("comments_done", 0)
        q.setdefault("votes_done", 0)
        q.setdefault("contribute_done", False)
        q.setdefault("contribute_ts", 0.0)

        next_at = float(hb.get("next_at", 0.0) or 0.0)
        if next_at <= 0.0:
            hb["next_at"] = now_ts + _hb_next_interval_sec(cfg)
            hb["active"] = False
            hb["last_at"] = float(hb.get("last_at", 0.0) or 0.0)
            return

        # If current cycle already satisfied, mark completed (idempotent)
        if bool(hb.get("active")):
            _hb_maybe_complete(state, now_ts=now_ts)

        # Start a new cycle if due and not currently active
        if (not bool(hb.get("active"))) and now_ts >= float(hb.get("next_at", 0.0) or 0.0):
            proto["cycle_id"] = int(proto.get("cycle_id", 0) or 0) + 1
            hb["cycle_id"] = int(proto.get("cycle_id", 0) or 0)

            hb["active"] = True
            hb["started_at"] = now_ts
            hb["completed_at"] = 0.0
            hb["last_at"] = now_ts

            # quotas
            cmin = int(getattr(hb_cfg, "comment_min", 2) or 0)
            cmax = int(getattr(hb_cfg, "comment_max", 3) or 0)
            if cmax < cmin:
                cmax = cmin
            q["comments_target"] = int(random.randint(cmin, cmax)) if (cmax > 0) else 0
            q["comments_target_clamped"] = False
            proto["hb_block_reason"] = ""
            q["comments_done"] = 0
            q["votes_done"] = 0
            q["contribute_done"] = False
            q["contribute_ts"] = 0.0

            # schedule next
            hb["next_at"] = now_ts + _hb_next_interval_sec(cfg)

            # credit an arena/post that happened just before the cycle boundary (within 10 minutes)
            recent = 600.0
            last_post_ts = float(state.get("last_post_ts", 0.0) or 0.0)
            last_arena_ts = float(state.get("arena_last_action_ts", 0.0) or 0.0)
            if (last_post_ts > 0 and (now_ts - last_post_ts) <= recent) or (last_arena_ts > 0 and (now_ts - last_arena_ts) <= recent):
                _hb_record_contribute(state, max(last_post_ts, last_arena_ts), kind="recent")

            try:
                log_info(
                    f"HB start cycle={hb.get('cycle_id')} target_comments={q.get('comments_target')} "
                    f"next_at={_hb_kst_str(float(hb.get('next_at') or 0.0))}"
                )
            except Exception as e:
                log_debug_exc("_heartbeat_tick:silent", e)
                pass

    except Exception:
        return

# -----------------------------------------------------------------------------
# v20.6 Observability: reason protocol helpers
# - Standardized "why" codes for vote/comment/arena decisions
# - Rolling 10-minute counter for health summaries (future wiring)
# -----------------------------------------------------------------------------
def _protocol_reason_reset_if_needed(proto: Dict[str, Any], now_ts: float) -> None:
    """Reset 10-minute reason window if expired or uninitialized."""
    try:
        started = float(proto.get("reason_window_started_ts", 0.0) or 0.0)
        if started <= 0.0 or (now_ts - started) >= 600.0:
            proto["reason_window_started_ts"] = float(now_ts)
            proto["reason_window_10m"] = {}
    except Exception:
        proto["reason_window_started_ts"] = float(now_ts)
        proto["reason_window_10m"] = {}

def protocol_bump_reason(state: Dict[str, Any], code: str, inc: int = 1) -> None:
    """Increment the rolling 10-minute reason counter (best-effort, never raises)."""
    try:
        proto = _sdict(state, "protocol")
        now_ts = float(time.time())
        _protocol_reason_reset_if_needed(proto, now_ts)

        win = proto.get("reason_window_10m")
        if not isinstance(win, dict):
            win = {}
            proto["reason_window_10m"] = win

        k = str(code or "unknown")
        win[k] = int(win.get(k, 0) or 0) + int(inc or 1)
    except Exception:
        return

def protocol_set_reason(state: Dict[str, Any], domain: str, code: str, detail: str = "") -> None:
    """Set last reason for a domain and bump the 10-minute window counter."""
    try:
        proto = _sdict(state, "protocol")
        now_ts = float(time.time())

        last = proto.get("reason_last")
        if not isinstance(last, dict):
            last = {}
            proto["reason_last"] = last
        last_ts = proto.get("reason_last_ts")
        if not isinstance(last_ts, dict):
            last_ts = {}
            proto["reason_last_ts"] = last_ts
        last_detail = proto.get("reason_last_detail")
        if not isinstance(last_detail, dict):
            last_detail = {}
            proto["reason_last_detail"] = last_detail

        d = str(domain or "misc")
        c = str(code or "unknown")
        det = one_line(detail, 220) if detail else ""

        last[d] = c
        last_ts[d] = now_ts
        if det:
            last_detail[d] = det
        else:
            last_detail.setdefault(d, "")

        protocol_bump_reason(state, c, 1)
    except Exception:
        return

def protocol_get_reason(state: Dict[str, Any], domain: str) -> str:
    """Get last reason code for a domain (returns 'unknown' if missing)."""
    try:
        proto = _safe_dict(state.get("protocol"))
        last = _safe_dict(proto.get("reason_last"))
        code = str(last.get(str(domain or "")) or "").strip()
        return code if code else "unknown"
    except Exception:
        return "unknown"

def protocol_get_reason_detail(state: Dict[str, Any], domain: str) -> str:
    """Get last reason detail for a domain (best-effort)."""
    try:
        proto = _safe_dict(state.get("protocol"))
        det = _safe_dict(proto.get("reason_last_detail"))
        return str(det.get(str(domain or "")) or "")
    except Exception:
        return ""




def _protocol_counter_reset_if_needed(proto: Dict[str, Any], now_ts: float) -> None:
    """Reset rolling 10-minute counters if the window is stale (best-effort, never raises)."""
    try:
        started = float(proto.get("counter_window_started_ts", 0.0) or 0.0)
        if started <= 0.0 or (now_ts - started) >= 600.0:
            proto["counter_window_started_ts"] = float(now_ts)
            proto["counter_window_10m"] = {}
    except Exception:
        proto["counter_window_started_ts"] = float(now_ts)
        proto["counter_window_10m"] = {}


def protocol_bump_counter(state: Dict[str, Any], key: str, inc: int = 1) -> None:
    """Increment a generic rolling 10-minute counter (best-effort, never raises)."""
    try:
        proto = _sdict(state, "protocol")
        now_ts = float(time.time())
        _protocol_counter_reset_if_needed(proto, now_ts)

        win = proto.get("counter_window_10m")
        if not isinstance(win, dict):
            win = {}
            proto["counter_window_10m"] = win

        k = str(key or "unknown")[:80]
        win[k] = int(win.get(k, 0) or 0) + int(inc or 1)
    except Exception:
        return


def protocol_get_counter_10m(state: Dict[str, Any], key: str) -> int:
    """Read a generic rolling 10-minute counter value (best-effort).

    Resets stale windows on read so HEALTH reflects *recent* activity even if no bumps occurred.
    """
    try:
        proto = _sdict(state, "protocol")
        now_ts = float(time.time())
        _protocol_counter_reset_if_needed(proto, now_ts)
        win = _safe_dict(proto.get("counter_window_10m"))
        return int(win.get(str(key or "")[:80], 0) or 0)
    except Exception:
        return 0


def protocol_top_counters_10m(state: Dict[str, Any], prefix: str = "", topn: int = 5) -> List[Dict[str, Any]]:
    """Return top counters (optionally by prefix) for debugging/health (best-effort)."""
    try:
        proto = _sdict(state, "protocol")
        now_ts = float(time.time())
        _protocol_counter_reset_if_needed(proto, now_ts)
        win = _safe_dict(proto.get("counter_window_10m"))
        out: List[Dict[str, Any]] = []
        pfx = str(prefix or "")
        for k, v in win.items():
            try:
                ks = str(k)
                if pfx and (not ks.startswith(pfx)):
                    continue
                out.append({"k": ks[:80], "n": int(v or 0)})
            except Exception:
                continue
        out = sorted(out, key=lambda d: int(d.get("n", 0) or 0), reverse=True)
        return out[: max(1, int(topn or 5))]
    except Exception:
        return []

def record_bm25_build(state: Dict[str, Any], *, build_ms: float, docs_indexed: int, mode: str, corpus_size: int, added_since: int) -> None:
    try:
        now_ts = time.time()
        rec = _safe_list(state.get("bm25_build_ms_recent"))
        rec = [it for it in rec if isinstance(it, list) and len(it) >= 2 and (now_ts - float(it[1] or 0.0)) <= 600.0]
        rec.append([float(build_ms), now_ts])
        state["bm25_build_ms_recent"] = rec[-200:]
        state["bm25_last_build_ms"] = float(build_ms)
        state["bm25_docs_indexed"] = int(docs_indexed)
        state["bm25_last_build_mode"] = str(mode or "full")
        state["bm25_last_build_corpus"] = int(corpus_size)
        state["bm25_last_build_added"] = int(added_since)
    except Exception:
        return

def bm25_build_p95_ms(state: Dict[str, Any]) -> int:
    try:
        rec = [float(it[0]) for it in _safe_list(state.get("bm25_build_ms_recent")) if isinstance(it, list) and len(it) >= 1]
        if not rec:
            return 0
        rec = sorted(rec)
        idx95 = int(0.95 * float(len(rec) - 1))
        idx95 = max(0, min(len(rec) - 1, idx95))
        return int(rec[idx95])
    except Exception:
        return 0

def protocol_tick(cfg: Config, state: Dict[str, Any], client: "HttpClient") -> None:
    """Update protocol bookkeeping (no-op behaviorally in v19.0).

    v19.x will implement:
      - Daily rules sync (skills.md) [implemented in v19.1]
      - Mandatory voting for seen posts
      - 4~5h heartbeat quota (votes/comments/contribute)
    """
    try:
        proto = _sdict(state, "protocol")
        # v19.1: daily rules sync (skills.md)
        rules_sync_if_due(cfg, state, client)
        # cfg.protocol is the source of truth (env-backed), but keep state flag for visibility
        enabled = True
        try:
            enabled = bool(getattr(getattr(cfg, "protocol", None), "enabled", True))
        except Exception:
            enabled = bool(_env_bool("MERSOOM_PROTOCOL_ENABLE", True))
        proto["enabled"] = enabled
        proto.setdefault("cycle_id", 0)
        proto["last_tick_at"] = float(time.time())
        # v19.3: 4~5h heartbeat scheduler + quotas
        _heartbeat_tick(cfg, state)
    except Exception:
        return

def run() -> None:
    cfg = load_config_from_env()
    _apply_runtime_globals(cfg)
    log_info("boot: starting mersoom_agent")
    tuning = load_tuning_from_env()
    # selftest mode (exits by default)
    if _env_bool("MERSOOM_SELFTEST", False):
        rc = _run_selftest()
        if _env_bool("MERSOOM_SELFTEST_EXIT", True):
            raise SystemExit(int(rc))


    # (Unit 13) prevent accidental double-run
    acquire_process_lock(_resolve_lock_path(cfg.paths.state))

    client = HttpClient(cfg.http)

    state = load_state(cfg.paths.state)
    memory = load_memory(cfg.paths.memory, tuning)
    policy = load_policy(cfg.paths.policy, tuning)
    semantic = load_semantic(cfg.paths.semantic)
    brain = load_brain(cfg.paths.brain)
    if TZ_FALLBACK_USED and not bool(state.get("tz_fallback_recorded")):
        protocol_bump_counter(state, "tz_fallback_used", 1)
        state["tz_fallback_recorded"] = True

    # load threads/users (expanded state mirrors)
    threads_split = load_threads(cfg.paths.threads)
    users_split = load_users(cfg.paths.users)

    # Prefer the newest payload by __meta__.tick_id (split files can be coalesced).
    st_threads = _safe_dict(state.get("threads"))
    if isinstance(threads_split, dict) and threads_split:
        if (not st_threads) or (_meta_tick_id(threads_split) >= _meta_tick_id(st_threads)):
            state["threads"] = threads_split
    elif st_threads:
        state["threads"] = st_threads

    st_users = _safe_dict(state.get("users"))
    if isinstance(users_split, dict) and users_split:
        if (not st_users) or (_meta_tick_id(users_split) >= _meta_tick_id(st_users)):
            state["users"] = users_split
    elif st_users:
        state["users"] = st_users

    # (P0) boot self-test (connectivity + challenge parsing)
    boot_self_test(client, cfg, state)

    # corpus + bm25
    corpus_docs = load_corpus_jsonl(cfg.paths.corpus_jsonl, max_docs=3000)
    # ✅ 중복 방지용 set (최근 범위만)
    doc_id_set = set()
    for d in corpus_docs[-2500:]:
        if isinstance(d, dict) and d.get("doc_id"):
            doc_id_set.add(str(d.get("doc_id")))

    bm25 = BM25Index()
    t0_bm25 = time.perf_counter()
    bm25.build(corpus_docs, max_docs=2500)
    record_bm25_build(
        state,
        build_ms=(time.perf_counter() - t0_bm25) * 1000.0,
        docs_indexed=len(corpus_docs),
        mode="full",
        corpus_size=len(corpus_docs),
        added_since=int(state.get("bm25_added_since_build", 0) or 0),
    )

    # v21.1: split/mirror file persist coalescing timer
    last_split_persist_ts: float = 0.0


    # (Unit 01) QA batch report (optional)
    if getattr(cfg, "quality", None) is not None and cfg.quality.batch_on_boot:
        try:
            report = qa_run_batch_report(client, cfg, tuning, state, policy, semantic, brain, bm25)
            qa_print_batch_report(report, show_worst=int(cfg.quality.batch_show_worst))
            if str(cfg.quality.batch_save_path or "").strip():
                qa_write_batch_report(str(cfg.quality.batch_save_path).strip(), report)
        except Exception as e:
            log_debug_exc("qa_batch", e)
        if cfg.quality.batch_exit:
            return

    post_limiter = SlidingWindowLimiter(cfg.limits.posts_per_window, cfg.limits.window_sec)
    comment_limiter = SlidingWindowLimiter(cfg.limits.comments_per_window, cfg.limits.window_sec)
    vote_limiter = SlidingWindowLimiter(cfg.limits.votes_per_window, cfg.limits.window_sec)

    vote_pace_sec = pace_interval(cfg.limits.votes_per_window, cfg.limits.window_sec, cfg.timing.global_vote_min_gap_sec, cfg.mode.activity_mode)
    comment_pace_sec = pace_interval(cfg.limits.comments_per_window, cfg.limits.window_sec, cfg.timing.global_comment_min_gap_sec, cfg.mode.activity_mode)
    post_pace_sec = pace_interval(
        cfg.limits.posts_per_window,
        cfg.limits.window_sec,
        max(cfg.timing.global_post_min_gap_sec, cfg.timing.post_min_gap_sec),
        cfg.mode.activity_mode
    )

    Console.cprint(Console.CYAN, f"[START] BASE={cfg.http.base_url} dry_run={cfg.http.dry_run}")
    Console.cprint(Console.CYAN, f"[START] nickname={cfg.nickname} always_on={cfg.mode.always_on} mode={cfg.mode.activity_mode}")
    Console.cprint(Console.MAGENTA, f"[LIMITS/window] posts={cfg.limits.posts_per_window}, comments={cfg.limits.comments_per_window}, votes={cfg.limits.votes_per_window} window={cfg.limits.window_sec}s")
    Console.cprint(Console.MAGENTA, f"[PACE] vote={vote_pace_sec}s comment={comment_pace_sec}s post={post_pace_sec}s")
    Console.cprint(Console.MAGENTA, f"[LEARN] eps={policy.get('epsilon')} lr={policy.get('lr')} eval_delay={tuning.eval_delay_min_sec//3600}-{tuning.eval_delay_max_sec//3600}h learn_period={tuning.learn_period_sec//3600}h")
    Console.cprint(Console.GRAY, "Ctrl+C to stop.\n")

    update_daily_counters(state)
    posts_cache: List[Dict[str, Any]] = []

    # snapshot scheduler
    script_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_script_path = os.path.join(script_dir, str(cfg.snapshot.script or "snapshot_mersoom_to_xlsx_v2.py"))
    snapshot_enabled = bool(cfg.snapshot.enabled)
    if snapshot_enabled and not os.path.exists(snapshot_script_path):
        log_warn(f"snapshot disabled: script not found: {snapshot_script_path}")
        snapshot_enabled = False

    snapshot_next_dt = now_kst() if (snapshot_enabled and cfg.snapshot.run_on_boot) else next_top_of_hour_kst()

    while True:
        # monotonic tick id for cross-file coherence (P0)
        state["tick_id"] = int(state.get("tick_id", 0) or 0) + 1
        if _env_bool("MERSOOM_HEALTH_V2", False):
            protocol_bump_counter(state, "loop_tick", 1)
        tick_stamp = _make_tick_stamp(state)

        # No-action reasons (personal-run observability)
        tick_no_action: Dict[str, int] = {}

        def _note_no_action(reason: str) -> None:
            try:
                r = str(reason or "").strip()[:80]
                if not r:
                    return
                tick_no_action[r] = int(tick_no_action.get(r, 0) or 0) + 1
            except Exception:
                pass

        acted_any = False
        today = _today_kst()
        # 0) Hourly snapshot at :00 (best-effort)
        if snapshot_enabled:
            try:
                now_dt = now_kst()
                hour_key = now_dt.strftime("%Y-%m-%d %H")
                if now_dt >= snapshot_next_dt and str(state.get("last_snapshot_hour_kst", "")) != hour_key:
                    log_info(f"snapshot: running {os.path.basename(snapshot_script_path)} @ {now_dt.strftime('%H:%M:%S')}")
                    rc, out = run_snapshot_script(snapshot_script_path, cfg.snapshot.timeout_sec)
                    state["last_snapshot_hour_kst"] = hour_key
                    state["last_snapshot_ts"] = time.time()
                    if rc == 0:
                        log_info("snapshot: OK")
                    else:
                        log_warn(f"snapshot: exit={rc} out={one_line(out, 200)}")
                    snapshot_next_dt = next_top_of_hour_kst(now_dt + timedelta(seconds=1))
            except Exception as e:
                log_error("snapshot", repr(e))
                snapshot_next_dt = next_top_of_hour_kst()

        # 1) Sync
        if ops_should_skip(state, "sync") and posts_cache:
            # Circuit breaker: keep last cache to avoid hammering API on repeated failures
            pass
        else:
            try:
                if (time.time() - float(state.get("last_sync_ts", 0.0) or 0.0)) >= cfg.timing.sync_min_interval_sec or not posts_cache:
                    posts_cache = do_sync_posts(client, cfg, state, tuning)
                    log_info(f"sync posts={len(posts_cache)} corpus={len(corpus_docs)}")
                ops_record_success(state, "sync")
            except Exception as e:
                ops_record_failure(state, "sync", repr(e))
                log_error("sync", repr(e))
                nap = random.randint(cfg.timing.idle_retry_min, cfg.timing.idle_retry_max)
                sleep_chunked(
                    nap,
                    hard_cap_sec=cfg.timing.sleep_hard_cap_sec,
                    why="(sync fail)",
                    wake_deadline_wall_ts=(snapshot_next_dt.timestamp() if snapshot_enabled else None),
                )
                continue

        # 1.5) Community flow + thread context scan (LLM-free 'perception' + 'compression')
        try:
            # share tuning into brain (so helper uses correct cap)
            brain["max_thoughts"] = int(tuning.max_thoughts)

            scan_n = int(tuning.scan_posts_per_sync)
            scan_posts = posts_cache[:max(0, scan_n)]
            update_community_flow(brain, scan_posts, half_life_hours=float(tuning.flow_half_life_hours))

            for p in scan_posts:
                ingest_post_into_context(state, p, brain=brain)
                pid = str(p.get("id") or "")
                if pid:
                    synthesize_thread(get_thread(state, pid))
        except Exception as e:
            log_error("flow_scan", repr(e))

        # 1.75) Arena (Colosseum) flow (Unit 09)
        if ops_should_skip(state, "arena"):
            protocol_set_reason(state, "arena", "arena:ops_disabled", "ops_should_skip")
            _note_no_action(protocol_get_reason(state, "arena"))
            pass
        else:
            try:
                a = do_arena_flow(client, cfg, tuning, state, memory, brain)
                if a > 0:
                    acted_any = True
                else:
                    r = protocol_get_reason(state, "arena")
                    _note_no_action(r if r != "unknown" else "arena:no_action")
                ops_record_success(state, "arena")
            except Exception as e:
                ops_record_failure(state, "arena", repr(e))
                log_error("arena_loop", repr(e))
                protocol_set_reason(state, "arena", "arena:error", one_line(repr(e), 120))
                _note_no_action(protocol_get_reason(state, "arena"))

        # 1.9) v19 protocol scaffold tick (no behavior change yet)
        try:
            protocol_tick(cfg, state, client)
        except Exception as e:
            log_error("protocol", repr(e))


        # 2) Votes
        if ops_should_skip(state, "vote"):
            protocol_set_reason(state, "vote", "vote:ops_disabled", "ops_should_skip")
            _note_no_action(protocol_get_reason(state, "vote"))
            pass
        else:
            try:
                voted = do_vote_main_feed(client, cfg, tuning, state, memory, semantic, brain, posts_cache, vote_limiter, vote_pace_sec)
                if voted > 0:
                    acted_any = True
                else:
                    r = protocol_get_reason(state, "vote")
                    _note_no_action(r if r != "unknown" else "vote:no_action")
                ops_record_success(state, "vote")
            except Exception as e:
                ops_record_failure(state, "vote", repr(e))
                log_error("vote_loop", repr(e))
                protocol_set_reason(state, "vote", "vote:error", one_line(repr(e), 120))
                _note_no_action(protocol_get_reason(state, "vote"))

        # 3) Contribution
        if ops_should_skip(state, "contrib"):
            protocol_set_reason(state, "comment", "comment:ops_disabled", "ops_should_skip")
            _note_no_action(protocol_get_reason(state, "comment"))
            pass
        else:
            try:
                c = do_contribution(
                    client, cfg, tuning,
                    state, memory, semantic, policy, brain,
                    posts_cache,
                    post_limiter, comment_limiter,
                    comment_pace_sec, post_pace_sec,
                    bm25=bm25
                )
                if c > 0:
                    acted_any = True
                else:
                    r = protocol_get_reason(state, "comment")
                    _note_no_action(r if r != "unknown" else "comment:no_action")
                ops_record_success(state, "contrib")
            except Exception as e:
                ops_record_failure(state, "contrib", repr(e))
                log_error("contrib", repr(e))
                protocol_set_reason(state, "comment", "comment:error", one_line(repr(e), 120))
                _note_no_action(protocol_get_reason(state, "comment"))

        # 4) Mine templates from rewarded comments (LLM-free)
        try:
            maturity = get_maturity_level(brain, state)
            evald = int(state.get("evaluated_count", 0) or 0)

            # cold-start: allow slightly lower threshold so templates accumulate
            thr_hi = 0.90
            if evald < 20 or maturity < 0.25:
                thr_hi = 0.75
            elif evald < 80 or maturity < 0.45:
                thr_hi = 0.82
            thr_lo = max(0.65, thr_hi - 0.12)

            mined = 0
            for it in memory[-60:]:
                if not isinstance(it, dict):
                    continue
                if it.get("evaluated") is not True:
                    continue
                if str(it.get("action") or "") not in ("comment", "reply"):
                    continue

                r = float(it.get("reward_scalar", 0.0) or 0.0)
                nov = float(it.get("novelty", 0.0) or 0.0)
                # main gate: high reward OR medium reward with strong novelty (avoids echo templates)
                if r < thr_hi and not (r >= thr_lo and nov >= 0.55):
                    continue

                txt = str(it.get("text") or "")
                if not txt:
                    continue
                tpl = mine_template_from_text(txt)
                if tpl:
                    tid_new = register_mined_template(policy, tpl, meta={"source": "rewarded_comment", "r": r, "nov": nov})
                    if tid_new:
                        mined += 1
                    if mined >= 3:
                        break

            # slow pruning to keep quality rising
            pr = prune_templates(policy, min_keep=10)
            if pr > 0:
                bump_semantic(semantic, _today_kst(), "tpl_prune", float(pr))
        except Exception as e:
            log_error("template_miner", repr(e))

# 5) Corpus growth (dedup + cap)
        try:
            added = 0
            for p in posts_cache[:8]:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("id") or "")
                if not pid:
                    continue
                txt = f"{p.get('title') or ''} {p.get('content') or ''}".strip()
                if len(txt) < 20:
                    continue

                doc_id = corpus_doc_id("post", pid, str(p.get("nickname") or ""), txt)
                if doc_id in doc_id_set:
                    continue

                doc = {
                    "doc_id": doc_id,
                    "kind": "post",
                    "post_id": pid,
                    "author": str(p.get("nickname") or ""),
                    "ts": time.time(),
                    "text": sanitize_plain_text(txt),
                }
                doc_id_set.add(doc_id)
                corpus_docs.append(doc)
                txt2 = str(doc.get("text") or "")
                append_corpus_doc(cfg.paths.corpus_jsonl, {**doc, "tokens": tokenize(txt2, max_tokens=200)})
                added += 1

            # cap in-memory corpus
            if len(corpus_docs) > 3000:
                corpus_docs = corpus_docs[-3000:]
                # doc_id_set도 최근 범위로 재구성
                doc_id_set = set(str(d.get("doc_id")) for d in corpus_docs[-2500:] if isinstance(d, dict) and d.get("doc_id"))

            # (P1) Rebuild BM25 only when enough new docs are added (CPU friendly)
            if added > 0:
                state["bm25_added_since_build"] = int(state.get("bm25_added_since_build", 0) or 0) + int(added)

            last_b = float(state.get("bm25_last_build_ts", 0.0) or 0.0)
            need_by_add = int(state.get("bm25_added_since_build", 0) or 0) >= int(getattr(tuning, "bm25_rebuild_min_add", 6))
            need_by_time = (time.time() - last_b) >= float(getattr(tuning, "bm25_rebuild_min_sec", 600))
            if (need_by_add or need_by_time) and int(state.get("bm25_added_since_build", 0) or 0) > 0:
                added_since = int(state.get("bm25_added_since_build", 0) or 0)
                use_v2 = _env_bool("MERSOOM_BM25_BUILD_V2", False)
                build_docs = corpus_docs
                mode = "full"
                if use_v2:
                    recent_n = _env_int("MERSOOM_BM25_BUILD_V2_RECENT_DOCS", 1200, 200, 3000)
                    build_docs = corpus_docs[-recent_n:]
                    mode = "partial"
                t0 = time.perf_counter()
                bm25.build(build_docs, max_docs=2500)
                build_ms = (time.perf_counter() - t0) * 1000.0
                record_bm25_build(
                    state,
                    build_ms=build_ms,
                    docs_indexed=len(build_docs),
                    mode=mode,
                    corpus_size=len(corpus_docs),
                    added_since=added_since,
                )
                state["bm25_added_since_build"] = 0
                state["bm25_last_build_ts"] = time.time()

        except Exception as e:
            log_error("corpus", repr(e))

        # 6) Evaluation + learning (batched)
        try:
            eval_n, eval_sum = evaluate_and_learn(client, tuning, memory, policy, semantic, state)
            if eval_n > 0:
                log_info(f"eval done={eval_n} reward_sum={eval_sum:.2f}")
                write_journal(cfg.paths.journal, f"eval {eval_n} items | reward_sum={eval_sum:.2f}")
        except Exception as e:
            log_error("eval_loop", repr(e))

        # 7) Brain update + note
        try:
            update_brain(brain, tuning, memory, policy)
            # persisted in the main persist block
        except Exception as e:
            log_error("brain", repr(e))

        # trim memory
        if len(memory) > tuning.memory_size:
            memory[:] = memory[-tuning.memory_size:]

        # tick journal + semantic
        try:
            contrib = int(state.get("contrib_count_today", 0))
            write_journal(
                cfg.paths.journal,
                f"tick | acted={acted_any} | contrib_today={contrib} | total_actions={state.get('total_actions', 0)} | eval_total={state.get('evaluated_count', 0)} | mem={len(memory)} | rem(v/c/p)={vote_limiter.remaining()}/{comment_limiter.remaining()}/{post_limiter.remaining()}"
            )
            bump_semantic(semantic, today, "ticks", 1.0)
        except Exception as e:
            log_debug_exc("run:silent", e)
            pass

        # persist (with cross-file generation stamp)
        try:
            _apply_stamp(state, tick_stamp)
            _apply_stamp(policy, tick_stamp)
            _apply_stamp(semantic, tick_stamp)

            th_obj = _safe_dict(state.get("threads"))
            us_obj = _safe_dict(state.get("users"))
            _apply_stamp(th_obj, tick_stamp)
            _apply_stamp(us_obj, tick_stamp)

            # brain is small; persist each tick so stamp stays coherent across files
            _apply_stamp(brain, tick_stamp)

            save_json_file_atomic(cfg.paths.state, state)
            save_json_file_atomic(cfg.paths.memory, memory)
            save_json_file_atomic(cfg.paths.policy, policy)
            save_json_file_atomic(cfg.paths.semantic, semantic)
            save_json_file_atomic(cfg.paths.brain, brain)

            # v21.1: coalesce "mirror"/derived files to reduce I/O
            try:
                split_every = float(_env_int("MERSOOM_PERSIST_SPLIT_EVERY_SEC", 60, 0, 86400))
                now_ts = float(time.time())
                do_split = (split_every <= 0.0) or (now_ts - float(last_split_persist_ts) >= split_every) or bool(acted_any)
                if do_split:
                    save_json_file_atomic(cfg.paths.threads, th_obj)
                    save_json_file_atomic(cfg.paths.users, us_obj)
                    save_text_file_atomic(cfg.paths.brain_note, render_brain_note(brain))

                    # meta manifests (out-of-band; doesn't touch memory list)
                    try:
                        meta = {
                            "__meta__": _safe_dict(tick_stamp),
                            "files": {
                                "state": os.path.abspath(cfg.paths.state),
                                "memory": os.path.abspath(cfg.paths.memory),
                                "policy": os.path.abspath(cfg.paths.policy),
                                "semantic": os.path.abspath(cfg.paths.semantic),
                                "brain": os.path.abspath(cfg.paths.brain),
                                "brain_note": os.path.abspath(cfg.paths.brain_note),
                                "threads": os.path.abspath(cfg.paths.threads),
                                "users": os.path.abspath(cfg.paths.users),
                            },
                        }
                        save_json_file_atomic(cfg.paths.meta, meta)
                        mem_meta = {"__meta__": _safe_dict(tick_stamp), "count": len(_safe_list(memory))}
                        save_json_file_atomic(cfg.paths.memory_meta, mem_meta)
                    except Exception as e:
                        log_debug_exc("persist_meta", e)

                    last_split_persist_ts = now_ts
            except Exception as e:
                log_debug_exc("persist_split", e)

            # flush buffered jsonl (if enabled)
            try:
                for pth in list(_JSONL_BUFFERS.keys()):
                    _flush_jsonl_buffer(pth, force=False)
            except Exception:
                pass

        except Exception as e:
            log_error("persist", repr(e))

        # (Unit 13) tick health summary (rate-limited to avoid log spam)
        try:
            ops = _ops_init(state)
            last_log = float(ops.get("last_tick_log_ts", 0.0) or 0.0)
            now_log = time.time()
            if acted_any or (now_log - last_log) >= float(_env_int("MERSOOM_TICK_LOG_EVERY_SEC", 180, 10, 3600)):
                ops["last_tick_log_ts"] = now_log
                m = getattr(client, "metrics", {}) or {}
                disabled = ops_disabled_keys(state)
                log_info(
                    "tick"
                    + f" acted={int(bool(acted_any))}"
                    + f" total_actions={int(state.get('total_actions', 0) or 0)}"
                    + f" total_reward={float(state.get('total_reward', 0.0) or 0.0):.3f}"
                    + f" eval={int(state.get('evaluated_count', 0) or 0)}"
                    + f" req_ok={int(m.get('req_ok', 0) or 0)}"
                    + f" 429={int(m.get('rate_limited', 0) or 0)}"
                    + f" last_status={int(m.get('last_status', 0) or 0)}"
                    + (f" ops_disabled={','.join(disabled)}" if disabled else " ops_disabled=-")
                )
        except Exception as e:
            log_debug_exc("run:silent", e)
            pass

        if not cfg.mode.always_on:
            log_info("ALWAYS_ON=false -> stop after one tick")
            break

        nap = next_sleep_from_limits(cfg, state, post_limiter, comment_limiter, vote_limiter, vote_pace_sec, comment_pace_sec, post_pace_sec)
        if not acted_any:
            nap = min(nap, random.randint(cfg.timing.idle_retry_min, cfg.timing.idle_retry_max))
        # No-action summary (only when nothing was done this tick)
        if not acted_any and tick_no_action:
            try:
                win = _safe_dict(state.get("no_action_window"))
                for k, v in tick_no_action.items():
                    win[k] = int(win.get(k, 0) or 0) + int(v or 0)
                # cap window keys
                if len(win) > 64:
                    # keep top 48
                    items = sorted(win.items(), key=lambda kv: int(kv[1] or 0), reverse=True)[:48]
                    win = {k: int(v) for k, v in items}
                state["no_action_window"] = win

                every = _env_int("MERSOOM_NO_ACTION_LOG_EVERY_SEC", 600, 60, 7200)
                last = float(state.get("no_action_last_log_ts", 0.0) or 0.0)
                now = time.time()
                if (now - last) >= float(every):
                    top = sorted(win.items(), key=lambda kv: int(kv[1] or 0), reverse=True)[:6]
                    msg = ", ".join([f"{k}={v}" for k, v in top]) if top else "n/a"
                    log_info(f"no_action: {msg}")
                    state["no_action_last_log_ts"] = now
                    # reset window after logging to keep it "recent"
                    state["no_action_window"] = {}
            except Exception as e:
                log_debug_exc("no_action_summary:silent", e)
                pass

        sleep_chunked(
            nap,
            hard_cap_sec=cfg.timing.sleep_hard_cap_sec,
            why="(tick)",
            wake_deadline_wall_ts=(snapshot_next_dt.timestamp() if snapshot_enabled else None),
        )

if __name__ == "__main__":
    try:
        _apply_cli_overrides(sys.argv[1:])
        run()
    except KeyboardInterrupt:
        log_warn("stopped by user (Ctrl+C)")
    except Exception as e:
        log_error("fatal", repr(e))
        raise
