#!/usr/bin/env python3
"""딥시크가 점유한 원격 GPU의 전력·가동률을 캐시로 모아 온다.

2026-09-19부터 모니터는 GPU·vLLM이 있는 NHN `NEXGEM`(GPU 4~7, 텐서 병렬)에서
**직접** 돌고 `mode: local`로 로컬 `nvidia-smi`를 읽는다. 이전에는 `nexgem` 로그인
노드에서 돌며 원격 `ssh nhn`으로 읽었다. `nexgem`에는 `nvidia-smi`가 없다.

브라우저가 2초마다 `/api/state`를 폴링하므로, **요청마다 SSH나 nvidia-smi를 새로
실행하면 안 된다.** 그러면 2초마다 SSH 세션이 열려 원격에 부하를 주고, 느린 SSH가
요청을 붙잡는다. 그래서 수집과 표시를 나눈다.

  수집 (`collect` -> `refresh_cache`)   별도 스레드 또는 `--once`가 TTL마다 한 번
                                        SSH로 `nvidia-smi`를 실행해 캐시 파일을 쓴다.
  표시 (`read_cache` -> `telemetry_state`)  모니터의 요청 경로는 **캐시 파일만**
                                        읽는다. SSH를 절대 부르지 않는다.

접근 경로는 `talks/ops/gpu.json`(git 추적)이 정하고, 기계별 override는
`talks/ops/gpu.local.json`(git-ignore, 이긴다)이 정한다. 주소를 코드에 적지 않는다
(D-053). 기본값은 로그인 노드(이 기계) 기준이라 `mode: ssh`, `ssh: nhn`이다.

정확한 수치를 못 얻으면 **0으로 채우지 않는다.** 각 GPU는 필드별로 `None`이 되고
화면은 `모름`으로 그린다. 0 W는 진짜 0(유휴)일 수 있으므로, 파싱 실패(`[N/A]`,
장치 누락)와 구분한다.

`nvidia-smi` 경로가 막히면(SSH 불가·키 만료·`nvidia-smi` 부재) 대안은 exporter다.
`talks/ops/gpu.json`에 `exporter_url`을 넣고 수집 모드를 HTTP로 바꾸는 것이 정석이며,
필요한 지표 이름은 이 파일 docstring 끝의 "필요한 exporter" 절에 적어 둔다.

    python scripts/ops/gpu_telemetry.py --once     # 캐시를 한 번 채운다
    python scripts/ops/gpu_telemetry.py --show      # 캐시된 상태를 사람이 읽게 출력

필요한 exporter (nvidia-smi 경로가 막혔을 때):
  - `nvidia_gpu_exporter`(utkuozdemir) — `nvidia_smi_power_draw_watts`,
    `nvidia_smi_power_limit_watts`, `nvidia_smi_utilization_gpu_ratio`.
  - 또는 NVIDIA `dcgm-exporter` — `DCGM_FI_DEV_POWER_USAGE`,
    `DCGM_FI_DEV_POWER_MGMT_LIMIT`, `DCGM_FI_DEV_GPU_UTIL`.
  Prometheus가 NEXGEM에서 scrape하고, 로그인 노드에서 그 주소에 닿게 한 뒤
  `gpu.json`에 `{"mode": "http", "exporter_url": "http://.../metrics"}`를 적는다.
  (HTTP 모드 수집기는 아직 구현하지 않았다 — 그 전까지는 도달 불가로 표시한다.)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPS = PROJECT_ROOT / "talks" / "ops"
CONFIG = OPS / "gpu.json"
LOCAL = OPS / "gpu.local.json"
CACHE = OPS / "gpu_telemetry_cache.json"
KST = timezone(timedelta(hours=9))

#: 딥시크가 텐서 병렬로 점유한 GPU. 원격 NEXGEM의 4~7번이다.
GPU_IDS = (4, 5, 6, 7)

DEFAULT_TTL = 15.0
#: 가동률이 이 시간(초) 이상 연속 0%일 때만 유휴로 본다. 표본 한 번으로는 확정하지 않는다.
IDLE_SECONDS = 10.0
DEFAULT_CONFIG = {
    "mode": "ssh",
    "ssh": "nhn",
    "ids": list(GPU_IDS),
    "ttl_seconds": DEFAULT_TTL,
    "timeout_seconds": 20,
    "exporter_url": None,
}

#: `[N/A]`, `[Not Supported]`, `[Unknown Error]` 등 값이 없는 표식.
_MISSING = ("[n/a]", "[not supported]", "[unknown error]", "[insufficient permissions]",
            "n/a", "unknown", "")


def load_config(config: Path | str = CONFIG,
                local: Path | str = LOCAL) -> dict:
    """기본값 위에 tracked 설정, 그 위에 기계별 override를 얹는다."""
    cfg = dict(DEFAULT_CONFIG)
    for path in (Path(config), Path(local)):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(raw, dict):
            cfg.update(raw)
    ids = cfg.get("ids")
    cfg["ids"] = [int(i) for i in ids] if ids else list(GPU_IDS)
    return cfg


# --------------------------------------------------------------------------
# 파싱 — 순수 함수. 표본으로 시험한다.
# --------------------------------------------------------------------------

def _value(token: str) -> float | None:
    token = (token or "").strip()
    if token.lower() in _MISSING:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_nvidia_smi(text: str) -> dict:
    """`--query-gpu=index,power.draw,power.limit,utilization.gpu` CSV를 읽는다.

    반환: `{"gpus": {4: {...}}, "errors": [...]}`. 값이 `[N/A]`면 그 필드는
    `None`이다 — 0이 아니다. 줄이 4칸이 아니거나 index가 정수가 아니면 error로
    보내고 조용히 버리지 않는다.
    """
    gpus: dict[int, dict] = {}
    errors: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            errors.append(line)
            continue
        try:
            index = int(parts[0])
        except ValueError:
            errors.append(line)
            continue
        gpus[index] = {
            "power_draw": _value(parts[1]),
            "power_limit": _value(parts[2]),
            "utilization": _value(parts[3]),
        }
    return {"gpus": gpus, "errors": errors}


def _query(gpus: list[int]) -> list[str]:
    ids = ",".join(str(g) for g in gpus)
    return ["nvidia-smi",
            "--query-gpu=index,power.draw,power.limit,utilization.gpu",
            "--format=csv,noheader,nounits", "-i", ids]


# --------------------------------------------------------------------------
# 수집
# --------------------------------------------------------------------------

def _stamp(now: datetime) -> str:
    return now.strftime("%Y-%m-%d %H:%M:%S")


def collect(cfg: dict | None = None, gpus: list[int] | None = None,
            now: datetime | None = None) -> dict:
    """한 번 수집해 캐시 레코드를 만든다. 실패해도 예외를 던지지 않는다."""
    cfg = cfg or load_config()
    want = list(gpus or cfg.get("ids") or GPU_IDS)
    now = now or datetime.now(KST)
    record = {
        "collected_at": _stamp(now),
        "epoch": now.timestamp(),
        "mode": cfg.get("mode", "ssh"),
        "source": None,
        "ok": False,
        "error": None,
        "gpus": {},
        "missing": want,
    }

    mode = cfg.get("mode", "ssh")
    if mode == "local":
        cmd, source = _query(want), "local"
    elif mode == "ssh":
        target = str(cfg.get("ssh") or "nhn")
        cmd = ["ssh",
               "-o", "BatchMode=yes",
               "-o", f"ConnectTimeout={int(cfg.get('timeout_seconds', 20))}",
               "-o", "StrictHostKeyChecking=accept-new",
               target] + _query(want)
        source = f"ssh:{target}"
    else:
        record["error"] = (f"수집 모드 {mode!r}는 아직 없다 — exporter를 쓰려면 "
                           f"gpu.json에 mode/http/exporter_url을 설정하라")
        return record
    record["source"] = source

    timeout = float(cfg.get("timeout_seconds", 20))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - 도달 불가는 상태이지 예외가 아니다
        record["error"] = f"{source}: {exc}"
        return record

    parsed = parse_nvidia_smi(out.stdout)
    gpus_out = {str(k): v for k, v in parsed["gpus"].items()}
    record["gpus"] = gpus_out
    record["missing"] = [g for g in want if g not in parsed["gpus"]]
    notes = list(parsed["errors"])
    if out.returncode != 0 and not gpus_out:
        notes.append((out.stderr or "").strip() or f"rc={out.returncode}")
    record["ok"] = bool(gpus_out)
    record["error"] = "; ".join(n for n in notes if n) or None
    return record


def carry_zero_since(prev: dict | None, record: dict, now: datetime,
                     ttl: float = DEFAULT_TTL,
                     idle_seconds: float = IDLE_SECONDS) -> dict:
    """새 수집 레코드에 GPU별 `zero_since`(0% 시작 수집 시각)를 붙인다.

    한 표본의 0%만으로 유휴를 주장하지 않기 위해, 직전 수집이 충분히 가까울
    때만 0% 구간을 이어 붙인다. 직전 수집이 없거나 오래됐으면(수집기 중단 등)
    구간을 `now`에서 다시 시작하므로, 수집 공백이 긴 유휴로 둔갑하지 않는다.
    가동률이 양수이거나 알 수 없으면 구간을 지운다.
    """
    prev_rows = (prev or {}).get("gpus") or {}
    prev_epoch = (prev or {}).get("epoch")
    now_epoch = now.timestamp()
    recent = (isinstance(prev_epoch, (int, float))
              and 0 <= now_epoch - float(prev_epoch) <= max(ttl, idle_seconds) * 1.5)
    for idx, row in (record.get("gpus") or {}).items():
        if not isinstance(row, dict):
            continue
        key = idx if idx in prev_rows else (int(idx) if str(idx).isdigit() and int(idx) in prev_rows else idx)
        prev_zero = (prev_rows.get(key) or {}).get("zero_since")
        util = row.get("utilization")
        if util is not None and util <= 0 and isinstance(prev_zero, (int, float)) and recent:
            row["zero_since"] = float(prev_zero)
        elif util is not None and util <= 0:
            row["zero_since"] = now_epoch
        else:
            row["zero_since"] = None
    return record


def write_cache(path: Path | str, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def read_cache(path: Path | str = CACHE) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def refresh_cache(path: Path | str = CACHE, cfg: dict | None = None,
                  gpus: list[int] | None = None, now: datetime | None = None,
                  collector=collect) -> dict:
    """TTL 안이면 기존 캐시를 돌려주고, 아니면 한 번 수집해 덮어쓴다.

    표시 경로가 아니라 **수집 스레드**가 부른다. `collector` 주입은 시험용이다.
    """
    cfg = cfg or load_config()
    now = now or datetime.now(KST)
    cached = read_cache(path)
    ttl = float(cfg.get("ttl_seconds", DEFAULT_TTL))
    epoch = cached.get("epoch")
    if isinstance(epoch, (int, float)) and now.timestamp() - float(epoch) < ttl:
        return cached
    record = collector(cfg, gpus, now)
    record = carry_zero_since(cached, record, now, ttl)
    write_cache(path, record)
    return record


# --------------------------------------------------------------------------
# 표시 — 순수 함수. 캐시만 본다.
# --------------------------------------------------------------------------

def gpu_activity_status(utilization: float | None, zero_since: float | None,
                        collected_epoch: float | None, stale: bool = False,
                        idle_seconds: float = IDLE_SECONDS) -> str:
    """가동률 하나가 아니라 **연속 0% 구간**으로 상태를 정하는 순수 함수.

    - `stale`이거나 가동률을 모르면 `모름` (0으로 대체하지 않는다).
    - 가동률이 0보다 크면 `사용 중`.
    - 0% 구간이 `idle_seconds` 이상 **수집 시각 기준**으로 이어졌을 때만 `유휴`.
      표본 한 번이나 브라우저 렌더 시각만으로는 `유휴`가 되지 않는다.
    """
    if stale or utilization is None:
        return "모름"
    if utilization > 0:
        return "사용 중"
    if (isinstance(zero_since, (int, float)) and isinstance(collected_epoch, (int, float))
            and collected_epoch - float(zero_since) >= idle_seconds):
        return "유휴"
    return "모름"


def _gpu_view(index: int, row: dict | None, collected_epoch: float | None,
              stale: bool) -> dict:
    if not row:
        return {"index": index, "power_draw": None, "power_limit": None,
                "utilization": None, "status": "모름"}
    return {
        "index": index,
        "power_draw": row.get("power_draw"),
        "power_limit": row.get("power_limit"),
        "utilization": row.get("utilization"),
        "status": gpu_activity_status(row.get("utilization"), row.get("zero_since"),
                                      collected_epoch, stale),
    }


def telemetry_state(cache: dict | None, now: datetime,
                    gpus: list[int] | tuple[int, ...] | None = None,
                    ttl: float = DEFAULT_TTL) -> dict:
    """캐시를 화면이 그릴 상태로 바꾼다. SSH를 부르지 않는다.

    - 캐시가 없으면 `available=False`, 각 GPU는 `모름`(0 아님).
    - 수집이 실패한 캐시면 `available=False`와 이유, 각 GPU는 `모름`.
    - 성공했어도 요청한 GPU가 빠져 있으면 그 GPU만 `모름`(부분 누락).
    - `ttl`의 두 배를 넘긴 캐시는 `stale=True`로 표시하되 값은 남긴다.
    """
    want = list(gpus if gpus is not None else GPU_IDS)
    if not cache:
        return {"available": False, "stale": False, "collected_at": None,
                "age_seconds": None, "mode": None, "source": None,
                "error": "GPU telemetry 캐시 없음 — 수집기가 아직 돌지 않았다",
                "gpus": [_gpu_view(i, None, None, False) for i in want]}
    epoch = cache.get("epoch")
    age = None
    collected_epoch = None
    if isinstance(epoch, (int, float)):
        collected_epoch = float(epoch)
        age = max(0.0, now.timestamp() - collected_epoch)
    stale = bool(age is not None and age > ttl * 2)
    rows = cache.get("gpus") or {}
    ok = bool(cache.get("ok"))
    return {
        "available": ok,
        "stale": stale,
        "collected_at": cache.get("collected_at"),
        "age_seconds": age,
        "mode": cache.get("mode"),
        "source": cache.get("source"),
        "error": cache.get("error"),
        "gpus": [_gpu_view(i, rows.get(str(i)) or rows.get(i), collected_epoch, stale)
                 for i in want],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--once", action="store_true", help="TTL을 무시하고 한 번 수집한다")
    ap.add_argument("--show", action="store_true", help="캐시된 상태만 출력한다")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.show:
        state = telemetry_state(read_cache(args.cache), datetime.now(KST),
                                cfg.get("ids"), float(cfg.get("ttl_seconds", DEFAULT_TTL)))
        print(json.dumps(state, ensure_ascii=False, indent=1))
        return 0
    record = collect(cfg) if args.once else refresh_cache(args.cache, cfg)
    write_cache(args.cache, record)
    print(json.dumps(record, ensure_ascii=False, indent=1))
    return 0 if record.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
