"""Every value the engine uses. Copied from ai/PRODUCT-BRIEF.md section 5 (Engine)."""
from datetime import datetime, timezone

SYMBOLS = ["NVDAUSDT", "TSLAUSDT", "QQQUSDT"]
ALIASES = {"NVDAUSDT": "Asset A", "TSLAUSDT": "Asset B", "QQQUSDT": "Asset C"}
GRANULARITY = "4H"
BAR_MS = 4 * 3600 * 1000
WINDOW_START = datetime(2026, 7, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 9, 26, 20, tzinfo=timezone.utc)  # last bar open included
BARS_PER_YEAR = 2190
TAKER_FEE = 0.0006
SLIPPAGE = 0.0002  # assumption, labelled in the report
AGENT_MODEL = "claude-haiku-4-5-20251001"  # released Oct 2025, training data to early 2025: before the window
AGENT_FALLBACK_MODEL = None  # no fallback: a later model could have seen the window
PARSER_MODEL = "gemini-3.5-flash-lite"  # parsing only; not contamination-sensitive
AGENT_RUNS = 2
AGENT_SEGMENTS = 4  # each run is replayed as 4 consecutive segments in parallel; each starts flat
LOOKBACK_RETURNS = 12
VOL_WINDOW = 30
LLM_RPM = 14  # free Gemini tier cap is 15 per minute per model (429 body, 2026-09-27)
LLM_TIMEOUT_S = 30
LLM_RETRIES = 2
PLACEBO_DRAWS = 2000
PLACEBO_SEED = 7
RULE_LOOKBACK = 6
MAX_CSV_BYTES = 1_000_000

VERDICT_DEPLOY = "DEPLOY"
VERDICT_HALF = "HALF SIZE"
VERDICT_NO = "DO NOT DEPLOY"

BITGET = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
