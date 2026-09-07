"""Shared HTTP session for the football-data.co.uk pulls.

football-data.co.uk sits behind an nginx front that, under load from a run
pulling a batch of files back to back, starts returning a 503 "temporarily
unavailable" HTML page - sometimes with a Retry-After of several *minutes*.
Every ingest module that hits that host goes through the session built here so
the retry/backoff policy is defined once.

respect_retry_after_header is off on purpose: urllib3 sleeps the raw header
value uncapped, and a multi-minute Retry-After across several attempts would
blow any CI job budget. A sustained throttle is meant to fall through to a
cached copy on disk (see ingest.historical / ingest.fixtures), not be waited
out here. raise_on_status is off so a persistent failure still surfaces as an
HTTPError from resp.raise_for_status() rather than a bare MaxRetryError.
"""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_UA = "football-predictor/0.1 (educational; contact via repo)"

_RETRY = Retry(
    total=3,
    backoff_factor=1.0,  # ~0s, 2s, 4s between attempts
    backoff_max=10.0,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    respect_retry_after_header=False,
    raise_on_status=False,
)

# A courtesy gap between files so a batch refresh doesn't hammer the front hard
# enough to trip the throttle in the first place.
INTER_REQUEST_PAUSE = 0.5


def _build() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = _UA
    adapter = HTTPAdapter(max_retries=_RETRY)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = _build()
