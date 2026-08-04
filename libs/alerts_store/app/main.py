import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PER_PAGE = int(os.environ.get("PER_PAGE", "500"))

_raw_dirs = os.environ.get("DATA_DIRS", "")
_DATA_DIRS: list[Path] = (
    [Path(p.strip()) for p in _raw_dirs.split(",") if p.strip()]
    if _raw_dirs else [DATA_DIR]
)
# "alert_{19-digit-id}.avro\n" ≈ 32 bytes — used to seek into index.txt without
# loading the whole file. Slightly wrong at page boundaries is fine; we re-align.
_AVG_LINE = 32

app = FastAPI(docs_url=None, redoc_url=None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _base_url(request: Request) -> str:
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", "localhost")
    return f"{scheme}://{host}/"

def _safe_mjd_dir(mjd: str) -> Path:
    """Resolve and validate that the path stays inside one of the data dirs."""
    valid_path = False
    for data_dir in _DATA_DIRS:
        resolved_root = data_dir.resolve()
        path = (data_dir / mjd).resolve()
        if not str(path).startswith(str(resolved_root) + "/"):
            continue
        valid_path = True
        if path.is_dir():
            return path
    if not valid_path:
        raise HTTPException(status_code=400, detail="Invalid night")
    raise HTTPException(status_code=404, detail="Night not found")


def _nights() -> list[Path]:
    seen: set[str] = set()
    nights: list[Path] = []
    for data_dir in _DATA_DIRS:
        if not data_dir.is_dir():
            continue
        for d in data_dir.iterdir():
            if d.is_dir() and not d.name.startswith(".") and d.name not in seen:
                seen.add(d.name)
                nights.append(d)
    return sorted(nights, key=lambda d: d.name, reverse=True)


def _count_estimate(mjd_dir: Path) -> str:
    idx = mjd_dir / "index.txt"
    if not idx.exists():
        return "indexing…"
    size = idx.stat().st_size
    if size == 0:
        return "0"
    n = size // _AVG_LINE
    if n >= 1_000_000:
        return f"~{n / 1_000_000:.1f} M"
    if n >= 1_000:
        return f"~{n / 1_000:.0f} K"
    return str(n)


def _read_page(mjd_dir: Path, page: int) -> tuple[list[str], int]:
    """
    Return (filenames, total_estimate) for the requested page.
    Uses a byte-offset seek so we never load the full index.txt into memory,
    even for 10-million-line files.
    """
    idx = mjd_dir / "index.txt"
    if not idx.exists():
        return [], 0

    size = idx.stat().st_size
    total_est = max(size // _AVG_LINE, 0)
    byte_offset = page * PER_PAGE * _AVG_LINE

    if byte_offset >= size:
        return [], total_est

    names: list[str] = []
    with open(idx, "rb") as f:
        f.seek(byte_offset)
        if byte_offset > 0:
            f.readline()  # discard the partial line we landed in the middle of
        for _ in range(PER_PAGE):
            raw = f.readline()
            if not raw:
                break
            name = raw.decode("utf-8", errors="replace").strip()
            if name:
                names.append(name)

    return names, total_est


# ---------------------------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; }
body { font-family: monospace; padding: 24px; max-width: 960px;
       color: #222; background: #fff; }
h2   { border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 0; }
a    { color: #0055cc; text-decoration: none; }
a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 0.92em; }
th, td { text-align: left; padding: 4px 14px; border-bottom: 1px solid #f0f0f0; }
th   { background: #f6f6f6; font-weight: bold; }
tr:hover td { background: #fafafa; }
.meta { color: #666; font-size: 0.88em; margin: 4px 0 12px; }
.pager { margin: 10px 0; }
.pager a, .pager span { margin-right: 10px; }
input[type=text], input[type=number] {
  font-family: monospace; padding: 4px 8px; border: 1px solid #ccc;
  border-radius: 3px; }
button { padding: 4px 12px; cursor: pointer; }
.search-box { margin: 14px 0; }
.warn { color: #c00; }
code { background: #f4f4f4; padding: 1px 5px; border-radius: 3px; }
"""


def _page(title: str, body: str) -> str:
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        f"<style>{_CSS}</style>"
        f"</head><body>{body}</body></html>"
    )


def _search_form(value: str = "") -> str:
    escaped = value.replace('"', "&quot;")
    return (
        f"<form class='search-box' action='/search' method='get'>"
        f"<input type='text' name='id' value='{escaped}' "
        f"  placeholder='diaSourceId or alert_NNN.avro' style='width:300px'>"
        f" <button type='submit'>Search</button>"
        f"</form>"
    )


def _pager_html(base_url: str, page: int, total_pages: int) -> str:
    sep = "&amp;" if "?" in base_url else "?"
    parts: list[str] = []
    if page > 0:
        parts.append(f"<a href='{base_url}{sep}page={page - 1}'>← prev</a>")
    parts.append(f"<span>page {page + 1:,} of {total_pages:,}</span>")
    if page < total_pages - 1:
        parts.append(f"<a href='{base_url}{sep}page={page + 1}'>next →</a>")
    # jump-to-page form (1-indexed for humans)
    parts.append(
        f"<form style='display:inline' action='{base_url.split('?')[0]}' method='get'>"
        f"<input type='number' name='page' value='{page + 1}' min='1' max='{total_pages}' "
        f"  style='width:70px' title='1-indexed page number'>"
        f" <button type='submit'>Go</button>"
        f"</form>"
    )
    return f"<p class='pager'>{'  '.join(parts)}</p>"


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/index.txt")
def root_index():
    nights = _nights()
    content = "".join(d.name + "\n" for d in sorted(nights, key=lambda d: d.name))
    return Response(content=content, media_type="text/plain")


@app.get("/", response_class=HTMLResponse)
def root():
    nights = _nights()
    rows = "".join(
        f"<tr><td><a href='/{d.name}'>{d.name}</a></td>"
        f"<td>{_count_estimate(d)}</td></tr>"
        for d in nights
    )
    body = (
        f"<h2>LSST Alerts</h2>"
        f"{_search_form()}"
        f"<p class='meta'>{len(nights)} nights available</p>"
        f"<table><thead><tr><th>Night (MJD)</th><th>Alerts (est.)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<p class='meta' style='margin-top:16px'>"
        f"Full night list: <a href='/index.txt'>index.txt</a></p>"
    )
    return _page("LSST Alerts", body)


@app.get("/search", response_class=HTMLResponse)
def search(id: str = Query(default="")):
    raw = id.strip()

    # Accept "alert_123.avro", "alert_123", or just "123"
    candidate = raw.removeprefix("alert_").removesuffix(".avro")

    if not candidate.isdigit():
        body = (
            f"<h2>Search</h2>"
            f"{_search_form(raw)}"
            f"<p class='warn'>Please enter a numeric diaSourceId.</p>"
            f"<p><a href='/'>← all nights</a></p>"
        )
        return _page("Search — LSST Alerts", body)

    filename = f"alert_{candidate}.avro"
    found: list[tuple[str, str]] = []

    for night_dir in _nights():
        if (night_dir / filename).exists():
            found.append((night_dir.name, filename))
            break  # diaSourceId is unique across the survey

    if not found:
        body = (
            f"<h2>Search: {candidate}</h2>"
            f"{_search_form(raw)}"
            f"<p>No results for <code>{filename}</code>.</p>"
            f"<p><a href='/'>← all nights</a></p>"
        )
    else:
        mjd, fname = found[0]
        body = (
            f"<h2>Search: {candidate}</h2>"
            f"{_search_form(raw)}"
            f"<p>Found in night <strong>{mjd}</strong>:</p>"
            f"<table><thead><tr><th>Night (MJD)</th><th>File</th></tr></thead>"
            f"<tbody><tr>"
            f"<td><a href='/{mjd}'>{mjd}</a></td>"
            f"<td><a href='/{mjd}/{fname}' download>{fname}</a></td>"
            f"</tr></tbody></table>"
            f"<p><a href='/'>← all nights</a></p>"
        )

    return _page(f"Search: {candidate} — LSST Alerts", body)


@app.get("/{mjd}", response_class=HTMLResponse)
def night(request: Request, mjd: str, page: int = Query(default=1, ge=1)):
    mjd_dir = _safe_mjd_dir(mjd)
    # Convert 1-indexed page from URL to 0-indexed internally
    page0 = page - 1

    files, total_est = _read_page(mjd_dir, page0)
    total_pages = max(1, (total_est + PER_PAGE - 1) // PER_PAGE)

    start = page0 * PER_PAGE + 1
    end = start + len(files) - 1

    rows = "".join(
        f"<tr><td><a href='/{mjd}/{f}' download>{f}</a></td></tr>"
        for f in files
    )

    idx_size = (mjd_dir / "index.txt").stat().st_size if (mjd_dir / "index.txt").exists() else 0
    idx_mb = f"{idx_size / 1_048_576:.1f} MB" if idx_size else "not yet generated"

    pager = _pager_html(f"/{mjd}", page, total_pages)

    body = (
        f"<p><a href='/'>← all nights</a></p>"
        f"<h2>LSST Alerts — MJD {mjd}</h2>"
        f"{_search_form()}"
        f"<p class='meta'>~{total_est:,} alerts total &nbsp;·&nbsp; "
        f"showing {start:,}–{end:,}</p>"
        f"<p class='meta'>"
        f"<a href='/{mjd}/index.txt'>index.txt</a> ({idx_mb}) — "
        f"one filename per line, for scripted bulk access"
        f"</p>"
        f"<details style='margin-bottom:10px'><summary>Bulk download commands</summary>"
        f"<pre style='background:#f6f6f6;padding:10px;font-size:0.85em'>"
        f"# Download all alerts for this night:\n"
        f"wget -i &lt;(curl -s {_base_url(request)}{mjd}/index.txt \\\n"
        f"     | sed 's|^|{_base_url(request)}{mjd}/|') -P ./{mjd}/\n\n"
        f"# Or with parallel downloads (4 at a time):\n"
        f"cat index.txt | xargs -P4 -I{{}} wget -q {_base_url(request)}{mjd}/{{}}"
        f"</pre></details>"
        f"{pager}"
        f"<table><thead><tr><th>Alert file</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"{pager}"
    )
    return _page(f"MJD {mjd} — LSST Alerts", body)
