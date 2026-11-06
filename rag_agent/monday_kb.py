"""Sync the monday.com "Knowledge base" into the RAG index.

Orlanda keeps company documents in monday under a board / folder / workspace called
"Knowledge base". This module finds it (by name, or by explicit ids from config), downloads
the files attached to its items (file columns + updates, including subitems) and the monday
Docs it contains, extracts their text and upserts them into the same pgvector tables as the
local PDFs. ``retrieve_context`` therefore searches them for every employee, independent of
whether that employee connected their own monday account.

Incremental by design:
- monday assets are immutable (a re-upload gets a new asset id), so an already-indexed asset
  is never downloaded again; only new files are downloaded and embedded.
- monday Docs are re-read every sync, but only re-embedded when their text changed
  (content hash in ``rag_documents``).
- Files/docs that disappeared from monday are removed from the index, but only after a sync
  that listed every source without errors, so a monday outage can never wipe the index.

Runs in a background thread (first sync shortly after startup, then every
RAG_MONDAY_KB_SYNC_INTERVAL_MINUTES) and on demand from the admin panel. A Postgres advisory
lock keeps two processes from syncing at once. CLI: ``python -m rag_agent.monday_kb``.
"""
from __future__ import annotations

import html
import io
import json
import logging
import re
import threading
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from langchain_core.documents import Document

from rag_agent.config import (
    RAG_MONDAY_API_VERSION,
    RAG_MONDAY_KB_API_TOKEN,
    RAG_MONDAY_KB_BOARD_IDS,
    RAG_MONDAY_KB_ENABLED,
    RAG_MONDAY_KB_FOLDER_IDS,
    RAG_MONDAY_KB_MAX_FILE_MB,
    RAG_MONDAY_KB_NAMES,
    RAG_MONDAY_KB_SYNC_INTERVAL_MINUTES,
    RAG_MONDAY_KB_TOKEN_USER,
    RAG_MONDAY_KB_WORKSPACE_IDS,
)

log = logging.getLogger(__name__)

MONDAY_API_URL = "https://api.monday.com/v2"
DOC_TYPE_FILE = "monday_file"
DOC_TYPE_DOC = "monday_doc"

# Arbitrary constant key for pg_try_advisory_lock (one monday KB sync cluster-wide).
_ADVISORY_LOCK_KEY = 7_340_051_201
_STARTUP_DELAY_SECONDS = 20
_ITEMS_PAGE_LIMIT = 50
_MAX_PAGES = 200
# Guards against pathological files: cap extracted text and uncompressed zip members.
_MAX_TEXT_CHARS = 2_000_000
_MAX_ZIP_MEMBER_BYTES = 200 * 1024 * 1024
_XLSX_MAX_ROWS_PER_SHEET = 5000

TEXT_EXTS = {"txt", "md", "markdown", "csv", "tsv", "html", "htm"}
SUPPORTED_EXTS = {"pdf", "docx", "pptx", "xlsx"} | TEXT_EXTS
IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "heic", "svg"}
LEGACY_OFFICE_EXTS = {"doc", "xls", "ppt", "rtf", "odt", "ods", "odp"}


class MondayKbError(RuntimeError):
    """A sync-level failure with a message that is safe to show to admins."""


# ── monday GraphQL client ────────────────────────────────────────────────────
class MondayClient:
    def __init__(self, token: str, *, api_url: str = MONDAY_API_URL, api_version: str = "") -> None:
        headers = {"Authorization": token, "Content-Type": "application/json"}
        if api_version:
            headers["API-Version"] = api_version
        self._api_url = api_url
        self._api = httpx.Client(timeout=60.0, headers=headers)
        # Separate client WITHOUT the monday token: asset public_urls are presigned S3 links
        # (S3 rejects a second auth mechanism, and the token must not leak to the storage host).
        self._dl = httpx.Client(timeout=120.0, follow_redirects=True)

    def close(self) -> None:
        self._api.close()
        self._dl.close()

    def query(self, query: str, variables: dict | None = None, *, attempts: int = 4) -> dict:
        last_error = "unknown error"
        for attempt in range(attempts):
            try:
                resp = self._api.post(self._api_url, json={"query": query, "variables": variables or {}})
            except httpx.HTTPError as exc:
                last_error = f"сеть: {exc.__class__.__name__}"
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code in (401, 403):
                raise MondayKbError(
                    "monday отклонил токен (HTTP %d). Проверьте RAG_MONDAY_KB_API_TOKEN." % resp.status_code
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(_retry_after(resp.headers.get("Retry-After"), default=5 * (attempt + 1)))
                continue
            try:
                body = resp.json()
            except ValueError:
                raise MondayKbError(f"monday вернул не-JSON ответ (HTTP {resp.status_code})")
            errors = list(body.get("errors") or [])
            if body.get("error_message"):
                errors.append({"message": body.get("error_message"), "code": body.get("error_code")})
            if errors:
                msg = "; ".join(str(e.get("message") or e) for e in errors)[:600]
                if _is_rate_limited(errors, msg) and attempt < attempts - 1:
                    last_error = msg
                    time.sleep(_retry_seconds_from(errors, msg))
                    continue
                raise MondayKbError(f"monday API: {msg}")
            return body.get("data") or {}
        raise MondayKbError(f"monday API недоступен: {last_error}")

    def download(self, url: str, *, max_bytes: int) -> bytes:
        with self._dl.stream("GET", url) as resp:
            resp.raise_for_status()
            buf = bytearray()
            for chunk in resp.iter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise MondayKbError("файл больше лимита")
            return bytes(buf)


def _retry_after(value: str | None, *, default: float) -> float:
    try:
        return max(1.0, min(60.0, float(value))) if value else default
    except ValueError:
        return default


def _is_rate_limited(errors: list[dict], msg: str) -> bool:
    codes = " ".join(
        str(e.get("code") or (e.get("extensions") or {}).get("code") or "") for e in errors
    ).upper()
    return (
        "COMPLEXITY" in codes
        or "RATE_LIMIT" in codes
        or "complexity budget" in msg.lower()
        or "rate limit" in msg.lower()
    )


def _retry_seconds_from(errors: list[dict], msg: str) -> float:
    for e in errors:
        secs = (e.get("extensions") or {}).get("retry_in_seconds")
        if isinstance(secs, (int, float)):
            return max(1.0, min(60.0, float(secs)))
    m = re.search(r"(\d+)\s*seconds?", msg)
    return max(1.0, min(60.0, float(m.group(1)))) if m else 20.0


# ── Discovery ────────────────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    """Casefold and drop spaces/punctuation/emoji: '📚 Knowledge-Base ' -> 'knowledgebase'."""
    return re.sub(r"[\W_]+", "", (name or "").casefold())


@dataclass(frozen=True)
class KbRoot:
    kind: str  # "board" | "folder" | "workspace"
    id: str
    name: str


@dataclass
class _Board:
    id: str
    name: str
    type: str  # "board" | "document" | "sub_items_board" | "custom_object"


_Q_WORKSPACES = "query ($page: Int!) { workspaces(limit: 100, page: $page) { id name } }"
_Q_BOARDS = (
    "query ($page: Int!) { boards(limit: 100, page: $page, state: active) { id name type } }"
)
_Q_BOARDS_IN_WS = (
    "query ($ws: [ID], $page: Int!) { boards(workspace_ids: $ws, limit: 100, page: $page, "
    "state: active) { id name type } }"
)
_Q_BOARDS_BY_ID = "query ($ids: [ID!]) { boards(ids: $ids) { id name type } }"
_Q_FOLDERS_IN_WS = (
    "query ($ws: [ID], $page: Int!) { folders(workspace_ids: $ws, limit: 100, page: $page) "
    "{ id name children { id name } sub_folders { id name } } }"
)
_Q_FOLDERS_BY_ID = (
    "query ($ids: [ID!]) { folders(ids: $ids) { id name children { id name } sub_folders { id name } } }"
)
_Q_DOCS_IN_WS = (
    "query ($ws: [ID], $page: Int!) { docs(workspace_ids: $ws, limit: 100, page: $page) "
    "{ id object_id name url } }"
)
_Q_DOCS_BY_OBJECT = "query ($ids: [ID!]) { docs(object_ids: $ids) { id object_id name url } }"
_Q_DOC_MARKDOWN = (
    "query ($id: ID!) { export_markdown_from_doc(docId: $id) { success markdown error } }"
)
_Q_DOC_BLOCKS = (
    "query ($ids: [ID!], $page: Int!) { docs(ids: $ids) { blocks(limit: 100, page: $page) "
    "{ id type content } } }"
)
_ASSET_FIELDS = "id name file_extension file_size public_url"
_ITEM_FIELDS = (
    "id name url assets { %s } column_values { id type value } "
    "subitems { id name url assets { %s } column_values { id type value } }"
) % (_ASSET_FIELDS, _ASSET_FIELDS)
# Column types whose value can reference monday Docs / external links.
_LINK_COLUMN_TYPES = {"file", "doc", "direct_doc"}
_Q_ITEMS_FIRST = (
    "query ($ids: [ID!], $limit: Int!) { boards(ids: $ids) { items_page(limit: $limit) "
    "{ cursor items { %s } } } }" % _ITEM_FIELDS
)
_Q_ITEMS_NEXT = (
    "query ($cursor: String!, $limit: Int!) { next_items_page(cursor: $cursor, limit: $limit) "
    "{ cursor items { %s } } }" % _ITEM_FIELDS
)


def _paged(client: MondayClient, query: str, key: str, variables: dict | None = None) -> list[dict]:
    out: list[dict] = []
    for page in range(1, _MAX_PAGES + 1):
        data = client.query(query, {**(variables or {}), "page": page})
        rows = [r for r in (data.get(key) or []) if r]
        out.extend(rows)
        if len(rows) < 100:
            break
    return out


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def discover_roots(client: MondayClient, names: list[str]) -> list[KbRoot]:
    """Find boards, folders and workspaces whose name matches one of ``names``."""
    wanted = {normalize_name(n) for n in names if normalize_name(n)}
    roots: list[KbRoot] = []

    workspaces = _paged(client, _Q_WORKSPACES, "workspaces")
    for ws in workspaces:
        if normalize_name(ws.get("name", "")) in wanted:
            roots.append(KbRoot("workspace", str(ws["id"]), ws.get("name") or ""))

    folders: list[dict] = []
    ws_ids = [str(ws["id"]) for ws in workspaces if ws.get("id")]
    for batch in _chunks(ws_ids, 25):
        folders.extend(_paged(client, _Q_FOLDERS_IN_WS, "folders", {"ws": batch}))
    try:  # null = the account's main workspace, which older accounts don't list in `workspaces`
        folders.extend(_paged(client, _Q_FOLDERS_IN_WS, "folders", {"ws": [None]}))
    except MondayKbError as exc:
        log.info("monday KB: main-workspace folder lookup skipped: %s", exc)
    for folder in folders:
        if normalize_name(folder.get("name", "")) in wanted:
            roots.append(KbRoot("folder", str(folder["id"]), folder.get("name") or ""))

    for board in _paged(client, _Q_BOARDS, "boards"):
        if board.get("type") == "sub_items_board":
            continue
        if normalize_name(board.get("name", "")) in wanted:
            roots.append(KbRoot("board", str(board["id"]), board.get("name") or ""))

    # Dedupe (the same object can match through several queries).
    seen: set[tuple[str, str]] = set()
    unique: list[KbRoot] = []
    for r in roots:
        if (r.kind, r.id) not in seen:
            seen.add((r.kind, r.id))
            unique.append(r)
    return unique


def explicit_roots() -> list[KbRoot]:
    return (
        [KbRoot("board", i, "") for i in RAG_MONDAY_KB_BOARD_IDS]
        + [KbRoot("folder", i, "") for i in RAG_MONDAY_KB_FOLDER_IDS]
        + [KbRoot("workspace", i, "") for i in RAG_MONDAY_KB_WORKSPACE_IDS]
    )


@dataclass
class _Expansion:
    boards: dict[str, _Board]
    doc_object_ids: set[str]
    # Docs listed directly (workspace roots): object_id -> doc row.
    docs: dict[str, dict]
    root_names: dict[tuple[str, str], str]


def expand_roots(client: MondayClient, roots: list[KbRoot]) -> _Expansion:
    """Resolve roots into concrete boards and doc object ids (recursing into sub-folders)."""
    board_ids: dict[str, str] = {}  # id -> root display name (for labels)
    docs: dict[str, dict] = {}
    root_names: dict[tuple[str, str], str] = {}

    pending_folders: list[tuple[str, str]] = []  # (folder id, root name)
    for r in roots:
        if r.kind == "board":
            board_ids.setdefault(r.id, r.name)
        elif r.kind == "folder":
            pending_folders.append((r.id, r.name))
        elif r.kind == "workspace":
            for b in _paged(client, _Q_BOARDS_IN_WS, "boards", {"ws": [r.id]}):
                if b.get("type") != "sub_items_board":
                    board_ids.setdefault(str(b["id"]), r.name)
            for d in _paged(client, _Q_DOCS_IN_WS, "docs", {"ws": [r.id]}):
                if d.get("object_id"):
                    docs.setdefault(str(d["object_id"]), d)

    visited: set[str] = set()
    while pending_folders:
        batch, pending_folders = pending_folders[:50], pending_folders[50:]
        names = {fid: name for fid, name in batch if fid not in visited}
        if not names:
            continue
        visited.update(names)
        data = client.query(_Q_FOLDERS_BY_ID, {"ids": list(names)})
        for folder in data.get("folders") or []:
            fid = str(folder.get("id"))
            root_name = names.get(fid) or folder.get("name") or ""
            root_names[("folder", fid)] = folder.get("name") or root_name
            for child in folder.get("children") or []:
                if child and child.get("id"):
                    board_ids.setdefault(str(child["id"]), folder.get("name") or root_name)
            for sub in folder.get("sub_folders") or []:
                if sub and sub.get("id") and str(sub["id"]) not in visited:
                    pending_folders.append((str(sub["id"]), root_name))

    boards: dict[str, _Board] = {}
    doc_object_ids: set[str] = set(docs)
    for batch in _chunks(list(board_ids), 50):
        data = client.query(_Q_BOARDS_BY_ID, {"ids": batch})
        returned = {str(b.get("id")) for b in data.get("boards") or [] if b}
        # Folder children that `boards` doesn't return may be monday Docs; resolved later.
        doc_object_ids.update(bid for bid in batch if bid not in returned)
        for b in data.get("boards") or []:
            bid = str(b.get("id"))
            btype = str(b.get("type") or "board")
            name = b.get("name") or board_ids.get(bid) or bid
            root_names[("board", bid)] = name
            if btype == "document":
                doc_object_ids.add(bid)
            elif btype in ("board", "custom_object"):
                boards[bid] = _Board(bid, name, btype)
    return _Expansion(boards=boards, doc_object_ids=doc_object_ids, docs=docs, root_names=root_names)


# ── Board items → assets & linked docs ───────────────────────────────────────
@dataclass
class AssetRef:
    asset_id: str
    name: str
    ext: str
    size: int
    public_url: str
    board_name: str
    item_name: str
    item_url: str


def _asset_ext(asset: dict) -> str:
    ext = str(asset.get("file_extension") or "").strip().lstrip(".").lower()
    if not ext and "." in str(asset.get("name") or ""):
        ext = str(asset["name"]).rsplit(".", 1)[-1].lower()
    return ext


def linked_from_column_value(raw) -> tuple[set[str], list[str]]:
    """Parse a file/doc column value: (monday doc object ids, names of external links)."""
    doc_ids: set[str] = set()
    external: list[str] = []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return doc_ids, external
    files = data.get("files") if isinstance(data, dict) else None
    for f in files or []:
        if not isinstance(f, dict):
            continue
        ftype = str(f.get("fileType") or "").upper()
        obj = f.get("objectId") or f.get("object_id")
        if ftype == "MONDAY_DOC" or (obj and ftype not in ("ASSET", "")):
            if obj:
                doc_ids.add(str(obj))
        elif ftype not in ("ASSET", "") and f.get("linkToFile"):
            external.append(str(f.get("name") or f.get("linkToFile")))
    return doc_ids, external


def collect_board_content(
    client: MondayClient, board: _Board
) -> tuple[list[AssetRef], set[str], list[dict]]:
    """Return (assets, linked doc object ids, skipped external links) for one board."""
    assets: list[AssetRef] = []
    doc_ids: set[str] = set()
    skipped: list[dict] = []

    def take(item: dict) -> None:
        for a in item.get("assets") or []:
            if not a or not a.get("id"):
                continue
            assets.append(
                AssetRef(
                    asset_id=str(a["id"]),
                    name=str(a.get("name") or f"file-{a['id']}"),
                    ext=_asset_ext(a),
                    size=int(a.get("file_size") or 0),
                    public_url=str(a.get("public_url") or ""),
                    board_name=board.name,
                    item_name=str(item.get("name") or ""),
                    item_url=str(item.get("url") or ""),
                )
            )
        for cv in item.get("column_values") or []:
            if str((cv or {}).get("type") or "") not in _LINK_COLUMN_TYPES:
                continue
            ids, external = linked_from_column_value((cv or {}).get("value"))
            doc_ids.update(ids)
            for name in external:
                skipped.append({"name": name, "where": f"{board.name} › {item.get('name')}",
                                "reason": "внешняя ссылка (Google Drive / Dropbox и т.п.) — не скачивается"})

    data = client.query(_Q_ITEMS_FIRST, {"ids": [board.id], "limit": _ITEMS_PAGE_LIMIT})
    page = ((data.get("boards") or [{}])[0] or {}).get("items_page") or {}
    for _ in range(_MAX_PAGES):
        for item in page.get("items") or []:
            if not item:
                continue
            take(item)
            for sub in item.get("subitems") or []:
                if sub:
                    take(sub)
        cursor = page.get("cursor")
        if not cursor:
            break
        data = client.query(_Q_ITEMS_NEXT, {"cursor": cursor, "limit": _ITEMS_PAGE_LIMIT})
        page = data.get("next_items_page") or {}
    return assets, doc_ids, skipped


# ── monday Docs → text ───────────────────────────────────────────────────────
_BLOCK_PREFIX = {
    "large_title": "# ",
    "medium_title": "## ",
    "small_title": "### ",
    "bulleted_list": "- ",
    "numbered_list": "1. ",
    "check_list": "- [ ] ",
    "quote": "> ",
}


def block_text(block: dict) -> str:
    """Text of one monday doc block (content is JSON with a Quill-style deltaFormat)."""
    raw = block.get("content")
    try:
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return ""
    delta = data.get("deltaFormat") if isinstance(data, dict) else None
    if not isinstance(delta, list):
        return ""
    text = "".join(
        op["insert"] for op in delta if isinstance(op, dict) and isinstance(op.get("insert"), str)
    ).strip()
    if not text:
        return ""
    return _BLOCK_PREFIX.get(str(block.get("type") or ""), "") + text


class _DocReader:
    def __init__(self, client: MondayClient) -> None:
        self._client = client
        self._markdown_supported = True

    def text(self, doc_id: str) -> str:
        if self._markdown_supported:
            try:
                data = self._client.query(_Q_DOC_MARKDOWN, {"id": doc_id})
                res = data.get("export_markdown_from_doc") or {}
                if res.get("success") and (res.get("markdown") or "").strip():
                    return str(res["markdown"]).strip()
            except MondayKbError as exc:
                # Older API versions lack the field: fall back to blocks for the rest of the run.
                if "export_markdown_from_doc" in str(exc) or "Cannot query field" in str(exc):
                    self._markdown_supported = False
                else:
                    raise
        lines: list[str] = []
        for page in range(1, _MAX_PAGES + 1):
            data = self._client.query(_Q_DOC_BLOCKS, {"ids": [doc_id], "page": page})
            blocks = ((data.get("docs") or [{}])[0] or {}).get("blocks") or []
            lines.extend(t for t in (block_text(b) for b in blocks if b) if t)
            if len(blocks) < 100:
                break
        return "\n".join(lines).strip()


# ── File text extraction (stdlib + pypdf; no extra dependencies) ─────────────
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _zip_read(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    if info.file_size > _MAX_ZIP_MEMBER_BYTES:
        raise MondayKbError("слишком большой файл внутри архива")
    return zf.read(info)


def _docx_paragraph(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        if node.tag == f"{_W}t":
            parts.append(node.text or "")
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts).strip()


def _extract_docx(data: bytes) -> list[tuple[int, str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        root = ET.fromstring(_zip_read(zf, "word/document.xml"))
    body = root.find(f"{_W}body")
    lines: list[str] = []
    for el in list(body) if body is not None else []:
        if el.tag == f"{_W}tbl":
            for tr in el.iter(f"{_W}tr"):
                cells = [
                    " ".join(t for t in (_docx_paragraph(p) for p in tc.iter(f"{_W}p")) if t)
                    for tc in tr.findall(f"{_W}tc")
                ]
                row = " | ".join(c for c in cells if c)
                if row:
                    lines.append(row)
        else:
            for p in ([el] if el.tag == f"{_W}p" else el.iter(f"{_W}p")):
                t = _docx_paragraph(p)
                if t:
                    lines.append(t)
    return [(0, "\n".join(lines))]


def _extract_pptx(data: bytes) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        slides = sorted(
            (n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)),
        )
        for idx, name in enumerate(slides):
            root = ET.fromstring(_zip_read(zf, name))
            paras = []
            for p in root.iter(f"{_A}p"):
                t = "".join(r.text or "" for r in p.iter(f"{_A}t")).strip()
                if t:
                    paras.append(t)
            if paras:
                pages.append((idx, "\n".join(paras)))
    return pages


def _extract_xlsx(data: bytes) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            for si in ET.fromstring(_zip_read(zf, "xl/sharedStrings.xml")).findall(f"{_S}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{_S}t")))
        sheet_names: list[str] = []
        if "xl/workbook.xml" in names:
            sheet_names = [
                s.get("name") or "" for s in ET.fromstring(_zip_read(zf, "xl/workbook.xml")).iter(f"{_S}sheet")
            ]
        sheets = sorted(
            (n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)),
        )
        for idx, name in enumerate(sheets):
            root = ET.fromstring(_zip_read(zf, name))
            lines: list[str] = []
            for row in root.iter(f"{_S}row"):
                cells: list[str] = []
                for c in row.findall(f"{_S}c"):
                    kind = c.get("t")
                    v = c.find(f"{_S}v")
                    if kind == "s" and v is not None and (v.text or "").isdigit():
                        i = int(v.text)
                        val = shared[i] if i < len(shared) else ""
                    elif kind == "inlineStr":
                        val = "".join(t.text or "" for t in c.iter(f"{_S}t"))
                    else:
                        val = (v.text or "") if v is not None else ""
                    val = val.strip()
                    if val:
                        cells.append(val)
                if cells:
                    lines.append(" | ".join(cells))
                if len(lines) >= _XLSX_MAX_ROWS_PER_SHEET:
                    break
            if lines:
                title = sheet_names[idx] if idx < len(sheet_names) and sheet_names[idx] else f"Лист {idx + 1}"
                pages.append((idx, f"Лист: {title}\n" + "\n".join(lines)))
    return pages


def _extract_pdf(data: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    from rag_agent.indexing import _reflow_pdf_text

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise MondayKbError("PDF защищён паролем") from exc
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one broken page must not drop the whole file
            continue
        t = _reflow_pdf_text(t)
        if t.strip():
            pages.append((idx, t))
    return pages


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _extract_plain(data: bytes, ext: str) -> list[tuple[int, str]]:
    text = _decode_text(data)
    if ext in ("html", "htm"):
        text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
        text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6])>", "\n", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        text = re.sub(r"[ \t]+", " ", text)
    return [(0, text.strip())]


def extract_pages(data: bytes, ext: str) -> list[tuple[int, str]]:
    """Return [(page_index, text), ...] for a supported file type (raises MondayKbError)."""
    ext = (ext or "").lower()
    try:
        if ext == "pdf":
            pages = _extract_pdf(data)
        elif ext == "docx":
            pages = _extract_docx(data)
        elif ext == "pptx":
            pages = _extract_pptx(data)
        elif ext == "xlsx":
            pages = _extract_xlsx(data)
        elif ext in TEXT_EXTS:
            pages = _extract_plain(data, ext)
        else:
            raise MondayKbError(f"формат .{ext} не поддерживается")
    except MondayKbError:
        raise
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as exc:
        raise MondayKbError(f"файл повреждён или не является .{ext}") from exc
    except Exception as exc:  # noqa: BLE001 - parser bugs on odd files must not stop the sync
        raise MondayKbError(f"не удалось прочитать файл ({exc.__class__.__name__})") from exc

    out: list[tuple[int, str]] = []
    total = 0
    for idx, text in pages:
        text = (text or "").strip()
        if not text:
            continue
        if total + len(text) > _MAX_TEXT_CHARS:
            text = text[: max(0, _MAX_TEXT_CHARS - total)]
        if text:
            out.append((idx, text))
            total += len(text)
        if total >= _MAX_TEXT_CHARS:
            break
    return out


def skip_reason(ext: str, size: int, max_bytes: int) -> str | None:
    """Why a file is not indexed without even downloading it (None = try it)."""
    if ext in IMAGE_EXTS:
        return "изображение — текст не извлекается (нет OCR)"
    if ext in LEGACY_OFFICE_EXTS:
        return f"старый формат .{ext} — сохраните как .docx/.xlsx/.pptx или PDF"
    if ext not in SUPPORTED_EXTS:
        return f"формат .{ext or '?'} не поддерживается"
    if size and size > max_bytes:
        return f"больше {max_bytes // (1024 * 1024)} МБ"
    return None


# ── Index documents ──────────────────────────────────────────────────────────
def asset_doc_id(asset_id: str) -> str:
    return f"monday_asset:{asset_id}"


def mdoc_doc_id(doc_id: str) -> str:
    return f"monday_doc:{doc_id}"


def asset_label(asset: AssetRef) -> str:
    return f"{asset.name} · monday/{asset.board_name}"[:500]


def build_documents(
    pages: list[tuple[int, str]], *, label: str, doc_type: str, header: str, meta: dict
) -> list[Document]:
    docs: list[Document] = []
    for n, (page, text) in enumerate(pages):
        content = f"{header}\n\n{text}" if (n == 0 and header) else text
        docs.append(
            Document(
                page_content=content,
                metadata={"source_file": label, "page": int(page), "type": doc_type, **meta},
            )
        )
    return docs


# ── Token resolution ─────────────────────────────────────────────────────────
def resolve_token() -> tuple[str | None, str]:
    """Return (token, human description of where it came from)."""
    if RAG_MONDAY_KB_API_TOKEN:
        return RAG_MONDAY_KB_API_TOKEN, "RAG_MONDAY_KB_API_TOKEN"
    try:
        from sqlalchemy import select

        from rag_agent.db.models import User, UserMondayToken
        from rag_agent.db.session import get_session_factory
        from rag_agent.monday_oauth import get_access_token

        candidates: list[str] = []
        if RAG_MONDAY_KB_TOKEN_USER:
            candidates.append(RAG_MONDAY_KB_TOKEN_USER)
        with get_session_factory()() as db:
            rows = db.execute(
                select(User.username)
                .join(UserMondayToken, UserMondayToken.user_id == User.id)
                .where(User.role == "admin", User.is_active.is_(True))
                .order_by(User.username.asc())
            ).all()
        candidates.extend(u for (u,) in rows if u not in candidates)
        for username in candidates:
            token = get_access_token(username)
            if token:
                return token, f"monday-подключение администратора {username}"
    except Exception as exc:  # noqa: BLE001 - status must render even if the DB is unhappy
        log.warning("monday KB: token lookup failed: %s", exc)
    return None, ""


# ── Sync ─────────────────────────────────────────────────────────────────────
_run_lock = threading.Lock()
_status_lock = threading.Lock()
_status: dict = {
    "running": False,
    "trigger": None,
    "last_started_at": None,
    "last_finished_at": None,
    "last_ok": None,
    "last_error": None,
    "token_source": None,
    "roots": [],
    "stats": {},
    "skipped": [],
    "progress": None,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_status(**kw) -> None:
    with _status_lock:
        _status.update(kw)


def _existing_monday_records() -> dict[str, str]:
    from sqlalchemy import select

    from rag_agent.db.models import DocumentIndexRecord
    from rag_agent.db.session import get_session_factory
    from rag_agent.indexing import EXTERNAL_DOC_TYPES

    with get_session_factory()() as db:
        rows = db.execute(
            select(DocumentIndexRecord.doc_id, DocumentIndexRecord.source_name).where(
                DocumentIndexRecord.doc_type.in_(EXTERNAL_DOC_TYPES)
            )
        ).all()
    return {doc_id: name for doc_id, name in rows}


def _sync_with_client(client: MondayClient) -> dict:
    from rag_agent.indexing import delete_document, upsert_external_document

    stats = {"files_indexed": 0, "files_unchanged": 0, "docs_indexed": 0, "docs_unchanged": 0,
             "removed": 0, "skipped": 0, "boards": 0, "docs": 0}
    skipped: list[dict] = []
    clean_listing = True  # deletions only after a complete, error-free listing

    roots = explicit_roots() or discover_roots(client, RAG_MONDAY_KB_NAMES)
    if not roots:
        raise MondayKbError(
            "В monday не найдено доски, папки или воркспейса с названием "
            + " / ".join(f"«{n}»" for n in RAG_MONDAY_KB_NAMES)
            + ". Проверьте название или задайте RAG_MONDAY_KB_BOARD_IDS / RAG_MONDAY_KB_FOLDER_IDS, "
            "и что у владельца токена есть доступ."
        )
    expansion = expand_roots(client, roots)
    _set_status(roots=[
        {"kind": r.kind, "id": r.id, "name": r.name or expansion.root_names.get((r.kind, r.id), "")}
        for r in roots
    ])

    assets: dict[str, AssetRef] = {}
    doc_object_ids = set(expansion.doc_object_ids)
    for board in expansion.boards.values():
        try:
            board_assets, linked_docs, board_skipped = collect_board_content(client, board)
        except MondayKbError as exc:
            clean_listing = False
            reason = f"ошибка чтения: {exc}"
            if "nauthorized" in str(exc) and not RAG_MONDAY_KB_API_TOKEN:
                reason += " — задайте RAG_MONDAY_KB_API_TOKEN (персональный API-токен monday)"
            skipped.append({"name": board.name, "where": "доска", "reason": reason})
            continue
        stats["boards"] += 1
        for a in board_assets:
            assets.setdefault(a.asset_id, a)
        doc_object_ids |= linked_docs
        skipped.extend(board_skipped)

    # Resolve doc object ids → doc rows (id, name, url).
    docs = dict(expansion.docs)
    missing = [oid for oid in doc_object_ids if oid not in docs]
    for batch in _chunks(missing, 50):
        try:
            for d in client.query(_Q_DOCS_BY_OBJECT, {"ids": batch}).get("docs") or []:
                if d and d.get("object_id"):
                    docs[str(d["object_id"])] = d
        except MondayKbError as exc:
            clean_listing = False
            skipped.append({"name": "monday Docs", "where": "", "reason": f"ошибка чтения: {exc}"})

    existing = _existing_monday_records()
    seen: set[str] = set()
    changed = False
    max_bytes = RAG_MONDAY_KB_MAX_FILE_MB * 1024 * 1024
    total = len(assets) + len(docs)
    done = 0

    for a in assets.values():
        done += 1
        _set_status(progress={"done": done, "total": total, "current": a.name})
        doc_id = asset_doc_id(a.asset_id)
        label = asset_label(a)
        where = f"{a.board_name} › {a.item_name}"
        reason = skip_reason(a.ext, a.size, max_bytes)
        if reason:
            stats["skipped"] += 1
            skipped.append({"name": a.name, "where": where, "reason": reason})
            continue
        seen.add(doc_id)
        if existing.get(doc_id) == label:
            stats["files_unchanged"] += 1
            continue
        try:
            if not a.public_url:
                raise MondayKbError("monday не выдал ссылку на файл")
            pages = extract_pages(client.download(a.public_url, max_bytes=max_bytes), a.ext)
            if not pages:
                raise MondayKbError("в файле нет текста (скан без OCR?)")
            documents = build_documents(
                pages,
                label=label,
                doc_type=DOC_TYPE_FILE,
                header=f"Документ: {a.name}\nmonday: {a.board_name} › {a.item_name}",
                meta={"monday_board": a.board_name, "monday_item": a.item_name,
                      "monday_file": a.name, "url": a.item_url},
            )
            upsert_external_document(doc_id=doc_id, doc_type=DOC_TYPE_FILE,
                                     source_ref=f"monday:asset:{a.asset_id}", source_name=label,
                                     documents=documents)
            stats["files_indexed"] += 1
            changed = True
        except (MondayKbError, httpx.HTTPError) as exc:
            stats["skipped"] += 1
            skipped.append({"name": a.name, "where": where, "reason": str(exc) or exc.__class__.__name__})
        except Exception as exc:  # noqa: BLE001 - e.g. embedding API hiccup: retry next sync
            log.exception("monday KB: failed to index asset %s", a.asset_id)
            stats["skipped"] += 1
            skipped.append({"name": a.name, "where": where,
                            "reason": f"ошибка индексации ({exc.__class__.__name__}), повторим при следующей синхронизации"})

    reader = _DocReader(client)
    for d in docs.values():
        done += 1
        name = str(d.get("name") or "monday Doc")
        _set_status(progress={"done": done, "total": total, "current": name})
        doc_id = mdoc_doc_id(str(d.get("id")))
        seen.add(doc_id)
        stats["docs"] += 1
        try:
            text = reader.text(str(d.get("id")))
        except MondayKbError as exc:
            clean_listing = False
            skipped.append({"name": name, "where": "monday Doc", "reason": f"ошибка чтения: {exc}"})
            continue
        if not text:
            seen.discard(doc_id)
            skipped.append({"name": name, "where": "monday Doc", "reason": "документ пустой"})
            continue
        label = f"{name} · monday Doc"[:500]
        try:
            result = upsert_external_document(
                doc_id=doc_id, doc_type=DOC_TYPE_DOC, source_ref=f"monday:doc:{d.get('id')}",
                source_name=label,
                documents=build_documents([(0, text)], label=label, doc_type=DOC_TYPE_DOC,
                                          header=f"Документ: {name} (monday Doc)",
                                          meta={"monday_file": name, "url": str(d.get("url") or "")}),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("monday KB: failed to index doc %s", d.get("id"))
            skipped.append({"name": name, "where": "monday Doc",
                            "reason": f"ошибка индексации ({exc.__class__.__name__})"})
            continue
        if result.get("status") == "unchanged":
            stats["docs_unchanged"] += 1
        else:
            stats["docs_indexed"] += 1
            changed = True

    if clean_listing:
        for doc_id in existing:
            if doc_id not in seen:
                delete_document(doc_id)
                stats["removed"] += 1
                changed = True

    if changed:
        from rag_agent.rag_tool import invalidate_vector_store

        invalidate_vector_store()
    stats["complete"] = clean_listing
    return {"stats": stats, "skipped": skipped[:300]}


def run_sync(trigger: str = "manual") -> dict:
    """Run one sync now (blocking). Returns the resulting status snapshot."""
    if not _run_lock.acquire(blocking=False):
        return {**get_status(include_files=False), "busy": True}
    try:
        _set_status(running=True, trigger=trigger, last_started_at=_now(), progress=None)
        token, source = resolve_token()
        _set_status(token_source=source or None)
        if not token:
            raise MondayKbError(
                "Нет токена monday: задайте RAG_MONDAY_KB_API_TOKEN или подключите monday "
                "в настройках под учётной записью администратора."
            )
        from sqlalchemy import text

        from rag_agent.db.session import get_engine

        with get_engine().connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            if not conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _ADVISORY_LOCK_KEY}).scalar():
                raise MondayKbError("синхронизация уже выполняется в другом процессе")
            client = MondayClient(token, api_version=RAG_MONDAY_API_VERSION)
            try:
                result = _sync_with_client(client)
            finally:
                client.close()
                try:
                    conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _ADVISORY_LOCK_KEY})
                except Exception:  # noqa: BLE001 - lock dies with the connection anyway
                    pass
        _set_status(last_ok=True, last_error=None, stats=result["stats"], skipped=result["skipped"])
        log.info("monday KB sync (%s) done: %s", trigger, result["stats"])
    except MondayKbError as exc:
        _set_status(last_ok=False, last_error=str(exc))
        log.warning("monday KB sync (%s) failed: %s", trigger, exc)
    except Exception as exc:  # noqa: BLE001 - never kill the background thread
        _set_status(last_ok=False, last_error=f"внутренняя ошибка: {exc.__class__.__name__}: {exc}"[:500])
        log.exception("monday KB sync (%s) crashed", trigger)
    finally:
        _set_status(running=False, last_finished_at=_now(), progress=None)
        _run_lock.release()
    return get_status(include_files=False)


def trigger_sync_async(trigger: str = "manual") -> bool:
    """Start a sync in a background thread. False if one is already running or disabled."""
    if not RAG_MONDAY_KB_ENABLED or _run_lock.locked():
        return False
    threading.Thread(target=run_sync, args=(trigger,), name="monday-kb-sync", daemon=True).start()
    return True


_stop_event = threading.Event()
_scheduler: threading.Thread | None = None


def _scheduler_loop() -> None:
    if _stop_event.wait(_STARTUP_DELAY_SECONDS):
        return
    trigger = "startup"
    while not _stop_event.is_set():
        run_sync(trigger)
        trigger = "scheduled"
        if RAG_MONDAY_KB_SYNC_INTERVAL_MINUTES <= 0:
            return
        if _stop_event.wait(RAG_MONDAY_KB_SYNC_INTERVAL_MINUTES * 60):
            return


def start_background_sync() -> None:
    """Start the periodic sync thread (idempotent). No-op when the feature is disabled."""
    global _scheduler
    if not RAG_MONDAY_KB_ENABLED or (_scheduler and _scheduler.is_alive()):
        return
    _stop_event.clear()
    _scheduler = threading.Thread(target=_scheduler_loop, name="monday-kb-scheduler", daemon=True)
    _scheduler.start()


def stop_background_sync() -> None:
    _stop_event.set()


def list_synced_documents() -> list[dict]:
    """Indexed monday files/docs for the admin panel."""
    from sqlalchemy import select

    from rag_agent.db.models import DocumentChunk, DocumentIndexRecord
    from rag_agent.db.session import get_session_factory
    from rag_agent.indexing import EXTERNAL_DOC_TYPES

    with get_session_factory()() as db:
        rows = db.execute(
            select(DocumentIndexRecord)
            .where(DocumentIndexRecord.doc_type.in_(EXTERNAL_DOC_TYPES))
            .order_by(DocumentIndexRecord.source_name.asc())
        ).scalars().all()
        ids = [r.doc_id for r in rows]
        metas: dict[str, dict] = {}
        if ids:
            for doc_id, meta in db.execute(
                select(DocumentChunk.doc_id, DocumentChunk.metadata_json).where(
                    DocumentChunk.doc_id.in_(ids), DocumentChunk.chunk_no == 0
                )
            ).all():
                metas[doc_id] = meta or {}
    out = []
    for r in rows:
        meta = metas.get(r.doc_id, {})
        out.append({
            "doc_id": r.doc_id,
            "type": r.doc_type,
            "name": meta.get("monday_file") or r.source_name,
            "board": meta.get("monday_board") or "",
            "item": meta.get("monday_item") or "",
            "url": meta.get("url") or "",
            "chunks": r.chunk_count,
            "updated_at": r.updated_at.isoformat() if r.updated_at else "",
        })
    return out


def get_status(*, include_files: bool = True) -> dict:
    with _status_lock:
        snap = json.loads(json.dumps(_status))
    if RAG_MONDAY_KB_API_TOKEN:
        source = "RAG_MONDAY_KB_API_TOKEN"
    elif include_files:
        source = resolve_token()[1]
    else:
        source = snap.get("token_source") or ""
    snap.update({
        "enabled": RAG_MONDAY_KB_ENABLED,
        "configured": bool(source),
        "token_source": source or None,
        "names": RAG_MONDAY_KB_NAMES,
        "interval_minutes": RAG_MONDAY_KB_SYNC_INTERVAL_MINUTES,
    })
    if include_files:
        try:
            snap["files"] = list_synced_documents()
        except Exception as exc:  # noqa: BLE001
            log.warning("monday KB: listing synced documents failed: %s", exc)
            snap["files"] = []
    return snap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_sync("cli")
    print(json.dumps({k: result.get(k) for k in ("last_ok", "last_error", "roots", "stats", "skipped")},
                     ensure_ascii=False, indent=2))
