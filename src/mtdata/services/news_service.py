"""MT5 local news database parser."""

import logging
import mmap
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.runtime_metadata import build_runtime_timezone_meta
from ..utils.mt5 import _mt5_epoch_to_utc
from ..utils.time import _resolve_client_tz, _use_client_tz, format_relative_time
from .news_text import normalize_news_text

logger = logging.getLogger(__name__)


def _finalize_news_payload(payload: Dict[str, Any], use_client_tz: bool) -> Dict[str, Any]:
    timezone_meta_input: Dict[str, Any] = dict(payload)
    if not use_client_tz:
        timezone_meta_input["timezone"] = "UTC"
        payload["timezone"] = "UTC"
    payload["meta"] = {
        "runtime": {
            "timezone": build_runtime_timezone_meta(
                timezone_meta_input,
                include_local=False,
                include_now=False,
            )
        }
    }
    return payload


class MT5NewsRecord:
    """Represents a single MT5 news item."""
    
    def __init__(
        self,
        timestamp: datetime,
        subject: str,
        category: str,
        source: str,
        flags: int = 0,
        priority: int = 0
    ):
        self.timestamp = timestamp
        self.subject = subject
        self.category = category
        self.source = source
        self.flags = flags
        self.priority = priority
    
    def to_dict(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        return {
            "published_at": self.timestamp.astimezone(timezone.utc).isoformat(),
            "relative_time": format_relative_time(self.timestamp, now=now),
            "subject": self.subject,
            "category": self.category,
            "source": self.source,
            "flags": self.flags,
            "priority": self.priority
        }


class MT5NewsParser:
    """
    Parser for MT5 news.dat binary format.
    
    Reverse-engineered layout used by current MT5 news.dat files.

    Each headline lives in an inline record with a 12-byte null prefix, followed
    by a subject field (type 3), source field (type 4), and metadata blob
    (type 5). The paired compact index record sits 0x118 bytes earlier and
    stores the Unix timestamp at bytes 20:24.
    """
    
    HEADER_SIZE = 504
    INDEX_RECORD_DELTA = 0x118
    MIN_TIMESTAMP = 1577836800  # 2020-01-01
    MAX_TIMESTAMP = 2051222400  # 2035-01-01
    LAYOUT_ID = "news_dat_header_504_index_0x118"
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.records: List[MT5NewsRecord] = []
        self.header_info: Dict[str, Any] = {}
        self.parse_stats: Dict[str, int] = {
            "candidates_scanned": 0,
            "records_parsed": 0,
            "duplicates_skipped": 0,
            "validation_failures": 0,
        }
        
    def parse(self) -> List[MT5NewsRecord]:
        """Parse the entire news.dat file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"News database not found: {self.filepath}")
            
        with open(self.filepath, 'rb') as file_handle:
            file_size = self.filepath.stat().st_size
            if file_size < self.HEADER_SIZE:
                self._parse_header(file_handle.read(self.HEADER_SIZE))
            with mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                # Parse header, index entries, and records without copying the
                # entire growing database into the Python heap.
                self._parse_header(data[:self.HEADER_SIZE])
                self.records = self._parse_records(data)
        
        return self.records
    
    def _parse_header(self, header: bytes) -> None:
        """Parse the 504-byte header section."""
        if len(header) < self.HEADER_SIZE:
            raise ValueError(
                f"Unsupported or truncated MT5 news.dat layout: expected a "
                f"{self.HEADER_SIZE}-byte header, found {len(header)} bytes."
            )
        # First 4 bytes: header size (should be 504)
        header_size = struct.unpack('<I', header[:4])[0]
        if header_size != self.HEADER_SIZE:
            raise ValueError(
                "Unsupported MT5 news.dat layout: "
                f"header declares {header_size} bytes, expected {self.HEADER_SIZE} "
                f"for parser layout {self.LAYOUT_ID}."
            )
        
        # Copyright string at offset 4 (UTF-16-LE, variable length, null-terminated)
        copyright_raw = header[4:132]
        try:
            copyright_str = copyright_raw.decode('utf-16-le').split('\x00')[0]
        except UnicodeDecodeError:
            copyright_str = "Unknown"
        
        # "News" identifier at offset 132
        news_id_raw = header[132:140]
        try:
            news_id = news_id_raw.decode('utf-16-le').split('\x00')[0]
        except UnicodeDecodeError:
            news_id = "Unknown"
        
        self.header_info = {
            "header_size": header_size,
            "copyright": copyright_str,
            "identifier": news_id,
            "parser_layout": self.LAYOUT_ID,
        }
        
        logger.debug(f"Parsed header: {self.header_info}")

    @property
    def parse_health(self) -> Dict[str, Any]:
        """Structured diagnostics for the most recent parse run."""
        stats = self.parse_stats
        scanned = stats["candidates_scanned"]
        parsed = stats["records_parsed"]
        failures = stats["validation_failures"]
        if scanned == 0:
            status = "empty"
        elif failures > 0 and parsed == 0:
            status = "failed"
        elif failures > 0:
            status = "degraded"
        else:
            status = "ok"
        return {"status": status, **stats}
    
    def _parse_records(self, data: Any) -> List[MT5NewsRecord]:
        """Parse all news records from the binary data."""
        records: List[MT5NewsRecord] = []
        seen_keys = set()
        prefix = b"\x00" * 12
        max_offset = len(data) - 80
        offset = self.HEADER_SIZE
        candidates = 0
        duplicates = 0
        failures = 0

        while offset < max_offset:
            offset = data.find(prefix, offset, max_offset)
            if offset < 0:
                break

            candidates += 1
            record = self._parse_inline_record(data, offset)
            if not record:
                failures += 1
                offset += len(prefix)
                continue

            record_key = (record.timestamp, record.subject, record.source)
            if record_key in seen_keys:
                duplicates += 1
                offset += len(prefix)
                continue

            seen_keys.add(record_key)
            records.append(record)
            offset += len(prefix)

        self.parse_stats["candidates_scanned"] = candidates
        self.parse_stats["records_parsed"] = len(records)
        self.parse_stats["duplicates_skipped"] = duplicates
        self.parse_stats["validation_failures"] = failures
        return records

    def _parse_inline_record(self, data: bytes, offset: int) -> Optional[MT5NewsRecord]:
        """Parse one inline headline record and its paired compact index record."""
        if data[offset:offset + 12] != b'\x00' * 12:
            return None

        try:
            subject_length = struct.unpack('<I', data[offset + 12:offset + 16])[0]
            subject_type = struct.unpack('<H', data[offset + 16:offset + 18])[0]
        except struct.error:
            return None

        if subject_type != 3 or subject_length < 2 or subject_length > 2000 or subject_length % 2 != 0:
            return None

        subject_start = offset + 18
        subject_end = subject_start + subject_length
        if subject_end + 12 > len(data):
            return None

        try:
            subject = data[subject_start:subject_end].decode('utf-16-le').rstrip('\x00')
        except UnicodeDecodeError:
            return None
        subject = normalize_news_text(subject)

        if not subject or not any(ch.isalpha() for ch in subject):
            return None

        try:
            source_length = struct.unpack('<I', data[subject_end:subject_end + 4])[0]
            source_type = struct.unpack('<H', data[subject_end + 4:subject_end + 6])[0]
        except struct.error:
            return None

        if source_type != 4 or source_length < 2 or source_length > 200 or source_length % 2 != 0:
            return None

        source_start = subject_end + 6
        source_end = source_start + source_length
        if source_end + 6 > len(data):
            return None

        try:
            source = data[source_start:source_end].decode('utf-16-le').rstrip('\x00')
        except UnicodeDecodeError:
            return None
        source = normalize_news_text(source)

        if not source:
            return None

        try:
            metadata_length = struct.unpack('<I', data[source_end:source_end + 4])[0]
            metadata_type = struct.unpack('<H', data[source_end + 4:source_end + 6])[0]
        except struct.error:
            return None

        if metadata_type != 5 or metadata_length > 4096:
            return None

        timestamp = self._parse_timestamp_for_record(data, offset)
        if not timestamp:
            return None

        flags = 0
        metadata_start = source_end + 6
        if metadata_length >= 4 and metadata_start + 4 <= len(data):
            flags = struct.unpack('<I', data[metadata_start:metadata_start + 4])[0]

        return MT5NewsRecord(
            timestamp=timestamp,
            subject=subject,
            category=source,
            source=source,
            flags=flags,
        )

    def _parse_timestamp_for_record(self, data: bytes, inline_offset: int) -> Optional[datetime]:
        """Read the compact index record paired with an inline headline record."""
        index_offset = inline_offset - self.INDEX_RECORD_DELTA
        if index_offset < 0 or index_offset + 24 > len(data):
            return None

        try:
            delta = struct.unpack('<H', data[index_offset + 6:index_offset + 8])[0]
            timestamp_raw = struct.unpack('<I', data[index_offset + 20:index_offset + 24])[0]
        except struct.error:
            return None

        if delta != self.INDEX_RECORD_DELTA:
            return None

        if not (self.MIN_TIMESTAMP <= timestamp_raw <= self.MAX_TIMESTAMP):
            return None

        timestamp_utc = _mt5_epoch_to_utc(float(timestamp_raw))
        return datetime.fromtimestamp(timestamp_utc, tz=timezone.utc)


def _parse_news_filter_datetime(value: str, client_tz: Any = None) -> datetime:
    """Interpret ISO filter inputs in the same display timezone used for output."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    if client_tz is not None:
        try:
            aware = dt.replace(tzinfo=client_tz)
            return aware.astimezone(timezone.utc)
        except Exception:
            pass
    return dt.replace(tzinfo=timezone.utc)


def _rank_news_candidates(candidates: List[Path]) -> Optional[str]:
    if not candidates:
        return None

    ranked: List[tuple[bool, float, int, str]] = []
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        ranked.append((stat.st_size > MT5NewsParser.HEADER_SIZE, stat.st_mtime, stat.st_size, str(path)))

    if not ranked:
        return None

    ranked.sort(reverse=True)
    return ranked[0][3]


def _news_candidates_from_terminal_root(terminal_root: Path) -> List[Path]:
    bases_path = terminal_root / "bases"
    if not bases_path.exists():
        return []

    candidates: List[Path] = []
    for broker_dir in bases_path.iterdir():
        if not broker_dir.is_dir():
            continue
        news_path = broker_dir / "news" / "news.dat"
        if news_path.exists():
            candidates.append(news_path)
    return candidates


def _is_mt5_terminal_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if "terminal64.exe" in output.lower():
            return True
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal.exe"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return "terminal.exe" in output.lower()
    except Exception:
        return False


def _terminal_root_from_mt5_api() -> Optional[Path]:
    """Ask the MT5 Python binding for the active terminal data path when available."""
    try:
        import MetaTrader5 as mt5_module  # type: ignore
    except Exception:
        return None

    mt5: Any = mt5_module

    try:
        terminal_info = mt5.terminal_info()
    except Exception:
        terminal_info = None

    if terminal_info is not None:
        data_path = getattr(terminal_info, "data_path", None)
        if data_path:
            path = Path(str(data_path))
            if path.exists():
                return path

    if not _is_mt5_terminal_running():
        return None

    initialized_here = False
    try:
        if not mt5.initialize():
            return None
        initialized_here = True
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            return None
        data_path = getattr(terminal_info, "data_path", None)
        if not data_path:
            return None
        path = Path(str(data_path))
        return path if path.exists() else None
    except Exception:
        return None
    finally:
        if initialized_here:
            try:
                mt5.shutdown()
            except Exception:
                pass


def _read_origin_path(origin_file: Path) -> Optional[str]:
    try:
        raw = origin_file.read_bytes()
    except OSError:
        return None

    for encoding in ("utf-16", "utf-8", "latin1"):
        try:
            text = raw.decode(encoding).strip().strip("\x00")
        except Exception:
            continue
        if text:
            return str(Path(text))
    return None


def _terminal_roots_from_origin(roaming_path: Path) -> List[Path]:
    roots: List[Path] = []
    for terminal_hash in roaming_path.iterdir():
        if not terminal_hash.is_dir():
            continue
        origin_file = terminal_hash / "origin.txt"
        if not origin_file.exists():
            continue
        if _read_origin_path(origin_file):
            roots.append(terminal_hash)
    return roots


def get_mt5_news(
    news_db_path: Optional[str] = None,
    limit: int = 100,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    category: Optional[str] = None,
    search_term: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch news from MT5's local news database.
    
    Parameters
    ----------
    news_db_path : str, optional
        Full path to news.dat file. If not provided, will attempt to auto-detect
        from MT5 installation directories.
    limit : int
        Maximum number of news items to return (default 100)
    from_date : str, optional
        Filter news from this date (ISO format: YYYY-MM-DD)
    to_date : str, optional
        Filter news up to this date (ISO format: YYYY-MM-DD)
    category : str, optional
        Filter by news category (e.g., "FXStreet", "DailyFX")
    search_term : str, optional
        Filter news items containing this term in the subject
        
    Returns
    -------
    dict
        Dictionary containing news items and metadata
    """
    try:
        try:
            safe_limit = int(limit)
        except Exception:
            safe_limit = 100
        safe_limit = max(1, min(500, safe_limit))

        # Auto-detect path if not provided
        if not news_db_path:
            news_db_path = _auto_detect_news_path()
            if not news_db_path:
                return {
                    "error": "Could not auto-detect MT5 news database path. Please provide news_db_path explicitly.",
                    "hint": "Look for 'news.dat' in your MT5 data folder under bases/<broker>/news/"
                }
        
        # Parse the database
        parser = MT5NewsParser(news_db_path)
        records = parser.parse()
        use_client_tz = _use_client_tz()
        client_tz = _resolve_client_tz() if use_client_tz else None
        
        try:
            from_dt = _parse_news_filter_datetime(from_date, client_tz) if from_date else None
            to_dt = _parse_news_filter_datetime(to_date, client_tz) if to_date else None
        except ValueError as e:
            return {
                "error": f"Invalid news date filter: {e}",
                "hint": "Use ISO dates such as YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
            }

        if from_dt and to_dt and from_dt > to_dt:
            all_categories = list(set(r.category for r in records if r.category))
            all_sources = list(set(r.source for r in records if r.source))
            payload = {
                "success": True,
                "count": 0,
                "total_records": len(records),
                "database_path": str(news_db_path),
                "header_info": parser.header_info,
                "parse_health": parser.parse_health,
                "available_categories": all_categories[:20],
                "available_sources": all_sources[:20],
                "news": [],
                "warning": "from_date is after to_date; returning no results",
            }
            return _finalize_news_payload(payload, use_client_tz)

        # Apply filters
        filtered = records

        if from_dt:
            filtered = [r for r in filtered if r.timestamp >= from_dt]

        if to_dt:
            filtered = [r for r in filtered if r.timestamp <= to_dt]
        
        if category:
            cat_lower = category.lower()
            filtered = [r for r in filtered if cat_lower in r.category.lower()]
        
        if search_term:
            term_lower = search_term.lower()
            filtered = [r for r in filtered if term_lower in r.subject.lower()]
        
        # Sort by timestamp (newest first) and apply limit
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        filtered = filtered[:safe_limit]
        
        # Build response
        now_utc = datetime.now(timezone.utc)
        news_list = [r.to_dict(now=now_utc) for r in filtered]
        
        # Get unique categories for reference
        all_categories = list(set(r.category for r in records if r.category))
        all_sources = list(set(r.source for r in records if r.source))

        payload = {
            "success": True,
            "count": len(news_list),
            "limit": safe_limit,
            "total_records": len(records),
            "database_path": str(news_db_path),
            "header_info": parser.header_info,
            "parse_health": parser.parse_health,
            "available_categories": all_categories[:20],
            "available_sources": all_sources[:20],
            "news": news_list
        }
        return _finalize_news_payload(payload, use_client_tz)
        
    except FileNotFoundError as e:
        logger.error(f"News database not found: {e}")
        return {
            "error": str(e),
            "hint": "Verify the path to news.dat or let the tool auto-detect from MT5 directories"
        }
    except Exception as e:
        logger.exception(f"Error parsing MT5 news database: {e}")
        return {
            "error": f"Failed to parse news database: {str(e)}",
            "hint": "The news.dat file format may have changed or the file may be corrupted"
        }


def _auto_detect_news_path() -> Optional[str]:
    """Attempt to auto-detect the MT5 news.dat file path."""
    # Common MT5 data directories
    roaming_path = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
    
    if not roaming_path.exists():
        return None

    terminal_root = _terminal_root_from_mt5_api()
    if terminal_root is not None:
        matched = _rank_news_candidates(_news_candidates_from_terminal_root(terminal_root))
        if matched:
            return matched

    origin_roots = _terminal_roots_from_origin(roaming_path)
    matched = _rank_news_candidates(
        [candidate for root in origin_roots for candidate in _news_candidates_from_terminal_root(root)]
    )
    if matched:
        return matched

    generic_roots = [path for path in roaming_path.iterdir() if path.is_dir()]
    return _rank_news_candidates(
        [candidate for root in generic_roots for candidate in _news_candidates_from_terminal_root(root)]
    )
