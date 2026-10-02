"""
kartikey/document_processing/extractor.py

Extracts plain text from PDF and DOCX tender documents.

Design principles:
  - Does NOT call external services or AI models.
    Text cleaning and semantic chunking are handled downstream by the AI/ML component.
  - Does NOT modify or annotate the text — returns raw extracted content.
  - Preserves structure hints where possible (page breaks, section numbers)
    so that downstream requirement extraction can identify source locations.
  - Handles the two most common government tender formats: PDF and DOCX.
  - Raises DocumentError with specific codes so callers can respond appropriately.

Known limitations (flagged, not hidden):
  - Scanned PDFs (images inside PDF): text extraction will return empty or near-empty.
    Detected and flagged via SCANNED_PDF error code — OCR is a future improvement.
  - Multi-column PDFs: pdfplumber handles these better than PyPDF2 but may still
    produce garbled column ordering in complex layouts.
  - Password-protected PDFs: rejected with ENCRYPTED_PDF error code.
"""

from __future__ import annotations

import re
from pathlib import Path

from shared.utils import DocumentError, get_logger

logger = get_logger(__name__)

# Minimum characters to consider extraction successful.
# Anything below this almost certainly means a scanned/image PDF.
_MIN_TEXT_LENGTH = 100


# ===========================================================================
# Public interface
# ===========================================================================

def extract_text(path: Path) -> str:
    """
    Extract plain text from a PDF or DOCX file.

    Parameters
    ----------
    path:
        Absolute path to the document file (saved by storage.py).

    Returns
    -------
    str
        Extracted plain text. Newlines preserved.
        Page breaks represented as '\\n\\n--- Page N ---\\n\\n' for PDFs.

    Raises
    ------
    DocumentError
        UNSUPPORTED_FILE_TYPE  — file extension not .pdf or .docx
        EXTRACTION_FAILED      — file could not be parsed
        SCANNED_PDF            — PDF appears to be image-only (no text layer)
        ENCRYPTED_PDF          — PDF is password-protected
        EMPTY_DOCUMENT         — document yielded no text after extraction
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)

    raise DocumentError(
        f"Cannot extract text from '{suffix}' files.",
        code="UNSUPPORTED_FILE_TYPE",
    )


# ===========================================================================
# PDF extraction
# ===========================================================================

def _extract_pdf(path: Path) -> str:
    try:
        import pymupdf  # PyMuPDF >= 1.24 (import name is pymupdf, not fitz)
    except ImportError as exc:
        raise DocumentError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF",
            code="MISSING_DEPENDENCY",
        ) from exc

    try:
        doc = pymupdf.open(str(path))
        pages: list[str] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text and text.strip():
                # Prepend a page marker so downstream can track source locations
                pages.append(f"--- Page {i} ---\n{text.strip()}")
        doc.close()

    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(
            f"Failed to parse PDF '{path.name}': {exc}",
            code="EXTRACTION_FAILED",
        ) from exc

    if not pages:
        raise DocumentError(
            f"'{path.name}' appears to contain no extractable text. "
            "It may be a scanned document (image-only PDF). "
            "OCR support is not yet implemented.",
            code="SCANNED_PDF",
        )

    full_text = "\n\n".join(pages)

    if len(full_text.strip()) < _MIN_TEXT_LENGTH:
        raise DocumentError(
            f"'{path.name}' yielded very little text ({len(full_text.strip())} chars). "
            "It may be a scanned document.",
            code="SCANNED_PDF",
        )

    logger.info(
        "PDF extracted: %s — %d pages, %d chars",
        path.name, len(pages), len(full_text),
    )
    return full_text


# ===========================================================================
# DOCX extraction
# ===========================================================================

def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise DocumentError(
            "python-docx is not installed. Run: pip install python-docx",
            code="MISSING_DEPENDENCY",
        ) from exc

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise DocumentError(
            f"Failed to parse DOCX '{path.name}': {exc}",
            code="EXTRACTION_FAILED",
        ) from exc

    sections: list[str] = []

    # Extract paragraphs — skip empty ones
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            sections.append(text)

    # Extract text from tables (common in tender BOQs and specification tables)
    for table in doc.tables:
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                # Join table row as tab-separated for readability
                sections.append("\t".join(row_cells))

    if not sections:
        raise DocumentError(
            f"'{path.name}' appears to be empty.",
            code="EMPTY_DOCUMENT",
        )

    full_text = "\n\n".join(sections)

    if len(full_text.strip()) < _MIN_TEXT_LENGTH:
        raise DocumentError(
            f"'{path.name}' yielded very little text ({len(full_text.strip())} chars). "
            "The document may be empty or corrupted.",
            code="EMPTY_DOCUMENT",
        )

    logger.info(
        "DOCX extracted: %s — %d paragraphs/rows, %d chars",
        path.name, len(sections), len(full_text),
    )
    return full_text


# ===========================================================================
# IS reference scanner — preliminary scan of extracted text
# ===========================================================================

# Regex to find IS standard references in tender text.
# Matches formats like:
#   IS 269:2015
#   IS 1180 (Part 1):2014
#   IS 10322 (Part 5/Sec 3):2012
#   IS 2062:2011 Amd.4
#   IS 269 (latest edition)
#   IS:2062   IS-2062   IS2062   IS : 2062 - 2011   IS 2062-2011
#   IS 1554 : Part 1 : 1988   IS 10322 : Part 5 : Sec 3   (the catalogue's own form)
#
# A range ("Part 1 and 2", "Sec 1 to 4") is not narrowed to its first member:
# the designation stops at the last level cited unambiguously.
#
# A year after "-" or "–" must look like one (19xx/20xx, not part of a longer
# number), so "IS 1239-1" does not read its part number as a year. After ":"
# any four digits are taken, as before.
#
# And must not match ordinary prose, which is full of "is <number>":
#   "voltage is 230 V", "warranty is 5 years"  — lower-case "is": the prefix is
#       matched case-sensitively, so only "IS" qualifies.
#   "basis 230", "this 5", "Analysis 2"        — "is" inside a word: \b requires
#       "IS" to start a word.
#   "WARRANTY IS 5 YEARS", "LENGTH IS 732 M"   — ALL-CAPS prose, where case does
#       not help: a number followed directly by a unit is a quantity, not a
#       standard. A real citation is followed by ":", "(", punctuation or a word.
# (?!\d) keeps the number whole, so a rejected "IS 230 V" cannot backtrack and
# match "IS 23" instead. The rest of the pattern is case-insensitive as before.
_PROSE_QUANTITY_UNIT = (
    r"%|°|(?:years?|yrs?|months?|days?|hours?|hrs?|nos"
    r"|mm|cm|km|m|kg|kv|kva|kw|ma|v|a|w|hz)\b"
)
_IS_REFERENCE_PATTERN = re.compile(
    r"\b(?-i:IS)\s*[:\-–]?\s*"       # "IS" — uppercase, whole word — then " ", ":", "-" or nothing
    r"(\d+)(?!\d)"                    # IS number
    rf"(?!\s*(?:{_PROSE_QUANTITY_UNIT}))"  # not a measured quantity
    r"(?:\s*\(([^)]+)\)"              # optional (Part N/Sec M) …
    r"|\s*:\s*(Part\s*\d+(?!\d|\s*(?:to|and|&)\s*\d)"             # … or : Part N
    r"(?:\s*:\s*Sec(?:tion)?\s*\d+(?!\d|\s*(?:to|and|&)\s*\d))?))?"  # [: Sec M] — not a range
    r"(?:\s*(?::|[\-–](?=\s*(?:19|20)\d{2}(?!\d)))\s*(\d{4}))?"  # optional :YYYY or -YYYY year
    r"(?:\s+Amd\.?\s*(\d+))?",        # optional Amd.N
    re.IGNORECASE,
)


def scan_is_references(text: str) -> list[dict]:
    """
    Do a quick regex scan of extracted text to find all IS standard references.

    This is a preliminary scan — it finds candidate references but does NOT
    validate that the standards actually exist. The requirement extraction
    step (AI/ML) does the authoritative extraction with semantic understanding.

    Returns a list of dicts with keys:
      matched_text, citation, is_number, part_section, designation, year,
      amendment_number, char_offset

    `is_number` is always the base number ("IS 10322"). `designation` is the
    catalogue form of what was cited — "IS 10322 : Part 5 : Sec 3" when a
    Latin-script part is given, otherwise the base number. `citation` is the
    tender's own wording with whitespace collapsed.

    `char_offset` is where the reference starts in `text`. Pair it with
    `clause_at()` to recover the sentence the citation sits in.
    """
    results = []
    for match in _IS_REFERENCE_PATTERN.finditer(text):
        part_section = match.group(2) or match.group(3)
        matched_text = match.group(0).strip()
        results.append({
            "matched_text": matched_text,
            "citation": " ".join(matched_text.split()),
            "is_number": f"IS {match.group(1)}",
            "part_section": part_section,
            "designation": _designation(match.group(1), part_section),
            "year": int(match.group(4)) if match.group(4) else None,
            "amendment_number": int(match.group(5)) if match.group(5) else None,
            "char_offset": match.start(),
        })
    return results


# "Part 5/Sec 3", "Part 5 : Sec 3", "Part 1" — the only part forms given a
# catalogue designation. Anything else in the brackets ("latest edition",
# "भाग 1") leaves the citation at its base number, as before.
_PART_SECTION = re.compile(
    r"^\s*Part\s*(\d+)\s*(?:[/:,]\s*Sec(?:tion)?\s*(\d+))?\s*$", re.IGNORECASE,
)


def _designation(number: str, part_section: str | None) -> str:
    """
    The cited standard in the catalogue's own spelling: "IS N", "IS N : Part P"
    or "IS N : Part P : Sec S". StandardsStore matches that exactly and falls
    back to the whole family when the catalogue has no such part.
    """
    base = f"IS {number}"
    m = _PART_SECTION.match(part_section or "")
    if not m:
        return base
    designation = f"{base} : Part {m.group(1)}"
    if m.group(2):
        designation += f" : Sec {m.group(2)}"
    return designation


_BASE_IS_NUMBER = re.compile(r"\s*IS\s*[:\-–]?\s*(\d+)")


def base_is_number(is_reference: str) -> str:
    """
    "IS 10322 : Part 5 : Sec 3" -> "IS 10322". A value that is already a base
    number, or is not an "IS N" reference at all, is returned unchanged.
    """
    m = _BASE_IS_NUMBER.match(is_reference)
    return f"IS {m.group(1)}" if m else is_reference


# ===========================================================================
# Clause context — the sentence a citation sits in
# ===========================================================================
#
# A bare IS number is not a requirement. "IS 1554" says which standard was
# cited; "1.1 kV grade XLPE insulated armoured power cables ... conforming to
# IS 1554 : Part 1" says what was actually asked for, and only the second lets a
# reader (or a scope check) see that XLPE was specified. When the LLM extractor
# is unavailable and the pipeline degrades to scan_is_references(), the
# surrounding clause is the difference between a requirement and a label.
#
# This is deliberately a boundary finder, not a sentence tokenizer. Tender text
# is full of periods that end nothing — "1.1 kV", "3.5 core", "IS 2062 Amd. 4",
# clause numbers like "4.2.1" — so a split on "." mangles exactly the documents
# this product reads.

# Sentence terminator followed by whitespace. ":" is excluded: BIS designations
# are written "IS 1554 : Part 1", and treating that colon as a boundary would
# cut the citation in half. Requiring trailing whitespace is what makes decimal
# quantities safe — the "." in "1.1 kV" is followed by a digit, not a space.
_SENTENCE_BOUNDARY = re.compile(r"([.;!?])(\s+)")

# A blank line always separates clauses.
_BLANK_LINE = re.compile(r"\n[ \t]*\n")

# A line break followed by something that starts a new item. The numeric form
# requires either a dot between digits ("4.2", "4.2.1") or trailing punctuation
# ("4.", "5)"), never a bare integer — a wrapped line beginning "110 lm/W with
# IP66 ingress protection" is the continuation of a sentence, not a new clause,
# and splitting there would drop the specification it belongs to.
_CLAUSE_MARKER = re.compile(
    r"\n[ \t]*(?="
    r"\d{1,3}(?:\.\d{1,3})+[.)]?\s"          # 4.2   4.2.1   4.2)
    r"|\d{1,3}[.)]\s"                        # 4.    5)
    r"|\(?[a-zA-Z]\)"                        # (a)   b)
    r"|[-–—•*]\s"             # bullets
    r"|(?:Clause|Section|Item|Annex|Note|Schedule|Sl\.)\b"
    r")"
)

# Tokens that end in a period without ending a sentence. Kept small and
# domain-specific: these are the ones that actually occur in Indian tender text.
_NON_TERMINAL_ABBREVIATIONS = frozenset({
    "amd", "amdt", "no", "nos", "sl", "sec", "cl", "cls", "pt", "para", "fig",
    "ref", "vol", "sr", "rs", "ltd", "pvt", "co", "govt", "dept", "approx",
    "max", "min", "qty", "wt", "viz", "vs", "i.e", "e.g",
})

# How far to look for a boundary on each side, and the most clause text worth
# storing. A clause longer than the cap is windowed around the citation rather
# than cut off at the front, so the reference always stays visible.
_CLAUSE_SCAN_WINDOW = 600
_MAX_CLAUSE_CHARS = 400


def _ends_with_abbreviation(text: str, dot_index: int) -> bool:
    """True if the period at `dot_index` closes an abbreviation, not a sentence."""
    if text[dot_index] != ".":
        return False
    i = dot_index
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] == "."):
        i -= 1
    token = text[i:dot_index].strip(".").casefold()
    return bool(token) and token in _NON_TERMINAL_ABBREVIATIONS


def _clause_boundaries(text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """
    Locate clause boundaries within `text[lo:hi]`.

    Each result is `(end_of_preceding_clause, start_of_next_clause)` in absolute
    offsets, ascending. The pair matters: the first is used when the boundary
    terminates a clause (so the full stop is kept), the second when it opens one
    (so the whitespace or bullet is dropped).
    """
    window = text[lo:hi]
    found: list[tuple[int, int]] = []

    for m in _BLANK_LINE.finditer(window):
        found.append((lo + m.start(), lo + m.end()))

    for m in _CLAUSE_MARKER.finditer(window):
        found.append((lo + m.start(), lo + m.end()))

    for m in _SENTENCE_BOUNDARY.finditer(window):
        if _ends_with_abbreviation(window, m.start(1)):
            continue
        found.append((lo + m.end(1), lo + m.end()))

    found.sort()
    return found


def clause_at(text: str, offset: int, match_length: int = 0) -> str:
    """
    Return the clause of `text` containing the span at `offset`.

    Whitespace is collapsed so a citation split across a line break reads as one
    sentence. If no boundary is found within the scan window the text is
    returned unbounded-but-capped rather than empty — some context beats none.

    Parameters
    ----------
    text:
        The document text the offset refers to.
    offset:
        Start of the span of interest, e.g. `char_offset` from
        `scan_is_references()`.
    match_length:
        Length of that span. Given, the clause is guaranteed to extend past the
        end of the match rather than stopping inside it.
    """
    if not text:
        return ""

    offset = max(0, min(offset, len(text)))
    match_end = min(len(text), offset + max(0, match_length))

    win_lo = max(0, offset - _CLAUSE_SCAN_WINDOW)
    win_hi = min(len(text), match_end + _CLAUSE_SCAN_WINDOW)

    start = win_lo
    for _, next_start in _clause_boundaries(text, win_lo, offset):
        start = next_start

    end = win_hi
    for clause_end, _ in _clause_boundaries(text, match_end, win_hi):
        end = clause_end
        break

    clause = " ".join(text[start:end].split())
    if len(clause) <= _MAX_CLAUSE_CHARS:
        return clause

    # Too long to store whole. Window it around the citation, which is at
    # roughly (offset - start) into the collapsed string, and mark both cuts so
    # nobody reads the excerpt as a complete clause.
    citation_at = len(" ".join(text[start:offset].split()))
    half = _MAX_CLAUSE_CHARS // 2
    lo = max(0, min(citation_at - half, len(clause) - _MAX_CLAUSE_CHARS))
    hi = lo + _MAX_CLAUSE_CHARS
    return (
        ("…" if lo > 0 else "")
        + clause[lo:hi].strip()
        + ("…" if hi < len(clause) else "")
    )


def iter_clauses(text: str) -> list[tuple[int, str]]:
    """
    Split `text` into clauses, returning `(char_offset, clause_text)` pairs.

    `clause_at()` answers "which clause is at this offset" — the right question
    when a regex has already found something. This answers "what clauses are
    there at all", which is what a caller needs when the document itself is the
    only input and there is nothing to anchor on yet.

    Both use the same boundary rules, so a clause returned here is the same
    string `clause_at()` would return for any offset inside it. That matters more
    than it looks: a second splitter tuned slightly differently would make the
    profile screen and the analysis disagree about where a requirement begins,
    for documents where the two are reading identical text.

    Whitespace is collapsed per clause, empty clauses are dropped, and offsets
    are into the original `text` so the caller can still recover context.
    """
    if not text:
        return []

    cuts = _clause_boundaries(text, 0, len(text))

    clauses: list[tuple[int, str]] = []
    start = 0
    for clause_end, next_start in cuts:
        if clause_end <= start:
            # Two rules fired on the same boundary (a full stop that is also a
            # blank line, say). The first already closed the clause.
            continue
        collapsed = " ".join(text[start:clause_end].split())
        if collapsed:
            clauses.append((start, collapsed))
        start = next_start

    tail = " ".join(text[start:].split())
    if tail:
        clauses.append((start, tail))

    return clauses
