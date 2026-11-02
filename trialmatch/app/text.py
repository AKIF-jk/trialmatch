"""Input handling for the app: note segmentation, criteria parsing, limits and the ClinicalTrials.gov lookup. No torch."""
import json
import re
import urllib.error
import urllib.request

from trialmatch.data.note import format_note, parse_note

MAX_NOTE_CHARS = 8000
MAX_SENTENCES = 80
MAX_CRITERIA = 12
MAX_CRITERION_CHARS = 600

ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "vs", "e.g", "i.e", "etc", "approx", "no", "st", "pt", "hx", "dx", "rx", "fx", "sx",
    "tx", "yo", "y.o", "c.f", "cf", "inc", "fig", "ca", "temp", "dept", "mg", "ml", "q.d", "b.i.d", "t.i.d", "q.i.d",
}


class InputError(ValueError):
    """Bad user input. The message is shown to the user as is."""


# ---- notes ---------------------------------------------------------------------------------------------------------

_CANDIDATE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][)\]\"']))\s+(?=[A-Z0-9(\[\"'])")
_LEADING_NUMBER = re.compile(r"^\s*\d+\s*[.)]\s+")


def _is_abbreviation(before):
    """True when the text before a full stop ends in an abbreviation, so the stop does not end a sentence."""
    word = before.split()[-1].rstrip(".").lower() if before.split() else ""
    return word in ABBREVIATIONS or bool(re.fullmatch(r"(\w\.)+\w", word))


def segment_note(text):
    """Split free text into sentences. Line breaks always split; inside a line a full stop, ! or ? followed by a capital letter,
    digit or opening bracket splits unless the word before it is a known abbreviation. Decimals such as 98.6 never split."""
    sentences = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        start = 0
        for match in _CANDIDATE.finditer(line):
            if line[match.start() - 1] == "." and _is_abbreviation(line[start:match.start()]):
                continue
            sentences.append(line[start:match.start()])
            start = match.end()
        sentences.append(line[start:])
    return [s for s in sentences if s]


def renumber(text):
    """Sentences from text with one sentence per line. Leading "12. " style numbers are dropped, so edited numbering is rebuilt."""
    return [" ".join(_LEADING_NUMBER.sub("", line).split()) for line in text.splitlines() if _LEADING_NUMBER.sub("", line).strip()]


def numbered(sentences):
    return format_note(sentences)


def parse_numbered_note(text):
    """Sentences from the editable numbered view ("0. ...\\n1. ..."). Raises InputError with a fix when numbering is broken."""
    if not text or not text.strip():
        raise InputError("The note is empty. Paste a patient note and press Segment note.")
    try:
        sentences = parse_note(text.strip())
    except ValueError:
        raise InputError(
            "The sentence numbering is broken. It must read 0. first, 1. second, 2. third without gaps. "
            "Press Renumber to rebuild it from the current lines."
        ) from None
    return validate_note(sentences)


def validate_note(sentences):
    if not sentences or not any(s.strip() for s in sentences):
        raise InputError("The note is empty. Paste a patient note and press Segment note.")
    if any(not s.strip() for s in sentences):
        raise InputError("A numbered sentence is empty. Delete that line and press Renumber.")
    if len(sentences) > MAX_SENTENCES:
        raise InputError(f"The note has {len(sentences)} sentences. The limit is {MAX_SENTENCES}. Shorten the note.")
    if sum(len(s) for s in sentences) > MAX_NOTE_CHARS:
        raise InputError(f"The note is longer than {MAX_NOTE_CHARS} characters. Shorten the note.")
    return [" ".join(s.split()) for s in sentences]


# ---- criteria ------------------------------------------------------------------------------------------------------

_BULLET = re.compile(r"^\s*(?:[-*•●]|\d+\s*[.)]|\(\w{1,2}\)|\w\))\s+")
_HEADER = re.compile(r"(?im)^[ \t]*(inclusion|exclusion)[ \t]+criteria[ \t]*:?[ \t]*(?:\n|$)")
_INLINE_HEADER = re.compile(r"(?i)\b(inclusion|exclusion)[ \t]+criteria[ \t]*:")


def split_items(block):
    """One criterion per item. If any line has a bullet or number, bullets start items and other lines continue the
    item above (the ClinicalTrials.gov layout). Without bullets every non-empty line is an item."""
    lines = [line.rstrip() for line in block.replace("\r\n", "\n").split("\n") if line.strip()]
    if not any(_BULLET.match(line) for line in lines):
        return [" ".join(line.split()) for line in lines]
    items = []
    for line in lines:
        if _BULLET.match(line) or not items:
            items.append(_BULLET.sub("", line, count=1))
        else:
            items[-1] += " " + line.strip()
    return [" ".join(item.split()) for item in items if item.strip()]


def split_eligibility(text):
    """(inclusion items, exclusion items, warnings) from an eligibility text with "Inclusion Criteria:" and
    "Exclusion Criteria:" headings. Without headings everything is read as inclusion, with a warning."""
    headers = list(_HEADER.finditer(text)) or list(_INLINE_HEADER.finditer(text))
    if not headers:
        return split_items(text), [], ["No Inclusion Criteria or Exclusion Criteria heading found. Everything was read as inclusion criteria."]
    sections = {"inclusion": [], "exclusion": []}
    for index, match in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        sections[match.group(1).lower()] += split_items(text[match.end():end])
    warnings = []
    before = text[:headers[0].start()].strip()
    if before:
        warnings.append("Text before the first heading was ignored.")
    return sections["inclusion"], sections["exclusion"], warnings


def validate_criteria(inclusion, exclusion):
    """Return [(type, text)] in order, inclusion first. Raises InputError for none, too many, or an item that is too long."""
    criteria = [("inclusion", t) for t in inclusion] + [("exclusion", t) for t in exclusion]
    if not criteria:
        raise InputError("No criteria. Enter at least one inclusion or exclusion criterion, one per line.")
    if len(criteria) > MAX_CRITERIA:
        raise InputError(f"{len(criteria)} criteria entered. The limit is {MAX_CRITERIA} per run to stay inside the GPU quota.")
    for kind, text in criteria:
        if len(text) < 3 or not any(c.isalpha() for c in text):
            raise InputError(f"Malformed {kind} criterion: {text!r}. A criterion needs words.")
        if len(text) > MAX_CRITERION_CHARS:
            raise InputError(f"A {kind} criterion is {len(text)} characters long. The limit is {MAX_CRITERION_CHARS}: {text[:60]!r}...")
    return criteria


# ---- ClinicalTrials.gov --------------------------------------------------------------------------------------------

NCT = re.compile(r"NCT\d{8}")


def fetch_trial(nct_id, timeout=20):
    """Public metadata only: the trial title and the eligibility text. Only the NCT id leaves the app."""
    nct_id = (nct_id or "").strip().upper()
    if not NCT.fullmatch(nct_id):
        raise InputError("An NCT id looks like NCT00000620: the letters NCT and eight digits.")
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?fields=NCTId%7CBriefTitle%7CEligibilityCriteria"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise InputError(f"ClinicalTrials.gov has no study {nct_id}." if exc.code == 404 else f"ClinicalTrials.gov returned HTTP {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        raise InputError("ClinicalTrials.gov could not be reached. Paste the criteria by hand instead.") from None
    section = data.get("protocolSection", {})
    text = section.get("eligibilityModule", {}).get("eligibilityCriteria")
    if not text:
        raise InputError(f"Study {nct_id} lists no eligibility criteria text.")
    return {"nct_id": nct_id, "title": section.get("identificationModule", {}).get("briefTitle", ""), "eligibility": text}
