# brand_ad_checker_app.py
# FlavCity Brand Ad Checker — Copy-only (no OCR)
# Channel-aware checks + Scoreboard + line/column pointers + inline highlights.

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import streamlit as st

# ---------- Optional grammar checker (quietly skipped if unavailable) ----------
try:
    import language_tool_python
    LT_TOOL = language_tool_python.LanguageToolPublicAPI("en-US")
    LT_AVAILABLE = True
except Exception:
    LT_TOOL = None
    LT_AVAILABLE = False

# ---------- Brand defaults (editable in sidebar) ----------
DEFAULT_COMPETITORS = [
    "starbucks", "dunkin", "nespresso", "peet's", "peets",
    "blue bottle", "keurig", "tim hortons", "philz",
    "intelligentsia", "lavazza"
]

DEFAULT_POSITIVE_CUES = [
    "whole-food", "whole food", "real-ingredient", "real ingredient", "real ingredients",
    "clean-label", "clean label",
    "exceptional nutrition", "nutrition and flavor",
    "high-quality", "crafted", "Bobby Approved", "Bobby-Approved",
    "better-for-you", "real food ingredients",
]

# Messaging pillars (we use for nudges)
PILLAR_FLAVOR = ["flavor-forward", "flavorful", "tastes phenomenal", "craveable", "taste you’ll love"]
PILLAR_TRANSPARENCY = ["no natural flavors", "no artificial sweeteners", "no seed oils", "no fillers", "transparent"]
PILLAR_CONVENIENCE = ["convenience", "without compromise", "no blender", "one scoop", "on-the-go"]
PILLAR_EDUCATION = ["learn", "guide", "break down", "simple", "digestible", "confidence", "expertise"]
PILLAR_COMMUNITY = ["community", "together", "feedback", "listen", "evolve", "household name"]

# Tone & Voice
VOICE_ENERGETIC_WORDS = [
    "bold", "vibrant", "flavorful", "excited", "enthusiastic", "joyful",
    "optimistic", "feel-good", "celebrate", "phenomenal"
]
VOICE_TRUST_WORDS = [
    "transparent", "no shortcuts", "quality first", "Bobby Approved", "crafted with care", "trust"
]
NEGATIVE_TONE = [
    "bad", "awful", "gross", "disgusting", "fake", "junk",
    "sorry", "apologize", "guilt"
]
FEAR_MONGER_WORDS = [
    "toxic", "dangerous", "poison", "unsafe", "carcinogen", "never eat",
    "avoid at all costs", "scary", "harmful", "chemical-laden"
]
INTIMIDATING_PHRASES = [
    "you must", "you have to", "or else", "never ever", "don’t you dare"
]

# ---------- Channel profiles (plain-English rules) ----------
CHANNEL_PROFILES = {
    "Email": {
        "desc": "Deeper education & storytelling with a clear action.",
        "max_chars": None,
        "max_sentence_len": 240,
        "require_cta": True,
        "cta_words": ["shop now", "learn more", "get started", "join the waitlist", "see how"],
        "allow_emoji": True,
        "encourage": ["education", "transparency", "simple explanations", "value"],
        "discourage": [],
    },
    "SMS": {
        "desc": "Quick, engaging, action-driven. Personal and to the point.",
        "max_chars": 300,
        "max_sentence_len": 140,
        "require_cta": True,
        "cta_words": ["shop now", "tap to see", "join now", "get yours", "save now"],
        "allow_emoji": True,
        "encourage": ["urgency", "value", "personal relevance"],
        "discourage": ["paragraphs", "too much detail"],
    },
    "Social": {
        "desc": "Visual storytelling, community energy, and engagement.",
        "max_chars": 400,
        "max_sentence_len": 200,
        "require_cta": False,
        "cta_words": ["see more", "learn more", "link in bio", "shop now"],
        "allow_emoji": True,
        "encourage": ["community", "conversation", "flavor", "fun"],
        "discourage": ["dense technical detail"],
    },
    "Website": {
        "desc": "Trust-building, transparency, and simple education.",
        "max_chars": None,
        "max_sentence_len": 240,
        "require_cta": False,
        "cta_words": ["add to cart", "learn more", "see ingredients", "compare", "see nutrition"],
        "allow_emoji": False,
        "encourage": ["transparency", "education", "value"],
        "discourage": ["hype without proof", "confusing jargon"],
    },
}

# ---------- Data classes ----------
@dataclass
class BrandRules:
    brand_name: str = "FlavCity"
    allowed_spellings: List[str] = field(default_factory=lambda: ["FlavCity"])
    competitors: List[str] = field(default_factory=lambda: DEFAULT_COMPETITORS.copy())
    required_phrases: List[str] = field(default_factory=list)
    positive_cues: List[str] = field(default_factory=lambda: DEFAULT_POSITIVE_CUES.copy())
    negative_tone: List[str] = field(default_factory=lambda: NEGATIVE_TONE.copy())
    product_terms: List[str] = field(default_factory=lambda: ["Latte","Mocha","Macchiato","Cappuccino","Espresso"])
    energetic_words: List[str] = field(default_factory=lambda: VOICE_ENERGETIC_WORDS.copy())
    trust_words: List[str] = field(default_factory=lambda: VOICE_TRUST_WORDS.copy())
    fear_words: List[str] = field(default_factory=lambda: FEAR_MONGER_WORDS.copy())
    intimidating_phrases: List[str] = field(default_factory=lambda: INTIMIDATING_PHRASES.copy())
    require_registered_mark: bool = False

@dataclass
class Issue:
    category: str
    message: str
    span: Optional[Tuple[int,int]] = None
    snippet: Optional[str] = None
    suggestion: Optional[str] = None

# ---------- Line/column helpers ----------
def _line_starts(text: str):
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts

def _idx_to_linecol(text: str, idx: int):
    starts = _line_starts(text)
    lo, hi = 0, len(starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if starts[mid] <= idx:
            lo = mid + 1
        else:
            hi = mid - 1
    line_no = hi + 1
    col_no  = idx - starts[hi] + 1
    return line_no, col_no, starts

def _line_slice(text: str, starts, line_no: int):
    start = starts[line_no - 1]
    end = starts[line_no] if line_no < len(starts) else len(text)
    return text[start:end]

def _with_line_numbers(text: str) -> str:
    out_lines = []
    for i, line in enumerate(text.splitlines(True), 1):
        out_lines.append(f"{i:>3} | {line}")
    return "".join(out_lines)

# ---------- Utilities ----------
def highlight_text(text: str, issues: List[Issue]) -> str:
    spans = [(i.span[0], i.span[1]) for i in issues if i.span is not None]
    if not spans:
        return text
    spans.sort()
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    out, last = [], 0
    for s, e in merged:
        out.append(text[last:s]); out.append("[!]"); out.append(text[s:e]); out.append("[/!]"); last = e
    out.append(text[last:])
    return "".join(out)

def find_all(pattern: re.Pattern, text: str):
    for m in pattern.finditer(text):
        yield m.start(), m.end(), m.group(0)

# ---------- Checks: Brand mechanics ----------
def check_brand_name(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    correct = rules.brand_name
    bad_variants = [r"flav\s?city", r"flavcity™", r"flavcity©", r"flav\s? city", r"flavicity"]
    for pat in bad_variants:
        for s, e, val in find_all(re.compile(pat, re.IGNORECASE), text):
            if text[s:e] != correct:
                issues.append(Issue("Brand Name", f"Incorrect brand usage: '{val}'.",
                                    (s, e), val, f"Use '{correct}' exactly."))
    if rules.require_registered_mark:
        first = re.search(re.escape(correct), text)
        if first and not re.search(re.escape(correct) + r"\s*®", text[first.start():first.end()+2]):
            issues.append(Issue("Trademark", f"Add '®' on first reference to {correct}.",
                                (first.start(), first.end())))
    return issues

def check_competitors(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    lower = text.lower()
    for c in rules.competitors:
        for m in re.finditer(re.escape(c.lower()), lower):
            s, e = m.span()
            issues.append(Issue("Competitor Mention",
                                f"Competitor mentioned: '{text[s:e]}' — consider removing or reframing.",
                                (s, e), text[s:e]))
    return issues

def check_required_phrases(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for phrase in rules.required_phrases:
        if re.search(re.escape(phrase), text, re.IGNORECASE) is None:
            issues.append(Issue("Required Phrase",
                                f"Consider including required phrase: “{phrase}”."))
    return issues

# ---------- Checks: Voice & Tone ----------
def _find_terms(text: str, terms: List[str]) -> List[Tuple[int,int,str]]:
    hits = []
    low = text.lower()
    for t in terms:
        for m in re.finditer(re.escape(t.lower()), low):
            s, e = m.span()
            hits.append((s, e, text[s:e]))
    return hits

def check_voice_alignment(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    if not _find_terms(text, rules.energetic_words):
        issues.append(Issue("Voice Cue",
                            "Opportunity: add a bit of enthusiasm/joy.",
                            suggestion="Try words like: bold, vibrant, flavorful, joyful, feel-good."))
    if not _find_terms(text, rules.trust_words):
        issues.append(Issue("Voice Cue",
                            "Opportunity: reinforce trust/transparency/quality.",
                            suggestion="Try cues like: Bobby-Approved, transparent, quality first, crafted with care."))
    return issues

def check_anti_fear(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for s, e, val in _find_terms(text, rules.fear_words):
        issues.append(Issue("Fear-mongering",
                            f"Fear-based language detected: “{val}”.",
                            (s, e), val,
                            "Avoid fear-mongering. Reframe the benefit positively."))
    for s, e, val in _find_terms(text, rules.intimidating_phrases):
        issues.append(Issue("Intimidating tone",
                            f"Phrase can feel intimidating: “{val}”.",
                            (s, e), val,
                            "Keep it approachable; guide rather than command."))
    return issues

# ---------- Checks: Cues, tone & capitalization ----------
def check_positive_cues(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    found = {p for p in rules.positive_cues if re.search(re.escape(p), text, re.IGNORECASE)}
    missing = set(rules.positive_cues[:3]) - found
    for m in missing:
        issues.append(Issue("Brand Cue", f"Opportunity: consider reinforcing '{m}'."))
    return issues

def check_negative_tone(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for n in rules.negative_tone:
        for m in re.finditer(rf"\b{re.escape(n)}\b", text, re.IGNORECASE):
            issues.append(Issue("Tone",
                                f"Negative tone word: '{m.group(0)}' — make sure it fits.",
                                m.span(), m.group(0)))
    return issues

def grammar_check(text: str) -> List[Issue]:
    issues = []
    if not LT_AVAILABLE:
        return issues
    try:
        matches = LT_TOOL.check(text)
    except Exception:
        matches = []
    for m in matches[:120]:
        s, e = m.offset, m.offset + m.errorLength
        rep = (m.replacements[0] if m.replacements else None)
        issues.append(Issue("Grammar/Spelling", m.message, (s, e), text[s:e], rep))
    return issues

def is_title_like(line: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", line)]
    if not words:
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    return caps / max(len(words), 1) >= 0.7

def is_sentence_like(line: str) -> bool:
    line_stripped = line.strip()
    if not line_stripped:
        return False
    first_alpha = re.search(r"[A-Za-z]", line_stripped)
    if not first_alpha:
        return False
    return line_stripped[first_alpha.start()].islower()

def check_headline_capitalization(text: str) -> List[Issue]:
    issues = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    titleish = [l for l in lines if is_title_like(l)]
    sentenceish = [l for l in lines if is_sentence_like(l)]
    if titleish and sentenceish:
        examples = f"Title Case ex: “{titleish[0]}”; sentence case ex: “{sentenceish[0]}”"
        issues.append(Issue("Capitalization",
                            f"Headline capitalization appears mixed. {examples}",
                            suggestion="Use a single style (Title Case or sentence case)."))
    return issues

def check_product_term_casing(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for term in rules.product_terms:
        variants = set(m.group(0) for m in re.finditer(rf"\b{term}\b", text, re.IGNORECASE))
        if len(variants) > 1:
            examples = ", ".join(sorted(variants))
            issues.append(Issue("Capitalization",
                                f"Inconsistent casing for '{term}': {examples}",
                                suggestion=f"Use '{term}' consistently."))
    return issues

# ---------- Channel-specific checks ----------
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")

def count_chars(text: str) -> int:
    return len(text.replace("\n", ""))

def longest_sentence_len(text: str) -> int:
    sentences = re.split(r"[.!?]\s+|\n+", text.strip())
    return max((len(s) for s in sentences if s), default=0)

def has_cta(text: str, cta_words: List[str]) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in cta_words)

def check_channel_rules(text: str, profile: dict) -> List[Issue]:
    issues = []
    if profile["max_chars"] is not None:
        chars = count_chars(text)
        if chars > profile["max_chars"]:
            issues.append(Issue("Channel: Length",
                                f"Too long for this channel (~{chars} characters).",
                                suggestion=f"Try under {profile['max_chars']} characters."))
    if profile["max_sentence_len"] is not None:
        ls = longest_sentence_len(text)
        if ls > profile["max_sentence_len"]:
            issues.append(Issue("Channel: Readability",
                                f"Some sentences are long for this channel (~{ls} characters).",
                                suggestion=f"Keep sentences under ~{profile['max_sentence_len']} characters."))
    if profile["require_cta"] and not has_cta(text, profile["cta_words"]):
        issues.append(Issue("Channel: CTA",
                            "Add a clear call-to-action for this channel.",
                            suggestion="Examples: " + ", ".join(profile["cta_words"][:4])))
    if not profile["allow_emoji"] and EMOJI_RE.search(text):
        issues.append(Issue("Channel: Emoji",
                            "Emojis aren’t recommended here.",
                            suggestion="Remove emojis for a cleaner, accessible look."))
    if profile["encourage"]:
        issues.append(Issue("Channel: Tips", f"Lean into: {', '.join(profile['encourage'])}."))
    if profile["discourage"]:
        issues.append(Issue("Channel: Tips", f"Try to avoid: {', '.join(profile['discourage'])}."))
    return issues

# ---------- Scoreboard ----------
def compute_scores(issues: List[Issue]) -> dict:
    weights = {"brand": 35, "voice": 25, "grammar": 15, "channel": 25}
    penalties = {"brand": 0, "voice": 0, "grammar": 0, "channel": 0}
    for i in issues:
        cat = i.category.lower()
        if cat.startswith("brand"):               penalties["brand"]   += 10
        elif "competitor" in cat:                 penalties["brand"]   += 8
        elif "trademark" in cat:                  penalties["brand"]   += 6
        elif "capitalization" in cat:             penalties["brand"]   += 4
        elif cat.startswith("voice") or "tone" in cat: penalties["voice"] += 6
        elif "fear" in cat or "intimidating" in cat:   penalties["voice"] += 8
        elif "brand cue" in cat:                  penalties["voice"]  += 3
        elif "grammar" in cat:                    penalties["grammar"] += 5
        elif cat.startswith("channel"):           penalties["channel"] += 6
    subs = {k: max(0, w - penalties[k]) for k, w in weights.items()}
    total = sum(subs.values())
    return {"subs": subs, "total": total, "weights": weights}

def render_scoreboard(scores: dict):
    subs = scores["subs"]; total = scores["total"]
    col1, col2, col3, col4, col5 = st.columns([1.1, 1, 1, 1, 1])
    with col1:
        st.markdown("### Scoreboard")
        st.metric("Overall", f"{total}/100")
        st.progress(total / 100)
    with col2: st.metric("Brand",      f"{subs['brand']}/{scores['weights']['brand']}")
    with col3: st.metric("Voice/Tone", f"{subs['voice']}/{scores['weights']['voice']}")
    with col4: st.metric("Grammar",    f"{subs['grammar']}/{scores['weights']['grammar']}")
    with col5: st.metric("Channel Fit",f"{subs['channel']}/{scores['weights']['channel']}")
    lowest = min(subs, key=subs.get)
    tips = {
        "brand": "Tighten the brand basics (name, capitalization, competitors).",
        "voice": "Dial up enthusiasm & trust words; avoid fear/intimidation.",
        "grammar": "Fix typos/punctuation for polished readability.",
        "channel": "Trim length, shorten sentences, and add a clear CTA where needed."
    }
    st.info(f"Focus tip → **{lowest.capitalize()}**: {tips[lowest]}")

# ---------- Always-visible renderer WITH line/column + inline highlight ----------
def render_issues(issues: List[Issue], source_text: str):
    if not issues:
        st.success("✅ Looks on-brand. No issues found.")
        return
    st.warning(f"⚠ {len(issues)} issues found:")

    _, _, line_starts = _idx_to_linecol(source_text, 0)  # precompute line starts

    for idx, i in enumerate(issues, 1):
        st.markdown(f"**{idx}. {i.category}** — {i.message}")

        if i.span is not None:
            s, e = i.span
            line_s, col_s, _ = _idx_to_linecol(source_text, s)
            line_e, col_e, _ = _idx_to_linecol(source_text, max(s, e - 1))
            st.caption(f"Line {line_s} (cols {col_s}–{col_e if line_e==line_s else '…'})")

            line_text = _line_slice(source_text, line_starts, line_s)
            line_start = line_starts[line_s - 1]
            ls = max(0, s - line_start)
            le = max(ls, e - line_start)
            marked = line_text[:ls] + "[!]" + line_text[ls:le] + "[!/!]" + line_text[le:]
            st.code(marked.rstrip("\n"))

        elif i.snippet:
            st.code(i.snippet)

        if i.suggestion:
            st.markdown(f"*Suggestion:* **{i.suggestion}**")
        st.markdown("---")

# ---------- App UI ----------
st.set_page_config(page_title="FlavCity Brand Ad Checker", page_icon="✅", layout="wide")
st.title("FlavCity Brand Ad Checker")
st.caption("Copy-only. Pick a channel and paste your copy. We’ll flag issues, show exact lines, and suggest fixes.")

# Persist results across reruns
if "issues_last" not in st.session_state:   st.session_state["issues_last"] = []
if "text_last"   not in st.session_state:   st.session_state["text_last"] = ""
if "channel_last"not in st.session_state:   st.session_state["channel_last"] = "Email"
if "scores_last" not in st.session_state:
    st.session_state["scores_last"] = {"subs": {"brand": 0, "voice": 0, "grammar": 0, "channel": 0},
                                       "total": 0,
                                       "weights": {"brand": 35, "voice": 25, "grammar": 15, "channel": 25}}

with st.sidebar:
    st.header("Channel")
    channel = st.selectbox(
        "Where will this run?",
        options=list(CHANNEL_PROFILES.keys()),
        index=list(CHANNEL_PROFILES.keys()).index(st.session_state["channel_last"])
    )
    st.caption(CHANNEL_PROFILES[channel]["desc"])

    st.header("Settings")
    brand_name = st.text_input("Brand Name", value="FlavCity")
    competitors = st.text_area("Competitor names (one per line)", value="\n".join(DEFAULT_COMPETITORS))
    required_phrases = st.text_area("Required phrases (optional, one per line)", value="")
    positive_cues = st.text_area("Positive brand cues (top 3 nudged)", value="\n".join(DEFAULT_POSITIVE_CUES))
    negative_tone = st.text_area("Negative tone words (warn)", value="\n".join(NEGATIVE_TONE))
    product_terms = st.text_input("Product terms (capitalization)", value="Latte,Mocha,Macchiato")

    st.markdown("**Voice & Tone**")
    energetic_words = st.text_area("Energetic/Joyful words", value="\n".join(VOICE_ENERGETIC_WORDS))
    trust_words = st.text_area("Trust/Transparency words", value="\n".join(VOICE_TRUST_WORDS))
    fear_words = st.text_area("Fear-mongering words (flag)", value="\n".join(FEAR_MONGER_WORDS))
    intimidating_phrases = st.text_area("Intimidating phrases (flag)", value="\n".join(INTIMIDATING_PHRASES))

    st.markdown("---")
    st.write(f"Grammar checker: {'✅ enabled' if LT_AVAILABLE else '❌ unavailable'}")

rules = BrandRules(
    brand_name=brand_name,
    allowed_spellings=[brand_name],
    competitors=[c.strip() for c in competitors.splitlines() if c.strip()],
    required_phrases=[p.strip() for p in required_phrases.splitlines() if p.strip()],
    positive_cues=[p.strip() for p in positive_cues.splitlines() if p.strip()],
    negative_tone=[n.strip() for n in negative_tone.splitlines() if n.strip()],
    product_terms=[t.strip() for t in product_terms.split(",") if t.strip()],
    energetic_words=[w.strip() for w in energetic_words.splitlines() if w.strip()],
    trust_words=[w.strip() for w in trust_words.splitlines() if w.strip()],
    fear_words=[w.strip() for w in fear_words.splitlines() if w.strip()],
    intimidating_phrases=[w.strip() for w in intimidating_phrases.splitlines() if w.strip()],
)

tab, = st.tabs(["📝 Check Copy"])

with tab:
    text = st.text_area(
        "Paste ad copy",
        height=220,
        placeholder="e.g., Bye, Bye, Barista. Say hello to instant lattes made with real food ingredients."
    )

    if st.button("Run Checks", type="primary"):
        profile = CHANNEL_PROFILES[channel]

        all_issues: List[Issue] = []
        # Mechanics
        all_issues += check_brand_name(text, rules)
        all_issues += check_competitors(text, rules)
        all_issues += check_required_phrases(text, rules)
        # Voice & Tone
        all_issues += check_voice_alignment(text, rules)
        all_issues += check_anti_fear(text, rules)
        # Cues, tone & capitalization
        all_issues += check_positive_cues(text, rules)
        all_issues += check_negative_tone(text, rules)
        all_issues += check_headline_capitalization(text)
        all_issues += check_product_term_casing(text, rules)
        # Grammar
        all_issues += grammar_check(text)
        # Channel Fit
        all_issues += check_channel_rules(text, profile)

        st.session_state["issues_last"] = all_issues
        st.session_state["text_last"] = text
        st.session_state["channel_last"] = channel
        st.session_state["scores_last"] = compute_scores(all_issues)

# ----- Scoreboard (always visible) -----
render_scoreboard(st.session_state["scores_last"])

# ----- Results (always visible) -----
st.subheader("Results")
render_issues(st.session_state["issues_last"], st.session_state["text_last"])

st.markdown("**Highlighted Copy**")
st.code(
    _with_line_numbers(
        highlight_text(
            st.session_state["text_last"],
            [i for i in st.session_state["issues_last"] if i.span]
        )
    )
)
