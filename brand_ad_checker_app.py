# brand_ad_checker_app.py
# FlavCity Brand Ad Checker (copy-only, no OCR)
# Shows ALL issues inline (no expanders) and keeps results visible after the button click.

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import streamlit as st

# ---------- Optional grammar checker (will quietly skip if not available) ----------
try:
    import language_tool_python
    LT_TOOL = language_tool_python.LanguageToolPublicAPI("en-US")
    LT_AVAILABLE = True
except Exception:
    LT_TOOL = None
    LT_AVAILABLE = False

# ---------- Defaults you can tweak in the sidebar ----------
DEFAULT_COMPETITORS = [
    "starbucks", "dunkin", "nespresso", "peet's", "peets",
    "blue bottle", "keurig", "tim hortons", "philz",
    "intelligentsia", "lavazza"
]

DEFAULT_POSITIVE_CUES = [
    "whole-food", "whole food", "real-ingredient", "real ingredient", "real ingredients",
    "clean-label", "clean label",
    "busy lifestyle", "busy lifestyles",
    "exceptional nutrition", "nutrition and flavor", "FLAVOR", "flavor-forward",
    "tastes phenomenal", "taste phenomenal", "high-quality", "clean-label products"
]

DEFAULT_NEGATIVE_TONE = [
    "bad", "awful", "gross", "disgusting", "fake", "chemical-laden", "synthetic",
    "sorry", "apologize", "guilt", "junk"
]

# ---------- Data classes ----------
@dataclass
class BrandRules:
    brand_name: str = "FlavCity"
    allowed_spellings: List[str] = field(default_factory=lambda: ["FlavCity"])
    competitors: List[str] = field(default_factory=lambda: DEFAULT_COMPETITORS.copy())
    required_phrases: List[str] = field(default_factory=list)  # optional
    positive_cues: List[str] = field(default_factory=lambda: DEFAULT_POSITIVE_CUES.copy())
    negative_tone: List[str] = field(default_factory=lambda: DEFAULT_NEGATIVE_TONE.copy())
    require_registered_mark: bool = False
    product_terms: List[str] = field(default_factory=lambda: ["Latte","Mocha","Macchiato","Cappuccino","Espresso"])

@dataclass
class Issue:
    category: str
    message: str
    span: Optional[Tuple[int,int]] = None  # (start, end) in the text
    snippet: Optional[str] = None          # a small offending piece to display
    suggestion: Optional[str] = None       # “do this instead”

# ---------- Utilities ----------
def highlight_text(text: str, issues: List[Issue]) -> str:
    """Surround offending spans with [!] ... [/!] so users can see them."""
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

# ---------- Checks ----------
def check_brand_name(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    correct = rules.brand_name
    # common incorrect variants (spacing/casing/symbols)
    bad_variants = [r"flav\s?city", r"flavcity™", r"flavcity©", r"flav\s? city", r"flavicity"]
    for pat in bad_variants:
        for s, e, val in find_all(re.compile(pat, re.IGNORECASE), text):
            if text[s:e] != correct:
                issues.append(Issue(
                    "Brand Name",
                    f"Incorrect brand usage: '{val}'.",
                    (s, e),
                    val,
                    f"Use '{correct}' exactly."
                ))
    if rules.require_registered_mark:
        first = re.search(re.escape(correct), text)
        if first and not re.search(re.escape(correct) + r"\s*®", text[first.start():first.end()+2]):
            issues.append(Issue(
                "Trademark",
                f"Add '®' on first reference to {correct}.",
                (first.start(), first.end())
            ))
    return issues

def check_competitors(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    lower = text.lower()
    for c in rules.competitors:
        for m in re.finditer(re.escape(c.lower()), lower):
            s, e = m.span()
            issues.append(Issue(
                "Competitor Mention",
                f"Competitor mentioned: '{text[s:e]}' — consider removing or reframing.",
                (s, e),
                text[s:e]
            ))
    return issues

def check_required_phrases(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for phrase in rules.required_phrases:
        if re.search(re.escape(phrase), text, re.IGNORECASE) is None:
            issues.append(Issue(
                "Required Phrase",
                f"Consider including required phrase: “{phrase}”."
            ))
    return issues

def check_positive_cues(text: str, rules: BrandRules) -> List[Issue]:
    # gentle nudge: only the first 3 cues are “encouraged”
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
            issues.append(Issue(
                "Tone",
                f"Negative tone word: '{m.group(0)}' — ensure this fits the context.",
                m.span(),
                m.group(0)
            ))
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
        issues.append(Issue(
            "Grammar/Spelling",
            m.message,
            (s, e),
            text[s:e],
            rep
        ))
    return issues

# ---- Capitalization consistency ----
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
        issues.append(Issue(
            "Capitalization",
            f"Headline capitalization appears mixed. {examples}",
            suggestion="Use a single style (Title Case or sentence case) across headlines."
        ))
    return issues

def check_product_term_casing(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for term in rules.product_terms:
        variants = set(m.group(0) for m in re.finditer(rf"\b{term}\b", text, re.IGNORECASE))
        if len(variants) > 1:
            examples = ", ".join(sorted(variants))
            issues.append(Issue(
                "Capitalization",
                f"Inconsistent casing for '{term}': {examples}",
                suggestion=f"Use '{term}' consistently."
            ))
    return issues

# ---------- Always-visible renderer ----------
def render_issues(issues: List[Issue]):
    if not issues:
        st.success("✅ Looks on-brand. No issues found.")
        return
    st.warning(f"⚠ {len(issues)} issues found:")
    for idx, i in enumerate(issues, 1):
        st.markdown(f"**{idx}. {i.category}** — {i.message}")
        if i.snippet:
            st.code(i.snippet)
        if i.suggestion:
            st.markdown(f"*Suggestion:* **{i.suggestion}**")
        st.markdown("---")

# ---------- App UI ----------
st.set_page_config(page_title="FlavCity Brand Ad Checker", page_icon="✅", layout="wide")
st.title("FlavCity Brand Ad Checker")
st.caption("Copy-only version (no OCR). Shows all issues inline and keeps them visible after checks.")

# Keep results between reruns so they don't disappear
if "issues_last" not in st.session_state:
    st.session_state["issues_last"] = []
if "text_last" not in st.session_state:
    st.session_state["text_last"] = ""

with st.sidebar:
    st.header("Settings")
    brand_name = st.text_input("Brand Name", value="FlavCity")
    competitors = st.text_area("Competitor names (one per line)", value="\n".join(DEFAULT_COMPETITORS))
    required_phrases = st.text_area("Required phrases (one per line, optional)", value="")
    positive_cues = st.text_area("Positive brand cues (top 3 nudged)", value="\n".join(DEFAULT_POSITIVE_CUES))
    negative_tone = st.text_area("Negative tone words (warn)", value="\n".join(DEFAULT_NEGATIVE_TONE))
    product_terms = st.text_input("Product terms to enforce capitalization (comma-separated)", value="Latte,Mocha,Macchiato")
    need_reg_mark = st.checkbox("Require ® on first 'FlavCity' usage", value=False)
    st.markdown("---")
    st.write(f"Grammar checker: {'✅ enabled' if LT_AVAILABLE else '❌ unavailable'}")

rules = BrandRules(
    brand_name=brand_name,
    allowed_spellings=[brand_name],
    competitors=[c.strip() for c in competitors.splitlines() if c.strip()],
    required_phrases=[p.strip() for p in required_phrases.splitlines() if p.strip()],
    positive_cues=[p.strip() for p in positive_cues.splitlines() if p.strip()],
    negative_tone=[n.strip() for n in negative_tone.splitlines() if n.strip()],
    require_registered_mark=need_reg_mark,
    product_terms=[t.strip() for t in product_terms.split(",") if t.strip()],
)

tab, = st.tabs(["📝 Check Copy"])

with tab:
    text = st.text_area(
        "Paste ad copy",
        height=220,
        placeholder="e.g., Bye, Bye, Barista. Say hello to instant lattes made with real food ingredients."
    )

    if st.button("Run Checks", type="primary"):
        all_issues: List[Issue] = []
        all_issues += check_brand_name(text, rules)
        all_issues += check_competitors(text, rules)
        all_issues += check_required_phrases(text, rules)
        all_issues += check_positive_cues(text, rules)
        all_issues += check_negative_tone(text, rules)
        all_issues += check_headline_capitalization(text)
        all_issues += check_product_term_casing(text, rules)
        all_issues += grammar_check(text)

        # Store so they remain visible after rerun
        st.session_state["issues_last"] = all_issues
        st.session_state["text_last"] = text

# Always show the latest results
st.subheader("Results")
render_issues(st.session_state["issues_last"])

st.markdown("**Highlighted Copy**")
st.code(
    highlight_text(
        st.session_state["text_last"],
        [i for i in st.session_state["issues_last"] if i.span]
    )
)
