
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import streamlit as st

# Optional deps
try:
    import pytesseract  # OCR
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

try:
    import language_tool_python  # Grammar
    TOOL = language_tool_python.LanguageToolPublicAPI('en-US')
    LT_AVAILABLE = True
except Exception:
    LT_AVAILABLE = False
    TOOL = None

###############################################################################
# Configuration
###############################################################################

DEFAULT_COMPETITORS = [
    "starbucks", "dunkin", "nespresso", "peet's", "peets", "blue bottle",
    "keurig", "tim hortons", "philz", "intelligentsia", "lavazza"
]

DEFAULT_POSITIVE_CUES = [
    "whole-food", "whole food", "real-ingredient", "real ingredient", "real ingredients",
    "clean-label", "clean label",
    "busy lifestyle", "busy lifestyles",
    "exceptional nutrition", "nutrition and flavor", "FLAVOR", "flavor-forward",
    "taste phenomenal", "high-quality", "clean-label products"
]

DEFAULT_NEGATIVE_TONE = [
    "bad", "awful", "gross", "disgusting", "fake", "chemical-laden", "synthetic",
    "sorry", "apologize", "guilt", "junk"
]

DEFAULT_ALLOWED_COLORS = []  # hex strings without '#'

REQUIRE_REGISTERED_MARK = False

###############################################################################
# Utils
###############################################################################

HEX_RE = re.compile(r"#?[0-9A-Fa-f]{6}")

def hex_to_rgb(h: str):
    h = h.strip().lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def color_distance(c1, c2):
    return float(np.linalg.norm(np.array(c1) - np.array(c2)))

def dominant_colors(img: Image.Image, k: int = 6):
    small = img.convert("RGB").resize((200, 200))
    quant = small.quantize(colors=k, method=0)
    palette = quant.getpalette()
    dom = []
    for count, idx in sorted(quant.getcolors(), reverse=True):
        r,g,b = palette[idx*3: idx*3+3]
        dom.append((int(r), int(g), int(b)))
    return dom

from dataclasses import dataclass

@dataclass
class BrandRules:
    brand_name: str = "FlavCity"
    allowed_spellings: List[str] = field(default_factory=lambda: ["FlavCity"])
    competitors: List[str] = field(default_factory=lambda: DEFAULT_COMPETITORS.copy())
    required_phrases: List[str] = field(default_factory=list)
    positive_cues: List[str] = field(default_factory=lambda: DEFAULT_POSITIVE_CUES.copy())
    negative_tone: List[str] = field(default_factory=lambda: DEFAULT_NEGATIVE_TONE.copy())
    allowed_colors: List[str] = field(default_factory=lambda: DEFAULT_ALLOWED_COLORS.copy())
    require_registered_mark: bool = REQUIRE_REGISTERED_MARK
    product_terms: List[str] = field(default_factory=lambda: ["Latte","Mocha","Macchiato","Cappuccino","Espresso"])

@dataclass
class Issue:
    category: str
    message: str
    span: Optional[Tuple[int,int]] = None
    snippet: Optional[str] = None
    suggestion: Optional[str] = None

def highlight_text(text: str, issues: List[Issue]) -> str:
    spans = [(i.span[0], i.span[1]) for i in issues if i.span is not None]
    if not spans: return text
    spans.sort()
    merged = []
    for s,e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    out, last = [], 0
    for s,e in merged:
        out.append(text[last:s]); out.append("[!]"); out.append(text[s:e]); out.append("[/!]"); last = e
    out.append(text[last:])
    return "".join(out)

def find_all(pattern: re.Pattern, text: str):
    for m in pattern.finditer(text):
        yield m.start(), m.end(), m.group(0)

def check_brand_name(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    correct = rules.brand_name
    bad_variants = [r"flav\s?city", r"flavcity™", r"flavcity©", r"flav\s? city", r"flavicity"]
    for pat in bad_variants:
        for s,e,val in find_all(re.compile(pat, re.IGNORECASE), text):
            if text[s:e] != correct:
                issues.append(Issue("Brand Name", f"Incorrect brand usage: '{val}'.", (s,e), val, f"Use '{correct}' exactly."))
    if rules.require_registered_mark:
        first = re.search(re.escape(correct), text)
        if first and not re.search(re.escape(correct) + r"\s*®", text[first.start():first.end()+2]):
            issues.append(Issue("Trademark", f"Add '®' on first reference to {correct}.", (first.start(), first.end())))
    return issues

def check_competitors(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    lower = text.lower()
    for c in rules.competitors:
        for m in re.finditer(re.escape(c.lower()), lower):
            s,e = m.span()
            issues.append(Issue("Competitor Mention", f"Competitor mentioned: '{text[s:e]}' — consider removing or reframing.", (s,e), text[s:e]))
    return issues

def check_required_phrases(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for phrase in rules.required_phrases:
        if re.search(re.escape(phrase), text, re.IGNORECASE) is None:
            issues.append(Issue("Required Phrase", f"Consider including required phrase: “{phrase}”.", None, None))
    return issues

def check_positive_cues(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    found = {p for p in rules.positive_cues if re.search(re.escape(p), text, re.IGNORECASE)}
    missing = set(rules.positive_cues[:3]) - found
    for m in missing:
        issues.append(Issue("Brand Cue", f"Opportunity: consider reinforcing '{m}'.", None, None))
    return issues

def check_negative_tone(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for n in rules.negative_tone:
        for m in re.finditer(rf"\b{re.escape(n)}\b", text, re.IGNORECASE):
            issues.append(Issue("Tone", f"Negative tone word: '{m.group(0)}' — ensure this fits the context.", m.span(), m.group(0)))
    return issues

def grammar_check(text: str) -> List[Issue]:
    issues = []
    try:
        import language_tool_python  # ensure available at runtime
        TOOL = language_tool_python.LanguageToolPublicAPI('en-US')
        matches = TOOL.check(text)
    except Exception:
        matches = []
    for m in matches[:100]:
        s, e = m.offset, m.offset + m.errorLength
        rep = (m.replacements[0] if m.replacements else None)
        issues.append(Issue("Grammar/Spelling", m.message, (s,e), text[s:e], rep))
    return issues

# ---- New: Capitalization consistency ----
def is_title_like(line: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", line)]
    if not words: return False
    caps = sum(1 for w in words if w[:1].isupper())
    return caps / max(len(words),1) >= 0.7

def is_sentence_like(line: str) -> bool:
    # mostly lowercase start except proper nouns
    line_stripped = line.strip()
    if not line_stripped: return False
    first_alpha = re.search(r"[A-Za-z]", line_stripped)
    if not first_alpha: return False
    return line_stripped[first_alpha.start()].islower()

def check_headline_capitalization(text: str) -> List[Issue]:
    issues = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    titleish = [l for l in lines if is_title_like(l)]
    sentenceish = [l for l in lines if is_sentence_like(l)]
    if titleish and sentenceish:
        # both styles appear -> potential inconsistency
        examples = f"Title Case ex: “{titleish[0]}”; Sentence case ex: “{sentenceish[0]}”"
        issues.append(Issue("Capitalization", f"Headline capitalization appears mixed. {examples}", None, None,
                            "Use a single style (Title Case or Sentence case) across headlines."))
    return issues

def check_product_term_casing(text: str, rules: BrandRules) -> List[Issue]:
    issues = []
    for term in rules.product_terms:
        # collect distinct casings of the same word
        variants = set(m.group(0) for m in re.finditer(rf"\b{term}\b", text, re.IGNORECASE))
        if len(variants) > 1:
            examples = ", ".join(sorted(variants))
            issues.append(Issue("Capitalization", f"Inconsistent casing for '{term}': {examples}", None, None,
                                f"Use '{term}' consistently."))
    return issues

# ---- OCR + Color checks ----
def ocr_image(img: Image.Image) -> str:
    if not TESS_AVAILABLE: return ""
    try:
        import pytesseract
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def color_check(img: Image.Image, allowed_hex: List[str], tolerance: float = 90.0) -> List[Issue]:
    issues = []
    if not allowed_hex: return issues
    dom = dominant_colors(img, k=6)
    allowed_rgb = [hex_to_rgb(h) for h in allowed_hex if HEX_RE.fullmatch(h) or HEX_RE.fullmatch("#"+h)]
    ok = False
    for c in dom:
        distances = [color_distance(c, a) for a in allowed_rgb]
        if distances and min(distances) <= tolerance:
            ok = True; break
    if not ok:
        issues.append(Issue("Color Palette", "Dominant colors appear far from configured brand palette.", None, None))
    return issues

###############################################################################
# UI
###############################################################################

st.set_page_config(page_title="FlavCity Brand Ad Checker", page_icon="✅", layout="wide")
st.title("FlavCity Brand Ad Checker")
st.caption("Paste copy or upload an ad image to check brand compliance, capitalization consistency, grammar, and visual cues.")

with st.sidebar:
    st.header("Settings")
    brand_name = st.text_input("Brand Name", value="FlavCity")
    competitors = st.text_area("Competitor names (one per line)", value="\n".join(DEFAULT_COMPETITORS))
    required_phrases = st.text_area("Required phrases (optional, one per line)", value="")
    positive_cues = st.text_area("Positive brand cues (top 3 used as nudges)", value="\n".join(DEFAULT_POSITIVE_CUES))
    negative_tone = st.text_area("Negative tone words (warn)", value="\n".join(DEFAULT_NEGATIVE_TONE))
    palette_hex = st.text_area("Allowed brand colors (hex, one per line, optional)", value="\n".join(DEFAULT_ALLOWED_COLORS))
    product_terms = st.text_input("Product terms to enforce capitalization (comma-separated)", value="Latte,Mocha,Macchiato")
    need_reg_mark = st.checkbox("Require ® on first 'FlavCity' usage", value=False)
    st.markdown("---")
    st.write("**Optional dependencies**")
    st.write(f"OCR available: {'✅' if TESS_AVAILABLE else '❌'} | Grammar available: {'✅' if LT_AVAILABLE else '❌'}")

rules = BrandRules(
    brand_name=brand_name,
    allowed_spellings=[brand_name],
    competitors=[c.strip() for c in competitors.splitlines() if c.strip()],
    required_phrases=[p.strip() for p in required_phrases.splitlines() if p.strip()],
    positive_cues=[p.strip() for p in positive_cues.splitlines() if p.strip()],
    negative_tone=[n.strip() for n in negative_tone.splitlines() if n.strip()],
    allowed_colors=[h.strip().lstrip("#") for h in palette_hex.splitlines() if h.strip()],
    require_registered_mark=need_reg_mark,
    product_terms=[t.strip() for t in product_terms.split(",") if t.strip()],
)

tab1, tab2 = st.tabs(["📝 Check Copy", "🖼️ Check Image Ad"])

with tab1:
    text = st.text_area("Paste ad copy", height=200, placeholder="e.g., Bye, Bye, Barista. Say hello to instant lattes made with real food ingredients.")
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
        st.subheader("Results")
        if not all_issues:
            st.success("✅ Looks on-brand. No issues found.")
        else:
            st.warning(f"⚠ {len(all_issues)} issues found:")
            for i in all_issues:
                with st.expander(f"{i.category}: {i.message}"):
                    if i.snippet: st.code(i.snippet)
                    if i.suggestion: st.write("Suggestion:", f"**{i.suggestion}**")
            st.markdown("**Highlighted Copy**")
            st.code(highlight_text(text, [i for i in all_issues if i.span]))

with tab2:
    uploaded = st.file_uploader("Upload a JPG/PNG of your ad", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded ad", use_column_width=True)
        if st.button("Analyze Image", type="primary"):
            issues_img: List[Issue] = []
            extracted = ocr_image(img) if TESS_AVAILABLE else ""
            if extracted.strip():
                issues_img += check_brand_name(extracted, rules)
                issues_img += check_competitors(extracted, rules)
                issues_img += check_required_phrases(extracted, rules)
                issues_img += check_positive_cues(extracted, rules)
                issues_img += check_negative_tone(extracted, rules)
                issues_img += check_headline_capitalization(extracted)
                issues_img += check_product_term_casing(extracted, rules)
                issues_img += grammar_check(extracted)
                st.markdown("**Extracted Text (OCR)**")
                st.code(extracted.strip())
            else:
                st.info("No OCR text (install Tesseract + pytesseract to enable).")
            issues_img += color_check(img, rules.allowed_colors)
            if not issues_img:
                st.success("✅ Looks on-brand. No issues found.")
            else:
                st.warning(f"⚠ {len(issues_img)} issues found:")
                for i in issues_img:
                    with st.expander(f"{i.category}: {i.message}"):
                        if i.snippet: st.code(i.snippet)
                        if i.suggestion: st.write("Suggestion:", f"**{i.suggestion}**")

st.markdown("---")
st.caption("Notes:\n- Grammar checking uses LanguageTool if installed (`pip install language-tool-python`).\n- OCR uses Tesseract + pytesseract (install system Tesseract, then `pip install pytesseract`).\n- Color palette checking is heuristic — add your brand hex values in Settings.\n")
