# brand_ad_checker_app.py
# FlavCity Brand Ad Checker — Copy-only (no OCR)
# Channel-aware + Scoreboard + line/column pointers + grammar + spelling/typos.

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import streamlit as st

# ---------- Grammar checker ----------
try:
    import language_tool_python
    LT_TOOL = language_tool_python.LanguageToolPublicAPI("en-US")
    LT_AVAILABLE = True
except Exception:
    LT_TOOL = None
    LT_AVAILABLE = False

# ---------- Defaults ----------
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

VOICE_ENERGETIC_WORDS = [
    "bold", "vibrant", "flavorful", "excited", "enthusiastic", "joyful",
    "optimistic", "feel-good", "celebrate", "phenomenal"
]
VOICE_TRUST_WORDS = [
    "transparent", "no shortcuts", "quality first", "Bobby Approved", "crafted with care", "trust"
]
NEGATIVE_TONE = ["bad","awful","gross","disgusting","fake","junk","sorry","apologize","guilt"]
FEAR_MONGER_WORDS = ["toxic","dangerous","poison","unsafe","carcinogen","never eat","avoid at all costs","scary","harmful","chemical-laden"]
INTIMIDATING_PHRASES = ["you must","you have to","or else","never ever","don’t you dare"]

CHANNEL_PROFILES = {
    "Email": {
        "desc":"Deeper education & storytelling with a clear action.",
        "max_chars":None,"max_sentence_len":240,"require_cta":True,
        "cta_words":["shop now","learn more","get started","join the waitlist","see how"],
        "allow_emoji":True,"encourage":["education","transparency","value"],"discourage":[]
    },
    "SMS": {
        "desc":"Quick, engaging, action-driven. Personal and to the point.",
        "max_chars":300,"max_sentence_len":140,"require_cta":True,
        "cta_words":["shop now","tap to see","join now","get yours","save now"],
        "allow_emoji":True,"encourage":["urgency","value"],"discourage":["paragraphs","too much detail"]
    },
    "Social": {
        "desc":"Visual storytelling, community energy, and engagement.",
        "max_chars":400,"max_sentence_len":200,"require_cta":False,
        "cta_words":["see more","learn more","link in bio","shop now"],
        "allow_emoji":True,"encourage":["community","conversation","flavor","fun"],"discourage":["dense technical detail"]
    },
    "Website": {
        "desc":"Trust-building, transparency, and simple education.",
        "max_chars":None,"max_sentence_len":240,"require_cta":False,
        "cta_words":["add to cart","learn more","see ingredients","compare","see nutrition"],
        "allow_emoji":False,"encourage":["transparency","education","value"],"discourage":["hype without proof","confusing jargon"]
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

# ---------- Helpers ----------
def _line_starts(text: str):
    starts = [0]
    for i,ch in enumerate(text):
        if ch == "\n": starts.append(i+1)
    return starts

def _idx_to_linecol(text: str, idx: int):
    starts = _line_starts(text)
    lo, hi = 0, len(starts)-1
    while lo<=hi:
        mid=(lo+hi)//2
        if starts[mid]<=idx: lo=mid+1
        else: hi=mid-1
    line=hi+1; col=idx-starts[hi]+1
    return line,col,starts

def _line_slice(text: str, starts, line: int):
    start=starts[line-1]; end=starts[line] if line<len(starts) else len(text)
    return text[start:end]

def _with_line_numbers(text: str) -> str:
    return "".join(f"{i:>3} | {line}" for i,line in enumerate(text.splitlines(True),1))

def highlight_text(text: str, issues: List[Issue]) -> str:
    spans=[(i.span[0],i.span[1]) for i in issues if i.span]
    if not spans: return text
    spans.sort(); merged=[]
    for s,e in spans:
        if not merged or s>merged[-1][1]: merged.append([s,e])
        else: merged[-1][1]=max(merged[-1][1],e)
    out,last=[],0
    for s,e in merged:
        out.append(text[last:s]); out.append("[!]"); out.append(text[s:e]); out.append("[/!]"); last=e
    out.append(text[last:]); return "".join(out)

def find_all(pattern: re.Pattern, text: str):
    for m in pattern.finditer(text): yield m.start(),m.end(),m.group(0)

# ---------- Checks ----------
def check_brand_name(text: str, rules: BrandRules)->List[Issue]:
    issues=[]; correct=rules.brand_name
    bad=[r"flav\s?city",r"flavcity™",r"flavcity©",r"flav\s? city",r"flavicity"]
    for pat in bad:
        for s,e,val in find_all(re.compile(pat,re.IGNORECASE),text):
            if text[s:e]!=correct:
                issues.append(Issue("Brand Name",f"Incorrect brand usage: '{val}'.",(s,e),val,f"Use '{correct}' exactly."))
    return issues

def check_competitors(text: str, rules: BrandRules)->List[Issue]:
    issues=[]; low=text.lower()
    for c in rules.competitors:
        for m in re.finditer(re.escape(c.lower()),low):
            s,e=m.span()
            issues.append(Issue("Competitor Mention",f"Competitor mentioned: '{text[s:e]}' — consider removing.",(s,e),text[s:e]))
    return issues

def check_positive_cues(text: str, rules: BrandRules)->List[Issue]:
    issues=[]; found={p for p in rules.positive_cues if re.search(re.escape(p),text,re.IGNORECASE)}
    missing=set(rules.positive_cues[:3])-found
    for m in missing: issues.append(Issue("Brand Cue",f"Opportunity: reinforce '{m}'."))
    return issues

def check_voice_alignment(text: str, rules: BrandRules)->List[Issue]:
    issues=[]
    if not any(re.search(re.escape(w),text,re.IGNORECASE) for w in rules.energetic_words):
        issues.append(Issue("Voice Cue","Could use more enthusiasm.",suggestion="Try: bold, vibrant, joyful."))
    if not any(re.search(re.escape(w),text,re.IGNORECASE) for w in rules.trust_words):
        issues.append(Issue("Voice Cue","Could use more trust/transparency cues.",suggestion="Try: transparent, crafted with care."))
    return issues

def check_negative_tone(text: str, rules: BrandRules)->List[Issue]:
    issues=[]
    for n in rules.negative_tone:
        for m in re.finditer(rf"\b{re.escape(n)}\b",text,re.IGNORECASE):
            issues.append(Issue("Tone",f"Negative tone word: '{m.group(0)}'",m.span(),m.group(0)))
    return issues

def grammar_check(text: str, rules: BrandRules|None=None)->List[Issue]:
    issues=[]
    # LanguageTool
    if LT_AVAILABLE:
        try: matches=LT_TOOL.check(text)
        except Exception: matches=[]
        for m in matches[:120]:
            s,e=m.offset,m.offset+m.errorLength
            rep=m.replacements[0] if m.replacements else None
            issues.append(Issue("Grammar/Spelling",m.message,(s,e),text[s:e],rep))
    # Spellchecker
    try:
        from spellchecker import SpellChecker
        sp=SpellChecker()
        allow={"FlavCity","Bobby","Approved","Bobby-Approved","Latte","Mocha","Macchiato","Cappuccino","Espresso"}
        if rules: allow.add(rules.brand_name); [allow.add(t) for t in rules.product_terms]
        tokens=[(m.group(0),m.start(),m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z'-]*",text)]
        unknown=sp.unknown([w.lower() for w,_,_ in tokens])
        seen=set()
        for word,s,e in tokens:
            wl=word.lower()
            if wl in {a.lower() for a in allow}: continue
            if wl in unknown:
                if (wl,s) in seen: continue
                seen.add((wl,s))
                sug=None
                try: sug=next(iter(sp.candidates(wl)))
                except Exception: pass
                issues.append(Issue("Grammar/Spelling","Possible misspelling.",(s,e),word,sug))
    except Exception: pass
    return issues

# ---------- Channel-specific ----------
def count_chars(text:str)->int: return len(text.replace("\n",""))
def longest_sentence_len(text:str)->int:
    sents=re.split(r"[.!?]\s+|\n+",text.strip())
    return max((len(s) for s in sents if s),default=0)
def has_cta(text:str,cta:list)->bool: return any(phrase in text.lower() for phrase in cta)

def check_channel_rules(text:str,profile:dict)->List[Issue]:
    issues=[]
    if profile["max_chars"] and count_chars(text)>profile["max_chars"]:
        issues.append(Issue("Channel: Length","Too long for this channel."))
    if profile["max_sentence_len"] and longest_sentence_len(text)>profile["max_sentence_len"]:
        issues.append(Issue("Channel: Readability","Some sentences are too long."))
    if profile["require_cta"] and not has_cta(text,profile["cta_words"]):
        issues.append(Issue("Channel: CTA","Missing a clear call-to-action."))
    return issues

# ---------- Scoreboard ----------
def compute_scores(issues: List[Issue])->dict:
    weights={"brand":35,"voice":25,"grammar":15,"channel":25}
    penalties={"brand":0,"voice":0,"grammar":0,"channel":0}
    for i in issues:
        cat=i.category.lower()
        if cat.startswith("brand"): penalties["brand"]+=10
        elif "competitor" in cat: penalties["brand"]+=8
        elif "voice" in cat or "tone" in cat: penalties["voice"]+=6
        elif "grammar" in cat: penalties["grammar"]+=5
        elif cat.startswith("channel"): penalties["channel"]+=6
    subs={k:max(0,w-penalties[k]) for k,w in weights.items()}
    return {"subs":subs,"total":sum(subs.values()),"weights":weights}

def render_scoreboard(scores:dict):
    subs=scores["subs"]; total=scores["total"]
    st.metric("Overall",f"{total}/100"); st.progress(total/100)
    st.write(f"Brand {subs['brand']}/{scores['weights']['brand']} | Voice {subs['voice']}/{scores['weights']['voice']} | Grammar {subs['grammar']}/{scores['weights']['grammar']} | Channel {subs['channel']}/{scores['weights']['channel']}")

# ---------- Renderer ----------
def render_issues(issues: List[Issue], source_text:str):
    if not issues: st.success("✅ Looks on-brand."); return
    st.warning(f"⚠ {len(issues)} issues found:")
    _,_,starts=_idx_to_linecol(source_text,0)
    for idx,i in enumerate(issues,1):
        st.markdown(f"**{idx}. {i.category}** — {i.message}")
        if i.span:
            s,e=i.span; line,col,_=_idx_to_linecol(source_text,s)
            st.caption(f"Line {line}, Col {col}")
            line_text=_line_slice(source_text,starts,line)
            ls,le=s-starts[line-1],e-starts[line-1]
            marked=line_text[:ls]+"[!]"+line_text[ls:le]+"[/!]"+line_text[le:]
            st.code(marked.rstrip("\n"))
        if i.suggestion: st.markdown(f"*Suggestion:* {i.suggestion}")
        st.markdown("---")

# ---------- App ----------
st.set_page_config(page_title="FlavCity Brand Ad Checker",page_icon="✅",layout="wide")
st.title("FlavCity Brand Ad Checker")

if "issues_last" not in st.session_state: st.session_state["issues_last"]=[]
if "text_last" not in st.session_state: st.session_state["text_last"]=""
if "channel_last" not in st.session_state: st.session_state["channel_last"]="Email"
if "scores_last" not in st.session_state: st.session_state["scores_last"]=compute_scores([])

with st.sidebar:
    st.header("Channel")
    channel=st.selectbox("Where will this run?",options=list(CHANNEL_PROFILES.keys()),
                         index=list(CHANNEL_PROFILES.keys()).index(st.session_state["channel_last"]))
    st.caption(CHANNEL_PROFILES[channel]["desc"])
    st.write(f"Grammar checker: {'✅' if LT_AVAILABLE else '❌'}")

rules=BrandRules()

tab,=st.tabs(["📝 Check Copy"])
with tab:
    text=st.text_area("Paste ad copy",height=220,
        placeholder="e.g., Bye, Bye, Barista. Say hello to instant lattes made with real food ingredients.")
    if st.button("Run Checks",type="primary"):
        profile=CHANNEL_PROFILES[channel]
        all_issues=[]
        all_issues+=check_brand_name(text,rules)
        all_issues+=check_competitors(text,rules)
        all_issues+=check_positive_cues(text,rules)
        all_issues+=check_voice_alignment(text,rules)
        all_issues+=check_negative_tone(text,rules)
        all_issues+=grammar_check(text,rules)
        all_issues+=check_channel_rules(text,profile)
        st.session_state["issues_last"]=all_issues
        st.session_state["text_last"]=text
        st.session_state["channel_last"]=channel
        st.session_state["scores_last"]=compute_scores(all_issues)

render_scoreboard(st.session_state["scores_last"])
st.subheader("Results")
render_issues(st.session_state["issues_last"],st.session_state["text_last"])
st.markdown("**Highlighted Copy**")
st.code(_with_line_numbers(highlight_text(st.session_state["text_last"],[i for i in st.session_state["issues_last"] if i.span])))
