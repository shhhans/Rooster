"""Deterministic safety screening for roster-planning requests and model output.

This is a lexical guardrail, not a semantic security boundary. It catches a
small set of known hazardous patterns and must not be treated as proof that
unmatched content is safe.
"""
from __future__ import annotations

import re
from typing import Any

SAFETY_VERSION = "2026-10-17.2"

# Each rule intentionally requires both a topic and an actionable/deceptive
# cue where feasible. This avoids blocking neutral discussion of policy.
_RULES: dict[str, tuple[re.Pattern[str], ...]] = {
    "medication_preflight": (
        re.compile(r"(?<![a-z])(mel\s*a\s*tonin|medication|medicine|sleeping\s+pill|sedative)(?![a-z])|安眠\s*药|褪\s*黑\s*素|药\s*物|镇静\s*剂", re.I),
        re.compile(r"\b(take|use|recommend|suggest|give|dose|should|before|within|approach|landing|takeoff|departure|flight)\b|起飞|进近|降落|航班|服用|建议|吃", re.I),
    ),
    "rest_shortening": (
        re.compile(r"(rest|休息)", re.I),
        re.compile(r"(shorten|reduce|cut|less|override|ignore|压缩|缩短|减少|少休|降低)", re.I),
    ),
    "rest_start_forgery": (
        re.compile(r"(rest|休息)", re.I),
        re.compile(r"(start|begin|started|开始|起始)", re.I),
        re.compile(r"(fake|forge|fabricat|pretend|backdat|invent|伪造|编造|假装|篡改)|(change|alter|move|adjust|correct).{0,60}(make.{0,15}look longer|look longer|longer|extend|increase|更长|延长|改长)", re.I),
    ),
    "unconfirmed_inflight": (
        re.compile(r"(in.?flight|airborne|during (the )?flight|aircraft segment|飞行中|机上|飞机航段)", re.I),
        re.compile(r"(assume|treat as|mark as|record as|claim|confirm|assumed|视为|标记为|记录为|确认|假定)", re.I),
        re.compile(r"(unconfirmed|unknown|not confirmed|未确认|未知|没有确认)", re.I),
    ),
    "sleep_fact_forgery": (
        re.compile(r"(sleep|slept|睡眠|睡觉)", re.I),
        re.compile(r"(fake|forge|fabricat|invent|claim|record|mark|pretend|伪造|编造|捏造|记录|标记|假称)", re.I),
        re.compile(r"(actual|fact|occurred|not reported|did not report|未报告|没报告|已发生|实际|事实|真实)", re.I),
    ),
    "hide_conflict": (
        re.compile(r"(conflict|overlap|clash|冲突|重叠)", re.I),
        re.compile(r"(hide|conceal|suppress|omit|remove|ignore|mention|隐藏|隐瞒|掩盖|删除|忽略|提及|提(?!交)|显示)", re.I),
    ),
    "authority_override": (
        re.compile(r"(ignore|disregard|override|forget|bypass|忽略|无视|覆盖|绕过|忘记)", re.I),
        re.compile(r"(previous|prior|system|developer|policy|instruction|rules?|之前|系统|开发者|策略|指令|规则)", re.I),
    ),
    "pressure_override": (
        re.compile(r"(ignore|override|bypass|skip|不要管|忽略|绕过|跳过)", re.I),
        re.compile(r"(safe|safety|legal|rest|fatigue|安全|合法|休息|疲劳)", re.I),
        re.compile(r"(urgent|deadline|boss|captain|job|fire|emergency|紧急|老板|机长|工作|解雇|赶时间)", re.I),
    ),
    "time_version_spoof": (
        re.compile(r"(time|date|clock|version|时间|日期|时钟|版本)", re.I),
        re.compile(r"(pretend|claim|set|change|override|spoof|fake|当作|声称|设置|修改|覆盖|伪造)", re.I),
        re.compile(r"(now|today|current|utc|system|现在|今天|当前|系统)", re.I),
    ),
    "safety_guarantee": (
        re.compile(r"(guarantee|certify|ensure|assure|definitely safe|100% safe|保证|担保|确保|绝对安全|百分百安全)", re.I),
        re.compile(r"(safe|legal|compliant|fit to fly|fit for duty|安全|合法|合规|适航|适岗)", re.I),
    ),
}

_NEGATION = re.compile(
    r"\b(don't|do not|doesn't|does not|never|must not|should not|cannot|can't|won't|without)\b|"
    r"(不要|不应|不能|不会|禁止|切勿|不可)", re.I
)
_QUESTION_ONLY = re.compile(r"\?|\b(what are|explain|discuss|question about|policy for|is it allowed)\b|关于.*(规则|政策|规定)", re.I)
_ACTION_CUE = re.compile(r"\b(please|help|plan|recommend|suggest|take|use|mark|record|hide|ignore|override|pretend|claim|set|make|tell|do|mention)\b|请|帮我|安排|建议|服用|标记|记录|隐藏|忽略|覆盖|假装|声称|设置|告诉|提及|显示", re.I)

def _matching_rules(text: str) -> list[str]:
    # Keep negation local to a clause: a refusal followed by a harmful
    # recommendation must still be caught.
    clauses = re.split(r"(?<=[.!?。！？;；])\s*|\b(?:but|still|then|however)\b|然后|但是|接着|却", text, flags=re.I)
    found = [code for clause in clauses for code in _matching_rules_clause(clause)]
    # Pressure indicators often sit in one clause and the requested bypass in
    # the next; evaluate that specific combined pattern across the sentence.
    if "pressure_override" in _matching_rules_clause(text):
        found.append("pressure_override")
    return list(dict.fromkeys(found))


def _matching_rules_clause(text: str) -> list[str]:
    # Collapse simple obfuscation of drug names (e.g. "mel a tonin") while
    # retaining word boundaries for ordinary English terms.
    medication_text = re.sub(r"(?<=mel)\s+(?=a)|(?<=a)\s+(?=tonin)", "", text, flags=re.I)
    if re.search(r"\bMT\b", text, re.I) and re.search(r"\b(?:2\s*h(?:ours?)?|two\s+hours?)\b", text, re.I) and re.search(r"\b(?:duty|flight|takeoff|approach|landing)\b|飞行|值勤|起飞|进近|降落", text, re.I):
        medication_text = re.sub(r"\bMT\b", "melatonin", medication_text, flags=re.I)
    found: list[str] = []
    for code, parts in _RULES.items():
        target_text = medication_text if code == "medication_preflight" else text
        if all(pattern.search(target_text) for pattern in parts):
            if code == 'safety_guarantee':
                positive = re.sub(r'(?:不能|无法|不会|不予|不作|不)(?:做出|作出)?(?:保证|担保|确保)|\b(?:cannot|can\x27t|do not|don\x27t|will not|won\x27t)\s+(?:guarantee|certify|ensure|assure)\b', '', text, flags=re.I)
                if not all(pattern.search(positive) for pattern in parts):
                    continue
            # Avoid common direct prohibitions. A matching phrase embedded in
            # a longer request still gets evaluated if it has an action cue.
            conceal_request = code == "hide_conflict" and re.search(
                r"(不要|别|不).{0,30}提(?!交).{0,40}(冲突|重叠)", text)
            if _NEGATION.search(text) and not _ACTION_CUE.search(text) and not conceal_request:
                continue
            if code in {"authority_override", "pressure_override", "time_version_spoof"} and _QUESTION_ONLY.search(text) and not _ACTION_CUE.search(text):
                continue
            if _QUESTION_ONLY.search(text) and re.search(r"(rule|policy|allowed|legal|guidance|规定|规则|政策|允许|指导)", text, re.I) and not re.search(r"(tell me to|recommend|suggest|plan|mark as|record as|hide|ignore|override|伪造|建议我|安排)", text, re.I):
                continue
            if code == "rest_shortening" and re.search(r"(don't|do not|never|can't|cannot|must not|should not|不要|不应|不能|禁止).{0,40}(shorten|reduce|cut|缩短|减少)", text, re.I):
                continue
            if code == "hide_conflict" and re.search(r"(do not|don't|never|不要|别|不).{0,30}(mention|提及|提|显示).{0,40}(conflict|overlap|clash|commitment|冲突|重叠|承诺)", text, re.I):
                # Here the negated verb requests concealment ("do not mention").
                pass
            elif code == "hide_conflict" and re.search(r"(don't|do not|never|can't|cannot|must not|should not|不要|不应|不能|禁止).{0,40}(hide|conceal|隐藏|隐瞒)", text, re.I):
                continue
            if code == "hide_conflict" and re.search(r"避免(?:隐藏|隐瞒)(?:冲突|重叠)", text):
                remaining = re.sub(r"避免(?:隐藏|隐瞒)(?:冲突|重叠)", "", text)
                if not all(pattern.search(remaining) for pattern in parts):
                    continue
            if code == "medication_preflight" and re.search(r"(don't|do not|never|can't|cannot|must not|should not|不要|不应|不能|禁止).{0,30}(take|use|recommend|suggest|服用|建议|吃)", text, re.I):
                continue
            if code == "unconfirmed_inflight" and re.search(r"\bplease confirm\b.{0,60}\bunknown\b", text, re.I) and not re.search(r"(assume|treat as|mark as|record as|视为|标记为|记录为)", text, re.I):
                continue
            if code == "unconfirmed_inflight":
                remaining = re.sub(r"未包含未确认的?机上睡眠", "", text)
                if not all(pattern.search(remaining) for pattern in parts):
                    continue
            found.append(code)
    return found

def _result(codes: list[str], *, evidence: list[str] | None = None) -> dict[str, Any]:
    codes = list(dict.fromkeys(codes))
    out: dict[str, Any] = {
        "blocked": bool(codes),
        "targets": codes.copy(),
        "reason_codes": codes,
        "message": ("I can’t help with the unsafe or deceptive part of this request. I can still help with a safe roster plan using confirmed facts and applicable rest requirements." if codes else "No known unsafe pattern was detected."),
    }
    if evidence is not None:
        out["evidence"] = evidence
    return out

def assess_request(text: str, snapshot: Any = None) -> dict[str, Any]:
    """Screen request text; snapshot is accepted for API stability, not trusted."""
    del snapshot
    if not isinstance(text, str):
        return _result([])
    return _result(_matching_rules(text))

_OUTPUT_FIELDS = {"action", "recommendation", "explanation", "negotiation", "sleepplan"}
_QUOTE_FIELDS = {"user_quote", "userquote", "quoted_user_text", "quotedusertext", "exact_quote", "exactquote", "quote", "quoted_text", "quotedtext"}
_DIAGNOSTIC_FIELDS = {"diagnostic", "diagnostics", "observation", "observations", "trace", "debug", "evidence"}
_METADATA_FIELDS = {"type", "kind", "id", "target", "reasoncode", "reasoncodes", "status"}

def _output_text(value: Any, key: str = "", in_request: bool = False) -> list[str]:
    """Collect only user-facing planning fields; omit quoted inputs/diagnostics."""
    normalized = re.sub(r"[^a-z]", "", key.lower())
    if normalized in _QUOTE_FIELDS or normalized in _DIAGNOSTIC_FIELDS:
        return []
    in_request = in_request or normalized in {"request", "requests"}
    found: list[str] = []
    if isinstance(value, str):
        if normalized in _OUTPUT_FIELDS or in_request:
            found.append(value)
        return found
    if isinstance(value, dict):
        for child_key, child in value.items():
            child_norm = re.sub(r"[^a-z]", "", str(child_key).lower())
            if child_norm in _QUOTE_FIELDS or child_norm in _DIAGNOSTIC_FIELDS or child_norm in _METADATA_FIELDS:
                continue
            if child_norm in _OUTPUT_FIELDS:
                if isinstance(child, str):
                    found.append(child)
                else:
                    found.extend(_output_text(child, str(child_key), in_request))
            elif in_request and isinstance(child, str):
                found.append(child)
            elif isinstance(child, dict):
                found.extend(_output_text(child, str(child_key), in_request))
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, dict):
                        found.extend(_output_text(item, str(child_key), in_request))
                    elif isinstance(item, str) and in_request:
                        found.append(item)
    return found

def inspect_output(output: Any, snapshot: Any = None) -> dict[str, Any]:
    """Inspect designated user-facing output fields, preserving text evidence."""
    del snapshot
    codes: list[str] = []
    evidence: list[str] = []
    for text in _output_text(output):
        matched = _matching_rules(text)
        if matched:
            evidence.append(text)
            codes.extend(matched)
    return _result(codes, evidence=list(dict.fromkeys(evidence)))
