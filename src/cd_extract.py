#!/usr/bin/env python3
"""Per-channel city extraction. The original regex required a 'City of death:' prefix and so
captured only burial+scaffold (25% of draws); the other six channels use different formats:
  plain / smallplace : the whole draw IS the city
  deathrec           : 'Place of death: X'
  hospital/dateanchor: 'City of death (<parenthetical>): X'
  obit               : prose, 'died ... in X'
"""
import re
KEYED=re.compile(r'(?:City of death|Death city|Place of death|Died at)\s*(?:\([^)]*\))?\s*:\s*([^\n|]+)',re.I)
PROSE=re.compile(r'\bdied\b[^.\n]{0,60}?\bin\s+([A-ZÀ-Ý][\w\.\'’\-]*(?:\s+[A-ZÀ-Ý][\w\.\'’\-]*){0,2})',re.I)
BAD={'n/a','na','none','unknown','alive','still alive','n/a, still alive',''}
def city_from(ch,text):
    t=str(text).strip()
    if not t: return None
    if ch in ('plain','smallplace'):
        v=t.split('\n')[0].split('|')[0].strip()
    else:
        m=KEYED.search(t)
        if m: v=m.group(1)
        elif ch=='obit':
            m=PROSE.search(t)
            if not m: return None
            v=m.group(1)
        else: return None
    v=v.split(',')[0].strip().rstrip('.').strip()
    if v.lower() in BAD or len(v)>40 or not v: return None
    if re.fullmatch(r'[\d\s\-\/]+',v): return None
    return v
