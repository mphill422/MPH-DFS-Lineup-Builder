"""
savant.py — Baseball Savant / FanGraphs enrichment for the MPH Lineup Builder
=============================================================================
PHASE 2. Pulls advanced metrics the RotoWire CSV doesn't carry and exposes them
to the gates:

  - OPPONENT TEAM K% ......... pitcher Gate 1 (the COL-whiffs read)
  - player xwOBA ............. bat form gate (is a hot streak REAL?)
  - player Barrel% ........... bat contact-quality gate
  - player Hard-Hit% ......... bat contact-quality gate

DESIGN PRINCIPLE — NEVER CRASH THE APP:
Every pull is wrapped. If Savant/FanGraphs is unreachable, rate-limited, or a
name doesn't match, the function returns None / empty and the caller falls back
to CSV-only behavior. Advanced metrics SHARPEN the gates when present; they are
never required. A bad Savant day must degrade to "still works on CSV," not a
crash.

⚠️ UNTESTED AGAINST LIVE DATA: the build environment cannot reach Savant/
FanGraphs (403), so name-matching and column names below are best-effort and
MUST be verified on first real deploy. See verify_savant() — run it once and
check the output before trusting any enriched gate.

Requires: pybaseball  (add to requirements.txt)
"""

import functools
import datetime as _dt

try:
    import pybaseball as _pb
    _pb.cache.enable()           # cache pulls to disk so repeat runs are fast
    HAS_PYBASEBALL = True
except Exception:
    HAS_PYBASEBALL = False

# Map RotoWire team abbreviations -> FanGraphs/Savant team abbreviations.
# RotoWire and FanGraphs mostly agree, but a few differ. Verify on first run.
TEAM_ALIAS = {
    "ATH": "OAK",   # Athletics — RotoWire 'ATH', FanGraphs often 'OAK'
    "AZ": "ARI", "ARI": "ARI",
    "CWS": "CHW", "CHW": "CHW",
    "SD": "SDP", "SDP": "SDP",
    "SF": "SFG", "SFG": "SFG",
    "TB": "TBR", "TBR": "TBR",
    "KC": "KCR", "KCR": "KCR",
    "WSH": "WSN", "WSN": "WSN",
}


def _team_key(rw_team):
    t = str(rw_team).strip().upper()
    return TEAM_ALIAS.get(t, t)


def _season():
    return _dt.date.today().year


# ============================================================
# TEAM-LEVEL: opponent strikeout rate  (cleanest — no name matching)
# ============================================================
@functools.lru_cache(maxsize=2)
def team_k_rates(season=None):
    """
    Return {team_abbr: k_pct} for all teams this season. k_pct in 0-100.
    Falls back to {} (empty) if anything goes wrong — caller then skips the gate.
    """
    if not HAS_PYBASEBALL:
        return {}
    season = season or _season()
    try:
        df = _pb.team_batting(season)
    except Exception:
        return {}
    # find a team column and a K% (or SO/PA) column defensively
    cols = {c.lower(): c for c in df.columns}
    team_col = cols.get("team") or cols.get("tm")
    if team_col is None:
        return {}
    out = {}
    if "k%" in cols:
        kcol = cols["k%"]
        for _, r in df.iterrows():
            try:
                v = float(r[kcol])
                out[str(r[team_col]).strip().upper()] = v * 100 if v < 1.5 else v
            except (TypeError, ValueError):
                continue
    elif "so" in cols and "pa" in cols:
        for _, r in df.iterrows():
            try:
                pa = float(r[cols["pa"]])
                so = float(r[cols["so"]])
                if pa > 0:
                    out[str(r[team_col]).strip().upper()] = round(so / pa * 100, 1)
            except (TypeError, ValueError):
                continue
    return out


def opponent_k_pct(rw_opp_team, season=None):
    """K% (0-100) of the offense a pitcher faces. None if unavailable."""
    rates = team_k_rates(season)
    if not rates:
        return None
    return rates.get(_team_key(rw_opp_team))


# ============================================================
# PLAYER-LEVEL: xwOBA, Barrel%, Hard-Hit%  (name matching — riskier)
# ============================================================
@functools.lru_cache(maxsize=2)
def _batting_table(season=None):
    """FanGraphs batting leaderboard with advanced metrics, keyed by lowercased name."""
    if not HAS_PYBASEBALL:
        return {}
    season = season or _season()
    try:
        df = _pb.batting_stats(season, qual=1)   # qual=1 -> include nearly everyone
    except Exception:
        return {}
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name")
    if name_col is None:
        return {}
    xwoba = cols.get("xwoba")
    barrel = cols.get("barrel%") or cols.get("barrel %")
    hardhit = cols.get("hardhit%") or cols.get("hard%") or cols.get("hardhit %")
    table = {}
    for _, r in df.iterrows():
        nm = str(r[name_col]).strip().lower()
        rec = {}
        for key, col in (("xwoba", xwoba), ("barrel", barrel), ("hardhit", hardhit)):
            if col is not None:
                try:
                    rec[key] = float(r[col])
                except (TypeError, ValueError):
                    rec[key] = None
        table[nm] = rec
    return table


def _norm_name(name):
    """Normalize for matching: lower, strip accents/punctuation."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().replace(".", "").replace("'", "").strip()


def player_metrics(rw_name, season=None):
    """
    Return {'xwoba','barrel','hardhit'} for a hitter, or {} if no confident match.
    Name-matches RotoWire name to FanGraphs name with accent/punct normalization.
    """
    table = _batting_table(season)
    if not table:
        return {}
    target = _norm_name(rw_name)
    # direct normalized match
    for nm, rec in table.items():
        if _norm_name(nm) == target:
            return rec
    # last-name + first-initial fallback (handles 'C. Sanchez' style differences)
    parts = target.split()
    if len(parts) >= 2:
        last, first_i = parts[-1], parts[0][:1]
        for nm, rec in table.items():
            np = _norm_name(nm).split()
            if len(np) >= 2 and np[-1] == last and np[0][:1] == first_i:
                return rec
    return {}


# ============================================================
# VERIFY — run once on first deploy, eyeball the output
# ============================================================
def verify_savant(sample_teams=("NYY", "COL", "ATH"), sample_players=("Aaron Judge",)):
    """
    Quick self-check to confirm Savant is reachable and matching. Returns a dict
    of results the app prints so Mike can SEE whether enrichment is live or fell
    back to CSV-only. Never raises.
    """
    out = {"pybaseball_installed": HAS_PYBASEBALL, "team_k": {}, "players": {},
           "status": "unknown"}
    if not HAS_PYBASEBALL:
        out["status"] = "pybaseball NOT installed — running CSV-only"
        return out
    rates = team_k_rates()
    if not rates:
        out["status"] = ("Savant/FanGraphs unreachable or empty — gates fall back "
                         "to CSV-only (this is safe, just no advanced metrics)")
        return out
    for t in sample_teams:
        out["team_k"][t] = opponent_k_pct(t)
    for p in sample_players:
        out["players"][p] = player_metrics(p)
    matched = sum(1 for v in out["team_k"].values() if v is not None)
    out["status"] = f"LIVE — matched {matched}/{len(sample_teams)} sample teams"
    return out
