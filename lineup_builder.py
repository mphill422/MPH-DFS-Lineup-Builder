"""
MPH DFS — MLB Lineup Builder  (v1, CSV-only)
============================================
WHAT THIS IS
------------
Your hour of manual lineup work, automated and auditable. It does NOT out-project
RotoWire. It takes their player pool CSV and applies YOUR named gates as hard
rules on top, shows its work, and gives you 2-3 lineup constructions to choose
from. You make every final call.

This is the "Option A" build we agreed on: RotoWire projection is the base, your
gates ADJUST it. No from-scratch projection engine.

WHAT'S IN v1 (all from the CSV, no API needed)
----------------------------------------------
PITCHERS — 4 gates:
  1. Opponent quality  : opponent's implied team runs (looked up by OPP, FIXED so
     a pitcher on a high-scoring team is NOT punished for his own offense — the
     Skenes bug). Lower opponent total = better spot.
  2. Quality/durability: IP, xFIP, K%  (true skill, not one lucky box score).
  3. Park/weather      : wind OUT/IN, precip, temp — PLUS GB/FB. A fly-ball
     pitcher in a hot/altitude run environment is downgraded (the Coors read).
  4. Win lean (+4)     : low opponent total => more likely his team wins => +4 DK
     bonus. (Real moneyline is a phase-2 upgrade.)
  + Three CONSTRUCTIONS surfaced: A) two studs  B) stud+value  C) two value.

BATS — form leads, team total amplifies, cold bats capped:
  - Individual form (L7 FPTS) + AVG + vs-hand OPS + batting-order slot LEAD.
  - Team implied total only AMPLIFIES a good bat (max ~25%); it cannot rescue a
    cold one. A slumping bat (cold L7 AND low AVG) is hard-capped so the team
    total can't float him over hot bats in good matchups (the .180-Yankee fix).
  - 7-8-9 batting slots de-prioritized by default (your rule); 7th allowed back
    in as a cheap punt if salary needs it.

LINEUP ASSEMBLY:
  - Fills a DK classic roster (2 P, 1 C, 1 1B, 1 2B, 1 3B, 1 SS, 3 OF) under the
    $50,000 cap, maximizing the GATE-ADJUSTED projection.
  - Outputs the 3 constructions so you can compare spend-up-on-arms vs save-for-bats.
  - Respects max 3 hitters per team (stack discipline) and excludes 7-8-9 slots
    (toggle to allow 7th as punt).

HONEST LIMITS (read this)
-------------------------
- This is v1 on CSV data only. It is NOT proven — track its lineups vs your own
  vs RotoWire for a few weeks before trusting it with real sizing.
- It can't see: late scratches (e.g. a star sat today), opponent strikeout rate,
  xwOBA / barrel% / hard-hit% (contact quality that says if a hot streak is REAL).
  Those are phase-2 Savant/Odds-API upgrades. Until then, YOUR manual override on
  breaking news and contact quality is still the edge. Override freely.
- L7 form is a noisy, small sample. Here it's used as a COLD-BAT VETO, not a
  promoter, on purpose. xwOBA (phase 2) is what will validate hot streaks.

USAGE
-----
    pip install streamlit pandas pulp
    streamlit run lineup_builder.py
Drop the RotoWire CSV (with the added columns: Opp ERA, T Runs, IP, xFIP, K%,
GB/FB, vH splits). Pick a construction. Audit. Override. Enter on DK yourself.
"""

import re
import pandas as pd
import streamlit as st

try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False

# ============================================================
# TUNABLES
# ============================================================
PITCHER_WEIGHTS = {"opponent": 0.28, "quality": 0.42, "weather": 0.15, "winprob": 0.15}
BAT_INDIV_WEIGHTS = {"form": 0.30, "avg": 0.20, "split": 0.30, "slot": 0.20}
TEAM_AMP_SHARE = 0.25          # team total contributes at most this much to a non-cold bat
COLD_L7 = 6.0                  # below this L7 FPTS = cold
COLD_AVG = 0.230               # AND below this AVG = cold -> hard cap
COLD_CAP = 45.0                # capped score ceiling for cold bats
SALARY_CAP = 50000
MAX_HITTERS_PER_TEAM = 3
CLOSE_CALL_PTS = 4.0

DK_SLOTS = ["P", "P", "C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]

st.set_page_config(page_title="MPH Lineup Builder", layout="wide")


# ============================================================
# PARSE HELPERS
# ============================================================
def num(x, d=None):
    if x is None:
        return d
    s = str(x).strip()
    if s in ("", "-", "--", "nan"):
        return d
    try:
        return float(s)
    except ValueError:
        return d


def parse_wind(w):
    if not w:
        return None, ""
    m = re.match(r"([\d.]+)\s*(.*)", str(w).strip())
    if not m:
        return None, ""
    return num(m.group(1)), (m.group(2) or "").strip()


def scale(v, lo, hi, inv=False):
    if v is None:
        return 50.0
    v = max(lo, min(hi, v))
    p = (v - lo) / (hi - lo) if hi != lo else 0.5
    return round((1 - p if inv else p) * 100, 1)


def build_team_runs(df):
    m = {}
    for _, r in df.iterrows():
        t = str(r.get("TEAM", "")).strip()
        v = num(r.get("T Runs"))
        if t and v is not None:
            m[t] = v
    return m


def positions_of(pos_str):
    """'2B/OF' -> ['2B','OF']; pitchers handled separately."""
    return [p.strip() for p in str(pos_str).split("/") if p.strip()]


# ============================================================
# PITCHER GATES
# ============================================================
def pitcher_gates(r, team_runs):
    opp = str(r.get("OPP", "")).strip()
    opp_runs = team_runs.get(opp)

    g_opp = scale(opp_runs, 2.8, 6.0, inv=True)

    ip, xfip, kpct = num(r.get("IP")), num(r.get("xFIP")), num(r.get("K%"))
    g_qual = round(scale(ip, 50, 110) * 0.30
                   + scale(xfip, 2.2, 5.2, inv=True) * 0.40
                   + scale(kpct, 14, 32) * 0.30, 1)

    sp, d = parse_wind(r.get("WIND"))
    precip, temp = num(r.get("PRECIP %")), num(r.get("TEMP"))
    gbfb = num(r.get("GB/FB"))
    base, wbits = 60.0, []
    d = d.lower()
    if "out" in d and sp:
        base -= min(30, sp * 1.6); wbits.append(f"⚠wind OUT {sp:.0f}")
    elif "in" in d and sp:
        base += min(20, sp * 1.2); wbits.append(f"wind IN {sp:.0f}")
    elif sp:
        wbits.append(f"cross {sp:.0f}")
    # Coors/altitude+heat fly-ball read: fly-ball pitcher (low GB/FB) in hot env = worse
    if gbfb is not None and gbfb < 1.0 and ((temp and temp >= 85) or (opp_runs and opp_runs >= 5)):
        base -= 8; wbits.append(f"⚠FB-pitcher in hi-run env (GB/FB {gbfb:.2f})")
    elif gbfb is not None and gbfb >= 1.4:
        base += 4; wbits.append(f"GB lean {gbfb:.2f}")
    if precip is not None and precip >= 50:
        base -= 10; wbits.append(f"⚠{precip:.0f}% precip")
    g_wx = round(max(0, min(100, base)), 1)

    g_win = scale(opp_runs, 3.0, 5.5, inv=True)

    composite = round(g_opp * PITCHER_WEIGHTS["opponent"]
                      + g_qual * PITCHER_WEIGHTS["quality"]
                      + g_wx * PITCHER_WEIGHTS["weather"]
                      + g_win * PITCHER_WEIGHTS["winprob"], 1)

    why = {
        "opp": f"opp {opp} implied {opp_runs:.1f} runs" if opp_runs is not None else "opp total n/a",
        "qual": ", ".join(b for b in [f"{ip:.0f}IP" if ip else None,
                                       f"{xfip:.2f}xFIP" if xfip else None,
                                       f"{kpct:.0f}%K" if kpct else None] if b),
        "wx": ", ".join(wbits) if wbits else "neutral",
        "win": "win-lean (opp scores little)" if (opp_runs and opp_runs <= 3.5)
               else ("⚠shootout risk" if (opp_runs and opp_runs >= 5.0) else "neutral"),
    }
    return composite, dict(opp=g_opp, qual=g_qual, wx=g_wx, win=g_win), why, opp_runs


# ============================================================
# BAT GATES  (form leads, team amplifies, cold capped)
# ============================================================
def bat_gates(r, team_runs):
    l7, avg, vops = num(r.get("L7 FPTS")), num(r.get("AVG")), num(r.get("vH OPS"))
    slot = str(r.get("LINEUP", "")).strip()
    t = str(r.get("TEAM", "")).strip()
    tr = team_runs.get(t)

    s_form = scale(l7, 3, 18)
    s_avg = scale(avg, 0.190, 0.320)
    s_split = scale(vops, 0.550, 1.000)
    s_slot = 100 if slot in ("1", "2", "3", "4") else (70 if slot == "5"
             else (45 if slot in ("6", "7") else 20))
    indiv = round(s_form * BAT_INDIV_WEIGHTS["form"]
                  + s_avg * BAT_INDIV_WEIGHTS["avg"]
                  + s_split * BAT_INDIV_WEIGHTS["split"]
                  + s_slot * BAT_INDIV_WEIGHTS["slot"], 1)
    amp = scale(tr, 3.0, 6.5)

    cold = (l7 is not None and l7 < COLD_L7) and (avg is not None and avg < COLD_AVG)
    if cold:
        score = round(min(indiv, COLD_CAP) * 0.9, 1)
        flag = "COLD-CAP"
    else:
        score = round(indiv * (1 - TEAM_AMP_SHARE) + amp * TEAM_AMP_SHARE, 1)
        flag = ""
    why = (f"L7 {l7:.1f}, AVG {avg:.3f}, vHand OPS "
           f"{vops:.3f}, slot {slot}, team {tr:.1f} runs"
           if all(v is not None for v in [l7, avg, vops, tr]) else "partial data")
    return score, indiv, amp, cold, flag, why


# ============================================================
# RANKERS
# ============================================================
def rank_pitchers(df, team_runs):
    rows = []
    for _, r in df.iterrows():
        comp, gates, why, opp_runs = pitcher_gates(r, team_runs)
        sal = num(r.get("SAL"), 0)
        rows.append({
            "Pitcher": r.get("PLAYER"), "Tm": r.get("TEAM"), "OppTeam": r.get("OPP"),
            "OppRuns": opp_runs, "Sal": int(sal) if sal else None,
            "RW": num(r.get("FPTS"), 0), "SCORE": comp,
            "Val": round(comp / (sal / 1000), 2) if sal else 0,
            "OppG": gates["opp"], "Qual": gates["qual"], "Wx": gates["wx"], "Win": gates["win"],
            "why": why,
        })
    return pd.DataFrame(rows).sort_values("SCORE", ascending=False).reset_index(drop=True)


def rank_bats(df, team_runs, allow_7th=False):
    rows = []
    for _, r in df.iterrows():
        slot = str(r.get("LINEUP", "")).strip()
        if slot == "BN":
            continue
        if slot in ("8", "9"):
            continue
        if slot == "7" and not allow_7th:
            continue
        sc, indiv, amp, cold, flag, why = bat_gates(r, team_runs)
        sal = num(r.get("SAL"), 0)
        rows.append({
            "Player": r.get("PLAYER"), "Tm": r.get("TEAM"), "Pos": r.get("POS"),
            "Slot": slot, "Sal": int(sal) if sal else None, "RW": num(r.get("FPTS"), 0),
            "L7": num(r.get("L7 FPTS")), "AVG": num(r.get("AVG")),
            "SCORE": sc, "flag": flag, "why": why,
        })
    return pd.DataFrame(rows).sort_values("SCORE", ascending=False).reset_index(drop=True)


# ============================================================
# LINEUP ASSEMBLY (salary-cap optimize on gate-adjusted score)
# ============================================================
def adjusted_proj(rw_fpts, gate_score):
    """RotoWire base nudged by gate score. gate 50 = neutral (x1.0), 100 = +25%, 0 = -25%."""
    if rw_fpts is None:
        return 0
    mult = 1 + (gate_score - 50) / 200.0  # 0->0.75x, 50->1.0x, 100->1.25x
    return round(rw_fpts * mult, 2)


def build_lineup(pitchers_df, bats_df, forced_pitchers=None):
    """MILP: pick 2 P + C/1B/2B/3B/SS + 3 OF under cap, max adjusted proj.
    forced_pitchers: list of pitcher names to lock (for the 3 constructions)."""
    if not HAS_PULP:
        return None, "pulp not installed"

    # build player universe
    players = []
    for _, r in pitchers_df.iterrows():
        players.append({"name": r["Pitcher"], "team": r["Tm"], "sal": r["Sal"] or 99999,
                        "proj": adjusted_proj(r["RW"], r["SCORE"]), "pos": ["P"]})
    for _, r in bats_df.iterrows():
        players.append({"name": r["Player"], "team": r["Tm"], "sal": r["Sal"] or 99999,
                        "proj": adjusted_proj(r["RW"], r["SCORE"]), "pos": positions_of(r["Pos"])})

    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    # decision var per (player, eligible-slot)
    x = {}
    for i, p in enumerate(players):
        for slot in set(p["pos"]) & set(DK_SLOTS):
            x[(i, slot)] = pulp.LpVariable(f"x_{i}_{slot}", cat="Binary")

    # objective
    prob += pulp.lpSum(x[(i, s)] * players[i]["proj"] for (i, s) in x)
    # exactly the slot counts
    from collections import Counter
    need = Counter(DK_SLOTS)
    for slot, cnt in need.items():
        prob += pulp.lpSum(x[(i, s)] for (i, s) in x if s == slot) == cnt
    # each player at most once
    for i in range(len(players)):
        vs = [x[(i, s)] for (i, s) in x if s == "P" or True if (i, s) in x]
        own = [x[(i, s)] for (j, s) in x if j == i]
        if own:
            prob += pulp.lpSum(own) <= 1
    # salary cap
    prob += pulp.lpSum(x[(i, s)] * players[i]["sal"] for (i, s) in x) <= SALARY_CAP
    # max hitters per team (non-pitcher slots)
    teams = set(p["team"] for p in players)
    for tm in teams:
        prob += pulp.lpSum(x[(i, s)] for (i, s) in x
                           if players[i]["team"] == tm and s != "P") <= MAX_HITTERS_PER_TEAM
    # forced pitchers
    if forced_pitchers:
        for fp in forced_pitchers:
            idxs = [i for i, p in enumerate(players) if p["name"] == fp]
            if idxs:
                i = idxs[0]
                if (i, "P") in x:
                    prob += x[(i, "P")] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None, f"no optimal lineup ({pulp.LpStatus[prob.status]})"

    chosen = []
    for (i, s) in x:
        if x[(i, s)].value() == 1:
            chosen.append({"slot": s, **players[i]})
    order = {s: k for k, s in enumerate(DK_SLOTS)}
    chosen.sort(key=lambda c: order.get(c["slot"], 99))
    total_sal = sum(c["sal"] for c in chosen)
    total_proj = round(sum(c["proj"] for c in chosen), 1)
    return {"players": chosen, "salary": total_sal, "proj": total_proj}, None


# ============================================================
# UI
# ============================================================
st.title("⚾ MPH MLB Lineup Builder")
st.caption("v1 — your gates on RotoWire's pool. Ranks, builds 3 constructions, you decide. "
           "Not proven yet — track vs your own picks before trusting it.")

uploaded = st.file_uploader("Drop the RotoWire player-pool CSV", type=["csv"])
if uploaded is None:
    st.info("Export the slate from RotoWire (with added columns: Opp ERA, T Runs, IP, "
            "xFIP, K%, GB/FB, vH AVG/OPS) and drop it here.")
    st.stop()

raw = pd.read_csv(uploaded)
team_runs = build_team_runs(raw)
pitchers_raw = raw[(raw["POS"].astype(str).str.strip() == "P") &
                   (raw["LINEUP"].astype(str).str.strip() == "SP")].copy()
allow_7th = st.checkbox("Allow 7th batting-order slot (cheap punt)", value=False)

if pitchers_raw.empty:
    st.error("No starting pitchers found (POS=P, LINEUP=SP).")
    st.stop()

pr = rank_pitchers(pitchers_raw, team_runs)
br = rank_bats(raw, team_runs, allow_7th=allow_7th)

# ---- pitcher board ----
st.subheader("Pitchers — ranked by your gates")
st.dataframe(pr[["Pitcher", "Tm", "OppTeam", "OppRuns", "Sal", "RW", "SCORE", "Val",
                 "OppG", "Qual", "Wx", "Win"]],
             use_container_width=True, hide_index=True)

with st.expander("Why each pitcher ranks where it does"):
    for i, r in pr.iterrows():
        st.write(f"**#{i+1} {r['Pitcher']} (SCORE {r['SCORE']}, ${r['Sal']}, RW {r['RW']})** — "
                 f"Opp: {r['why']['opp']} · Qual: {r['why']['qual']} · "
                 f"Wx: {r['why']['wx']} · Win: {r['why']['win']}")

# ---- three constructions ----
st.subheader("Three pitcher constructions")
studs = pr.head(2)["Pitcher"].tolist()
stud_one = pr.head(1)["Pitcher"].tolist()
# value = best Val score among cheaper arms (sal <= 7000)
value_pool = pr[pr["Sal"] <= 7000].sort_values("SCORE", ascending=False)
value_arms = value_pool["Pitcher"].head(2).tolist()
stud_plus_value = stud_one + (value_pool["Pitcher"].head(1).tolist())

constructions = {
    "A) Two studs": studs,
    "B) Stud + value": stud_plus_value if len(stud_plus_value) == 2 else studs,
    "C) Two value": value_arms if len(value_arms) == 2 else studs,
}
choice = st.radio("Pick a pitcher construction to build around:",
                  list(constructions.keys()), index=1)
forced = constructions[choice]
st.write(f"Locking pitchers: **{', '.join(forced)}**")

# ---- bats board ----
st.subheader("Bats — form leads, team total amplifies, cold bats capped")
st.dataframe(br[["Player", "Tm", "Pos", "Slot", "Sal", "RW", "L7", "AVG", "SCORE", "flag"]],
             use_container_width=True, hide_index=True)
st.caption("COLD-CAP = slumping bat held down so a hot team total can't float it up (the .180 fix).")

# ---- assemble ----
st.subheader("Built lineup")
if not HAS_PULP:
    st.warning("Install `pulp` to enable salary-cap lineup assembly: pip install pulp")
else:
    lineup, err = build_lineup(pr, br, forced_pitchers=forced)
    if err:
        st.error(f"Couldn't build: {err}. Try a different construction or allow the 7th slot.")
    else:
        rows = [{"Slot": c["slot"], "Player": c["name"], "Team": c["team"],
                 "Sal": c["sal"], "AdjProj": c["proj"]} for c in lineup["players"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.write(f"**Salary:** ${lineup['salary']:,} / ${SALARY_CAP:,}  ·  "
                 f"**Adj projection:** {lineup['proj']}")
        st.caption("AdjProj = RotoWire FPTS nudged ±25% by your gate score. This is the tool's "
                   "lineup — override any spot freely before entering on DK.")

st.divider()
st.caption("Phase 2 (next): wire Baseball Savant (xwOBA, barrel%, opp K%) + Odds API moneyline "
           "to sharpen the gates. Until then: override on breaking news (scratches) and contact quality.")
