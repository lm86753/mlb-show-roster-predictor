# MLB The Show Roster Predictor Dashboard [v5 - Redesigned]
import sys, os, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import init_db
import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime, timedelta

TEAM_COLORS = {
    "Angels": "#BA0C2F", "Astros": "#183469", "Giants": "#FD5A1E", "Dodgers": "#005A9C",
    "Braves": "#CE1141", "Phillies": "#E81828", "Orioles": "#DF4601", "Rays": "#008080",
    "Twins": "#002B5C", "Blue Jays": "#134A8E", "Red Sox": "#BD3039", "White Sox": "#27251F",
    "Yankees": "#003087", "Athletics": "#003831", "Guardians": "#0C2340", "Tigers": "#0C1B38",
    "Royals": "#004687", "Rockies": "#33006F", "Marlins": "#00A3E0", "Brewers": "#FFC72C",
    "Cardinals": "#C41E3A", "Nationals": "#AB0003", "Mets": "#002D72", "Pirates": "#27251F",
    "Padres": "#2F241D", "Rangers": "#003278", "Reds": "#C6011F", "Cubs": "#0E3386",
    "Diamondbacks": "#A71930", "Mariners": "#0C2C56",
}
RARITY_COLORS = {"Red Diamond": "#FF0044", "Diamond": "#00BFFF", "Gold": "#FFD700", "Silver": "#C0C0C0", "Bronze": "#CD7F32", "Common": "#808080"}

HITTER_STATS = [
    ("contact_right", "Con R"), ("contact_left", "Con L"),
    ("power_right", "Pow R"), ("power_left", "Pow L"),
    ("plate_vision", "Vis"), ("batting_clutch", "Clutch"),
    ("plate_discipline", "Disc"), ("speed", "Spd"),
]
PITCHER_STATS = [
    ("pitch_control", "Ctrl"), ("pitch_movement", "Mov"),
    ("pitch_velocity", "Vel"), ("pitching_clutch", "P Clutch"),
    ("stamina", "Stam"), ("k/9_r", "K/9 R"), ("k/9_l", "K/9 L"),
    ("h/9_r", "H/9 R"), ("h/9", "H/9"), ("bb/9", "BB/9"), ("bb_per_bf", "BB/9"),
]

def get_team_color(team):
    return TEAM_COLORS.get(team, "#2d3748")

def get_rarity_color(rarity):
    return RARITY_COLORS.get(rarity, "#808080")

@st.cache_data(ttl=60)
def load_card_images():
    img_cache = {}
    for img_dir in [PROJECT_ROOT / "data" / "card_images", PROJECT_ROOT / "data" / "card_images_real"]:
        if img_dir.exists():
            for f_path in sorted(img_dir.glob("*.png")):
                try:
                    with open(f_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    img_cache[f_path.stem] = f"data:image/png;base64,{b64}"
                except:
                    pass
    return img_cache

@st.cache_data(ttl=300)
def load_and_prepare_data():
    try:
        SessionLocal = init_db()
        engine = SessionLocal.kw.get('bind')
        if engine is None:
            from src.config import DB_PATH
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

        query = """
            SELECT
                p.card_uuid,
                p.player_name,
                p.mlb_player_id,
                p.current_ovr,
                p.current_rarity,
                p.predicted_ovr_delta,
                p.upgrade_probability,
                p.downgrade_probability,
                p.tier_jump_probability,
                p.sample_size_ok,
                p.attributes_json,
                p.avg_gap,
                p.direction_consensus,
                c.team,
                c.position,
                c.is_hitter
            FROM predictions p
            LEFT JOIN card_snapshots c ON p.card_uuid = c.card_uuid
            WHERE p.horizon_days = 1
        """
        df = pd.read_sql(query, engine)

        img_cache = load_card_images()
        def get_card_img(row):
            cu = row.get('card_uuid')
            return img_cache.get(cu) if cu and cu in img_cache else None

        df['card_image'] = df.apply(get_card_img, axis=1)
        return df
    except Exception as e:
        st.error(f"DB Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_update_status():
    try:
        from sqlalchemy import create_engine, text
        from src.config import DB_PATH
        engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT update_date FROM attribute_changes
                WHERE update_date IS NOT NULL
                ORDER BY update_date DESC LIMIT 1
            """)).fetchone()
        if row:
            latest = datetime.strptime(str(row[0]), "%Y-%m-%d").date()
            today = datetime.utcnow().date()
            days_since = (today - latest).days
            next_update = latest + timedelta(days=14)
            days_until = (next_update - today).days
            return {
                "latest": latest,
                "days_since": days_since,
                "next_expected": next_update,
                "days_until": days_until,
                "is_update_today": days_since == 0,
            }
    except:
        pass
    return {"latest": None, "days_since": None, "next_expected": None, "days_until": None, "is_update_today": False}

def get_rarity_order(r):
    order = {"Common": 0, "Bronze": 1, "Silver": 2, "Gold": 3, "Diamond": 4, "Red Diamond": 5}
    return order.get(r, -1)

def format_delta(delta):
    if pd.isna(delta): return "\u2014"
    return f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"

def delta_color(delta):
    if pd.isna(delta): return "#a0aec0"
    if delta > 0: return "#48bb78"
    if delta < 0: return "#f56565"
    return "#a0aec0"

def parse_attributes(attributes_json_str, is_hitter):
    if pd.isna(attributes_json_str) or not attributes_json_str:
        return []
    try:
        data = json.loads(attributes_json_str) if isinstance(attributes_json_str, str) else attributes_json_str
    except:
        return []
    if isinstance(data, dict):
        stats_order = HITTER_STATS if is_hitter == 1 else PITCHER_STATS
        rows = []
        for attr_name, label in stats_order:
            val = data.get(attr_name, None)
            if val is None:
                alt = attr_name.replace('batting_clutch', 'pitching_clutch')
                if alt != attr_name:
                    val = data.get(alt, None)
            if val is not None and val > 0:
                rows.append((label, val, '\u2014', '\u2014', 0))
        return rows
    if isinstance(data, list):
        attr_dict = {}
        for item in data:
            name = item.get('attribute_name', '')
            if name:
                attr_dict[name] = item
        stats_order = HITTER_STATS if is_hitter == 1 else PITCHER_STATS
        rows = []
        for attr_name, label in stats_order:
            item = attr_dict.get(attr_name)
            if item is None and 'clutch' in attr_name:
                item = attr_dict.get(attr_name.replace('batting_clutch', 'pitching_clutch'))
            if item:
                current = item.get('rating_before', '\u2014')
                projected = item.get('projected_rating', current)
                delta_val = item.get('gap', None)
                if delta_val is None:
                    delta_val = item.get('predicted_delta', None)
                if delta_val is not None:
                    delta_str = format_delta(float(delta_val))
                    delta_float = float(delta_val)
                else:
                    delta_str = "\u2014"
                    delta_float = 0
                rows.append((label, current, projected, delta_str, delta_float))
        return rows
    return []

def compute_stat_data_quality(attrs_json):
    """Compute a score 0-1 for how much stat data supports the prediction."""
    if pd.isna(attrs_json) or not attrs_json:
        return 0, "No data"
    try:
        data = json.loads(attrs_json) if isinstance(attrs_json, str) else attrs_json
        if isinstance(data, list) and len(data) > 0:
            has_data = sum(1 for a in data if a.get('has_stat_data', 0))
            return has_data / len(data), f"{has_data}/{len(data)} attrs"
        return 0, "No attrs"
    except:
        return 0, "Error"

def render_player_card(row, img_cache):
    player_name = row.get('player_name', 'Unknown')
    team = row.get('team', '')
    position = row.get('position', '')
    card_uuid = row.get('card_uuid', '')
    current_ovr = row.get('current_ovr', 0)
    predicted_delta = row.get('predicted_ovr_delta', 0)
    current_rarity = row.get('current_rarity', 'Common')
    upgrade_prob = row.get('upgrade_probability', 0)
    downgrade_prob = row.get('downgrade_probability', 0)
    tier_jump_prob = row.get('tier_jump_probability', 0)
    is_hitter = row.get('is_hitter', 1)
    attributes_json_str = row.get('attributes_json')

    team_color = get_team_color(team)
    rarity_color = get_rarity_color(current_rarity)
    delta_str = format_delta(predicted_delta)
    delta_col = delta_color(predicted_delta)
    new_ovr = round(current_ovr + predicted_delta) if not pd.isna(predicted_delta) else current_ovr

    if upgrade_prob > 0.75 and predicted_delta > 1.0:
        signal, signal_color = "BUY", "#48bb78"
    elif downgrade_prob > 0.75 and predicted_delta < -1.0:
        signal, signal_color = "SELL", "#f56565"
    else:
        signal, signal_color = "HOLD", "#ed8936"

    roi_pct = (predicted_delta * 100 / current_ovr) if not pd.isna(predicted_delta) and current_ovr else 0
    roi_color = "#48bb78" if roi_pct > 0 else ("#f56565" if roi_pct < 0 else "#a0aec0")

    card_img_b64 = img_cache.get(card_uuid) if card_uuid else None

    # Stat data quality
    data_quality, quality_label = compute_stat_data_quality(attributes_json_str)
    quality_color = "#48bb78" if data_quality > 0.7 else ("#ed8936" if data_quality > 0.3 else "#f56565")

    if card_img_b64:
        card_html = f"""
        <div style="background:#1a202c url('{card_img_b64}') no-repeat center center / cover;border:1px solid {team_color}44;border-radius:10px;overflow:hidden;width:100%;aspect-ratio:363/512;transition:all 0.2s ease;position:relative;"
             onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 8px 25px {team_color}44';this.style.borderColor='{team_color}88'"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 2px 8px rgba(0,0,0,0.3)';this.style.borderColor='{team_color}44'">
            <div style="position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,0.1) 40%,rgba(0,0,0,0.65) 100%);"></div>
            <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{rarity_color};z-index:2;"></div>
            <div style="position:absolute;top:8px;left:10px;display:flex;align-items:center;gap:4px;z-index:2;">
                <span style="font-weight:600;font-size:11px;color:#f7fafc;text-shadow:0 1px 8px rgba(0,0,0,0.9);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{player_name}</span>
                <span style="font-size:9px;color:#cbd5e0;text-shadow:0 1px 8px rgba(0,0,0,0.9);">\u2022</span>
                <span style="font-size:9px;color:#cbd5e0;text-shadow:0 1px 8px rgba(0,0,0,0.9);white-space:nowrap;">{team}</span>
                <span style="font-size:9px;color:#cbd5e0;text-shadow:0 1px 8px rgba(0,0,0,0.9);">\u2022</span>
                <span style="font-size:9px;color:#cbd5e0;text-shadow:0 1px 8px rgba(0,0,0,0.9);">{position}</span>
            </div>
            <div style="position:absolute;top:8px;right:10px;display:flex;gap:4px;z-index:2;">
                <span style="background:{rarity_color}55;color:{rarity_color};padding:2px 8px;border-radius:8px;font-size:9px;font-weight:700;text-transform:uppercase;text-shadow:0 1px 4px rgba(0,0,0,0.6);">{current_rarity}</span>
            </div>
            <div style="position:absolute;bottom:10px;left:10px;display:flex;align-items:baseline;gap:8px;z-index:2;">
                <span style="font-size:48px;font-weight:900;color:#f7fafc;text-shadow:0 2px 14px rgba(0,0,0,0.95);line-height:1;">{current_ovr}</span>
                <span style="font-size:26px;font-weight:700;color:{delta_col};text-shadow:0 2px 12px rgba(0,0,0,0.95);">{delta_str}</span>
                <span style="font-size:18px;font-weight:600;color:#a0aec0;text-shadow:0 2px 12px rgba(0,0,0,0.95);">\u2192</span>
                <span style="font-size:26px;font-weight:700;color:#f7fafc;text-shadow:0 2px 12px rgba(0,0,0,0.95);">{new_ovr}</span>
                <span style="margin-left:8px;background:{roi_color}44;color:{roi_color};padding:2px 8px;border-radius:8px;font-size:11px;font-weight:700;text-shadow:0 1px 4px rgba(0,0,0,0.6);">{roi_pct:+.1f}%</span>
                <span style="background:{signal_color}55;color:{signal_color};padding:2px 8px;border-radius:8px;font-size:11px;font-weight:700;text-shadow:0 1px 4px rgba(0,0,0,0.6);">{signal}</span>
            </div>
        </div>
        """
    else:
        rarity_bg = f"linear-gradient(135deg, {rarity_color}33, {rarity_color}55)"
        card_html = f"""
        <div style="background:#1a202c;border:1px solid {team_color}44;border-radius:10px;overflow:hidden;height:100%;display:flex;flex-direction:column;transition:all 0.2s ease;position:relative;"
             onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 8px 25px {team_color}44';this.style.borderColor='{team_color}88'"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 2px 8px rgba(0,0,0,0.3)';this.style.borderColor='{team_color}44'">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{rarity_color};"></div>
            <div style="padding:10px 10px 6px 10px;display:flex;gap:10px;align-items:center;">
                <div style="width:40px;height:56px;border-radius:4px;flex-shrink:0;background:{rarity_bg};display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:800;color:{rarity_color};">
                    {current_ovr}
                </div>
                <div style="min-width:0;flex:1;">
                    <div style="font-weight:700;font-size:13px;color:#f7fafc;line-height:1.2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{player_name}</div>
                    <div style="font-size:10px;color:#a0aec0;margin-top:1px;">{team} \u2022 {position}</div>
                </div>
                <div style="background:{rarity_bg};color:{rarity_color};padding:1px 8px;border-radius:10px;font-size:9px;font-weight:700;text-transform:uppercase;white-space:nowrap;">{current_rarity}</div>
            </div>
            <div style="padding:0 10px 6px 10px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:24px;font-weight:800;color:#f7fafc;line-height:1;">{current_ovr}</span>
                    <span style="font-size:16px;font-weight:700;color:{delta_col};">{delta_str}</span>
                    <span style="font-size:12px;font-weight:600;color:#a0aec0;">\u2192</span>
                    <span style="font-size:16px;font-weight:700;color:#f7fafc;">{new_ovr}</span>
                    <div style="margin-left:auto;display:flex;gap:3px;">
                        <span style="background:{roi_color}18;color:{roi_color};padding:1px 6px;border-radius:8px;font-size:9px;font-weight:700;">{roi_pct:+.1f}%</span>
                        <span style="background:{signal_color}18;color:{signal_color};padding:1px 6px;border-radius:8px;font-size:9px;font-weight:700;">{signal}</span>
                    </div>
                </div>
            </div>
        </div>
        """
    return card_html

def render_detail_expander(player_row, df_filtered):
    card_uuid = player_row.get('card_uuid', '')
    st.markdown(f"**Player:** {player_row.get('player_name', 'N/A')}")
    st.markdown(f"**Team:** {player_row.get('team', 'N/A')}  \u2022 **Position:** {player_row.get('position', 'N/A')}")
    st.markdown(f"**Current OVR:** {player_row.get('current_ovr', 'N/A')}  \u2022 **Rarity:** {player_row.get('current_rarity', 'N/A')}")

    delta = player_row.get('predicted_ovr_delta', 0)
    upgrade_prob = player_row.get('upgrade_probability', 0)
    downgrade_prob = player_row.get('downgrade_probability', 0)
    tier_jump_prob = player_row.get('tier_jump_probability', 0)
    current_ovr = player_row.get('current_ovr', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        dc = delta_color(delta)
        st.markdown(f"<span style='color:{dc};font-size:24px;font-weight:800;'>{format_delta(delta)}</span>", unsafe_allow_html=True)
        st.caption("Predicted OVR Change")
    with col2:
        up_color = "#48bb78" if upgrade_prob > 0.5 else "#a0aec0"
        st.markdown(f"<span style='color:{up_color};font-size:24px;font-weight:800;'>{upgrade_prob*100:.1f}%</span>", unsafe_allow_html=True)
        st.caption("Upgrade Probability")
    with col3:
        dn_color = "#f56565" if downgrade_prob > 0.5 else "#a0aec0"
        st.markdown(f"<span style='color:{dn_color};font-size:24px;font-weight:800;'>{downgrade_prob*100:.1f}%</span>", unsafe_allow_html=True)
        st.caption("Downgrade Probability")

    if tier_jump_prob > 0:
        st.markdown(f"**Tier Jump Probability:** {tier_jump_prob*100:.1f}%")

    if not pd.isna(delta):
        qs_tiers = [(0, 25), (65, 100), (75, 300), (85, 1000), (90, 5000), (95, 10000)]
        ovr = player_row.get('current_ovr', 0)
        cur_qs = max((v for k, v in qs_tiers if ovr >= k), default=0)
        new_ovr = max(0, min(99, int(ovr + round(delta))))
        new_qs = max((v for k, v in qs_tiers if new_ovr >= k), default=0)
        profit = new_qs - cur_qs
        total_profit = profit * 20
        st.markdown("---")
        st.markdown(f"**QS Value:** {cur_qs:,} \u2192 {new_qs:,} stubs")
        st.markdown(f"**Profit/card:** {profit:,} stubs  \u2022 **Total (x20):** {total_profit:,} stubs")
        roi = (profit / cur_qs * 100) if cur_qs else 0
        st.markdown(f"**ROI:** {roi:+.1f}%")

    # Attribute projections table — editable
    attrs_json = player_row.get('attributes_json')
    if attrs_json and isinstance(attrs_json, str):
        try:
            attrs = json.loads(attrs_json)
            if isinstance(attrs, list) and attrs:
                st.markdown("---")
                st.markdown("**Attribute Projections** *(edit Projected values to recalculate)*")
                attr_data = []
                for item in attrs[:15]:
                    name = item.get('attribute_name', '').replace('_', ' ').title()
                    before = item.get('rating_before', 0)
                    projected = item.get('projected_rating', 0)
                    has_stat = item.get('has_stat_data', 0)
                    if projected == '\u2014' or pd.isna(projected):
                        projected = before
                    attr_data.append({
                        'Stat': name,
                        'Current': int(before),
                        'Projected': int(round(projected)),
                        'Data': '\U0001F7E2' if has_stat else '\u26AA',
                    })

                df_attrs = pd.DataFrame(attr_data)
                editor_key = f"attr_editor_{card_uuid}"

                # Pre-fill with session state edits from last run
                edit_state_key = f"edit_state_{card_uuid}"
                if edit_state_key not in st.session_state:
                    st.session_state[edit_state_key] = {}
                for i, row in df_attrs.iterrows():
                    stat_name = row['Stat']
                    if stat_name in st.session_state[edit_state_key]:
                        df_attrs.at[i, 'Projected'] = st.session_state[edit_state_key][stat_name]

                edited_df = st.data_editor(
                    df_attrs,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Stat": st.column_config.TextColumn("Stat", disabled=True, width="medium"),
                        "Current": st.column_config.NumberColumn("Current", disabled=True, width="small"),
                        "Projected": st.column_config.NumberColumn("Projected", min_value=0, max_value=99, step=1, width="small"),
                        "Data": st.column_config.TextColumn("Data", disabled=True, width="small"),
                    },
                    key=editor_key,
                )

                # Persist edits to session state and show recalculated overall
                deltas = []
                for i, row in edited_df.iterrows():
                    orig_proj = int(round(attrs[i].get('projected_rating', 0))) if i < len(attrs) else row['Projected']
                    if orig_proj == '\u2014' or pd.isna(orig_proj):
                        orig_proj = int(attrs[i].get('rating_before', 0))
                    if row['Projected'] != orig_proj:
                        st.session_state[edit_state_key][row['Stat']] = row['Projected']
                        before = attrs[i].get('rating_before', 0)
                        deltas.append(row['Projected'] - before)

                if deltas:
                    avg_delta = sum(deltas) / len(deltas)
                    recalc_ovr = max(0, min(99, int(round(current_ovr + avg_delta))))
                    dc2 = delta_color(avg_delta)
                    delta_str2 = format_delta(avg_delta)
                    st.markdown(f"<div style='background:#2d3748;border:1px solid #4a5568;border-radius:8px;padding:10px;text-align:center;margin-top:8px;'>"
                                f"<span style='color:#a0aec0;font-size:12px;'>Recalculated Overall</span><br>"
                                f"<span style='font-size:28px;font-weight:900;color:#f7fafc;'>{current_ovr}</span> "
                                f"<span style='font-size:20px;font-weight:700;color:{dc2};'>{delta_str2}</span> "
                                f"<span style='font-size:14px;color:#a0aec0;'>\u2192</span> "
                                f"<span style='font-size:28px;font-weight:900;color:#f7fafc;'>{recalc_ovr}</span>"
                                f"</div>", unsafe_allow_html=True)
                    with st.expander("Edit details"):
                        st.caption(f"Avg \u0394: {avg_delta:+.2f} across {len(deltas)} edited attributes")
                        for i, row in edited_df.iterrows():
                            orig_proj = int(round(attrs[i].get('projected_rating', 0))) if i < len(attrs) else row['Projected']
                            if orig_proj == '\u2014' or pd.isna(orig_proj):
                                orig_proj = int(attrs[i].get('rating_before', 0))
                            if row['Projected'] != orig_proj:
                                st.markdown(f"- {row['Stat']}: {row['Current']} \u2192 **{row['Projected']}** (orig: {orig_proj})")
                else:
                    st.caption("\U0001F7E2 = stat data available  \u26AA = formula only")
        except:
            pass

    # Mismatch scores
    mismatch_scores = []
    if attrs_json and isinstance(attrs_json, str):
        try:
            attrs = json.loads(attrs_json)
            if isinstance(attrs, list):
                for item in attrs:
                    if 'mismatch_score' in item and item['mismatch_score'] != 0:
                        mismatch_scores.append(float(item['mismatch_score']))
        except:
            pass
    if mismatch_scores:
        avg_mm = sum(mismatch_scores) / len(mismatch_scores)
        st.markdown(f"**Mismatch Score:** {avg_mm:+.3f} (avg of {len(mismatch_scores)} attrs)")
        st.caption("Formula-vs-rating gap (positive = underrated, negative = overrated)")

    st.markdown(f"**Avg Gap:** {player_row.get('avg_gap', 0):.2f}")
    st.markdown(f"**Direction Consensus:** {player_row.get('direction_consensus', 0):.2f}")

def main():
    st.set_page_config(page_title="MLB The Show 26 Roster Predictor", layout="wide")

    st.markdown("""
    <style>
    .main { background-color: #0f1419; }
    .stApp { background-color: #0f1419; }
    [data-testid="stSidebar"] { background-color: #1a202c; }
    .stSelectbox > div > div { background-color: #2d3748; color: #f7fafc; }
    .stTextInput > div > div > input { background-color: #2d3748; color: #f7fafc; }
    .stMultiSelect > div > div { background-color: #2d3748; color: #f7fafc; }
    h1, h2, h3 { color: #f7fafc !important; }
    .stMarkdown { color: #e2e8f0; }
    div[data-testid="stButton"] button { font-size: 12px !important; padding: 4px 12px !important; }
    .row-widget.stColumns > div { padding: 0 4px !important; }
    div[data-testid="stExpander"] { border: 1px solid #2d3748 !important; border-radius: 8px !important; }
    div[data-testid="stExpander"] summary { background: #1a202c !important; border-radius: 8px !important; }
    .stMetric { background: #1a202c; padding: 8px; border-radius: 8px; border: 1px solid #2d3748; }
    .stMetric label { color: #a0aec0 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #f7fafc !important; }
    </style>
    """, unsafe_allow_html=True)

    # Top header with update status
    status = load_update_status()
    col_title, col_status, col_meta = st.columns([2, 2, 1])
    with col_title:
        st.title("\U000026BE MLB The Show 26 Roster Predictor")
        st.caption("Real card images \u2022 Grid view v5 \u2022 MLB 26 Live Series")
    with col_status:
        if status["latest"]:
            if status["is_update_today"]:
                st.markdown(f"<div style='background:#48bb7822;border:1px solid #48bb78;border-radius:8px;padding:8px;text-align:center;'><span style='color:#48bb78;font-weight:700;'>\U0001F7E2 Update Day!</span><br><span style='color:#a0aec0;font-size:11px;'>Expected weekly on Thursdays</span></div>", unsafe_allow_html=True)
            else:
                next_str = f"{status['days_until']}d" if status["days_until"] is not None else "?"
                st.markdown(f"<div style='background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:8px;'><span style='color:#a0aec0;font-size:11px;'>Last Update</span><br><span style='color:#f7fafc;font-weight:700;'>{status['latest']}</span><br><span style='color:#a0aec0;font-size:11px;'>{status['days_since']}d ago \u2022 Next ~{next_str}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#1a202c;border:1px solid #2d3748;border-radius:8px;padding:8px;text-align:center;'><span style='color:#a0aec0;'>No update data yet</span></div>", unsafe_allow_html=True)
    with col_meta:
        st.metric("MLB 26 Updates", "15 so far", help="Roster updates this season")
        st.caption("Last refresh: just now")

    with st.spinner("Loading predictions..."):
        df = load_and_prepare_data()
        img_cache = load_card_images()

    if df.empty:
        st.warning("No data available. Run the prediction pipeline first.")
        return

    # Sidebar filters
    st.sidebar.header("Filters & Sort")
    search_text = st.sidebar.text_input("Search Player / Team", placeholder="Type to filter...")

    delta_range = st.sidebar.slider("OVR Delta Range", -15.0, 15.0, (-15.0, 15.0), step=0.25, key="delta_range")

    sort_options = {
        "Overall (Current)": "current_ovr",
        "Predicted Delta": "predicted_ovr_delta",
        "Upgrade Probability": "upgrade_probability",
        "Downgrade Probability": "downgrade_probability",
        "Player Name": "player_name",
        "Direction Consensus": "direction_consensus",
        "Avg Gap": "avg_gap",
    }
    sort_label = st.sidebar.selectbox("Sort By", list(sort_options.keys()), index=0)
    sort_col = sort_options[sort_label]
    sort_ascending = st.sidebar.checkbox("Ascending", value=(sort_col == "player_name"))

    teams = sorted([t for t in df['team'].dropna().unique() if t])
    selected_teams = st.sidebar.multiselect("Teams", teams, default=[])
    rarities = sorted([r for r in df['current_rarity'].dropna().unique() if r], key=get_rarity_order)
    selected_rarities = st.sidebar.multiselect("Rarities", rarities, default=[])

    def compute_change_prob(attrs_json):
        if pd.isna(attrs_json) or not attrs_json:
            return 0.0
        try:
            data = json.loads(attrs_json) if isinstance(attrs_json, str) else attrs_json
            if isinstance(data, list) and len(data) > 0:
                probs = [float(a.get('change_prob', 0)) for a in data if 'change_prob' in a]
                return sum(probs) / len(probs) if probs else 0.0
            return 0.0
        except:
            return 0.0

    df['change_prob'] = df['attributes_json'].apply(compute_change_prob)

    change_prob_range = st.sidebar.slider("Change Probability", 0.0, 1.0, (0.0, 1.0), step=0.05, key="change_prob_range")

    dir_consensus_range = st.sidebar.slider("Direction Consensus", -1.0, 1.0, (-1.0, 1.0), step=0.05, key="dir_range", help="-1=all down, 1=all up")

    cols_per_row = st.sidebar.selectbox("Columns per Row", [2, 3, 4], index=1)
    PAGE_SIZE = st.sidebar.selectbox("Cards per Page", [12, 24, 48, 96], index=1)

    # Filter
    df_filtered = df.copy()
    if search_text:
        mask = (
            df_filtered['player_name'].str.contains(search_text, case=False, na=False) |
            df_filtered['team'].str.contains(search_text, case=False, na=False)
        )
        df_filtered = df_filtered[mask]
    if selected_teams:
        df_filtered = df_filtered[df_filtered['team'].isin(selected_teams)]
    if selected_rarities:
        df_filtered = df_filtered[df_filtered['current_rarity'].isin(selected_rarities)]
    df_filtered = df_filtered[
        (df_filtered['predicted_ovr_delta'] >= delta_range[0]) &
        (df_filtered['predicted_ovr_delta'] <= delta_range[1]) &
        (df_filtered['change_prob'] >= change_prob_range[0]) &
        (df_filtered['change_prob'] <= change_prob_range[1]) &
        (df_filtered['direction_consensus'] >= dir_consensus_range[0]) &
        (df_filtered['direction_consensus'] <= dir_consensus_range[1])
    ]
    if sort_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by=sort_col, ascending=sort_ascending)

    total = len(df_filtered)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    if "page" not in st.session_state:
        st.session_state.page = 1
    st.session_state.page = max(1, min(st.session_state.page, total_pages))

    # Quick stats summary bar
    up_count = len(df_filtered[df_filtered['predicted_ovr_delta'] > 0.5])
    dn_count = len(df_filtered[df_filtered['predicted_ovr_delta'] < -0.5])
    st.markdown(f"<div style='display:flex;gap:12px;margin-bottom:8px;flex-wrap:wrap;'>"
                f"<span style='background:#1a202c;padding:4px 12px;border-radius:8px;border:1px solid #2d3748;color:#a0aec0;font-size:12px;'><strong style='color:#f7fafc;'>{total}</strong> players</span>"
                f"<span style='background:#48bb7818;padding:4px 12px;border-radius:8px;border:1px solid #48bb7844;color:#48bb78;font-size:12px;'>\u25b2 <strong>{up_count}</strong> upgrades</span>"
                f"<span style='background:#f5656518;padding:4px 12px;border-radius:8px;border:1px solid #f5656544;color:#f56565;font-size:12px;'>\u25bc <strong>{dn_count}</strong> downgrades</span>"
                f"</div>", unsafe_allow_html=True)

    start_idx = (st.session_state.page - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total)
    page_df = df_filtered.iloc[start_idx:end_idx]

    # Top info bar (showing range, NO pagination)
    st.markdown(f"<div style='color:#718096;font-size:12px;margin-bottom:6px;'>Showing {start_idx+1}\u2013{end_idx} of {total}</div>", unsafe_allow_html=True)

    if page_df.empty:
        st.info("No players match the current filters.")
        return

    # Player grid
    for i in range(0, len(page_df), cols_per_row):
        row_players = page_df.iloc[i:i+cols_per_row]
        cols = st.columns(cols_per_row)
        for j, (_, player_row) in enumerate(row_players.iterrows()):
            with cols[j]:
                card_html = render_player_card(player_row, img_cache)
                st.markdown(card_html, unsafe_allow_html=True)
                with st.expander("Details & Projections"):
                    render_detail_expander(player_row, df_filtered)

    # ━━━ BOTTOM PAGINATION (moved from top) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("<hr style='border-color:#2d3748;margin:16px 0 8px 0;'>", unsafe_allow_html=True)
    if total_pages > 1:
        pag_col1, pag_col2, pag_col3, pag_col4, pag_col5 = st.columns([1, 3, 4, 3, 1])
        with pag_col1:
            if st.session_state.page > 1:
                st.button("\u00ab First", key="first_btn", use_container_width=True,
                         on_click=lambda: setattr(st.session_state, 'page', 1))
        with pag_col2:
            if st.session_state.page > 1:
                st.button("\u25c0 Previous", key="prev_btn", use_container_width=True,
                         on_click=lambda: setattr(st.session_state, 'page', st.session_state.page - 1))
        with pag_col3:
            st.markdown(f"<div style='text-align:center;color:#a0aec0;font-size:14px;padding-top:6px;'><strong>{st.session_state.page}</strong> / {total_pages}</div>", unsafe_allow_html=True)
        with pag_col4:
            if st.session_state.page < total_pages:
                st.button("Next \u25b6", key="next_btn", use_container_width=True,
                         on_click=lambda: setattr(st.session_state, 'page', st.session_state.page + 1))
        with pag_col5:
            if st.session_state.page < total_pages:
                st.button("Last \u00bb", key="last_btn", use_container_width=True,
                         on_click=lambda: setattr(st.session_state, 'page', total_pages))
    else:
        st.markdown(f"<div style='text-align:center;color:#718096;font-size:13px;'>Page 1 / 1</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='text-align:center;color:#4a5568;font-size:11px;margin-top:4px;'>Showing {start_idx+1}\u2013{end_idx} of {total} players</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.caption("Data from MLB The Show 26 roster updates \u2022 Real card images from The Show CDN \u2022 Predictions are formula-based estimates, not financial advice")

if __name__ == "__main__":
    main()
