export interface AttributeItem {
  attribute_name: string
  rating_before: number
  projected_rating: number
  /** Rating the card's stats usually earn, per the fitted projector */
  stat_projection: number | null
  predicted_delta: number
  gap: number | null
  /** 80% range for the attribute change */
  confidence_low: number
  confidence_high: number
  upgrade_prob_attr: number
  downgrade_prob_attr: number
  change_prob: number
  last_delta: number
}

export interface Prediction {
  card_uuid: string
  player_name: string
  mlb_player_id: number | null
  current_ovr: number
  current_rarity: string
  current_qs: number
  predicted_ovr_delta: number
  ovr_delta_sd: number | null
  upgrade_probability: number
  downgrade_probability: number
  tier_jump_probability: number
  tier_down_probability: number
  /** Chance the card is Gold (80+) after the update */
  gold_probability?: number
  /** Chance the card is Diamond (85+) after the update */
  diamond_probability?: number
  /** P(OVR moves by k) for k = −4…+4 */
  ovr_move_probs?: number[]
  sample_size_ok: boolean
  team: string | null
  position: string | null
  is_hitter: number | null
  attributes: AttributeItem[]
  created_at: string
  /** Probability-weighted quicksell change per card (stubs) */
  expected_value_per_card: number | null
  stats?: PlayerStats
}

export type StatLine = Record<string, number | null>

export interface PlayerStats {
  group?: 'hitting' | 'pitching'
  season?: StatLine
  last30?: StatLine
  last_season?: StatLine
}

export interface ReviewPick {
  card_uuid: string
  player_name: string
  ovr_before: number
  ovr_after: number
  predicted_delta: number
  upgrade_probability: number
  downgrade_probability: number
  expected_qs_change: number
  gold_probability?: number
  realized_qs_change: number
}

export interface UpdateStatus {
  latest: string | null
  days_since: number | null
  days_until: number | null
  next_expected: string | null
  is_update_today: boolean
  last_attribute_update?: string | null
  cadence_days?: number
}

export interface BacktestMetrics {
  ovr_spearman: number
  attr_direction_acc: number
  interval_coverage: number
  upgrade_brier: number
  upgrade_brier_baseline: number
  upgrade_prob_mean: number
  base_rate_up: number
  base_rate_down: number
  top25_up_hit_rate: number
  top25_up_avg_ovr_delta: number
  top25_down_hit_rate: number
  top25_ev_avg_qs_gain: number
  all_avg_qs_gain: number
  ovr_mae?: number
  downgrade_prob_mean?: number
  ovr_mae_baseline?: number
  downgrade_brier?: number
  downgrade_brier_baseline?: number
  tier_up_brier?: number
  tier_up_brier_baseline?: number
  tier_up_prob_mean?: number
  base_rate_tier_up?: number
  /** Share of OVR changes inside the card's 80% range */
  ovr_interval_coverage?: number
  qs_ev_mae?: number
  qs_ev_mae_baseline?: number
  qs_ev_spearman?: number | null
  s2g_n?: number
  s2g_base_rate?: number
  s2g_prob_mean?: number
  s2g_brier?: number
  s2g_brier_baseline?: number
  /** Share of the top 10 Silver → Gold picks that reached Gold */
  s2g_top_hit_rate?: number
  s2g_top_avg_qs_gain?: number
}

/** Backtest accuracy for one predicted attribute, pooled over calibrated updates. */
export interface AttributeAccuracy {
  group: 'hitting' | 'pitching'
  n: number
  attr_mae: number
  attr_mae_baseline: number
  attr_skill: number
  attr_spearman: number
  attr_direction_acc: number | null
  interval_coverage: number
  moved_rate: number
  bias: number
}

/** Forecast-probability bucket vs how often the event happened. */
export interface ReliabilityRow {
  lo: number
  hi: number
  forecast: number
  actual: number
  n: number
}

export interface BacktestFold extends Partial<BacktestMetrics> {
  test_update: string
  train_updates: number
}

export interface ModelSummary {
  trained_on?: string[]
  metrics?: BacktestMetrics
  folds?: BacktestFold[]
  ovr_weights?: Record<'hitting' | 'pitching', Record<string, number>>
  stat_cutoff_days?: number
  last_update_review?: {
    update?: string
    top_upgrades?: ReviewPick[]
    top_downgrades?: ReviewPick[]
    top_value?: ReviewPick[]
    silver_to_gold?: ReviewPick[]
  }
  by_attribute?: Record<string, AttributeAccuracy>
  calibration?: Partial<Record<'upgrade' | 'downgrade' | 'tier_up' | 'silver_to_gold', ReliabilityRow[]>>
}

export interface DashboardResponse {
  count: number
  predictions: Prediction[]
  update_status: UpdateStatus
  model?: ModelSummary
}

export interface HistoryUpdate {
  update_date: string
  update_name: string
  ovr_before: number | null
  ovr_after: number | null
  changes: { attribute: string; rating_before: number; rating_after: number; delta: number }[]
}

/* ── Attributes ── */

export const HITTER_ATTRS: [string, string][] = [
  ['contact_right', 'Contact R'], ['contact_left', 'Contact L'],
  ['power_right', 'Power R'], ['power_left', 'Power L'],
  ['plate_vision', 'Vision'], ['batting_clutch', 'Clutch'],
]

export const PITCHER_ATTRS: [string, string][] = [
  ['h_per_9_r', 'H/9 R'], ['k_per_9_r', 'K/9 R'], ['k_per_9_l', 'K/9 L'],
  ['pitching_clutch', 'Clutch'], ['stamina', 'Stamina'],
]

export const ATTR_LABELS: Record<string, string> = Object.fromEntries([...HITTER_ATTRS, ...PITCHER_ATTRS])

export const RATING_MAX = 125
export const MAX_CARD_COPIES = 20

export const RARITY_ORDER = ['Common', 'Bronze', 'Silver', 'Gold', 'Diamond']

export const RARITY_COLORS: Record<string, string> = {
  Diamond: '#3d9fd3', Gold: '#c99a2e', Silver: '#9aa3ad', Bronze: '#a8703f', Common: '#7d8288',
}

/** Current card art straight from the SDS CDN (re-baked after every update). */
export function cardImage(uuid: string, size: 'sm' | 'lg' = 'sm'): string {
  return `https://cards.theshow.com/mlb26/${uuid}-baked-${size}.webp`
}

/* ── Formatting ── */

export function signed(n: number | null | undefined, dp = 1): string {
  if (n == null || isNaN(n)) return '—'
  const s = Math.abs(n).toFixed(dp)
  if (Number(s) === 0) return (0).toFixed(dp)
  return n > 0 ? `+${s}` : `−${s}`
}

export function pct(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return '—'
  const p = v * 100
  if (p > 0 && p < 1) return '<1%'
  if (p < 100 && p > 99) return '>99%'
  return `${Math.round(p)}%`
}

export function stubs(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return '—'
  const r = Math.round(n)
  return `${r > 0 ? '+' : r < 0 ? '−' : ''}${Math.abs(r).toLocaleString()}`
}

export function toneOf(n: number | null | undefined, threshold = 0.05): '' | 'up' | 'down' {
  if (n == null || isNaN(n)) return ''
  return n > threshold ? 'up' : n < -threshold ? 'down' : ''
}

export function shortDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(`${iso.slice(0, 10)}T12:00:00`)
  return isNaN(d.getTime()) ? iso : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

/** Batting/pitching rates: .xxx for slash stats, % for rates, 2dp for ERA/WHIP. */
export function fmtStat(key: string, v: number | null | undefined): string {
  if (v == null || isNaN(v)) return '—'
  if (key === 'pa' || key === 'bf') return Math.round(v).toLocaleString()
  if (['avg', 'obp', 'slg', 'iso'].includes(key)) return v.toFixed(3).replace(/^0/, '')
  if (key.endsWith('_pct')) return `${(v * 100).toFixed(1)}%`
  if (key === 'ip_per_g') return v.toFixed(1)
  return v.toFixed(2)
}
