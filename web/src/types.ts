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
  sample_size_ok: boolean
  team: string | null
  position: string | null
  is_hitter: number | null
  attributes: AttributeItem[]
  created_at: string
  /** Probability-weighted quicksell change per card (stubs) */
  expected_value_per_card: number | null
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
