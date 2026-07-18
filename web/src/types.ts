export interface AttributeItem {
  attribute_name: string
  rating_before: number
  projected_rating: number
  predicted_delta: number
  gap: number
  change_prob: number
  has_stat_data: number
  mismatch_score: number
}

export interface Prediction {
  card_uuid: string
  player_name: string
  mlb_player_id: number | null
  current_ovr: number
  current_rarity: string
  predicted_ovr_delta: number
  upgrade_probability: number
  downgrade_probability: number
  tier_jump_probability: number
  sample_size_ok: boolean
  avg_gap: number
  direction_consensus: number
  team: string | null
  position: string | null
  is_hitter: number | null
  has_card_image: boolean
  attributes: AttributeItem[]
  created_at: string
}

export interface UpdateStatus {
  latest: string | null
  days_since: number | null
  days_until: number | null
  next_expected: string | null
  is_update_today: boolean
}

export interface DashboardResponse {
  count: number
  predictions: Prediction[]
  update_status: UpdateStatus
}

export const TEAM_COLORS: Record<string, string> = {
  Angels: "#BA0C2F", Astros: "#183469", Giants: "#FD5A1E", Dodgers: "#005A9C",
  Braves: "#CE1141", Phillies: "#E81828", Orioles: "#DF4601", Rays: "#008080",
  Twins: "#002B5C", "Blue Jays": "#134A8E", "Red Sox": "#BD3039", "White Sox": "#27251F",
  Yankees: "#003087", Athletics: "#003831", Guardians: "#0C2340", Tigers: "#0C1B38",
  Royals: "#004687", Rockies: "#33006F", Marlins: "#00A3E0", Brewers: "#FFC72C",
  Cardinals: "#C41E3A", Nationals: "#AB0003", Mets: "#002D72", Pirates: "#27251F",
  Padres: "#2F241D", Rangers: "#003278", Reds: "#C6011F", Cubs: "#0E3386",
  Diamondbacks: "#A71930", Mariners: "#0C2C56",
}

export const RARITY_COLORS: Record<string, string> = {
  "Red Diamond": "#FF0044", Diamond: "#00BFFF", Gold: "#FFD700", Silver: "#C0C0C0", Bronze: "#CD7F32", Common: "#808080",
}

export const RARITY_ORDER: Record<string, number> = {
  Common: 0, Bronze: 1, Silver: 2, Gold: 3, Diamond: 4, "Red Diamond": 5,
}

export const HITTER_ATTRS: [string, string][] = [
  ["contact_right", "Con R"], ["contact_left", "Con L"],
  ["power_right", "Pow R"], ["power_left", "Pow L"],
  ["plate_vision", "Vis"], ["batting_clutch", "Clutch"],
  ["plate_discipline", "Disc"], ["speed", "Spd"],
]

export const PITCHER_ATTRS: [string, string][] = [
  ["pitch_control", "Ctrl"], ["pitch_movement", "Mov"],
  ["pitch_velocity", "Vel"], ["pitching_clutch", "P Clutch"],
  ["stamina", "Stam"], ["k/9_r", "K/9 R"], ["k/9_l", "K/9 L"],
  ["h/9_r", "H/9 R"], ["h/9", "H/9"], ["bb/9", "BB/9"],
]

export interface Filters {
  searchText: string
  deltaRange: [number, number]
  changeProbRange: [number, number]
  consensusRange: [number, number]
  selectedTeams: string[]
  selectedRarities: string[]
  sortBy: string
  sortAsc: boolean
  colsPerRow: number
  pageSize: number
}

export const QS_TIERS: [number, number][] = [
  [0, 25], [65, 100], [75, 300], [85, 1000], [90, 5000], [95, 10000],
]

export function getTeamColor(team: string | null): string {
  return team ? TEAM_COLORS[team] || "#2d3748" : "#2d3748"
}

export function getRarityColor(rarity: string): string {
  return RARITY_COLORS[rarity] || "#808080"
}

export function formatDelta(delta: number | null | undefined): string {
  if (delta == null || isNaN(delta)) return "\u2014"
  return delta > 0 ? `+${delta.toFixed(1)}` : `${delta.toFixed(1)}`
}

export function deltaColor(delta: number | null | undefined): string {
  if (delta == null || isNaN(delta)) return "#a0aec0"
  if (delta > 0) return "#48bb78"
  if (delta < 0) return "#f56565"
  return "#a0aec0"
}

export function getCardImageUrl(cardUuid: string): string {
  return `/api/card-image/${cardUuid}`
}
