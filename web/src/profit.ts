import type { Prediction } from './types'

/** Integer OVR moves the model assigns probabilities to (mirrors OVR_MOVES in train.py). */
export const OVR_MOVES = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

/** Community Market sales lose 10% to tax. */
export const MARKET_TAX = 0.1

// Live Series quicksell values by OVR (mirrors src/config.py).
const QS_BY_OVR: Record<number, number> = {
  75: 50, 76: 75, 77: 100, 78: 125, 79: 150,
  80: 400, 81: 600, 82: 900, 83: 1200, 84: 1500,
  85: 3000, 86: 3750, 87: 4500, 88: 5500, 89: 7000,
  90: 8000, 91: 9000,
}

export function quicksellValue(ovr: number): number {
  if (ovr < 65) return 5
  if (ovr < 75) return 25
  if (ovr >= 92) return 10000
  return QS_BY_OVR[ovr]
}

export const SILVER_FLOOR = 75
export const GOLD_FLOOR = 80

/** Live Series rarity for an OVR: Bronze 65, Silver 75, Gold 80, Diamond 85. */
export function rarityFor(ovr: number): string {
  if (ovr >= 85) return 'Diamond'
  if (ovr >= 80) return 'Gold'
  if (ovr >= 75) return 'Silver'
  if (ovr >= 65) return 'Bronze'
  return 'Common'
}

export const NEXT_RARITY: Record<string, string> = { Common: 'Bronze', Bronze: 'Silver', Silver: 'Gold', Gold: 'Diamond' }

export function isSilver(p: Prediction): boolean {
  return p.current_ovr >= SILVER_FLOOR && p.current_ovr < GOLD_FLOOR
}

export interface Outcome {
  move: number
  ovr: number
  prob: number
  qs: number
  profit: number
}

export interface FlipResult {
  outcomes: Outcome[]
  expectedPayout: number
  expectedProfit: number
  /** Chance the sale beats the buy price */
  pProfit: number
  /** Chance of losing more than nothing */
  pLoss: number
  /** Highest buy price with zero expected profit */
  breakEven: number
  worst: number
  best: number
}

/** The card's OVR-move distribution, capped at 99, as a list of outcomes. */
function moveDistribution(p: Prediction): { move: number; ovr: number; prob: number }[] {
  const probs = p.ovr_move_probs?.length === OVR_MOVES.length ? p.ovr_move_probs : null
  if (!probs) {
    // Older payloads without the distribution: fall back to up / flat / down by one.
    const flat = Math.max(0, 1 - p.upgrade_probability - p.downgrade_probability)
    return [
      { move: -1, ovr: p.current_ovr - 1, prob: p.downgrade_probability },
      { move: 0, ovr: p.current_ovr, prob: flat },
      { move: 1, ovr: Math.min(99, p.current_ovr + 1), prob: p.upgrade_probability },
    ]
  }
  return OVR_MOVES.map((move, i) => ({ move, ovr: Math.max(40, Math.min(99, p.current_ovr + move)), prob: probs[i] }))
}

/**
 * Profit per card from buying at `buy` and quickselling after the update. If
 * `marketPrice` is given, the exit is instead a market sale at that price
 * (after tax) whenever it beats the post-update quicksell.
 */
export function flipOutcomes(p: Prediction, buy: number, marketPrice?: number | null): FlipResult {
  const outcomes = moveDistribution(p)
    .filter(o => o.prob > 0)
    .map(o => {
      const qs = quicksellValue(o.ovr)
      const exit = marketPrice ? Math.max(qs, marketPrice * (1 - MARKET_TAX)) : qs
      return { ...o, qs: exit, profit: exit - buy }
    })
  const total = outcomes.reduce((s, o) => s + o.prob, 0) || 1
  outcomes.forEach(o => { o.prob /= total })
  const expectedPayout = outcomes.reduce((s, o) => s + o.prob * o.qs, 0)
  const likely = outcomes.filter(o => o.prob >= 0.02)
  return {
    outcomes,
    expectedPayout,
    expectedProfit: expectedPayout - buy,
    pProfit: outcomes.filter(o => o.profit > 0).reduce((s, o) => s + o.prob, 0),
    pLoss: outcomes.filter(o => o.profit < 0).reduce((s, o) => s + o.prob, 0),
    breakEven: expectedPayout,
    worst: Math.min(...(likely.length ? likely : outcomes).map(o => o.profit)),
    best: Math.max(...(likely.length ? likely : outcomes).map(o => o.profit)),
  }
}
