export const ROUTES = [
  { id: 'projections', label: 'Projections' },
  { id: 'buy-lists', label: 'Buy Lists' },
  { id: 'silver-gold', label: 'Silver to Gold' },
  { id: 'profit', label: 'Profit Calculator' },
  { id: 'stats', label: 'Player Stats' },
  { id: 'track-record', label: 'Track Record' },
] as const

export type Route = (typeof ROUTES)[number]['id']
