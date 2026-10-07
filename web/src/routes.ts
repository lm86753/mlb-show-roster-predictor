export const ROUTES = [
  { id: 'projections', label: 'Projections' },
  { id: 'buy-lists', label: 'Buy Lists' },
  { id: 'stats', label: 'Player Stats' },
  { id: 'track-record', label: 'Track Record' },
] as const

export type Route = (typeof ROUTES)[number]['id']
