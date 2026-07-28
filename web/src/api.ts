import type { DashboardResponse, Prediction } from './types'

const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '') || '/api'

export async function fetchDashboard(): Promise<DashboardResponse> {
  const res = await fetch(`${API_BASE}/dashboard`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchPlayer(cardUuid: string): Promise<Prediction> {
  const res = await fetch(`${API_BASE}/player/${cardUuid}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}
