import type { DashboardResponse, HistoryUpdate } from './types'

const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '') || '/api'

export async function fetchDashboard(): Promise<DashboardResponse> {
  const res = await fetch(`${API_BASE}/dashboard`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchHistory(cardUuid: string): Promise<HistoryUpdate[]> {
  const res = await fetch(`${API_BASE}/player-history/${cardUuid}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()).updates
}
