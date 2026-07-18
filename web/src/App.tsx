import { useState, useEffect, useMemo } from 'react'
import { fetchDashboard } from './api'
import type { Prediction, Filters, UpdateStatus } from './types'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import PlayerCard from './components/PlayerCard'
import Pagination from './components/Pagination'

const defaultFilters: Filters = {
  searchText: '',
  deltaRange: [-15, 15] as [number, number],
  changeProbRange: [0, 1] as [number, number],
  consensusRange: [-1, 1] as [number, number],
  selectedTeams: [],
  selectedRarities: [],
  sortBy: 'current_ovr',
  sortAsc: false,
  colsPerRow: 3,
  pageSize: 24,
}

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filters, setFilters] = useState<Filters>(defaultFilters)
  const [page, setPage] = useState(1)

  useEffect(() => {
    fetchDashboard()
      .then(data => {
        setPredictions(data.predictions)
        setUpdateStatus(data.update_status)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const filtered = useMemo(() => {
    let result = [...predictions]

    if (filters.searchText) {
      const q = filters.searchText.toLowerCase()
      result = result.filter(p =>
        p.player_name?.toLowerCase().includes(q) ||
        p.team?.toLowerCase().includes(q)
      )
    }

    if (filters.selectedTeams.length > 0) {
      result = result.filter(p => p.team && filters.selectedTeams.includes(p.team))
    }

    if (filters.selectedRarities.length > 0) {
      result = result.filter(p => filters.selectedRarities.includes(p.current_rarity))
    }

    result = result.filter(p => {
      const delta = p.predicted_ovr_delta ?? 0
      const consensus = p.direction_consensus ?? 0
      const changeProb = p.attributes?.length
        ? p.attributes.reduce((s, a) => s + (a.change_prob || 0), 0) / p.attributes.length
        : 0
      return (
        delta >= filters.deltaRange[0] && delta <= filters.deltaRange[1] &&
        changeProb >= filters.changeProbRange[0] && changeProb <= filters.changeProbRange[1] &&
        consensus >= filters.consensusRange[0] && consensus <= filters.consensusRange[1]
      )
    })

    result.sort((a, b) => {
      const getVal = (p: Prediction): number | string => {
        const val = p[filters.sortBy as keyof Prediction]
        if (typeof val === 'number') return val
        if (typeof val === 'string') return val
        return 0
      }
      const aVal = getVal(a)
      const bVal = getVal(b)
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return filters.sortAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return filters.sortAsc
        ? (aVal as number) - (bVal as number)
        : (bVal as number) - (aVal as number)
    })

    return result
  }, [predictions, filters])

  const totalPages = Math.max(1, Math.ceil(filtered.length / filters.pageSize))
  const safePage = Math.max(1, Math.min(page, totalPages))
  const startIdx = (safePage - 1) * filters.pageSize
  const endIdx = Math.min(startIdx + filters.pageSize, filtered.length)
  const pageData = filtered.slice(startIdx, endIdx)

  useEffect(() => {
    setPage(1)
  }, [filters])

  const upCount = filtered.filter(p => (p.predicted_ovr_delta ?? 0) > 0.5).length
  const dnCount = filtered.filter(p => (p.predicted_ovr_delta ?? 0) < -0.5).length

  if (loading) {
    return (
      <div style={{ background: '#0f1419', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: '#a0aec0', fontSize: 16 }}>Loading predictions...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ background: '#0f1419', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: '#f56565', fontSize: 16 }}>
          Error loading data: {error}. Make sure the API server is running.
        </div>
      </div>
    )
  }

  return (
    <div style={{ background: '#0f1419', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <Header status={updateStatus} />

      <div style={{ display: 'flex' }}>
        <Sidebar predictions={predictions} filters={filters} onFilterChange={filters => { setFilters(filters); setPage(1) }} />

        <div style={{ flex: 1, padding: 16, overflow: 'hidden' }}>
          {/* Summary bar */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 8, flexWrap: 'wrap' }}>
            <span style={{ background: '#1a202c', padding: '4px 12px', borderRadius: 8, border: '1px solid #2d3748', color: '#a0aec0', fontSize: 12 }}>
              <strong style={{ color: '#f7fafc' }}>{filtered.length}</strong> players
            </span>
            <span style={{ background: '#48bb7818', padding: '4px 12px', borderRadius: 8, border: '1px solid #48bb7844', color: '#48bb78', fontSize: 12 }}>
              &#9650; <strong>{upCount}</strong> upgrades
            </span>
            <span style={{ background: '#f5656518', padding: '4px 12px', borderRadius: 8, border: '1px solid #f5656544', color: '#f56565', fontSize: 12 }}>
              &#9660; <strong>{dnCount}</strong> downgrades
            </span>
          </div>

          <div style={{ color: '#718096', fontSize: 12, marginBottom: 6 }}>
            Showing {startIdx + 1}&ndash;{endIdx} of {filtered.length}
          </div>

          {pageData.length === 0 ? (
            <div style={{ color: '#a0aec0', textAlign: 'center', padding: 40 }}>
              No players match the current filters.
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: `repeat(${filters.colsPerRow}, 1fr)`,
              gap: 12,
            }}>
              {pageData.map(p => (
                <PlayerCard key={p.card_uuid} prediction={p} />
              ))}
            </div>
          )}

          <Pagination
            page={safePage}
            totalPages={totalPages}
            startIdx={startIdx}
            endIdx={endIdx}
            total={filtered.length}
            onPageChange={setPage}
          />

          <hr style={{ borderColor: '#2d3748', margin: '16px 0' }} />
          <div style={{ textAlign: 'center', color: '#4a5568', fontSize: 11 }}>
            Data from MLB The Show 26 roster updates &bull; Real card images from The Show CDN &bull; Predictions are formula-based estimates, not financial advice
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
