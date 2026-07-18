import { useMemo } from 'react'
import type { Prediction, Filters } from '../types'
import { RARITY_ORDER } from '../types'

interface SidebarProps {
  predictions: Prediction[]
  filters: Filters
  onFilterChange: (filters: Filters) => void
}

const SORT_OPTIONS: Record<string, string> = {
  'Overall (Current)': 'current_ovr',
  'Predicted Delta': 'predicted_ovr_delta',
  'Upgrade Probability': 'upgrade_probability',
  'Downgrade Probability': 'downgrade_probability',
  'Player Name': 'player_name',
  'Direction Consensus': 'direction_consensus',
  'Avg Gap': 'avg_gap',
}

export default function Sidebar({ predictions, filters, onFilterChange }: SidebarProps) {
  const teams = useMemo(() => {
    const t = new Set(predictions.map(p => p.team).filter(Boolean) as string[])
    return Array.from(t).sort()
  }, [predictions])

  const rarities = useMemo(() => {
    const r = new Set(predictions.map(p => p.current_rarity).filter(Boolean))
    return Array.from(r).sort((a, b) => (RARITY_ORDER[a] ?? 0) - (RARITY_ORDER[b] ?? 0))
  }, [predictions])

  const update = (patch: Partial<Filters>) => {
    onFilterChange({ ...filters, ...patch })
  }

  return (
    <div style={{
      width: 260, minWidth: 260, background: '#1a202c',
      borderRight: '1px solid #2d3748', padding: 16,
      display: 'flex', flexDirection: 'column', gap: 12,
      overflowY: 'auto', height: '100vh',
    }}>
      <h2 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: '#f7fafc' }}>Filters & Sort</h2>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>Search Player / Team</label>
        <input
          type="text"
          placeholder="Type to filter..."
          value={filters.searchText}
          onChange={e => update({ searchText: e.target.value })}
          style={{
            width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid #4a5568',
            background: '#2d3748', color: '#f7fafc', fontSize: 13, outline: 'none',
            boxSizing: 'border-box',
          }}
        />
      </div>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>
          OVR Delta Range: {filters.deltaRange[0].toFixed(1)} &ndash; {filters.deltaRange[1].toFixed(1)}
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            type="range" min={-15} max={15} step={0.25}
            value={filters.deltaRange[0]}
            onChange={e => update({ deltaRange: [parseFloat(e.target.value), filters.deltaRange[1]] })}
            style={{ flex: 1 }}
          />
          <input
            type="range" min={-15} max={15} step={0.25}
            value={filters.deltaRange[1]}
            onChange={e => update({ deltaRange: [filters.deltaRange[0], parseFloat(e.target.value)] })}
            style={{ flex: 1 }}
          />
        </div>
      </div>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>
          Change Probability: {filters.changeProbRange[0].toFixed(2)} &ndash; {filters.changeProbRange[1].toFixed(2)}
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            type="range" min={0} max={1} step={0.05}
            value={filters.changeProbRange[0]}
            onChange={e => update({ changeProbRange: [parseFloat(e.target.value), filters.changeProbRange[1]] })}
            style={{ flex: 1 }}
          />
          <input
            type="range" min={0} max={1} step={0.05}
            value={filters.changeProbRange[1]}
            onChange={e => update({ changeProbRange: [filters.changeProbRange[0], parseFloat(e.target.value)] })}
            style={{ flex: 1 }}
          />
        </div>
      </div>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>
          Direction Consensus: {filters.consensusRange[0].toFixed(2)} &ndash; {filters.consensusRange[1].toFixed(2)}
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            type="range" min={-1} max={1} step={0.05}
            value={filters.consensusRange[0]}
            onChange={e => update({ consensusRange: [parseFloat(e.target.value), filters.consensusRange[1]] })}
            style={{ flex: 1 }}
          />
          <input
            type="range" min={-1} max={1} step={0.05}
            value={filters.consensusRange[1]}
            onChange={e => update({ consensusRange: [filters.consensusRange[0], parseFloat(e.target.value)] })}
            style={{ flex: 1 }}
          />
        </div>
      </div>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>Sort By</label>
        <select
          value={Object.entries(SORT_OPTIONS).find(([, v]) => v === filters.sortBy)?.[0] || 'Overall (Current)'}
          onChange={e => {
            const val = SORT_OPTIONS[e.target.value]
            if (val) update({ sortBy: val })
          }}
          style={{
            width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid #4a5568',
            background: '#2d3748', color: '#f7fafc', fontSize: 13, outline: 'none',
          }}
        >
          {Object.entries(SORT_OPTIONS).map(([label, val]) => (
            <option key={val} value={label}>{label}</option>
          ))}
        </select>
      </div>

      <label style={{ fontSize: 12, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: 6 }}>
        <input
          type="checkbox"
          checked={filters.sortAsc}
          onChange={e => update({ sortAsc: e.target.checked })}
        />
        Ascending
      </label>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>Teams</label>
        <div style={{ maxHeight: 120, overflowY: 'auto' }}>
          {teams.map(team => (
            <label key={team} style={{ fontSize: 12, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: 4, padding: '1px 0' }}>
              <input
                type="checkbox"
                checked={filters.selectedTeams.includes(team)}
                onChange={e => {
                  update({
                    selectedTeams: e.target.checked
                      ? [...filters.selectedTeams, team]
                      : filters.selectedTeams.filter(t => t !== team),
                  })
                }}
              />
              {team}
            </label>
          ))}
        </div>
      </div>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>Rarities</label>
        <div style={{ maxHeight: 120, overflowY: 'auto' }}>
          {rarities.map(r => (
            <label key={r} style={{ fontSize: 12, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: 4, padding: '1px 0' }}>
              <input
                type="checkbox"
                checked={filters.selectedRarities.includes(r)}
                onChange={e => {
                  update({
                    selectedRarities: e.target.checked
                      ? [...filters.selectedRarities, r]
                      : filters.selectedRarities.filter(x => x !== r),
                  })
                }}
              />
              {r}
            </label>
          ))}
        </div>
      </div>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>Columns per Row</label>
        <select
          value={filters.colsPerRow}
          onChange={e => update({ colsPerRow: parseInt(e.target.value) })}
          style={{
            width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid #4a5568',
            background: '#2d3748', color: '#f7fafc', fontSize: 13, outline: 'none',
          }}
        >
          {[2, 3, 4].map(n => <option key={n} value={n}>{n}</option>)}
        </select>
      </div>

      <div>
        <label style={{ fontSize: 11, color: '#a0aec0', display: 'block', marginBottom: 2 }}>Cards per Page</label>
        <select
          value={filters.pageSize}
          onChange={e => update({ pageSize: parseInt(e.target.value) })}
          style={{
            width: '100%', padding: '6px 10px', borderRadius: 6, border: '1px solid #4a5568',
            background: '#2d3748', color: '#f7fafc', fontSize: 13, outline: 'none',
          }}
        >
          {[12, 24, 48, 96].map(n => <option key={n} value={n}>{n}</option>)}
        </select>
      </div>
    </div>
  )
}
