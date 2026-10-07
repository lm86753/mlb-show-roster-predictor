import { useEffect, useMemo, useState } from 'react'
import type { DashboardResponse, Prediction } from '../types'
import { fmtStat, signed, toneOf } from '../types'
import { CardArt, Icon, Segmented } from '../components/ui'

type Group = 'hitting' | 'pitching'
type Window = 'season' | 'last30'

const PAGE_SIZE = 50

const COLUMNS: Record<Group, { key: string; label: string; title: string; higherIsBetter: boolean }[]> = {
  hitting: [
    { key: 'pa', label: 'PA', title: 'Plate appearances', higherIsBetter: true },
    { key: 'avg', label: 'AVG', title: 'Batting average', higherIsBetter: true },
    { key: 'obp', label: 'OBP', title: 'On-base percentage', higherIsBetter: true },
    { key: 'slg', label: 'SLG', title: 'Slugging percentage', higherIsBetter: true },
    { key: 'iso', label: 'ISO', title: 'Isolated power', higherIsBetter: true },
    { key: 'k_pct', label: 'K%', title: 'Strikeout rate', higherIsBetter: false },
    { key: 'bb_pct', label: 'BB%', title: 'Walk rate', higherIsBetter: true },
  ],
  pitching: [
    { key: 'bf', label: 'BF', title: 'Batters faced', higherIsBetter: true },
    { key: 'era', label: 'ERA', title: 'Earned run average', higherIsBetter: false },
    { key: 'whip', label: 'WHIP', title: 'Walks plus hits per inning', higherIsBetter: false },
    { key: 'k_pct', label: 'K%', title: 'Strikeout rate', higherIsBetter: true },
    { key: 'bb_pct', label: 'BB%', title: 'Walk rate', higherIsBetter: false },
    { key: 'hr_pct', label: 'HR%', title: 'Home runs per batter faced', higherIsBetter: false },
    { key: 'ip_per_g', label: 'IP/G', title: 'Innings per appearance', higherIsBetter: true },
  ],
}

export default function PlayerStatsPage({ data, onSelect, selected }: {
  data: DashboardResponse; onSelect: (uuid: string) => void; selected: string | null
}) {
  const [group, setGroup] = useState<Group>('hitting')
  const [window, setWindow] = useState<Window>('season')
  // After the regular season ends the 30-day window is empty for everyone; hide it rather than show a blank table.
  const hasLast30 = useMemo(() => data.predictions.some(p => Object.values(p.stats?.last30 ?? {}).some(v => v != null)), [data.predictions])
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: 'pa', dir: -1 })
  const [page, setPage] = useState(0)

  const cols = COLUMNS[group]
  const minSample = window === 'season' ? 50 : 15

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const n = group === 'hitting' ? 'pa' : 'bf'
    const value = (p: Prediction, k: string) => {
      if (k === 'ovr') return p.current_ovr
      if (k === 'delta') return p.predicted_ovr_delta
      return p.stats?.[window]?.[k] ?? null
    }
    return data.predictions
      .filter(p => p.stats?.group === group && (p.stats?.[window]?.[n] ?? 0) >= minSample)
      .filter(p => !q || p.player_name.toLowerCase().includes(q) || (p.team ?? '').toLowerCase().includes(q))
      .sort((a, b) => {
        if (sort.key === 'name') return sort.dir * a.player_name.localeCompare(b.player_name)
        const va = value(a, sort.key), vb = value(b, sort.key)
        if (va == null) return 1
        if (vb == null) return -1
        return sort.dir * (va - vb)
      })
  }, [data.predictions, group, window, query, sort, minSample])

  useEffect(() => { setPage(0) }, [group, window, query, sort])

  const onGroup = (g: Group) => { setGroup(g); setSort({ key: g === 'hitting' ? 'pa' : 'bf', dir: -1 }) }
  const onSort = (key: string, higherIsBetter = true) =>
    setSort(s => (s.key === key ? { key, dir: (s.dir * -1) as 1 | -1 } : { key, dir: higherIsBetter ? -1 : 1 }))

  const header = (key: string, label: string, title?: string, higherIsBetter = true, className = 'r') => (
    <th key={key} className={className} title={title} aria-sort={sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : undefined}>
      <button type="button" onClick={() => onSort(key, higherIsBetter)}>
        {label}{sort.key === key && <Icon name={sort.dir === 1 ? 'sortUp' : 'sortDown'} size={13} />}
      </button>
    </th>
  )

  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const lastPage = Math.max(0, Math.ceil(rows.length / PAGE_SIZE) - 1)

  return (
    <>
      <div className="page-head">
        <h1>Player stats</h1>
        <p>
          Real 2026 MLB numbers for every Live Series player, from the MLB Stats API game logs the model is
          trained on, next to the card's rating and forecast. Minimum {minSample} {group === 'hitting' ? 'plate appearances' : 'batters faced'}.
        </p>
      </div>

      <div className="toolbar">
        <label className="visually-hidden" htmlFor="stat-search">Search players or teams</label>
        <input id="stat-search" className="input search" type="search" placeholder="Search players or teams"
          value={query} onChange={e => setQuery(e.target.value)} />
        <Segmented<Group> label="Player type" value={group} onChange={onGroup} options={[
          { value: 'hitting', label: 'Hitters' }, { value: 'pitching', label: 'Pitchers' },
        ]} />
        {hasLast30 && (
          <Segmented<Window> label="Stat window" value={window} onChange={setWindow} options={[
            { value: 'season', label: 'Season' }, { value: 'last30', label: 'Last 30 days' },
          ]} />
        )}
        <span className="grow" />
        <span className="muted num" style={{ fontSize: 13 }}>{rows.length.toLocaleString()} players</span>
      </div>

      {rows.length === 0 ? (
        <div className="table-wrap empty">No players match.</div>
      ) : (
        <div className="table-wrap">
          <table className="grid clickable">
            <thead>
              <tr>
                {header('name', 'Player', undefined, false, '')}
                {header('ovr', 'OVR', 'Current overall')}
                {header('delta', 'Proj.', 'Projected OVR change at the next update')}
                {cols.map((c, i) => header(c.key, c.label, c.title, c.higherIsBetter, i === 0 ? 'r group-start' : 'r'))}
              </tr>
            </thead>
            <tbody>
              {pageRows.map(p => {
                const line = p.stats?.[window] ?? {}
                return (
                  <tr key={p.card_uuid} aria-selected={selected === p.card_uuid} tabIndex={0}
                    onClick={() => onSelect(p.card_uuid)} onKeyDown={e => { if (e.key === 'Enter') onSelect(p.card_uuid) }}>
                    <td>
                      <div className="player-cell">
                        <CardArt uuid={p.card_uuid} className="thumb" />
                        <div>
                          <div className="player-name">{p.player_name}</div>
                          <div className="player-meta">{[p.team, p.position].filter(Boolean).join(' · ')}</div>
                        </div>
                      </div>
                    </td>
                    <td className="r num">{p.current_ovr}</td>
                    <td className={`r num delta ${toneOf(p.predicted_ovr_delta, 0.25)}`}>{signed(p.predicted_ovr_delta)}</td>
                    {cols.map((c, i) => <td key={c.key} className={`r num${i === 0 ? ' group-start muted' : ''}`}>{fmtStat(c.key, line[c.key])}</td>)}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {rows.length > PAGE_SIZE && (
        <div className="pager">
          <span className="num">{(page * PAGE_SIZE + 1).toLocaleString()}&ndash;{Math.min(rows.length, (page + 1) * PAGE_SIZE).toLocaleString()} of {rows.length.toLocaleString()}</span>
          <div className="pager-buttons">
            <button className="btn" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Previous</button>
            <button className="btn" disabled={page >= lastPage} onClick={() => setPage(p => p + 1)}>Next</button>
          </div>
        </div>
      )}
    </>
  )
}
