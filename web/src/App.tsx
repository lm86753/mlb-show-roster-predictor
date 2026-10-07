import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchDashboard } from './api'
import type { DashboardResponse, Prediction } from './types'
import { RARITY_ORDER, pct, shortDate } from './types'
import TopBar from './components/TopBar'
import PlayerTable, { type Sort, type SortKey } from './components/PlayerTable'
import CardGrid from './components/CardGrid'
import PlayerDrawer from './components/PlayerDrawer'
import Methodology from './components/Methodology'
import { Icon, Segmented } from './components/ui'

type Tab = 'up' | 'down' | 'value' | 'all'
type Group = 'all' | 'hitters' | 'pitchers'
type Layout = 'table' | 'cards'

const PAGE_SIZE = 50

const TABS: { id: Tab; label: string; filter: (p: Prediction) => boolean; sort: Sort }[] = [
  { id: 'up', label: 'Likely upgrades', filter: p => p.upgrade_probability >= 0.3, sort: { key: 'upgrade_probability', dir: 'desc' } },
  { id: 'down', label: 'Likely downgrades', filter: p => p.downgrade_probability >= 0.3, sort: { key: 'downgrade_probability', dir: 'desc' } },
  { id: 'value', label: 'Quicksell value', filter: p => (p.expected_value_per_card ?? 0) >= 1, sort: { key: 'expected_value_per_card', dir: 'desc' } },
  { id: 'all', label: 'All cards', filter: () => true, sort: { key: 'current_ovr', dir: 'desc' } },
]

function compare(a: Prediction, b: Prediction, key: SortKey): number {
  if (key === 'player_name') return a.player_name.localeCompare(b.player_name)
  return (a[key] ?? 0) - (b[key] ?? 0)
}

export default function App() {
  const [data, setData] = useState<DashboardResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [tab, setTab] = useState<Tab>('up')
  const [sort, setSort] = useState<Sort>(TABS[0].sort)
  const [query, setQuery] = useState('')
  const [group, setGroup] = useState<Group>('all')
  const [rarity, setRarity] = useState('')
  const [team, setTeam] = useState('')
  const [layout, setLayout] = useState<Layout>('table')
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState<string | null>(null)
  const [showMethod, setShowMethod] = useState(false)

  useEffect(() => {
    fetchDashboard().then(setData).catch(err => setError(err.message))
  }, [])

  const predictions = useMemo(() => data?.predictions ?? [], [data])
  const teams = useMemo(() => [...new Set(predictions.map(p => p.team).filter(Boolean) as string[])].sort(), [predictions])

  // Everything except the tab filter, so tab counts reflect the other filters.
  const base = useMemo(() => {
    const q = query.trim().toLowerCase()
    return predictions.filter(p =>
      (!q || p.player_name.toLowerCase().includes(q) || (p.team ?? '').toLowerCase().includes(q)) &&
      (group === 'all' || (group === 'hitters' ? !!p.is_hitter : !p.is_hitter)) &&
      (!rarity || p.current_rarity === rarity) &&
      (!team || p.team === team),
    )
  }, [predictions, query, group, rarity, team])

  const counts = useMemo(() => Object.fromEntries(TABS.map(t => [t.id, base.filter(t.filter).length])) as Record<Tab, number>, [base])

  const rows = useMemo(() => {
    const t = TABS.find(x => x.id === tab)!
    const dir = sort.dir === 'asc' ? 1 : -1
    return base.filter(t.filter).sort((a, b) => dir * compare(a, b, sort.key) || b.current_ovr - a.current_ovr)
  }, [base, tab, sort])

  useEffect(() => { setPage(0) }, [query, group, rarity, team, tab, sort])

  const summary = useMemo(() => ({
    up: predictions.filter(p => p.upgrade_probability >= 0.5).length,
    down: predictions.filter(p => p.downgrade_probability >= 0.5).length,
    tier: predictions.filter(p => p.tier_jump_probability >= 0.25).length,
  }), [predictions])

  const dataAsOf = useMemo(() => {
    const t = Math.max(0, ...predictions.map(p => Date.parse(p.created_at) || 0))
    return t ? new Date(t).toISOString() : null
  }, [predictions])

  const onTab = (t: Tab) => { setTab(t); setSort(TABS.find(x => x.id === t)!.sort) }
  const onSort = (key: SortKey) =>
    setSort(s => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'player_name' ? 'asc' : 'desc' }))
  const closeDrawer = useCallback(() => setSelected(null), [])
  const closeMethod = useCallback(() => setShowMethod(false), [])

  if (error) {
    return (
      <div className="state">
        <strong style={{ color: 'var(--text)' }}>Couldn't load forecasts</strong>
        <span>{error}. Check that the API is running, then reload.</span>
        <button className="btn" onClick={() => location.reload()}>Reload</button>
      </div>
    )
  }
  if (!data) return <div className="state">Loading forecasts&hellip;</div>

  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const lastPage = Math.max(0, Math.ceil(rows.length / PAGE_SIZE) - 1)
  const m = data.model?.metrics
  const status = data.update_status
  const selectedPrediction = predictions.find(p => p.card_uuid === selected)

  return (
    <>
      <TopBar status={status} dataAsOf={dataAsOf} onMethodology={() => setShowMethod(true)} />

      <main className="page">
        <h1 className="page-title">Next attribute update forecast</h1>
        <p className="page-lede">
          Every Live Series card, scored on how its ratings are likely to move when San Diego Studio
          next re-rates players{status.last_attribute_update ? ` (last update ${shortDate(status.last_attribute_update)})` : ''}.
        </p>

        <div className="summary">
          <div className="summary-item">
            <div className="summary-value num up">{summary.up}</div>
            <div className="summary-label">cards 50%+ likely to upgrade</div>
          </div>
          <div className="summary-item">
            <div className="summary-value num down">{summary.down}</div>
            <div className="summary-label">cards 50%+ likely to downgrade</div>
          </div>
          <div className="summary-item">
            <div className="summary-value num tier">{summary.tier}</div>
            <div className="summary-label">cards 25%+ likely to jump a rarity tier</div>
          </div>
          {m && (
            <div className="summary-item">
              <div className="summary-value num">{pct(m.top25_up_hit_rate)}</div>
              <div className="summary-label">
                of top-25 upgrade picks went up in backtests{' '}
                <button className="btn-link" onClick={() => setShowMethod(true)}>Details</button>
              </div>
            </div>
          )}
        </div>

        <div className="tabs" role="tablist" aria-label="Forecast views">
          {TABS.map(t => (
            <button key={t.id} role="tab" className="tab" aria-selected={tab === t.id} onClick={() => onTab(t.id)}>
              {t.label}<span className="tab-count num">{counts[t.id].toLocaleString()}</span>
            </button>
          ))}
        </div>

        <div className="toolbar">
          <label className="visually-hidden" htmlFor="search">Search players or teams</label>
          <input id="search" className="input search" type="search" placeholder="Search players or teams"
            value={query} onChange={e => setQuery(e.target.value)} />
          <Segmented<Group> label="Player type" value={group} onChange={setGroup} options={[
            { value: 'all', label: 'All' }, { value: 'hitters', label: 'Hitters' }, { value: 'pitchers', label: 'Pitchers' },
          ]} />
          <select className="select" aria-label="Rarity" value={rarity} onChange={e => setRarity(e.target.value)}>
            <option value="">All rarities</option>
            {RARITY_ORDER.slice().reverse().map(r => <option key={r} value={r}>{r}</option>)}
          </select>
          <select className="select" aria-label="Team" value={team} onChange={e => setTeam(e.target.value)}>
            <option value="">All teams</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <span className="grow" />
          <Segmented<Layout> label="Layout" value={layout} onChange={setLayout} options={[
            { value: 'table', label: <Icon name="table" />, title: 'Table' },
            { value: 'cards', label: <Icon name="cards" />, title: 'Cards' },
          ]} />
        </div>

        {rows.length === 0 ? (
          <div className="table-wrap empty">No cards match these filters.</div>
        ) : layout === 'table' ? (
          <PlayerTable rows={pageRows} sort={sort} onSort={onSort} selected={selected} onSelect={setSelected} />
        ) : (
          <CardGrid rows={pageRows} onSelect={setSelected} />
        )}

        {rows.length > PAGE_SIZE && (
          <div className="pager">
            <span className="num">
              {(page * PAGE_SIZE + 1).toLocaleString()}&ndash;{Math.min(rows.length, (page + 1) * PAGE_SIZE).toLocaleString()} of {rows.length.toLocaleString()}
            </span>
            <div className="pager-buttons">
              <button className="btn" disabled={page === 0} onClick={() => setPage(p => p - 1)}>Previous</button>
              <button className="btn" disabled={page >= lastPage} onClick={() => setPage(p => p + 1)}>Next</button>
            </div>
          </div>
        )}

        <footer className="footer">
          <span>Ratings and card art from The Show's public API. Stats from the MLB Stats API.</span>
          <span>Fan project, not affiliated with San Diego Studio or MLB. Forecasts are estimates.</span>
        </footer>
      </main>

      {selectedPrediction && <PlayerDrawer prediction={selectedPrediction} model={data.model} onClose={closeDrawer} />}
      {showMethod && <Methodology model={data.model} onClose={closeMethod} />}
    </>
  )
}
