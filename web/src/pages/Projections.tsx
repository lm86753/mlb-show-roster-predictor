import { useEffect, useMemo, useState } from 'react'
import type { DashboardResponse, Prediction } from '../types'
import { RARITY_ORDER, pct, shortDate } from '../types'
import PlayerTable, { type Sort, type SortKey } from '../components/PlayerTable'
import CardGrid from '../components/CardGrid'
import { Icon, Segmented } from '../components/ui'

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

export default function Projections({ data, onSelect, selected }: {
  data: DashboardResponse; onSelect: (uuid: string) => void; selected: string | null
}) {
  const [tab, setTab] = useState<Tab>('up')
  const [sort, setSort] = useState<Sort>(TABS[0].sort)
  const [query, setQuery] = useState('')
  const [group, setGroup] = useState<Group>('all')
  const [rarity, setRarity] = useState('')
  const [team, setTeam] = useState('')
  const [layout, setLayout] = useState<Layout>('table')
  const [page, setPage] = useState(0)

  const predictions = data.predictions
  const teams = useMemo(() => [...new Set(predictions.map(p => p.team).filter(Boolean) as string[])].sort(), [predictions])

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

  const likelyUp = predictions.filter(p => p.upgrade_probability >= 0.5).length
  const likelyDown = predictions.filter(p => p.downgrade_probability >= 0.5).length
  const status = data.update_status
  const m = data.model?.metrics

  const onTab = (t: Tab) => { setTab(t); setSort(TABS.find(x => x.id === t)!.sort) }
  const onSort = (key: SortKey) =>
    setSort(s => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: key === 'player_name' ? 'asc' : 'desc' }))

  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const lastPage = Math.max(0, Math.ceil(rows.length / PAGE_SIZE) - 1)

  return (
    <>
      <div className="page-head">
        <h1>Player projections</h1>
        <p>
          For the next attribute update{status.next_expected ? <> (expected around <span className="strong">{shortDate(status.next_expected)}</span>)</> : null},{' '}
          <span className="up strong num">{likelyUp}</span> cards are more likely than not to upgrade and{' '}
          <span className="down strong num">{likelyDown}</span> to downgrade.
          {m && <> In backtests, {pct(m.top25_up_hit_rate)} of the top 25 upgrade picks went up. <a href="#/track-record">See the track record</a>.</>}
        </p>
      </div>

      <div className="tabs" role="tablist" aria-label="Projection lists">
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
          { value: 'table', label: <Icon name="table" />, title: 'Table view' },
          { value: 'cards', label: <Icon name="cards" />, title: 'Card view' },
        ]} />
      </div>

      {rows.length === 0 ? (
        <div className="table-wrap empty">
          No cards match these filters. <button className="btn-link" onClick={() => { setQuery(''); setGroup('all'); setRarity(''); setTeam('') }}>Clear filters</button>
        </div>
      ) : layout === 'table' ? (
        <PlayerTable rows={pageRows} sort={sort} onSort={onSort} selected={selected} onSelect={onSelect} />
      ) : (
        <CardGrid rows={pageRows} onSelect={onSelect} />
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
    </>
  )
}
