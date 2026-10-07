import type { Prediction } from '../types'
import { pct, signed, stubs, toneOf } from '../types'
import { CardArt, Icon, ProbBar, Rarity } from './ui'

export type SortKey =
  | 'player_name' | 'current_ovr' | 'predicted_ovr_delta' | 'upgrade_probability'
  | 'downgrade_probability' | 'tier_jump_probability' | 'expected_value_per_card'

export interface Sort { key: SortKey; dir: 'asc' | 'desc' }

interface Props {
  rows: Prediction[]
  sort: Sort
  onSort: (key: SortKey) => void
  selected: string | null
  onSelect: (uuid: string) => void
}

function Header({ label, k, sort, onSort, align, className }: {
  label: string; k: SortKey; sort: Sort; onSort: (k: SortKey) => void; align?: 'r'; className?: string
}) {
  const active = sort.key === k
  return (
    <th className={[align, className].filter(Boolean).join(' ')} aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}>
      <button type="button" onClick={() => onSort(k)}>
        {label}
        {active && <Icon name={sort.dir === 'asc' ? 'sortUp' : 'sortDown'} size={13} />}
      </button>
    </th>
  )
}

function TierCell({ p }: { p: Prediction }) {
  if (p.tier_jump_probability >= 0.05) return <span className="tier">{pct(p.tier_jump_probability)} up</span>
  if (p.tier_down_probability >= 0.05) return <span className="down">{pct(p.tier_down_probability)} down</span>
  return <span className="faint">&mdash;</span>
}

export default function PlayerTable({ rows, sort, onSort, selected, onSelect }: Props) {
  return (
    <div className="table-wrap">
      <table className="grid">
        <thead>
          <tr>
            <Header label="Player" k="player_name" sort={sort} onSort={onSort} />
            <Header label="OVR" k="current_ovr" sort={sort} onSort={onSort} align="r" />
            <th className="hide-sm">Rarity</th>
            <Header label="Proj. change" k="predicted_ovr_delta" sort={sort} onSort={onSort} align="r" />
            <Header label="Chance up / down" k="upgrade_probability" sort={sort} onSort={onSort} />
            <Header label="Tier move" k="tier_jump_probability" sort={sort} onSort={onSort} align="r" className="hide-md" />
            <Header label="Exp. quicksell" k="expected_value_per_card" sort={sort} onSort={onSort} align="r" />
          </tr>
        </thead>
        <tbody>
          {rows.map(p => {
            const ev = p.expected_value_per_card ?? 0
            return (
              <tr
                key={p.card_uuid}
                aria-selected={selected === p.card_uuid}
                onClick={() => onSelect(p.card_uuid)}
                onKeyDown={e => { if (e.key === 'Enter') onSelect(p.card_uuid) }}
                tabIndex={0}
              >
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
                <td className="hide-sm"><Rarity rarity={p.current_rarity} /></td>
                <td className={`r num delta ${toneOf(p.predicted_ovr_delta, 0.25)}`}>{signed(p.predicted_ovr_delta)}</td>
                <td><ProbBar up={p.upgrade_probability} down={p.downgrade_probability} /></td>
                <td className="r num hide-md"><TierCell p={p} /></td>
                <td className={`r num ${toneOf(ev, 0.5)}`}>{Math.abs(ev) >= 1 ? stubs(ev) : <span className="faint">&mdash;</span>}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
