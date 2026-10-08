import { useMemo, useState } from 'react'
import type { DashboardResponse, Prediction } from '../types'
import { pct, stubs, toneOf } from '../types'
import { flipOutcomes } from '../profit'
import FlipCalculator from '../components/FlipCalculator'
import ListTable from '../components/ListTable'
import { CardArt, Rarity } from '../components/ui'

const meta = (p: Prediction) => [p.team, p.position, p.current_rarity].filter(Boolean).join(' · ')

export default function Profit({ data, onSelect }: { data: DashboardResponse; onSelect: (uuid: string) => void }) {
  const [query, setQuery] = useState('')
  const [picked, setPicked] = useState<string | null>(null)

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return []
    return data.predictions.filter(p => p.player_name.toLowerCase().includes(q)).slice(0, 8)
  }, [data.predictions, query])

  // Best flips at the quicksell floor: highest expected profit with better-than-even odds of not losing.
  const best = useMemo(() => data.predictions
    .map(p => ({ p, flip: flipOutcomes(p, p.current_qs) }))
    .filter(r => r.flip.expectedProfit > 0 && r.flip.pLoss < 0.5)
    .sort((a, b) => b.flip.expectedProfit - a.flip.expectedProfit)
    .slice(0, 15),
  [data.predictions])

  const card = data.predictions.find(p => p.card_uuid === picked) ?? null

  return (
    <>
      <div className="page-head">
        <h1>Profit calculator</h1>
        <p>
          Pick a card, enter what you'd pay, and see the profit for every way its OVR could move at the next
          attribute update, with the chance of each. Quicksell is the default exit; add a market price to model
          selling on the Community Market instead.
        </p>
      </div>

      <div className="list-grid">
        <section>
          <h2 className="section">Choose a card</h2>
          <div className="toolbar" style={{ paddingTop: 0 }}>
            <label className="visually-hidden" htmlFor="profit-search">Search players</label>
            <input id="profit-search" className="input search" type="search" placeholder="Search players" value={query}
              onChange={e => setQuery(e.target.value)} />
          </div>
          {matches.length > 0 && (
            <div className="table-wrap">
              <table className="grid compact clickable">
                <caption className="visually-hidden">Matching cards</caption>
                <tbody>
                  {matches.map(p => (
                    <tr key={p.card_uuid} tabIndex={0} aria-selected={p.card_uuid === picked}
                      onClick={() => setPicked(p.card_uuid)} onKeyDown={e => { if (e.key === 'Enter') setPicked(p.card_uuid) }}>
                      <td>
                        <div className="player-cell">
                          <CardArt uuid={p.card_uuid} className="thumb" />
                          <div><div className="player-name">{p.player_name}</div><div className="player-meta">{meta(p)}</div></div>
                        </div>
                      </td>
                      <td className="r num">{p.current_ovr}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {card ? (
            <div style={{ marginTop: 20 }}>
              <div className="player-cell" style={{ marginBottom: 12 }}>
                <CardArt uuid={card.card_uuid} className="thumb" />
                <div>
                  <button className="btn-link player-name" onClick={() => onSelect(card.card_uuid)}>{card.player_name}</button>
                  <div className="player-meta">{card.current_ovr} OVR &middot; <Rarity rarity={card.current_rarity} /> &middot; quicksell {card.current_qs.toLocaleString()}</div>
                </div>
              </div>
              <FlipCalculator key={card.card_uuid} p={card} />
            </div>
          ) : (
            <p className="note">Search for a player, or pick one of the best flips.</p>
          )}
        </section>

        <section>
          <h2 className="section">Best flips at the quicksell floor</h2>
          <p className="section-note">Highest expected profit buying at today's quicksell value and quickselling after the update, among cards less likely than not to lose stubs.</p>
          <ListTable
            caption="Best flips at the quicksell floor"
            rows={best}
            rowKey={r => r.p.card_uuid}
            name={r => r.p.player_name}
            meta={r => meta(r.p)}
            onSelect={r => setPicked(r.p.card_uuid)}
            columns={[
              { label: 'OVR', align: 'r', cell: r => r.p.current_ovr },
              { label: 'Buy', align: 'r', className: 'hide-sm', cell: r => r.p.current_qs.toLocaleString() },
              { label: 'Profit chance', align: 'r', cell: r => pct(r.flip.pProfit) },
              { label: 'Exp. profit', align: 'r', cell: r => <span className={toneOf(r.flip.expectedProfit, 0.5)}>{stubs(r.flip.expectedProfit)}</span> },
            ]}
          />
        </section>
      </div>
    </>
  )
}
