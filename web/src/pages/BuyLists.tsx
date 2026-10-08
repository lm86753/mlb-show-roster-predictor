import { useMemo } from 'react'
import type { DashboardResponse, Prediction } from '../types'
import { MAX_CARD_COPIES, pct, signed, stubs, toneOf } from '../types'
import ListTable, { type ListColumn } from '../components/ListTable'
import { NEXT_RARITY, rarityFor } from '../profit'

const LIST_SIZE = 15


const meta = (p: Prediction) => [p.team, p.position, p.current_rarity].filter(Boolean).join(' · ')

const COL = {
  ovr: { label: 'OVR', align: 'r', cell: (p: Prediction) => p.current_ovr } as ListColumn<Prediction>,
  change: { label: 'Proj.', align: 'r', cell: (p: Prediction) => <span className={`delta ${toneOf(p.predicted_ovr_delta, 0.25)}`}>{signed(p.predicted_ovr_delta)}</span> } as ListColumn<Prediction>,
  up: { label: 'Up', align: 'r', cell: (p: Prediction) => <span className={p.upgrade_probability >= 0.5 ? 'up' : ''}>{pct(p.upgrade_probability)}</span> } as ListColumn<Prediction>,
  down: { label: 'Down', align: 'r', cell: (p: Prediction) => <span className={p.downgrade_probability >= 0.5 ? 'down' : ''}>{pct(p.downgrade_probability)}</span> } as ListColumn<Prediction>,
  qs: { label: 'QS now', align: 'r', className: 'hide-sm', cell: (p: Prediction) => p.current_qs.toLocaleString() } as ListColumn<Prediction>,
  ev: { label: 'Exp. QS', align: 'r', cell: (p: Prediction) => <span className={toneOf(p.expected_value_per_card, 0.5)}>{stubs(p.expected_value_per_card)}</span> } as ListColumn<Prediction>,
}

export default function BuyLists({ data, onSelect }: { data: DashboardResponse; onSelect: (uuid: string) => void }) {
  const lists = useMemo(() => {
    const ps = data.predictions
    const byDesc = (f: (p: Prediction) => number) => (a: Prediction, b: Prediction) => f(b) - f(a)
    return {
      value: ps.filter(p => (p.expected_value_per_card ?? 0) >= 25).sort(byDesc(p => p.expected_value_per_card ?? 0)).slice(0, LIST_SIZE),
      tiers: ps.filter(p => p.tier_jump_probability >= 0.25).sort(byDesc(p => p.tier_jump_probability)).slice(0, LIST_SIZE),
      cheap: ps.filter(p => p.current_qs <= 150 && p.upgrade_probability >= 0.5).sort(byDesc(p => p.upgrade_probability)).slice(0, LIST_SIZE),
      sell: ps.filter(p => p.current_qs >= 400 && p.downgrade_probability >= 0.3).sort(byDesc(p => -(p.expected_value_per_card ?? 0))).slice(0, LIST_SIZE),
    }
  }, [data.predictions])

  const select = (p: Prediction) => onSelect(p.card_uuid)
  const common = { rowKey: (p: Prediction) => p.card_uuid, name: (p: Prediction) => p.player_name, meta, onSelect: select }

  return (
    <>
      <div className="page-head">
        <h1>Buy lists</h1>
        <p>
          Shortlists built from the model's forecasts for the next attribute update. Quicksell figures assume you
          buy at the quicksell floor; at market prices, treat them as the downside.
        </p>
      </div>

      <div className="list-grid">
        <section>
          <h2 className="section">Best expected quicksell value</h2>
          <p className="section-note">Highest probability-weighted quicksell gain per card. Across {MAX_CARD_COPIES} copies the top pick is worth {stubs((lists.value[0]?.expected_value_per_card ?? 0) * MAX_CARD_COPIES)} stubs on average.</p>
          <ListTable caption="Best expected quicksell value" rows={lists.value} columns={[COL.ovr, COL.change, COL.up, COL.qs, COL.ev]} {...common} />
        </section>

        <section>
          <h2 className="section">Likely rarity tier jumps</h2>
          <p className="section-note">Cards with at least a 25% chance of crossing into the next rarity, where quicksell value steps up the most.</p>
          <ListTable
            caption="Likely rarity tier jumps"
            rows={lists.tiers}
            columns={[
              COL.ovr,
              { label: 'Move', cell: p => <span className="muted">{rarityFor(p.current_ovr)} &rarr; {NEXT_RARITY[rarityFor(p.current_ovr)] ?? 'Diamond'}</span> },
              { label: 'Chance', align: 'r', cell: p => <span className="tier strong">{pct(p.tier_jump_probability)}</span> },
              COL.ev,
            ]}
            {...common}
            empty="No card is 25% likely to change rarity right now."
          />
        </section>

        <section>
          <h2 className="section">Cheap and likely to rise</h2>
          <p className="section-note">Bronze and low Silver cards (quicksell 150 or less) that are more likely than not to upgrade. Low cost, good for collection progress.</p>
          <ListTable caption="Cheap and likely to rise" rows={lists.cheap} columns={[COL.ovr, COL.change, COL.up, COL.qs]} {...common} />
        </section>

        <section>
          <h2 className="section">Sell before the update</h2>
          <p className="section-note">Silver and above cards with at least a 30% chance of a downgrade, ranked by expected quicksell loss.</p>
          <ListTable caption="Sell before the update" rows={lists.sell} columns={[COL.ovr, COL.change, COL.down, COL.qs, COL.ev]} {...common}
            empty="No Silver-or-better card has a meaningful downgrade risk right now." />
        </section>
      </div>
    </>
  )
}
