import { useMemo, useState } from 'react'
import type { DashboardResponse, Prediction } from '../types'
import { MAX_CARD_COPIES, pct, signed, stubs, toneOf } from '../types'
import { GOLD_FLOOR, flipOutcomes, isSilver, quicksellValue } from '../profit'
import ListTable from '../components/ListTable'
import { Segmented } from '../components/ui'

type MinChance = '0.05' | '0.25' | '0.5'

const meta = (p: Prediction) => [p.team, p.position].filter(Boolean).join(' · ')

export default function SilverToGold({ data, onSelect }: { data: DashboardResponse; onSelect: (uuid: string) => void }) {
  const [minChance, setMinChance] = useState<MinChance>('0.25')
  const m = data.model?.metrics
  const review = data.model?.last_update_review

  const rows = useMemo(() => data.predictions
    .filter(p => isSilver(p) && (p.gold_probability ?? 0) >= Number(minChance))
    .map(p => ({ p, flip: flipOutcomes(p, p.current_qs) }))
    .sort((a, b) => (b.p.gold_probability ?? 0) - (a.p.gold_probability ?? 0)),
  [data.predictions, minChance])

  const silverCount = data.predictions.filter(isSilver).length
  const pastPicks = review?.silver_to_gold?.slice(0, 10) ?? []
  const pastHits = pastPicks.filter(r => r.ovr_after >= GOLD_FLOOR).length

  return (
    <>
      <div className="page-head">
        <h1>Silver to Gold</h1>
        <p>
          Every Silver card (75–79 OVR) ranked by its chance of being Gold (80+) after the next attribute update.
          A 79 that reaches 80 jumps from 150 to 400 quicksell, so these are the cheapest big swings in the market.
          Profit assumes you buy at the quicksell floor and quicksell after the update.
        </p>
      </div>

      {m?.s2g_top_hit_rate != null && (
        <div className="figures">
          <div className="figure">
            <div className="figure-value num">{pct(m.s2g_top_hit_rate)}</div>
            <div className="figure-label">of the top 10 Silver to Gold picks reached Gold in past updates (average Silver: {pct(m.s2g_base_rate)})</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{stubs(m.s2g_top_avg_qs_gain)}</div>
            <div className="figure-label">average quicksell gain per card for those top 10 picks</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{pct(m.s2g_prob_mean)}</div>
            <div className="figure-label">average forecast Gold chance per Silver, against {pct(m.s2g_base_rate)} that made it</div>
          </div>
        </div>
      )}

      <section className="block">
        <div className="toolbar">
          <Segmented<MinChance> label="Minimum Gold chance" value={minChance} onChange={setMinChance} options={[
            { value: '0.05', label: '5%+' },
            { value: '0.25', label: '25%+' },
            { value: '0.5', label: '50%+' },
          ]} />
          <span className="muted" style={{ fontSize: 13 }}>{rows.length} of {silverCount} Silver cards</span>
        </div>
        <ListTable
          caption="Silver cards ranked by chance of reaching Gold"
          rows={rows}
          rowKey={r => r.p.card_uuid}
          name={r => r.p.player_name}
          meta={r => meta(r.p)}
          onSelect={r => onSelect(r.p.card_uuid)}
          empty="No Silver card clears that Gold chance right now."
          columns={[
            { label: 'OVR', align: 'r', cell: r => r.p.current_ovr },
            { label: 'Proj.', align: 'r', cell: r => <span className={`delta ${toneOf(r.p.predicted_ovr_delta, 0.25)}`}>{signed(r.p.predicted_ovr_delta)}</span> },
            { label: 'Gold chance', align: 'r', cell: r => <span className="tier strong">{pct(r.p.gold_probability)}</span> },
            { label: 'QS now → Gold', align: 'r', className: 'hide-sm', cell: r => <span>{r.p.current_qs} &rarr; {quicksellValue(GOLD_FLOOR).toLocaleString()}+</span> },
            { label: 'Exp. profit', align: 'r', cell: r => <span className={toneOf(r.flip.expectedProfit, 0.5)}>{stubs(r.flip.expectedProfit)}</span> },
            { label: 'Max buy', align: 'r', className: 'hide-sm', cell: r => <span title="Highest price with zero expected profit at quicksell">{Math.floor(r.flip.breakEven).toLocaleString()}</span> },
          ]}
        />
        <p className="note">
          "Max buy" is the price where the expected quicksell after the update just covers the cost. With {MAX_CARD_COPIES} copies, multiply the expected profit by {MAX_CARD_COPIES}.
        </p>
      </section>

      {pastPicks.length > 0 && review?.update && (
        <section className="block">
          <h2 className="section">How these picks did last update</h2>
          <p className="section-note">
            The model's top 10 Silver to Gold picks before the {review.update} update, trained only on earlier updates.{' '}
            <span className="strong num">{pastHits} of {pastPicks.length}</span> reached Gold.
          </p>
          <ListTable
            caption={`Silver to Gold picks for the ${review.update} update`}
            rows={pastPicks}
            rowKey={r => r.card_uuid}
            name={r => r.player_name}
            columns={[
              { label: 'Gold chance', align: 'r', cell: r => pct(r.gold_probability) },
              { label: 'OVR before → after', align: 'r', cell: r => <span>{r.ovr_before} &rarr; <span className={`strong ${r.ovr_after >= GOLD_FLOOR ? 'tier' : ''}`}>{r.ovr_after}</span></span> },
              { label: 'QS change', align: 'r', cell: r => <span className={toneOf(r.realized_qs_change, 0.5)}>{stubs(r.realized_qs_change)}</span> },
            ]}
          />
        </section>
      )}
    </>
  )
}
