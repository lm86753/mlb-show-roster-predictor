import { useState } from 'react'
import type { DashboardResponse, ReviewPick } from '../types'
import { pct, shortDate, signed, stubs, toneOf } from '../types'
import ListTable from '../components/ListTable'
import { Segmented } from '../components/ui'

type ReviewList = 'top_upgrades' | 'top_downgrades' | 'top_value'

export default function TrackRecord({ data, onSelect }: { data: DashboardResponse; onSelect: (uuid: string) => void }) {
  const [list, setList] = useState<ReviewList>('top_upgrades')
  const model = data.model
  const m = model?.metrics
  const folds = (model?.folds ?? []).slice(1) // the first fold has no earlier predictions to calibrate on
  const review = model?.last_update_review
  const picks = review?.[list] ?? []
  const hits = picks.filter(p => (list === 'top_downgrades' ? p.ovr_after < p.ovr_before : list === 'top_upgrades' ? p.ovr_after > p.ovr_before : p.realized_qs_change > 0)).length
  const known = new Set(data.predictions.map(p => p.card_uuid))

  return (
    <>
      <div className="page-head">
        <h1>Track record</h1>
        <p>
          Every past attribute update was re-predicted using only the information available before it: the
          ratings at the time, MLB stats through {model?.stat_cutoff_days ?? 3} days before the update, and the
          updates that came earlier. These are the results{folds.length ? `, averaged over the ${folds.length} most recent updates` : ''}.
        </p>
      </div>

      {m && (
        <div className="figures">
          <div className="figure">
            <div className="figure-value num">{pct(m.top25_up_hit_rate)}</div>
            <div className="figure-label">of the top 25 upgrade picks went up (average card: {pct(m.base_rate_up)})</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{pct(m.top25_down_hit_rate)}</div>
            <div className="figure-label">of the top 25 downgrade picks went down (average card: {pct(m.base_rate_down)})</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{stubs(m.top25_ev_avg_qs_gain)}</div>
            <div className="figure-label">quicksell change per card for the top 25 value picks (average card: {stubs(m.all_avg_qs_gain)})</div>
          </div>
          <div className="figure">
            <div className="figure-value num">+{m.top25_up_avg_ovr_delta.toFixed(1)}</div>
            <div className="figure-label">average OVR change of the top 25 upgrade picks</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{pct(m.upgrade_prob_mean)}</div>
            <div className="figure-label">average forecast chance of an upgrade, against {pct(m.base_rate_up)} that actually upgraded</div>
          </div>
          <div className="figure">
            <div className="figure-value num">{pct(m.interval_coverage)}</div>
            <div className="figure-label">of attribute changes landed inside the 80% range shown</div>
          </div>
        </div>
      )}

      {review?.update && (
        <section className="block">
          <h2 className="section">The {shortDate(review.update)} update: predicted vs actual</h2>
          <p className="section-note">
            What the model, trained only on earlier updates, picked before the {shortDate(review.update)} update, and what happened.
            {picks.length > 0 && <> <span className="strong num">{hits} of {picks.length}</span> {list === 'top_value' ? 'gained quicksell value' : list === 'top_upgrades' ? 'went up' : 'went down'}.</>}
          </p>
          <div className="toolbar" style={{ paddingTop: 0 }}>
            <Segmented<ReviewList> label="Pick list" value={list} onChange={setList} options={[
              { value: 'top_upgrades', label: 'Upgrade picks' },
              { value: 'top_downgrades', label: 'Downgrade picks' },
              { value: 'top_value', label: 'Value picks' },
            ]} />
          </div>
          <ListTable<ReviewPick>
            caption={`Picks for the ${review.update} update`}
            rows={picks}
            rowKey={p => p.card_uuid}
            name={p => p.player_name}
            onSelect={p => { if (known.has(p.card_uuid)) onSelect(p.card_uuid) }}
            columns={[
              { label: 'Forecast', align: 'r', cell: p => <span className={toneOf(p.predicted_delta, 0.25)}>{signed(p.predicted_delta)}</span> },
              { label: list === 'top_downgrades' ? 'Chance down' : 'Chance up', align: 'r', cell: p => pct(list === 'top_downgrades' ? p.downgrade_probability : p.upgrade_probability) },
              { label: 'OVR before → after', align: 'r', cell: p => <span>{p.ovr_before} &rarr; <span className="strong">{p.ovr_after}</span></span> },
              { label: 'Actual', align: 'r', cell: p => <span className={`delta ${toneOf(p.ovr_after - p.ovr_before)}`}>{signed(p.ovr_after - p.ovr_before, 0)}</span> },
              { label: 'QS change', align: 'r', className: 'hide-sm', cell: p => <span className={toneOf(p.realized_qs_change, 0.5)}>{stubs(p.realized_qs_change)}</span> },
            ]}
          />
        </section>
      )}

      {folds.length > 0 && (
        <section className="block">
          <h2 className="section">By update</h2>
          <p className="section-note">Each row is one update, predicted by a model trained on the updates before it.</p>
          <div className="table-wrap">
            <table className="grid compact">
              <thead>
                <tr>
                  <th>Update</th>
                  <th className="r">Trained on</th>
                  <th className="r">Top 25 up</th>
                  <th className="r">Top 25 down</th>
                  <th className="r">Top 25 value (QS/card)</th>
                  <th className="r hide-sm">Rank correlation</th>
                </tr>
              </thead>
              <tbody>
                {folds.map(f => (
                  <tr key={f.test_update}>
                    <td>{shortDate(f.test_update)}</td>
                    <td className="r num muted">{f.train_updates} updates</td>
                    <td className="r num">{pct(f.top25_up_hit_rate)}</td>
                    <td className="r num">{pct(f.top25_down_hit_rate)}</td>
                    <td className="r num">{stubs(f.top25_ev_avg_qs_gain)}</td>
                    <td className="r num hide-sm">{f.ovr_spearman?.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="block prose">
        <h2 className="section" style={{ color: 'var(--text)' }}>How the forecast works</h2>
        <p>
          San Diego Studio re-rates Live Series cards in a monthly attribute update, based mostly on how players
          have performed. For every card the model estimates how each rated attribute will move, then turns that
          into the chance its overall rating goes up, down or stays put, and the quicksell value that implies.
        </p>
        <ul>
          <li>Ratings and history come from every MLB 26 roster update in The Show's public API.</li>
          <li>Stats come from MLB game logs: last 7, 14 and 30 days, season to date, and last season. Totals are checked against MLB's official season stats.</li>
          <li>Each card is matched to its MLB player by name, pitcher or hitter role, and team, so players who share a name don't get mixed up.</li>
        </ul>
        <p>
          Limits: only this season has stats lined up with past updates, so the model has learned from {model?.trained_on?.length ?? 'a handful of'} updates.
          Quicksell figures assume you buy at the quicksell floor; market prices are higher.
        </p>
      </section>
    </>
  )
}
