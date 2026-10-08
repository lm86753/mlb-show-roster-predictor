import { useState } from 'react'
import type { DashboardResponse, ReliabilityRow, ReviewPick } from '../types'
import { ATTR_LABELS, HITTER_ATTRS, PITCHER_ATTRS, pct, shortDate, signed, stubs, toneOf } from '../types'
import ListTable from '../components/ListTable'
import { Segmented } from '../components/ui'

type ReviewList = 'top_upgrades' | 'top_downgrades' | 'top_value'
type CalibrationKey = 'upgrade' | 'downgrade' | 'tier_up' | 'silver_to_gold'

const CALIBRATION_LABELS: Record<CalibrationKey, string> = {
  upgrade: 'Upgrade', downgrade: 'Downgrade', tier_up: 'Rarity tier up', silver_to_gold: 'Silver to Gold',
}

/** Brier skill: how much better than always forecasting the base rate (100% = perfect, 0% = no better). */
function skill(score?: number, baseline?: number): string {
  if (score == null || !baseline) return '—'
  return pct(1 - score / baseline)
}

const unsigned = (n?: number) => (n == null ? '—' : Math.round(n).toLocaleString())

function CalibrationTable({ rows, label }: { rows: ReliabilityRow[]; label: string }) {
  return (
    <div className="table-wrap">
      <table className="grid compact">
        <caption className="visually-hidden">{label} forecast calibration</caption>
        <thead>
          <tr>
            <th>Forecast chance</th>
            <th className="r">Cards</th>
            <th className="r">Average forecast</th>
            <th className="r">Actually happened</th>
            <th className="hide-sm">Forecast vs actual</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.lo}>
              <td className="num">{Math.round(r.lo * 100)}–{Math.round(r.hi * 100)}%</td>
              <td className="r num muted">{r.n.toLocaleString()}</td>
              <td className="r num">{pct(r.forecast)}</td>
              <td className="r num strong">{pct(r.actual)}</td>
              <td className="hide-sm">
                <div className="calib" aria-hidden="true">
                  <span className="calib-actual" style={{ width: `${r.actual * 100}%` }} />
                  <span className="calib-forecast" style={{ left: `${r.forecast * 100}%` }} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function TrackRecord({ data, onSelect }: { data: DashboardResponse; onSelect: (uuid: string) => void }) {
  const [list, setList] = useState<ReviewList>('top_upgrades')
  const [calChoice, setCalKey] = useState<CalibrationKey>('upgrade')
  const model = data.model
  const m = model?.metrics
  const folds = (model?.folds ?? []).slice(1) // the first fold has no earlier predictions to calibrate on
  const review = model?.last_update_review
  const picks = review?.[list] ?? []
  const hits = picks.filter(p => (list === 'top_downgrades' ? p.ovr_after < p.ovr_before : list === 'top_upgrades' ? p.ovr_after > p.ovr_before : p.realized_qs_change > 0)).length
  const known = new Set(data.predictions.map(p => p.card_uuid))
  const byAttr = model?.by_attribute ?? {}
  const attrRows = [...HITTER_ATTRS, ...PITCHER_ATTRS].filter(([k]) => byAttr[k]).map(([k]) => ({ key: k, ...byAttr[k] }))
  const calibration = model?.calibration ?? {}
  const calKeys = (Object.keys(CALIBRATION_LABELS) as CalibrationKey[]).filter(k => calibration[k]?.length)
  const calKey = calKeys.includes(calChoice) ? calChoice : calKeys[0]

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
          {m.ovr_interval_coverage != null && (
            <div className="figure">
              <div className="figure-value num">{pct(m.ovr_interval_coverage)}</div>
              <div className="figure-label">of OVR changes landed inside the card's 80% range</div>
            </div>
          )}
          {m.s2g_top_hit_rate != null && (
            <div className="figure">
              <div className="figure-value num">{pct(m.s2g_top_hit_rate)}</div>
              <div className="figure-label">of the top 10 Silver to Gold picks reached Gold (average Silver: {pct(m.s2g_base_rate)})</div>
            </div>
          )}
          {m.tier_up_prob_mean != null && (
            <div className="figure">
              <div className="figure-value num">{pct(m.tier_up_prob_mean)}</div>
              <div className="figure-label">average forecast chance of a rarity tier jump, against {pct(m.base_rate_tier_up)} that jumped</div>
            </div>
          )}
        </div>
      )}

      {m && (
        <section className="block">
          <h2 className="section">Every forecast, scored</h2>
          <p className="section-note">
            Each probability on the site is checked against what happened. Skill is how much better the forecast scored
            (Brier score) than always guessing the average rate: 0% means no better, 100% means perfect.
          </p>
          <div className="table-wrap">
            <table className="grid compact">
              <caption className="visually-hidden">Forecast skill by type</caption>
              <thead>
                <tr><th>Forecast</th><th className="r">Average forecast</th><th className="r">Actual rate</th><th className="r">Skill</th></tr>
              </thead>
              <tbody>
                <tr><td>OVR goes up</td><td className="r num">{pct(m.upgrade_prob_mean)}</td><td className="r num">{pct(m.base_rate_up)}</td><td className="r num strong">{skill(m.upgrade_brier, m.upgrade_brier_baseline)}</td></tr>
                <tr><td>OVR goes down</td><td className="r num">{pct(m.downgrade_prob_mean)}</td><td className="r num">{pct(m.base_rate_down)}</td><td className="r num strong">{skill(m.downgrade_brier, m.downgrade_brier_baseline)}</td></tr>
                <tr><td>Rarity tier jump</td><td className="r num">{pct(m.tier_up_prob_mean)}</td><td className="r num">{pct(m.base_rate_tier_up)}</td><td className="r num strong">{skill(m.tier_up_brier, m.tier_up_brier_baseline)}</td></tr>
                <tr><td>Silver reaches Gold</td><td className="r num">{pct(m.s2g_prob_mean)}</td><td className="r num">{pct(m.s2g_base_rate)}</td><td className="r num strong">{skill(m.s2g_brier, m.s2g_brier_baseline)}</td></tr>
              </tbody>
            </table>
          </div>
          {m.qs_ev_mae != null && (
            <p className="note">
              Expected quicksell change: off by {unsigned(m.qs_ev_mae)} stubs per card on average
              (guessing no change: {unsigned(m.qs_ev_mae_baseline)}), rank correlation with what happened {m.qs_ev_spearman?.toFixed(2) ?? '—'}.
              {m.ovr_mae != null && <> OVR change: off by {m.ovr_mae.toFixed(2)} on average (guessing no change: {m.ovr_mae_baseline?.toFixed(2)}).</>}
            </p>
          )}
        </section>
      )}

      {attrRows.length > 0 && (
        <section className="block">
          <h2 className="section">Accuracy by attribute</h2>
          <p className="section-note">
            Every attribute the model forecasts, backtested on its own. Error is the average miss in rating points,
            next to the miss from predicting no change. Direction is how often the forecast called up or down right
            when the attribute moved.
          </p>
          <div className="table-wrap">
            <table className="grid compact">
              <caption className="visually-hidden">Backtest accuracy by attribute</caption>
              <thead>
                <tr>
                  <th>Attribute</th>
                  <th className="r hide-sm">Forecasts</th>
                  <th className="r hide-sm">Moved</th>
                  <th className="r">Error</th>
                  <th className="r">No-change error</th>
                  <th className="r" title="Share of squared error removed versus predicting no change">Skill</th>
                  <th className="r">Direction</th>
                  <th className="r hide-sm">In 80% range</th>
                </tr>
              </thead>
              <tbody>
                {attrRows.map(a => (
                  <tr key={a.key}>
                    <td>{ATTR_LABELS[a.key] ?? a.key} <span className="faint">{a.group === 'hitting' ? 'hitter' : 'pitcher'}</span></td>
                    <td className="r num muted hide-sm">{a.n.toLocaleString()}</td>
                    <td className="r num muted hide-sm">{pct(a.moved_rate)}</td>
                    <td className="r num">{a.attr_mae.toFixed(2)}</td>
                    <td className="r num muted">{a.attr_mae_baseline.toFixed(2)}</td>
                    <td className="r num strong">{pct(a.attr_skill)}</td>
                    <td className="r num">{pct(a.attr_direction_acc)}</td>
                    <td className="r num hide-sm">{pct(a.interval_coverage)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            Most attributes don't move at a given update, so guessing no change has a low average error. The model
            earns its keep on the big moves, which is what skill measures.
          </p>
        </section>
      )}

      {calKey && (
        <section className="block">
          <h2 className="section">Are the percentages honest?</h2>
          <p className="section-note">
            Forecasts grouped by the chance shown. When the model says 60%, about 60% of those cards should do it.
            The bar is what actually happened; the tick is the average forecast.
          </p>
          <div className="toolbar" style={{ paddingTop: 0 }}>
            <Segmented<CalibrationKey> label="Forecast type" value={calKey} onChange={setCalKey}
              options={calKeys.map(k => ({ value: k, label: CALIBRATION_LABELS[k] }))} />
          </div>
          <CalibrationTable rows={calibration[calKey] ?? []} label={CALIBRATION_LABELS[calKey]} />
        </section>
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
