import { useEffect } from 'react'
import type { ModelSummary } from '../types'
import { pct, shortDate, stubs } from '../types'
import { Icon } from './ui'

export default function Methodology({ model, onClose }: { model?: ModelSummary; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const m = model?.metrics
  const folds = (model?.folds ?? []).slice(1) // the first fold has no earlier predictions to calibrate on

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby="method-title">
        <div className="drawer-head" style={{ padding: '14px 24px' }}>
          <h2 id="method-title">How the forecast works</h2>
          <button className="icon-btn" style={{ marginLeft: 'auto' }} onClick={onClose} aria-label="Close">
            <Icon name="close" />
          </button>
        </div>
        <div className="dialog-body">
          <p>
            San Diego Studio re-rates Live Series cards in a monthly attribute update, based mostly on how
            players have performed. For every card, the model estimates how each rated attribute will move
            in the next one, then rolls that up into the chance its overall rating goes up, down or stays put.
          </p>

          <h3>What goes in</h3>
          <ul>
            <li>Every MLB 26 roster update from The Show's public API, with each card's rating history.</li>
            <li>Game-by-game MLB stats: last 7, 14 and 30 days, season to date, and last season.</li>
            <li>How far a card's ratings sit from what comparable stat lines have earned before.</li>
          </ul>

          <h3>How it's tested</h3>
          <p>
            Each past attribute update was predicted using only information available before it: the stats
            through that date and the earlier updates. These are the results, averaged over{' '}
            {folds.length} updates.
          </p>

          {m && (
            <div className="table-wrap" style={{ marginTop: 10 }}>
              <table className="grid compact">
                <tbody>
                  <tr><td>Top 25 predicted upgrades that went up</td><td className="r num"><strong>{pct(m.top25_up_hit_rate)}</strong></td><td className="r faint">{pct(m.base_rate_up)} of all cards</td></tr>
                  <tr><td>Their average OVR change</td><td className="r num"><strong>{m.top25_up_avg_ovr_delta >= 0 ? '+' : ''}{m.top25_up_avg_ovr_delta.toFixed(1)}</strong></td><td /></tr>
                  <tr><td>Top 25 predicted downgrades that went down</td><td className="r num"><strong>{pct(m.top25_down_hit_rate)}</strong></td><td className="r faint">{pct(m.base_rate_down)} of all cards</td></tr>
                  <tr><td>Quicksell change per card, top 25 by expected value</td><td className="r num"><strong>{stubs(m.top25_ev_avg_qs_gain)}</strong></td><td className="r faint">{stubs(m.all_avg_qs_gain)} average card</td></tr>
                  <tr><td>Average forecast chance of an upgrade</td><td className="r num"><strong>{pct(m.upgrade_prob_mean)}</strong></td><td className="r faint">{pct(m.base_rate_up)} actually upgraded</td></tr>
                  <tr><td>Attribute changes inside the shown 80% range</td><td className="r num"><strong>{pct(m.interval_coverage)}</strong></td><td /></tr>
                </tbody>
              </table>
            </div>
          )}

          {folds.length > 0 && (
            <>
              <h3>By update</h3>
              <div className="table-wrap">
                <table className="grid compact">
                  <thead>
                    <tr>
                      <th>Update</th>
                      <th className="r">Trained on</th>
                      <th className="r">Top 25 up</th>
                      <th className="r">Top 25 down</th>
                      <th className="r">Top 25 by value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {folds.map(f => (
                      <tr key={f.test_update} style={{ cursor: 'default' }}>
                        <td>{shortDate(f.test_update)}</td>
                        <td className="r num muted">{f.train_updates} updates</td>
                        <td className="r num">{pct(f.top25_up_hit_rate)}</td>
                        <td className="r num">{pct(f.top25_down_hit_rate)}</td>
                        <td className="r num">{stubs(f.top25_ev_avg_qs_gain)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <h3>Limits</h3>
          <ul>
            <li>Only this season has point-in-time stats, so the model has learned from {model?.trained_on?.length ?? 'a handful of'} attribute updates.</li>
            <li>Quicksell figures assume you buy at the quicksell floor. Market prices are higher.</li>
            <li>This is a fan project and isn't affiliated with San Diego Studio or MLB.</li>
          </ul>
        </div>
      </div>
    </>
  )
}
