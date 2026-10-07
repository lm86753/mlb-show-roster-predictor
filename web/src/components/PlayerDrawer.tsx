import { useEffect, useMemo, useState } from 'react'
import type { HistoryUpdate, ModelSummary, Prediction } from '../types'
import {
  ATTR_LABELS, HITTER_ATTRS, MAX_CARD_COPIES, PITCHER_ATTRS, RATING_MAX,
  pct, shortDate, signed, stubs, toneOf,
} from '../types'
import { fetchHistory } from '../api'
import { CardArt, Icon, Rarity } from './ui'

const RANGE_SPAN = 10 // attribute range bars cover −10…+10

function RangeBar({ lo, hi, mid }: { lo: number; hi: number; mid: number }) {
  const x = (v: number) => `${((Math.max(-RANGE_SPAN, Math.min(RANGE_SPAN, v)) + RANGE_SPAN) / (2 * RANGE_SPAN)) * 100}%`
  const color = mid > 0.25 ? 'var(--up)' : mid < -0.25 ? 'var(--down)' : 'var(--text-3)'
  return (
    <div className="range" aria-hidden="true">
      <span className="range-span" style={{ left: x(lo), right: `calc(100% - ${x(hi)})` }} />
      <span className="range-zero" style={{ left: '50%' }} />
      <span className="range-dot" style={{ left: x(mid), background: color }} />
    </div>
  )
}

function useHistory(uuid: string) {
  const [state, setState] = useState<{ uuid: string; updates: HistoryUpdate[] } | null>(null)
  useEffect(() => {
    let alive = true
    fetchHistory(uuid)
      .then(updates => { if (alive) setState({ uuid, updates }) })
      .catch(() => { if (alive) setState({ uuid, updates: [] }) })
    return () => { alive = false }
  }, [uuid])
  return state?.uuid === uuid ? state.updates : null
}

export default function PlayerDrawer({ prediction: p, model, onClose }: {
  prediction: Prediction; model?: ModelSummary; onClose: () => void
}) {
  const history = useHistory(p.card_uuid)
  const [edits, setEdits] = useState<Record<string, number>>({})

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const attrs = useMemo(() => {
    const byName = Object.fromEntries((p.attributes ?? []).map(a => [a.attribute_name, a]))
    const order = p.is_hitter ? [...HITTER_ATTRS, ...PITCHER_ATTRS] : [...PITCHER_ATTRS, ...HITTER_ATTRS]
    return order.filter(([k]) => byName[k]).map(([k]) => ({ ...byName[k], key: k }))
  }, [p.attributes, p.is_hitter])

  const weights = model?.ovr_weights?.[p.is_hitter ? 'hitting' : 'pitching']
  const scenario = useMemo(() => {
    if (!weights || Object.keys(edits).length === 0) return null
    const delta = attrs.reduce((sum, a) => {
      const next = edits[a.key] ?? a.rating_before + a.predicted_delta
      return sum + (weights[a.key] ?? 0) * (next - a.rating_before)
    }, 0)
    return { delta, ovr: Math.max(40, Math.min(99, Math.round(p.current_ovr + delta))) }
  }, [edits, attrs, weights, p.current_ovr])

  const up = p.upgrade_probability
  const down = p.downgrade_probability
  const none = Math.max(0, 1 - up - down)
  const ev = p.expected_value_per_card ?? 0
  const projected = Math.min(99, Math.round(p.current_ovr + p.predicted_ovr_delta))

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`${p.player_name} forecast`}>
        <div className="drawer-head">
          <span className="muted" style={{ fontSize: 13 }}>Player forecast</span>
          <button className="icon-btn" style={{ marginLeft: 'auto' }} onClick={onClose} aria-label="Close">
            <Icon name="close" />
          </button>
        </div>

        <div className="drawer-body">
          <section className="player-hero">
            <CardArt uuid={p.card_uuid} size="lg" alt={`${p.player_name} card`} />
            <div style={{ minWidth: 0, flex: 1 }}>
              <h2>{p.player_name}</h2>
              <div className="muted" style={{ marginTop: 2 }}>
                {[p.team, p.position].filter(Boolean).join(' · ')} &middot; <Rarity rarity={p.current_rarity} />
              </div>
              <div className="kv">
                <div>
                  <div className="kv-value num">
                    {p.current_ovr} &rarr; {projected}
                  </div>
                  <div className="kv-label">
                    OVR, expected change <span className={toneOf(p.predicted_ovr_delta, 0.25)}>{signed(p.predicted_ovr_delta)}</span>
                    {p.ovr_delta_sd != null && <> &plusmn;{p.ovr_delta_sd.toFixed(1)}</>}
                  </div>
                </div>
                <div>
                  <div className={`kv-value num ${toneOf(ev, 0.5)}`}>{stubs(ev)}</div>
                  <div className="kv-label">Expected quicksell change / card</div>
                </div>
                <div>
                  <div className="kv-value num">{p.current_qs.toLocaleString()}</div>
                  <div className="kv-label">Quicksell now</div>
                </div>
                <div>
                  <div className={`kv-value num ${p.tier_jump_probability >= 0.05 ? 'tier' : p.tier_down_probability >= 0.05 ? 'down' : ''}`}>
                    {p.tier_jump_probability >= p.tier_down_probability ? pct(p.tier_jump_probability) : pct(p.tier_down_probability)}
                  </div>
                  <div className="kv-label">
                    Chance of {p.tier_jump_probability >= p.tier_down_probability ? 'moving up' : 'dropping'} a rarity tier
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section>
            <h3 className="section-title">Next attribute update</h3>
            <div className="outcome" role="img" aria-label={`Up ${pct(up)}, no change ${pct(none)}, down ${pct(down)}`}>
              <span style={{ width: `${up * 100}%`, background: 'var(--up)' }} />
              <span style={{ width: `${none * 100}%`, background: 'var(--border-strong)' }} />
              <span style={{ width: `${down * 100}%`, background: 'var(--down)' }} />
            </div>
            <div className="outcome-legend num">
              <span><span className="up">{pct(up)}</span> OVR goes up</span>
              <span>{pct(none)} no change</span>
              <span><span className="down">{pct(down)}</span> goes down</span>
            </div>
            {!p.sample_size_ok && (
              <p className="note" style={{ marginBottom: 0 }}>Little MLB playing time this season, so this forecast leans on rating history.</p>
            )}
          </section>

          {attrs.length > 0 && (
            <section>
              <h3 className="section-title">Attributes</h3>
              <div className="table-wrap">
                <table className="grid compact">
                  <thead>
                    <tr>
                      <th>Attribute</th>
                      <th className="r">Now</th>
                      <th className="r" title="Rating these stats usually earn">Stats</th>
                      <th className="r">Change</th>
                      <th className="hide-sm">80% range</th>
                      {weights && <th className="r" title="Edit to see the OVR impact">Scenario</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {attrs.map(a => (
                      <tr key={a.key} style={{ cursor: 'default' }}>
                        <td>{ATTR_LABELS[a.key] ?? a.key}</td>
                        <td className="r num">{a.rating_before}</td>
                        <td className="r num muted">
                          {a.stat_projection != null ? Math.round(a.stat_projection) : '—'}
                        </td>
                        <td className={`r num delta ${toneOf(a.predicted_delta, 0.25)}`}>{signed(a.predicted_delta)}</td>
                        <td className="hide-sm">
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <RangeBar lo={a.confidence_low} hi={a.confidence_high} mid={a.predicted_delta} />
                            <span className="faint num" style={{ fontSize: 12 }}>{signed(a.confidence_low, 0)} to {signed(a.confidence_high, 0)}</span>
                          </div>
                        </td>
                        {weights && (
                          <td className="r">
                            <input
                              type="number" min={0} max={RATING_MAX}
                              aria-label={`${ATTR_LABELS[a.key]} scenario rating`}
                              value={edits[a.key] ?? a.projected_rating}
                              onChange={e => {
                                const v = parseInt(e.target.value)
                                if (!isNaN(v)) setEdits(prev => ({ ...prev, [a.key]: Math.max(0, Math.min(RATING_MAX, v)) }))
                              }}
                            />
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {scenario ? (
                <p className="note" style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <span>
                    Scenario OVR: <strong className="num" style={{ color: 'var(--text)' }}>{scenario.ovr}</strong>{' '}
                    (<span className={toneOf(scenario.delta, 0.25)}>{signed(scenario.delta)}</span>)
                  </span>
                  <button className="btn-link" onClick={() => setEdits({})}>Reset</button>
                </p>
              ) : (
                <p className="note">"Stats" is the rating this season's numbers usually earn. Edit a scenario rating to see how it would move the OVR.</p>
              )}
            </section>
          )}

          <section>
            <h3 className="section-title">Recent roster updates</h3>
            {history === null ? (
              <p className="note">Loading&hellip;</p>
            ) : history.length === 0 ? (
              <p className="note">No changes recorded for this card this season.</p>
            ) : (
              <div className="history">
                {history.slice(0, 6).map(u => {
                  const d = u.ovr_before != null && u.ovr_after != null ? u.ovr_after - u.ovr_before : 0
                  return (
                    <div key={u.update_date} className="history-row">
                      <span className="muted">{shortDate(u.update_date)}</span>
                      <span className="num">
                        {u.ovr_before} &rarr; {u.ovr_after} {d !== 0 && <span className={toneOf(d)}>({signed(d, 0)})</span>}
                      </span>
                      <span className="history-changes">
                        {u.changes.map((c, i) => (
                          <span key={i}>
                            {i > 0 && ', '}
                            {ATTR_LABELS[c.attribute] ?? c.attribute} <span className={`num ${toneOf(c.delta)}`}>{signed(c.delta, 0)}</span>
                          </span>
                        ))}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </section>

          <p className="note" style={{ margin: 0 }}>
            Expected quicksell change is weighted over every possible OVR outcome and assumes buying at the
            quicksell floor; &times;{MAX_CARD_COPIES} copies that's {stubs(ev * MAX_CARD_COPIES)} stubs.
          </p>
        </div>
      </aside>
    </>
  )
}
