import { useMemo, useState } from 'react'
import type { Prediction, AttributeItem } from '../types'
import { formatDelta, deltaColor, HITTER_ATTRS, PITCHER_ATTRS, QS_TIERS } from '../types'

interface Props {
  prediction: Prediction
}

function getQSValue(ovr: number): number {
  let val = 0
  for (const [minOvr, qs] of QS_TIERS) {
    if (ovr >= minOvr) val = qs
  }
  return val
}

export default function DetailPanel({ prediction }: Props) {
  const {
    player_name, team, position, current_ovr,
    predicted_ovr_delta, current_rarity, upgrade_probability,
    downgrade_probability, tier_jump_probability, attributes,
    is_hitter, avg_gap, direction_consensus,
  } = prediction

  const [edits, setEdits] = useState<Record<string, number>>({})

  const attrList = useMemo(() => {
    if (!attributes || attributes.length === 0) return []
    const attrMap: Record<string, AttributeItem> = {}
    for (const a of attributes) {
      if (a.attribute_name) attrMap[a.attribute_name] = a
    }
    const order = is_hitter ? HITTER_ATTRS : PITCHER_ATTRS
    return order
      .map(([key, label]) => {
        const item = attrMap[key]
        if (!item) return null
        return { label, ...item }
      })
      .filter(Boolean) as (AttributeItem & { label: string })[]
  }, [attributes, is_hitter])

  const deltas = useMemo(() => {
    const d: number[] = []
    for (let i = 0; i < attrList.length; i++) {
      const a = attrList[i]
      const edited = edits[a.label]
      if (edited != null && edited !== a.rating_before) {
        d.push(edited - a.rating_before)
      }
    }
    return d
  }, [edits, attrList])

  const recalcOvr = useMemo(() => {
    if (deltas.length === 0) return null
    const avgDelta = deltas.reduce((s, d) => s + d, 0) / deltas.length
    const ovrMult = 2.0 - 0.5 * (current_ovr / 99.0)
    const modelDelta = avgDelta * ovrMult
    return { delta: modelDelta, ovr: Math.max(0, Math.min(99, Math.round(current_ovr + modelDelta))) }
  }, [deltas, current_ovr])

  const dc = deltaColor(predicted_ovr_delta)
  const upColor = upgrade_probability > 0.5 ? '#48bb78' : '#a0aec0'
  const dnColor = downgrade_probability > 0.5 ? '#f56565' : '#a0aec0'

  const newOvr = predicted_ovr_delta != null && !isNaN(predicted_ovr_delta)
    ? Math.round(current_ovr + predicted_ovr_delta) : current_ovr

  const curQS = getQSValue(current_ovr)
  const newQS = getQSValue(newOvr)
  const profit = newQS - curQS
  const roi = curQS > 0 ? (profit / curQS) * 100 : 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 12, color: '#e2e8f0' }}>
        <strong>{player_name}</strong> &bull; {team} &bull; {position}
        <span style={{ marginLeft: 8, color: '#718096' }}>
          OVR: {current_ovr} &bull; {current_rarity}
        </span>
      </div>

      {/* Metric cards */}
      <div style={{ display: 'flex', gap: 12 }}>
        <div style={{ flex: 1, background: '#1a202c', borderRadius: 8, padding: 8, textAlign: 'center', border: '1px solid #2d3748' }}>
          <div style={{ color: dc, fontSize: 20, fontWeight: 800 }}>{formatDelta(predicted_ovr_delta)}</div>
          <div style={{ color: '#a0aec0', fontSize: 10 }}>Predicted OVR Change</div>
        </div>
        <div style={{ flex: 1, background: '#1a202c', borderRadius: 8, padding: 8, textAlign: 'center', border: '1px solid #2d3748' }}>
          <div style={{ color: upColor, fontSize: 20, fontWeight: 800 }}>{(upgrade_probability * 100).toFixed(1)}%</div>
          <div style={{ color: '#a0aec0', fontSize: 10 }}>Upgrade Probability</div>
        </div>
        <div style={{ flex: 1, background: '#1a202c', borderRadius: 8, padding: 8, textAlign: 'center', border: '1px solid #2d3748' }}>
          <div style={{ color: dnColor, fontSize: 20, fontWeight: 800 }}>{(downgrade_probability * 100).toFixed(1)}%</div>
          <div style={{ color: '#a0aec0', fontSize: 10 }}>Downgrade Probability</div>
        </div>
      </div>

      {tier_jump_probability > 0 && (
        <div style={{ color: '#e2e8f0', fontSize: 12 }}>
          <strong>Tier Jump Probability:</strong> {(tier_jump_probability * 100).toFixed(1)}%
        </div>
      )}

      {/* Stub calculator */}
      <div style={{ background: '#1a202c', borderRadius: 8, padding: 10, border: '1px solid #2d3748' }}>
        <div style={{ color: '#a0aec0', fontSize: 10, marginBottom: 4 }}>Stub Profit Calculator</div>
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#e2e8f0', flexWrap: 'wrap' }}>
          <span>QS: <strong>{curQS.toLocaleString()}</strong> &rarr; <strong>{newQS.toLocaleString()}</strong></span>
          <span>Profit/card: <strong style={{ color: profit >= 0 ? '#48bb78' : '#f56565' }}>{profit.toLocaleString()}</strong></span>
          <span>x20: <strong style={{ color: profit >= 0 ? '#48bb78' : '#f56565' }}>{(profit * 20).toLocaleString()}</strong></span>
          <span>ROI: <strong style={{ color: roi >= 0 ? '#48bb78' : '#f56565' }}>{roi >= 0 ? '+' : ''}{roi.toFixed(1)}%</strong></span>
        </div>
      </div>

      {/* Attribute editor */}
      {attrList.length > 0 && (
        <div>
          <div style={{ color: '#e2e8f0', fontSize: 11, marginBottom: 4 }}>
            Attribute Projections <span style={{ color: '#718096' }}>(edit Projected values to recalculate)</span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: '#718096', borderBottom: '1px solid #2d3748' }}>
                  <th style={{ textAlign: 'left', padding: '4px 6px' }}>Stat</th>
                  <th style={{ textAlign: 'center', padding: '4px 6px' }}>Current</th>
                  <th style={{ textAlign: 'center', padding: '4px 6px' }}>Projected</th>
                  <th style={{ textAlign: 'center', padding: '4px 6px' }}>±Δ</th>
                  <th style={{ textAlign: 'center', padding: '4px 6px' }}>Data</th>
                </tr>
              </thead>
              <tbody>
                  {attrList.map((attr) => {
                    const editedVal = edits[attr.label]
                    const displayVal = editedVal != null ? editedVal : (attr.projected_rating || attr.rating_before)
                    const diff = displayVal - attr.rating_before
                    const diffColor = diff > 0 ? '#48bb78' : diff < 0 ? '#f56565' : '#a0aec0'
                    return (
                      <tr key={attr.label} style={{ borderBottom: '1px solid #1a202c' }}>
                        <td style={{ padding: '3px 6px', color: '#e2e8f0' }}>{attr.label}</td>
                        <td style={{ padding: '3px 6px', textAlign: 'center', color: '#e2e8f0' }}>{attr.rating_before}</td>
                        <td style={{ padding: '3px 6px', textAlign: 'center' }}>
                          <input
                            type="number"
                            min={0} max={99}
                            value={displayVal}
                            onChange={e => {
                              const v = parseInt(e.target.value) || 0
                              setEdits(prev => ({ ...prev, [attr.label]: Math.max(0, Math.min(99, v)) }))
                            }}
                            style={{
                              width: 48, padding: '2px 4px', borderRadius: 4, border: '1px solid #4a5568',
                              background: '#1a202c', color: '#f7fafc', fontSize: 11, textAlign: 'center',
                            }}
                          />
                        </td>
                        <td style={{ padding: '3px 6px', textAlign: 'center', color: diffColor, fontWeight: 700 }}>
                          {diff > 0 ? '+' : ''}{diff}
                        </td>
                        <td style={{ padding: '3px 6px', textAlign: 'center' }}>
                          {attr.has_stat_data ? (
                            <span style={{ color: '#48bb78', fontSize: 13 }}>&#9679;</span>
                          ) : (
                            <span style={{ color: '#a0aec0', fontSize: 13 }}>&#9679;</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recalculated overall */}
      {recalcOvr && (
        <div style={{
          background: '#2d3748', border: '1px solid #4a5568', borderRadius: 8,
          padding: 10, textAlign: 'center', marginTop: 4,
        }}>
          <div style={{ color: '#a0aec0', fontSize: 11 }}>Recalculated Overall</div>
          <div>
            <span style={{ fontSize: 22, fontWeight: 900, color: '#f7fafc' }}>{current_ovr}</span>{' '}
            <span style={{ fontSize: 16, fontWeight: 700, color: deltaColor(recalcOvr.delta) }}>
              {recalcOvr.delta >= 0 ? '+' : ''}{recalcOvr.delta.toFixed(1)}
            </span>{' '}
            <span style={{ fontSize: 12, color: '#a0aec0' }}>&rarr;</span>{' '}
            <span style={{ fontSize: 22, fontWeight: 900, color: '#f7fafc' }}>{recalcOvr.ovr}</span>
          </div>
          <div style={{ color: '#718096', fontSize: 10 }}>
            Avg &Delta;: {recalcOvr.delta >= 0 ? '+' : ''}{recalcOvr.delta.toFixed(2)} across {deltas.length} edited attributes
          </div>
        </div>
      )}

      <div style={{ fontSize: 11, color: '#e2e8f0' }}>
        <strong>Avg Gap:</strong> {avg_gap?.toFixed(2) ?? '\u2014'}
      </div>
      <div style={{ fontSize: 11, color: '#e2e8f0' }}>
        <strong>Direction Consensus:</strong> {direction_consensus?.toFixed(2) ?? '\u2014'}
      </div>
    </div>
  )
}
