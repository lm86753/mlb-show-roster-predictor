import { useState } from 'react'
import type { Prediction } from '../types'
import { getTeamColor, getRarityColor, formatDelta, deltaColor, getCardImageUrl } from '../types'
import DetailPanel from './DetailPanel'

interface Props {
  prediction: Prediction
}

export default function PlayerCard({ prediction }: Props) {
  const [expanded, setExpanded] = useState(false)
  const {
    player_name, team, position, card_uuid, current_ovr,
    predicted_ovr_delta, current_rarity, upgrade_probability,
    downgrade_probability, has_card_image,
  } = prediction

  const teamColor = getTeamColor(team)
  const rarityColor = getRarityColor(current_rarity)
  const deltaStr = formatDelta(predicted_ovr_delta)
  const deltaCol = deltaColor(predicted_ovr_delta)
  const newOvr = predicted_ovr_delta != null && !isNaN(predicted_ovr_delta)
    ? Math.round(current_ovr + predicted_ovr_delta)
    : current_ovr
  const roiPct = predicted_ovr_delta != null && !isNaN(predicted_ovr_delta) && current_ovr > 0
    ? (predicted_ovr_delta * 100 / current_ovr) : 0
  const roiColor = roiPct > 0 ? '#48bb78' : roiPct < 0 ? '#f56565' : '#a0aec0'

  let signal: string, signalColor: string
  if (upgrade_probability > 0.75 && predicted_ovr_delta > 1.0) {
    signal = 'BUY'; signalColor = '#48bb78'
  } else if (downgrade_probability > 0.75 && predicted_ovr_delta < -1.0) {
    signal = 'SELL'; signalColor = '#f56565'
  } else {
    signal = 'HOLD'; signalColor = '#ed8936'
  }

  const cardStyle: React.CSSProperties = {
    background: '#1a202c', border: `1px solid ${teamColor}44`, borderRadius: 10,
    overflow: 'hidden', cursor: 'pointer', transition: 'all 0.2s ease',
    position: 'relative',
  }

  return (
    <div
      style={cardStyle}
      onMouseEnter={e => {
        e.currentTarget.style.transform = 'translateY(-3px)'
        e.currentTarget.style.boxShadow = `0 8px 25px ${teamColor}44`
        e.currentTarget.style.borderColor = `${teamColor}88`
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = 'translateY(0)'
        e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)'
        e.currentTarget.style.borderColor = `${teamColor}44`
      }}
    >
      {has_card_image ? (
        <div style={{
          background: `#1a202c url(${getCardImageUrl(card_uuid)}) no-repeat center center / cover`,
          width: '100%', aspectRatio: '363/512', position: 'relative',
        }}>
          <div style={{
            position: 'absolute', inset: 0,
            background: 'linear-gradient(180deg,rgba(0,0,0,0.1) 40%,rgba(0,0,0,0.65) 100%)',
          }} />
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 3, background: rarityColor, zIndex: 2 }} />
          <div style={{ position: 'absolute', top: 8, left: 10, display: 'flex', alignItems: 'center', gap: 4, zIndex: 2 }}>
            <span style={{ fontWeight: 600, fontSize: 11, color: '#f7fafc', textShadow: '0 1px 8px rgba(0,0,0,0.9)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {player_name}
            </span>
            <span style={{ fontSize: 9, color: '#cbd5e0', textShadow: '0 1px 8px rgba(0,0,0,0.9)' }}>&bull;</span>
            <span style={{ fontSize: 9, color: '#cbd5e0', textShadow: '0 1px 8px rgba(0,0,0,0.9)', whiteSpace: 'nowrap' }}>{team}</span>
            <span style={{ fontSize: 9, color: '#cbd5e0', textShadow: '0 1px 8px rgba(0,0,0,0.9)' }}>&bull;</span>
            <span style={{ fontSize: 9, color: '#cbd5e0', textShadow: '0 1px 8px rgba(0,0,0,0.9)' }}>{position}</span>
          </div>
          <div style={{ position: 'absolute', top: 8, right: 10, zIndex: 2 }}>
            <span style={{
              background: `${rarityColor}55`, color: rarityColor, padding: '2px 8px',
              borderRadius: 8, fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
              textShadow: '0 1px 4px rgba(0,0,0,0.6)',
            }}>
              {current_rarity}
            </span>
          </div>
          <div style={{ position: 'absolute', bottom: 10, left: 10, display: 'flex', alignItems: 'baseline', gap: 8, zIndex: 2 }}>
            <span style={{ fontSize: 48, fontWeight: 900, color: '#f7fafc', textShadow: '0 2px 14px rgba(0,0,0,0.95)', lineHeight: 1 }}>
              {current_ovr}
            </span>
            <span style={{ fontSize: 26, fontWeight: 700, color: deltaCol, textShadow: '0 2px 12px rgba(0,0,0,0.95)' }}>
              {deltaStr}
            </span>
            <span style={{ fontSize: 18, fontWeight: 600, color: '#a0aec0', textShadow: '0 2px 12px rgba(0,0,0,0.95)' }}>
              &rarr;
            </span>
            <span style={{ fontSize: 26, fontWeight: 700, color: '#f7fafc', textShadow: '0 2px 12px rgba(0,0,0,0.95)' }}>
              {newOvr}
            </span>
            <span style={{
              marginLeft: 8, background: `${roiColor}44`, color: roiColor,
              padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 700,
              textShadow: '0 1px 4px rgba(0,0,0,0.6)',
            }}>
              {roiPct > 0 ? '+' : ''}{roiPct.toFixed(1)}%
            </span>
            <span style={{
              background: `${signalColor}55`, color: signalColor,
              padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 700,
              textShadow: '0 1px 4px rgba(0,0,0,0.6)',
            }}>
              {signal}
            </span>
          </div>
        </div>
      ) : (
        <div onClick={() => setExpanded(!expanded)} style={{ padding: 0 }}>
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 3, background: rarityColor }} />
          <div style={{ padding: '10px 10px 6px 10px', display: 'flex', gap: 10, alignItems: 'center' }}>
            <div style={{
              width: 40, height: 56, borderRadius: 4, flexShrink: 0,
              background: `linear-gradient(135deg, ${rarityColor}33, ${rarityColor}55)`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 18, fontWeight: 800, color: rarityColor,
            }}>
              {current_ovr}
            </div>
            <div style={{ minWidth: 0, flex: 1 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#f7fafc', lineHeight: 1.2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {player_name}
              </div>
              <div style={{ fontSize: 10, color: '#a0aec0', marginTop: 1 }}>
                {team} &bull; {position}
              </div>
            </div>
            <div style={{
              background: `linear-gradient(135deg, ${rarityColor}33, ${rarityColor}55)`,
              color: rarityColor, padding: '1px 8px', borderRadius: 10,
              fontSize: 9, fontWeight: 700, textTransform: 'uppercase', whiteSpace: 'nowrap',
            }}>
              {current_rarity}
            </div>
          </div>
          <div style={{ padding: '0 10px 6px 10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 24, fontWeight: 800, color: '#f7fafc', lineHeight: 1 }}>{current_ovr}</span>
              <span style={{ fontSize: 16, fontWeight: 700, color: deltaCol }}>{deltaStr}</span>
              <span style={{ fontSize: 12, fontWeight: 600, color: '#a0aec0' }}>&rarr;</span>
              <span style={{ fontSize: 16, fontWeight: 700, color: '#f7fafc' }}>{newOvr}</span>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 3 }}>
                <span style={{ background: `${roiColor}18`, color: roiColor, padding: '1px 6px', borderRadius: 8, fontSize: 9, fontWeight: 700 }}>
                  {roiPct > 0 ? '+' : ''}{roiPct.toFixed(1)}%
                </span>
                <span style={{ background: `${signalColor}18`, color: signalColor, padding: '1px 6px', borderRadius: 8, fontSize: 9, fontWeight: 700 }}>
                  {signal}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {has_card_image && (
        <div onClick={() => setExpanded(!expanded)} style={{ padding: '8px 10px', borderTop: '1px solid #2d3748' }}>
          <span style={{ color: '#718096', fontSize: 11 }}>{expanded ? '▲' : '▼'} Details & Projections</span>
        </div>
      )}

      {expanded && (
        <div style={{ borderTop: '1px solid #2d3748', padding: 12, background: '#162330' }}>
          <DetailPanel prediction={prediction} />
        </div>
      )}
    </div>
  )
}
