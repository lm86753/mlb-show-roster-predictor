import type { Prediction } from '../types'
import { pct, signed, toneOf } from '../types'
import { CardArt } from './ui'

export default function CardGrid({ rows, onSelect }: { rows: Prediction[]; onSelect: (uuid: string) => void }) {
  return (
    <div className="cards">
      {rows.map(p => {
        const projected = Math.min(99, Math.round(p.current_ovr + p.predicted_ovr_delta))
        return (
          <button key={p.card_uuid} type="button" className="card-tile" onClick={() => onSelect(p.card_uuid)}>
            <CardArt uuid={p.card_uuid} alt={`${p.player_name} card`} />
            <div>
              <div className="card-tile-name">{p.player_name}</div>
              <div className="card-tile-row num">
                <span>
                  {p.current_ovr} &rarr; {projected}{' '}
                  <span className={toneOf(p.predicted_ovr_delta, 0.25)}>({signed(p.predicted_ovr_delta)})</span>
                </span>
                <span>
                  <span className={p.upgrade_probability >= 0.5 ? 'up' : ''}>{pct(p.upgrade_probability)}</span>
                  {' / '}
                  <span className={p.downgrade_probability >= 0.5 ? 'down' : ''}>{pct(p.downgrade_probability)}</span>
                </span>
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
