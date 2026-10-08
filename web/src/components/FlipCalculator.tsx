import { useState } from 'react'
import type { Prediction } from '../types'
import { MAX_CARD_COPIES, pct, signed, stubs, toneOf } from '../types'
import { MARKET_TAX, flipOutcomes, rarityFor } from '../profit'

const digits = (s: string) => s.replace(/[^0-9]/g, '')

/** Buy-before / sell-after profit from the card's full OVR-move distribution. */
export default function FlipCalculator({ p, heading = 'h3' }: { p: Prediction; heading?: 'h2' | 'h3' }) {
  const [price, setPrice] = useState(String(p.current_qs))
  const [copies, setCopies] = useState('1')
  const [market, setMarket] = useState('')
  const buy = Math.max(0, Number(price) || 0)
  const n = Math.max(1, Math.min(MAX_CARD_COPIES, Math.round(Number(copies) || 1)))
  const marketPrice = Number(market) || null
  const r = flipOutcomes(p, buy, marketPrice)
  const H = heading

  return (
    <section>
      <H className={heading === 'h2' ? 'section' : undefined}>Profit calculator</H>
      <div className="calc calc-3">
        <label>Buy price per card
          <input className="input num" inputMode="numeric" value={price} onChange={e => setPrice(digits(e.target.value))} aria-label="Buy price per card in stubs" />
        </label>
        <label>Copies (max {MAX_CARD_COPIES})
          <input className="input num" inputMode="numeric" value={copies} onChange={e => setCopies(digits(e.target.value))} aria-label="Number of copies" />
        </label>
        <label>Market sell price (optional)
          <input className="input num" inputMode="numeric" placeholder="Quicksell" value={market} onChange={e => setMarket(digits(e.target.value))} aria-label="Expected market sell price after the update" />
        </label>
      </div>
      <div className="calc-out">
        <div><div className={`kv-value num ${toneOf(r.expectedProfit, 0.5)}`}>{stubs(r.expectedProfit)}</div><div className="kv-label">Expected profit per card</div></div>
        <div><div className={`kv-value num ${toneOf(r.expectedProfit, 0.5)}`}>{stubs(r.expectedProfit * n)}</div><div className="kv-label">Total for {n} {n === 1 ? 'copy' : 'copies'}</div></div>
        <div><div className="kv-value num">{pct(r.pProfit)}</div><div className="kv-label">Chance of a profit ({pct(r.pLoss)} chance of a loss)</div></div>
        <div><div className="kv-value num">{Math.floor(r.breakEven).toLocaleString()}</div><div className="kv-label">Break-even buy price</div></div>
        <div><div className={`kv-value num ${toneOf(r.worst, 0.5)}`}>{stubs(r.worst * n)}</div><div className="kv-label">Likely worst case, all copies</div></div>
        <div><div className={`kv-value num ${toneOf(r.best, 0.5)}`}>{stubs(r.best * n)}</div><div className="kv-label">Likely best case, all copies</div></div>
      </div>
      <div className="table-wrap" style={{ marginTop: 14 }}>
        <table className="grid compact">
          <caption className="visually-hidden">Outcomes after the update</caption>
          <thead>
            <tr>
              <th>OVR after</th>
              <th className="r">Chance</th>
              <th className="r">{marketPrice ? 'Sale' : 'Quicksell'}</th>
              <th className="r">Profit / card</th>
            </tr>
          </thead>
          <tbody>
            {r.outcomes.filter(o => o.prob >= 0.005).map(o => (
              <tr key={o.move}>
                <td className="num">{o.ovr} <span className="faint">({signed(o.move, 0)}, {rarityFor(o.ovr)})</span></td>
                <td className="r num">{pct(o.prob)}</td>
                <td className="r num">{Math.round(o.qs).toLocaleString()}</td>
                <td className={`r num ${toneOf(o.profit, 0.5)}`}>{stubs(o.profit)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Chances come from the model's backtested OVR forecast. With a market price, each outcome sells for that price
        less the {Math.round(MARKET_TAX * 100)}% market tax, or quicksells if that pays more.
      </p>
    </section>
  )
}
