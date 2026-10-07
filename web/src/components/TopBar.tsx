import type { UpdateStatus } from '../types'
import { shortDate } from '../types'

export default function TopBar({ status, dataAsOf, onMethodology }: {
  status: UpdateStatus | null; dataAsOf: string | null; onMethodology: () => void
}) {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand">
          <span className="brand-name">Roster Forecast</span>
          <span className="brand-sub">MLB The Show 26 &middot; Live Series</span>
        </div>
        <div className="topbar-meta">
          {status?.next_expected && (
            <span className="hide-sm">
              Next attribute update <strong className="num">~{shortDate(status.next_expected)}</strong>
              {status.days_until != null && <span className="faint hide-sm"> &middot; {status.days_until} days</span>}
            </span>
          )}
          {dataAsOf && <span className="hide-sm">Updated {shortDate(dataAsOf)}</span>}
          <button type="button" className="btn" onClick={onMethodology}>How it works</button>
        </div>
      </div>
    </header>
  )
}
