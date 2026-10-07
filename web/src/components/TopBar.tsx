import type { UpdateStatus } from '../types'
import { shortDate } from '../types'
import { ROUTES, type Route } from '../routes'

function Mark() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <rect width="22" height="22" rx="5" fill="currentColor" />
      <path d="M5.5 14.5l3.5-3.5 2.8 2.2 4.7-5.7" fill="none" stroke="var(--surface)" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export default function TopBar({ route, status }: { route: Route; status: UpdateStatus | null }) {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <a className="wordmark" href="#/" aria-label="Roster Forecast home">
          <Mark /><span>Roster Forecast</span>
        </a>
        <nav className="nav" aria-label="Sections">
          {ROUTES.map(r => (
            <a key={r.id} href={`#/${r.id}`} aria-current={route === r.id ? 'page' : undefined}>{r.label}</a>
          ))}
        </nav>
        {status?.next_expected && (
          <span className="topbar-meta">
            Next attribute update <strong className="num">~{shortDate(status.next_expected)}</strong>
            {status.days_until != null && <span className="faint"> &middot; in {status.days_until} days</span>}
          </span>
        )}
      </div>
    </header>
  )
}
