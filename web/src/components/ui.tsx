import { useState, type ReactNode } from 'react'
import { RARITY_COLORS, cardImage, pct } from '../types'

const ICON_PATHS = {
  close: <path d="M6 6l12 12M18 6L6 18" />,
  search: <><circle cx="11" cy="11" r="6.5" /><path d="M20 20l-4.2-4.2" /></>,
  table: <><path d="M4 6h16M4 12h16M4 18h16" /></>,
  cards: <><rect x="4" y="4" width="6.5" height="9" rx="1" /><rect x="13.5" y="4" width="6.5" height="9" rx="1" /><rect x="4" y="15" width="6.5" height="5" rx="1" /><rect x="13.5" y="15" width="6.5" height="5" rx="1" /></>,
  sortDown: <path d="M7 10l5 5 5-5" />,
  sortUp: <path d="M7 14l5-5 5 5" />,
}

export function Icon({ name, size = 16 }: { name: keyof typeof ICON_PATHS; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICON_PATHS[name]}
    </svg>
  )
}

export function Rarity({ rarity }: { rarity: string }) {
  return (
    <span className="rarity" style={{ ['--rarity' as string]: RARITY_COLORS[rarity] ?? 'var(--text-3)' }}>
      {rarity}
    </span>
  )
}

/** Card art with a quiet placeholder if the CDN doesn't have it. */
export function CardArt({ uuid, size = 'sm', className, alt = '' }: {
  uuid: string; size?: 'sm' | 'lg'; className?: string; alt?: string
}) {
  const [failed, setFailed] = useState(false)
  if (failed) return <span className={className} aria-hidden="true" style={{ display: 'block', background: 'var(--surface-2)' }} />
  return <img className={className} src={cardImage(uuid, size)} alt={alt} loading="lazy" decoding="async" onError={() => setFailed(true)} />
}

/** Up probability grows from the left, down from the right. */
export function ProbBar({ up, down }: { up: number; down: number }) {
  return (
    <div className="pbar">
      <div className="pbar-track" aria-hidden="true">
        <span className="pbar-up" style={{ width: `${up * 100}%` }} />
        <span className="pbar-down" style={{ width: `${down * 100}%` }} />
      </div>
      <span className="pbar-label num">
        <span className={up >= 0.5 ? 'up' : 'muted'}>{pct(up)}</span>
        <span className="faint"> / </span>
        <span className={down >= 0.5 ? 'down' : 'muted'}>{pct(down)}</span>
      </span>
    </div>
  )
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: ReactNode; title?: string }[]; onChange: (v: T) => void; label: string
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map(o => (
        <button key={o.value} type="button" aria-pressed={value === o.value} title={o.title} aria-label={o.title} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}
