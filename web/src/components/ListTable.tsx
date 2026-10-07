import type { ReactNode } from 'react'
import { CardArt } from './ui'

export interface ListColumn<T> {
  label: string
  align?: 'r'
  className?: string
  cell: (row: T) => ReactNode
}

interface Props<T> {
  rows: T[]
  columns: ListColumn<T>[]
  rowKey: (row: T) => string
  name: (row: T) => string
  meta?: (row: T) => string
  onSelect?: (row: T) => void
  caption: string
  empty?: string
}

/** Compact ranked table with a card thumbnail, used by buy lists and the track record. */
export default function ListTable<T>({ rows, columns, rowKey, name, meta, onSelect, caption, empty }: Props<T>) {
  if (rows.length === 0) return <div className="table-wrap empty">{empty ?? 'Nothing qualifies right now.'}</div>
  return (
    <div className="table-wrap">
      <table className={`grid compact${onSelect ? ' clickable' : ''}`}>
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          <tr>
            <th className="r" style={{ width: 36 }}>#</th>
            <th>Player</th>
            {columns.map(c => <th key={c.label} className={[c.align, c.className].filter(Boolean).join(' ')}>{c.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={rowKey(r)}
              onClick={onSelect ? () => onSelect(r) : undefined}
              onKeyDown={onSelect ? e => { if (e.key === 'Enter') onSelect(r) } : undefined}
              tabIndex={onSelect ? 0 : undefined}
            >
              <td className="r num faint">{i + 1}</td>
              <td>
                <div className="player-cell" style={{ minWidth: 160 }}>
                  <CardArt uuid={rowKey(r)} className="thumb" />
                  <div>
                    <div className="player-name">{name(r)}</div>
                    {meta && <div className="player-meta">{meta(r)}</div>}
                  </div>
                </div>
              </td>
              {columns.map(c => <td key={c.label} className={['num', c.align, c.className].filter(Boolean).join(' ')}>{c.cell(r)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
