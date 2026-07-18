import type { UpdateStatus } from '../types'

interface HeaderProps {
  status: UpdateStatus | null
}

export default function Header({ status }: HeaderProps) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 24,
      padding: '16px 24px', borderBottom: '1px solid #2d3748', flexWrap: 'wrap',
    }}>
      <div style={{ flex: 1, minWidth: 200 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f7fafc' }}>
          MLB The Show 26 Roster Predictor
        </h1>
        <span style={{ color: '#718096', fontSize: 12 }}>
          Real card images &bull; Grid view v5 &bull; MLB 26 Live Series
        </span>
      </div>

      {status && (
        <div style={{ minWidth: 180 }}>
          {status.is_update_today ? (
            <div style={{
              background: '#48bb7822', border: '1px solid #48bb78', borderRadius: 8,
              padding: '8px 12px', textAlign: 'center',
            }}>
              <span style={{ color: '#48bb78', fontWeight: 700, fontSize: 13 }}>Update Day!</span>
              <div style={{ color: '#a0aec0', fontSize: 11 }}>Expected weekly on Thursdays</div>
            </div>
          ) : (
            <div style={{
              background: '#1a202c', border: '1px solid #2d3748', borderRadius: 8,
              padding: '8px 12px',
            }}>
              <div style={{ color: '#a0aec0', fontSize: 11 }}>Last Update</div>
              <div style={{ color: '#f7fafc', fontWeight: 700, fontSize: 14 }}>
                {status.latest || 'N/A'}
              </div>
              <div style={{ color: '#a0aec0', fontSize: 11 }}>
                {status.days_since != null ? `${status.days_since}d ago` : '?'}
                {status.days_until != null ? ` \u2022 Next ~${status.days_until}d` : ''}
              </div>
            </div>
          )}
        </div>
      )}

      <div style={{
        background: '#1a202c', border: '1px solid #2d3748', borderRadius: 8,
        padding: '8px 16px', textAlign: 'center',
      }}>
        <div style={{ color: '#a0aec0', fontSize: 11 }}>MLB 26 Updates</div>
        <div style={{ color: '#f7fafc', fontWeight: 700, fontSize: 16 }}>15 so far</div>
      </div>
    </div>
  )
}
