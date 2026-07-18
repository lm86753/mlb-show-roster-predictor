interface PaginationProps {
  page: number
  totalPages: number
  startIdx: number
  endIdx: number
  total: number
  onPageChange: (page: number) => void
}

export default function Pagination({ page, totalPages, startIdx, endIdx, total, onPageChange }: PaginationProps) {
  return (
    <div>
      <hr style={{ borderColor: '#2d3748', margin: '16px 0 8px 0' }} />
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, flexWrap: 'wrap' }}>
        {totalPages > 1 && (
          <>
            <button
              disabled={page <= 1}
              onClick={() => onPageChange(1)}
              style={{
                padding: '6px 14px', borderRadius: 6, border: '1px solid #4a5568',
                background: page <= 1 ? '#1a202c' : '#2d3748', color: page <= 1 ? '#4a5568' : '#f7fafc',
                fontSize: 12, cursor: page <= 1 ? 'default' : 'pointer', fontWeight: 600,
              }}
            >
              &laquo; First
            </button>
            <button
              disabled={page <= 1}
              onClick={() => onPageChange(page - 1)}
              style={{
                padding: '6px 14px', borderRadius: 6, border: '1px solid #4a5568',
                background: page <= 1 ? '#1a202c' : '#2d3748', color: page <= 1 ? '#4a5568' : '#f7fafc',
                fontSize: 12, cursor: page <= 1 ? 'default' : 'pointer', fontWeight: 600,
              }}
            >
              &#9664; Previous
            </button>
          </>
        )}

        <span style={{ color: '#a0aec0', fontSize: 14, padding: '0 12px', fontWeight: 600 }}>
          {page} / {totalPages}
        </span>

        {totalPages > 1 && (
          <>
            <button
              disabled={page >= totalPages}
              onClick={() => onPageChange(page + 1)}
              style={{
                padding: '6px 14px', borderRadius: 6, border: '1px solid #4a5568',
                background: page >= totalPages ? '#1a202c' : '#2d3748',
                color: page >= totalPages ? '#4a5568' : '#f7fafc',
                fontSize: 12, cursor: page >= totalPages ? 'default' : 'pointer', fontWeight: 600,
              }}
            >
              Next &#9654;
            </button>
            <button
              disabled={page >= totalPages}
              onClick={() => onPageChange(totalPages)}
              style={{
                padding: '6px 14px', borderRadius: 6, border: '1px solid #4a5568',
                background: page >= totalPages ? '#1a202c' : '#2d3748',
                color: page >= totalPages ? '#4a5568' : '#f7fafc',
                fontSize: 12, cursor: page >= totalPages ? 'default' : 'pointer', fontWeight: 600,
              }}
            >
              Last &raquo;
            </button>
          </>
        )}
      </div>
      <div style={{ textAlign: 'center', color: '#718096', fontSize: 11, marginTop: 4 }}>
        Showing {startIdx + 1}&ndash;{endIdx} of {total} players
      </div>
    </div>
  )
}
