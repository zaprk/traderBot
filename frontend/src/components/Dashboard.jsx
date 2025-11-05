import PerformanceChart from './PerformanceChart'
import './Dashboard.css'

function Dashboard({ metrics }) {
  if (!metrics) {
    return null
  }

  const summary = metrics.summary || {}

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="dashboard-icon">📊</div>
        <div>
          <h2 className="dashboard-title">Performance Analytics</h2>
          <p className="dashboard-subtitle">Real-time trading metrics</p>
        </div>
      </div>
      
      <div className="metrics-grid">
        {/* Total Trades */}
        <div className="metric-card blue">
          <div className="metric-header">
            <div className="metric-label">Total Trades</div>
            <span className="metric-emoji">🎯</span>
          </div>
          <div className="metric-value">
            {summary.total_trades || 0}
          </div>
          <div className="metric-subtitle">All time</div>
        </div>

        {/* Win Rate */}
        <div className={`metric-card ${(summary.win_rate || 0) > 0.5 ? 'green' : 'red'}`}>
          <div className="metric-header">
            <div className="metric-label">Win Rate</div>
            <span className="metric-emoji">{(summary.win_rate || 0) > 0.5 ? '✨' : '📉'}</span>
          </div>
          <div className="metric-value">
            {((summary.win_rate || 0) * 100).toFixed(1)}%
          </div>
          <div className="metric-subtitle">
            {summary.winning_trades || 0}W / {summary.losing_trades || 0}L
          </div>
        </div>

        {/* Total PnL */}
        <div className={`metric-card ${(summary.total_pnl || 0) >= 0 ? 'emerald' : 'rose'}`}>
          <div className="metric-header">
            <div className="metric-label">Total P&L</div>
            <span className="metric-emoji">{(summary.total_pnl || 0) >= 0 ? '💰' : '📊'}</span>
          </div>
          <div className="metric-value">
            {(summary.total_pnl || 0) >= 0 ? '+' : ''}${(summary.total_pnl || 0).toLocaleString('en-US', { 
              minimumFractionDigits: 2, 
              maximumFractionDigits: 2 
            })}
          </div>
          <div className="metric-subtitle">Net profit/loss</div>
        </div>

        {/* W/L Ratio */}
        <div className="metric-card purple">
          <div className="metric-header">
            <div className="metric-label">Wins / Losses</div>
            <span className="metric-emoji">⚡</span>
          </div>
          <div className="metric-value">
            {summary.winning_trades || 0} / {summary.losing_trades || 0}
          </div>
          <div className="metric-subtitle">Trade distribution</div>
        </div>
      </div>

      {/* Performance Chart */}
      <PerformanceChart metrics={metrics} />

      {/* Metrics History Table */}
      {metrics.metrics && metrics.metrics.length > 0 && (
        <div className="metrics-history">
          <h3>📅 Recent Performance</h3>
          <div className="metrics-table-container">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Balance</th>
                  <th>Trades</th>
                  <th>Win Rate</th>
                  <th>Return</th>
                </tr>
              </thead>
              <tbody>
                {metrics.metrics.slice(0, 10).map((m, idx) => (
                  <tr key={idx}>
                    <td>
                      {new Date(m.date).toLocaleDateString('en-US', { 
                        month: 'short', 
                        day: 'numeric',
                        year: 'numeric'
                      })}
                    </td>
                    <td>${m.balance.toLocaleString('en-US', { 
                      minimumFractionDigits: 2, 
                      maximumFractionDigits: 2 
                    })}</td>
                    <td>{m.total_trades}</td>
                    <td>{(m.win_rate * 100).toFixed(1)}%</td>
                    <td className={
                      (m.cumulative_return || 0) >= 0 ? 'positive' : 'negative'
                    }>
                      {m.cumulative_return 
                        ? `${m.cumulative_return >= 0 ? '+' : ''}${m.cumulative_return.toFixed(2)}%`
                        : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard

