import { useState, useEffect } from 'react'
import { getTrades, getOpenPositions } from '../api'
import './TradeLog.css'

function TradeLog() {
  const [trades, setTrades] = useState([])
  const [openPositions, setOpenPositions] = useState({})
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('all') // 'all', 'open', 'closed', 'winning', 'losing'
  const [selectedTrade, setSelectedTrade] = useState(null)

  useEffect(() => {
    loadTrades()
    
    // Refresh every 30 seconds
    const interval = setInterval(loadTrades, 30000)
    return () => clearInterval(interval)
  }, [])

  const loadTrades = async () => {
    try {
      const [tradesData, positionsData] = await Promise.all([
        getTrades(100),
        getOpenPositions()
      ])
      
      setTrades(tradesData.trades || [])
      setOpenPositions(positionsData.positions || {})
      setLoading(false)
    } catch (error) {
      console.error('Error loading trades:', error)
      setLoading(false)
    }
  }

  const getFilteredTrades = () => {
    switch (filter) {
      case 'open':
        return trades.filter(t => t.status === 'open')
      case 'closed':
        return trades.filter(t => t.status === 'closed')
      case 'winning':
        return trades.filter(t => t.pnl_usd && t.pnl_usd > 0)
      case 'losing':
        return trades.filter(t => t.pnl_usd && t.pnl_usd < 0)
      default:
        return trades
    }
  }

  const filteredTrades = getFilteredTrades()

  if (loading) {
    return (
      <div className="tradelog-container loading">
        <div>Loading trades...</div>
      </div>
    )
  }

  return (
    <div className="tradelog-container">
      <div className="tradelog-header">
        <h2 className="tradelog-title">Trade Log</h2>
        
        <div className="filter-buttons">
          {['all', 'open', 'closed', 'winning', 'losing'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`filter-btn ${filter === f ? 'active' : ''}`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Open Positions Summary */}
      {Object.keys(openPositions).length > 0 && (
        <div className="open-positions-banner">
          <div className="positions-count">
            {Object.keys(openPositions).length} Open Position(s)
          </div>
          {Object.entries(openPositions).map(([id, pos]) => (
            <div key={id} className="position-item">
              {pos.symbol} - {pos.side.toUpperCase()} @ ${pos.entry_price}
            </div>
          ))}
        </div>
      )}

      {/* Trade Table */}
      {filteredTrades.length === 0 ? (
        <div className="no-trades">
          <p className="no-trades-title">📊 No trades found</p>
          <p className="no-trades-subtitle">
            Trades will appear here once you start trading
          </p>
        </div>
      ) : (
        <div className="trades-table-container">
          <table className="trades-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L</th>
                <th>%</th>
                <th>Status</th>
                <th>Mode</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {filteredTrades.map((trade) => (
                <tr key={trade.id}>
                  <td className="symbol-cell">{trade.symbol}</td>
                  <td className="side-cell">
                    <span className={`side-badge ${trade.side}`}>
                      {trade.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="price-cell">${trade.entry_price.toFixed(2)}</td>
                  <td className="price-cell">
                    {trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : '-'}
                  </td>
                  <td className={`pnl-cell ${
                    trade.pnl_usd > 0 ? 'positive' : 
                    trade.pnl_usd < 0 ? 'negative' : 
                    'neutral'
                  }`}>
                    {trade.pnl_usd 
                      ? `${trade.pnl_usd >= 0 ? '+' : ''}$${trade.pnl_usd.toFixed(2)}`
                      : '-'}
                  </td>
                  <td className={`pnl-cell ${
                    trade.pnl_pct > 0 ? 'positive' : 
                    trade.pnl_pct < 0 ? 'negative' : 
                    'neutral'
                  }`}>
                    {trade.pnl_pct 
                      ? `${trade.pnl_pct >= 0 ? '+' : ''}${trade.pnl_pct.toFixed(2)}%`
                      : '-'}
                  </td>
                  <td className="status-cell">
                    <span className={`status-badge ${trade.status}`}>
                      {trade.status}
                    </span>
                  </td>
                  <td className="mode-cell">
                    <span className={`mode-badge ${trade.paper_mode ? 'paper' : 'live'}`}>
                      {trade.paper_mode ? 'PAPER' : 'LIVE'}
                    </span>
                  </td>
                  <td className="details-cell">
                    <button
                      onClick={() => setSelectedTrade(trade)}
                      className="view-btn"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Trade Details Modal */}
      {selectedTrade && (
        <div 
          className="modal-overlay"
          onClick={() => setSelectedTrade(null)}
        >
          <div 
            className="modal-content"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <h3 className="modal-title">
                Trade Details - {selectedTrade.symbol}
              </h3>
              <button
                onClick={() => setSelectedTrade(null)}
                className="modal-close"
              >
                ×
              </button>
            </div>

            <div className="modal-body">
              <div className="detail-grid">
                <div className="detail-item">
                  <div className="detail-label">Side</div>
                  <div className="detail-value">{selectedTrade.side.toUpperCase()}</div>
                </div>
                <div className="detail-item">
                  <div className="detail-label">Status</div>
                  <div className="detail-value">{selectedTrade.status}</div>
                </div>
                <div className="detail-item">
                  <div className="detail-label">Entry Price</div>
                  <div className="detail-value">${selectedTrade.entry_price.toFixed(2)}</div>
                </div>
                <div className="detail-item">
                  <div className="detail-label">Exit Price</div>
                  <div className="detail-value">
                    {selectedTrade.exit_price ? `$${selectedTrade.exit_price.toFixed(2)}` : '-'}
                  </div>
                </div>
                <div className="detail-item">
                  <div className="detail-label">Stop Loss</div>
                  <div className="detail-value">${selectedTrade.stop_loss.toFixed(2)}</div>
                </div>
                <div className="detail-item">
                  <div className="detail-label">Take Profit</div>
                  <div className="detail-value">${selectedTrade.take_profit.toFixed(2)}</div>
                </div>
                <div className="detail-item">
                  <div className="detail-label">Units</div>
                  <div className="detail-value">{selectedTrade.units.toFixed(6)}</div>
                </div>
                <div className="detail-item">
                  <div className="detail-label">Confidence</div>
                  <div className="detail-value">
                    {selectedTrade.confidence ? `${(selectedTrade.confidence * 100).toFixed(0)}%` : 'N/A'}
                  </div>
                </div>
              </div>

              {selectedTrade.pnl_usd && (
                <div className="pnl-section">
                  <div className="detail-label">P&L</div>
                  <div className={`pnl-amount ${selectedTrade.pnl_usd >= 0 ? 'positive' : 'negative'}`}>
                    {selectedTrade.pnl_usd >= 0 ? '+' : ''}${selectedTrade.pnl_usd.toFixed(2)} 
                    ({selectedTrade.pnl_pct >= 0 ? '+' : ''}{selectedTrade.pnl_pct.toFixed(2)}%)
                  </div>
                </div>
              )}

              {selectedTrade.llm_reason && (
                <div className="reason-section">
                  <div className="detail-label">AI Reasoning</div>
                  <div className="reason-text">
                    {selectedTrade.llm_reason}
                  </div>
                </div>
              )}

              <div className="timestamp-section">
                <div className="detail-label">Timestamps</div>
                <div className="timestamp-text">
                  Entry: {new Date(selectedTrade.entry_time).toLocaleString()}
                  {selectedTrade.exit_time && (
                    <><br />Exit: {new Date(selectedTrade.exit_time).toLocaleString()}</>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default TradeLog
