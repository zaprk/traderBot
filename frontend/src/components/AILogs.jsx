import { useState, useEffect } from 'react'
import { getAILogs } from '../api'
import './AILogs.css'

export default function AILogs() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedLog, setSelectedLog] = useState(null)
  const [filter, setFilter] = useState('all') // 'all', 'long', 'short', 'none'

  useEffect(() => {
    loadLogs()
    const interval = setInterval(loadLogs, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const loadLogs = async () => {
    try {
      const data = await getAILogs(7)
      setLogs(data.logs || [])
      setLoading(false)
    } catch (error) {
      console.error('Error loading AI logs:', error)
      setLoading(false)
    }
  }

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  const getActionSummary = (response) => {
    if (!response || !response.decisions) return 'No decisions'
    
    const decisions = Object.values(response.decisions)
    const longs = decisions.filter(d => d.action === 'long').length
    const shorts = decisions.filter(d => d.action === 'short').length
    const none = decisions.filter(d => d.action === 'none').length
    
    return `${longs} Long, ${shorts} Short, ${none} Hold`
  }

  const filteredLogs = logs.filter(log => {
    if (filter === 'all') return true
    if (!log.response || !log.response.decisions) return false
    
    const decisions = Object.values(log.response.decisions)
    return decisions.some(d => d.action === filter)
  })

  if (loading) {
    return <div className="ai-logs-loading">Loading AI reasoning logs...</div>
  }

  return (
    <div className="ai-logs-container">
      <div className="ai-logs-header">
        <h2>🧠 AI Reasoning History</h2>
        <div className="ai-logs-filters">
          <button 
            className={filter === 'all' ? 'active' : ''} 
            onClick={() => setFilter('all')}
          >
            All
          </button>
          <button 
            className={filter === 'long' ? 'active' : ''} 
            onClick={() => setFilter('long')}
          >
            Longs
          </button>
          <button 
            className={filter === 'short' ? 'active' : ''} 
            onClick={() => setFilter('short')}
          >
            Shorts
          </button>
          <button 
            className={filter === 'none' ? 'active' : ''} 
            onClick={() => setFilter('none')}
          >
            Holds
          </button>
        </div>
      </div>

      {filteredLogs.length === 0 ? (
        <div className="ai-logs-empty">No AI analysis logs yet. Enable auto-trading to start!</div>
      ) : (
        <div className="ai-logs-list">
          {filteredLogs.map((log, idx) => (
            <div 
              key={idx} 
              className="ai-log-card"
              onClick={() => setSelectedLog(log)}
            >
              <div className="ai-log-header">
                <span className="ai-log-time">{formatTimestamp(log.timestamp)}</span>
                <span className="ai-log-summary">{getActionSummary(log.response)}</span>
              </div>
              {log.response && log.response.summary && (
                <div className="ai-log-summary-text">{log.response.summary}</div>
              )}
              <div className="ai-log-symbols">
                Analyzed: {log.symbols?.join(', ') || 'N/A'}
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedLog && (
        <div className="ai-log-modal" onClick={() => setSelectedLog(null)}>
          <div className="ai-log-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="ai-log-modal-header">
              <h3>AI Reasoning Details</h3>
              <button onClick={() => setSelectedLog(null)}>×</button>
            </div>
            <div className="ai-log-modal-body">
              <div className="ai-log-detail-section">
                <strong>Timestamp:</strong> {formatTimestamp(selectedLog.timestamp)}
              </div>
              <div className="ai-log-detail-section">
                <strong>Symbols Analyzed:</strong> {selectedLog.symbols?.join(', ')}
              </div>
              
              {selectedLog.response?._raw_response?.reasoning && (
                <div className="ai-log-detail-section">
                  <strong>🧠 AI Reasoning:</strong>
                  <div className="ai-reasoning-text">
                    {selectedLog.response._raw_response.reasoning}
                  </div>
                </div>
              )}

              {selectedLog.response?.summary && (
                <div className="ai-log-detail-section">
                  <strong>Market Summary:</strong>
                  <p>{selectedLog.response.summary}</p>
                </div>
              )}

              {selectedLog.response?.decisions && (
                <div className="ai-log-detail-section">
                  <strong>Decisions:</strong>
                  <div className="ai-decisions-grid">
                    {Object.entries(selectedLog.response.decisions).map(([symbol, decision]) => (
                      <div key={symbol} className={`ai-decision-card ${decision.action}`}>
                        <div className="ai-decision-symbol">{symbol}</div>
                        <div className="ai-decision-action">
                          Action: <span className={`action-${decision.action}`}>{decision.action.toUpperCase()}</span>
                        </div>
                        {decision.confidence && (
                          <div className="ai-decision-confidence">
                            Confidence: {(decision.confidence * 100).toFixed(0)}%
                          </div>
                        )}
                        {decision.reason && (
                          <div className="ai-decision-reason">{decision.reason}</div>
                        )}
                        {decision.entry_price && (
                          <div className="ai-decision-prices">
                            Entry: ${decision.entry_price?.toLocaleString()}<br/>
                            Stop: ${decision.stop_loss?.toLocaleString()}<br/>
                            Target: ${decision.take_profit?.toLocaleString()}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

