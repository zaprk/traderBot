import { useState, useEffect, useRef } from 'react'
import './ActivityLog.css'

function ActivityLog() {
  const [logs, setLogs] = useState([])
  const logIdCounter = useRef(0)

  // Listen for custom log events
  useEffect(() => {
    const handleLog = (event) => {
      logIdCounter.current += 1
      const newLog = {
        id: `${Date.now()}-${logIdCounter.current}`, // Unique ID
        timestamp: new Date().toLocaleTimeString(),
        message: event.detail.message,
        type: event.detail.type || 'info' // 'info', 'success', 'error', 'warning'
      }
      setLogs(prev => [newLog, ...prev].slice(0, 100)) // Keep last 100 logs
    }

    window.addEventListener('app-log', handleLog)
    return () => window.removeEventListener('app-log', handleLog)
  }, [])

  const clearLogs = () => {
    setLogs([])
  }

  return (
    <div className="activity-log-container">
      <div className="log-header">
        <div className="log-title-section">
          <span className="log-icon">📋</span>
          <h3 className="log-title">Activity Log</h3>
          <span className="log-count">{logs.length} entries</span>
        </div>
        <button onClick={clearLogs} className="clear-btn">Clear</button>
      </div>
      
      <div className="log-content">
        {logs.length === 0 ? (
          <div className="no-logs">No activity yet...</div>
        ) : (
          logs.map(log => (
            <div key={log.id} className={`log-entry ${log.type}`}>
              <span className="log-timestamp">{log.timestamp}</span>
              <span className={`log-badge ${log.type}`}>
                {log.type === 'success' && '✓'}
                {log.type === 'error' && '✗'}
                {log.type === 'warning' && '⚠'}
                {log.type === 'info' && 'ℹ'}
              </span>
              <span className="log-message">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

// Helper function to log from anywhere
export const addLog = (message, type = 'info') => {
  window.dispatchEvent(new CustomEvent('app-log', { 
    detail: { message, type } 
  }))
}

export default ActivityLog

