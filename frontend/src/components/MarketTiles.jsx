import { useState, useEffect } from 'react'
import { getMarketData, getDecision, executeTrade } from '../api'
import './MarketTiles.css'

function MarketTile({ symbol }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(false)
  const [decision, setDecision] = useState(null)
  const [gettingDecision, setGettingDecision] = useState(false)

  useEffect(() => {
    loadMarketData()
    
    // Refresh every 60 seconds
    const interval = setInterval(loadMarketData, 60000)
    return () => clearInterval(interval)
  }, [symbol])

  const loadMarketData = async () => {
    try {
      const marketData = await getMarketData(symbol)
      setData(marketData)
      setLoading(false)
    } catch (error) {
      console.error(`Error loading data for ${symbol}:`, error)
      setLoading(false)
    }
  }

  const handleGetDecision = async () => {
    setGettingDecision(true)
    try {
      const llmDecision = await getDecision(symbol)
      setDecision(llmDecision)
    } catch (error) {
      alert('Error getting decision: ' + error.message)
    }
    setGettingDecision(false)
  }

  const handleExecuteTrade = async () => {
    if (!decision || decision.action === 'none') {
      alert('No trade to execute')
      return
    }

    if (!window.confirm(`Execute ${decision.action} trade for ${symbol}?`)) {
      return
    }

    try {
      const result = await executeTrade({
        symbol: symbol,
        side: decision.action,
        entry_price: decision.entry_price,
        stop_loss: decision.stop_loss,
        take_profit: decision.take_profit,
        confidence: decision.confidence,
        reason: decision.reason
      })

      if (result.success) {
        alert('Trade executed successfully!')
        setDecision(null)
      } else {
        alert('Trade validation failed: ' + result.reason)
      }
    } catch (error) {
      alert('Error executing trade: ' + error.message)
    }
  }

  if (loading) {
    return (
      <div className="market-tile loading-tile">
        <div>Loading...</div>
      </div>
    )
  }

  if (!data) {
    return null
  }

  const indicators1h = data.indicators['1h'] || {}
  const rsi = indicators1h.rsi || 'N/A'
  const macdPositive = indicators1h.macd_positive

  return (
    <div className={`market-tile ${expanded ? 'expanded' : ''}`}>
      <div className="tile-glow"></div>
      
      <div className="tile-header" onClick={() => setExpanded(!expanded)}>
        <div className="tile-title">
          <div className="tile-icon">{symbol.split('/')[0].charAt(0)}</div>
          <h3 className="tile-symbol">{symbol.split('/')[0]}</h3>
        </div>
        <span className={`expand-arrow ${expanded ? 'rotated' : ''}`}>▼</span>
      </div>
      
      <div className="tile-price" onClick={() => setExpanded(!expanded)}>
        ${data.current_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </div>
      
      <div className="tile-indicators" onClick={() => setExpanded(!expanded)}>
        <span className={`indicator-badge ${
          rsi === 'N/A' ? 'gray' :
          rsi < 30 ? 'green' :
          rsi > 70 ? 'red' :
          'blue'
        }`}>
          RSI {rsi}
        </span>
        <span className={`indicator-badge ${macdPositive ? 'green' : 'red'}`}>
          {macdPositive ? '↗' : '↘'} MACD
        </span>
      </div>

      {expanded && (
        <div className="tile-expanded">
          <div className="timeframe-indicators">
            {Object.entries(data.indicators).map(([tf, ind]) => (
              <div key={tf} className="timeframe-row">
                <span className="timeframe-label">{tf}</span>
                <div className="timeframe-values">
                  <span>RSI <strong>{ind.rsi || 'N/A'}</strong></span>
                  <span>EMA <strong>{ind.ema_20?.toFixed(0) || 'N/A'}</strong></span>
                  <span>ATR <strong>{ind.atr?.toFixed(2) || 'N/A'}</strong></span>
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={(e) => {
              e.stopPropagation()
              handleGetDecision()
            }}
            disabled={gettingDecision}
            className={`ai-button ${gettingDecision ? 'loading' : ''}`}
          >
            {gettingDecision ? (
              <>
                <span className="spinner">⚡</span>
                Analyzing...
              </>
            ) : (
              <>🤖 AI Decision</>
            )}
          </button>

          {decision && (
            <div className="decision-box">
              <div className="decision-header">
                <span className="decision-icon">🎯</span>
                <div>
                  AI Recommendation: <span className={`decision-action ${decision.action}`}>
                    {decision.action.toUpperCase()}
                  </span>
                </div>
              </div>
              
              {decision.action !== 'none' && (
                <>
                  <div className="decision-details">
                    <div>Entry: ${decision.entry_price}</div>
                    <div>Stop: ${decision.stop_loss}</div>
                    <div>Target: ${decision.take_profit}</div>
                    <div>Confidence: {(decision.confidence * 100).toFixed(0)}%</div>
                  </div>
                  <div className="decision-reason">{decision.reason}</div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleExecuteTrade()
                    }}
                    className="execute-button"
                  >
                    Execute Trade
                  </button>
                </>
              )}
              
              {decision.action === 'none' && (
                <div className="decision-reason">{decision.reason}</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MarketTiles({ symbols }) {
  if (!symbols || symbols.length === 0) {
    return null
  }

  return (
    <div className="market-tiles-container">
      <div className="market-tiles-header">
        <div className="market-tiles-icon">📈</div>
        <div>
          <h2 className="market-tiles-title">Live Markets</h2>
          <p className="market-tiles-subtitle">Real-time cryptocurrency prices</p>
        </div>
      </div>
      <div className="market-tiles-grid">
        {symbols.map(symbol => (
          <MarketTile key={symbol} symbol={symbol} />
        ))}
      </div>
    </div>
  )
}

export default MarketTiles

