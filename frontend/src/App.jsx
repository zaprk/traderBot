import { useState, useEffect } from 'react'
import Dashboard from './components/Dashboard'
import Balance from './components/Balance'
import TradeLog from './components/TradeLog'
import Controls from './components/Controls'
import MarketTiles from './components/MarketTiles'
import ActivityLog, { addLog } from './components/ActivityLog'
import AILogs from './components/AILogs'
import { getBalance, getSymbols, getMetrics, controlBot, getBatchDecisions, getAutoTrading, setAutoTrading } from './api'
import './App.css'

function App() {
  const [balance, setBalance] = useState(null)
  const [symbols, setSymbols] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [botStatus, setBotStatus] = useState('active')
  const [loading, setLoading] = useState(true)
  const [autoTrade, setAutoTrade] = useState(false)
  const [activeTab, setActiveTab] = useState('dashboard') // 'dashboard', 'trades', 'ai-logs'

  useEffect(() => {
    loadInitialData()
    addLog('Application started', 'success')
    
    // Refresh data every 10 seconds for live updates
    const interval = setInterval(() => {
      loadInitialData()
    }, 10000)
    
    return () => clearInterval(interval)
  }, [])

  // Auto-trading is now handled by the backend!
  // The backend scheduler checks the database every hour and auto-executes trades.
  // Frontend just displays the current state.

  const loadInitialData = async () => {
    try {
      const [balanceData, symbolsData, metricsData, autoTradingData] = await Promise.all([
        getBalance(),
        getSymbols(),
        getMetrics(),
        getAutoTrading()
      ])
      
      setBalance(balanceData)
      setSymbols(symbolsData.symbols || [])
      setMetrics(metricsData)
      setAutoTrade(autoTradingData.enabled)
      setLoading(false)
    } catch (error) {
      console.error('Error loading data:', error)
      addLog(`Error loading data: ${error.message}`, 'error')
      setLoading(false)
    }
  }

  const handleControl = async (action) => {
    try {
      addLog(`Sending ${action} command to bot...`, 'info')
      const result = await controlBot(action)
      setBotStatus(result.status)
      addLog(result.message, 'success')
      
      // Stop auto-trading if bot is paused/killed
      if (action === 'pause' || action === 'kill') {
        setAutoTrade(false)
      }
    } catch (error) {
      addLog(`Error controlling bot: ${error.message}`, 'error')
    }
  }

  const toggleAutoTrade = async () => {
    if (!autoTrade) {
      if (botStatus !== 'active') {
        addLog('Cannot enable auto-trading: bot is not active', 'error')
        return
      }
      try {
        await setAutoTrading(true)
        setAutoTrade(true)
        addLog('✅ Auto-trading enabled - backend will analyze markets every hour', 'success')
      } catch (error) {
        addLog(`Error enabling auto-trading: ${error.message}`, 'error')
      }
    } else {
      try {
        await setAutoTrading(false)
        setAutoTrade(false)
        addLog('⏸️ Auto-trading disabled', 'warning')
      } catch (error) {
        addLog(`Error disabling auto-trading: ${error.message}`, 'error')
      }
    }
  }

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <div className="loading-title">DeepSeek Trader</div>
        <div className="loading-subtitle">Initializing AI Trading System...</div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <div className="app-background">
        <div className="bg-orb bg-orb-1"></div>
        <div className="bg-orb bg-orb-2"></div>
      </div>

      <div className="app-content">
        <header className="app-header">
          <div className="header-content">
            <div>
              <h1 className="app-title">🤖 DeepSeek Trader</h1>
              <p className="app-subtitle">AI-Powered Crypto Trading • Real-time Analytics</p>
            </div>
            <div className="header-badges">
              <span className={`status-badge ${balance?.paper_mode ? 'paper' : 'live'}`}>
                <span className="status-dot"></span>
                {balance?.paper_mode ? '📄 PAPER MODE' : '🔴 LIVE TRADING'}
              </span>
              <span className={`status-badge ${botStatus === 'active' ? 'active' : 'paused'}`}>
                <span className="status-dot"></span>
                {botStatus === 'active' ? '▶ Active' : '⏸ Paused'}
              </span>
            </div>
          </div>
        </header>

        <Controls 
          onControl={handleControl} 
          status={botStatus}
          autoTrade={autoTrade}
          onToggleAutoTrade={toggleAutoTrade}
        />
        <Balance balance={balance} />
        <ActivityLog />
        <MarketTiles symbols={symbols} />
        
        {/* Tabs Navigation */}
        <div className="tabs-navigation">
          <button 
            className={`tab-button ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            📊 Dashboard
          </button>
          <button 
            className={`tab-button ${activeTab === 'trades' ? 'active' : ''}`}
            onClick={() => setActiveTab('trades')}
          >
            📈 Trade Log
          </button>
          <button 
            className={`tab-button ${activeTab === 'ai-logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('ai-logs')}
          >
            🧠 AI Reasoning
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === 'dashboard' && <Dashboard metrics={metrics} />}
          {activeTab === 'trades' && <TradeLog />}
          {activeTab === 'ai-logs' && <AILogs />}
        </div>
      </div>
    </div>
  )
}

export default App

