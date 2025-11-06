/**
 * API client for DeepSeek Trader backend
 */
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for reasoning mode (it thinks deeply!)
  headers: {
    'Content-Type': 'application/json'
  }
})

// Request interceptor for debugging
api.interceptors.request.use(
  config => {
    console.log('🚀 API Request:', {
      method: config.method.toUpperCase(),
      url: config.baseURL + config.url,
      data: config.data,
      params: config.params
    })
    return config
  },
  error => {
    console.error('❌ API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for debugging
api.interceptors.response.use(
  response => {
    console.log('✅ API Response:', {
      status: response.status,
      url: response.config.url,
      data: response.data
    })
    return response
  },
  error => {
    console.error('❌ API Response Error:', {
      status: error.response?.status,
      url: error.config?.url,
      data: error.response?.data,
      message: error.message
    })
    return Promise.reject(error)
  }
)

// Health check
export const getHealth = async () => {
  const response = await api.get('/')
  return response.data
}

// Get balance
export const getBalance = async () => {
  const response = await api.get('/balance')
  return response.data
}

// Get symbols
export const getSymbols = async () => {
  const response = await api.get('/symbols')
  return response.data
}

// Get market data for symbol
export const getMarketData = async (symbol) => {
  // Use query parameter instead of path parameter
  const response = await api.get(`/market?symbol=${encodeURIComponent(symbol)}`)
  return response.data
}

// Get LLM decision
export const getDecision = async (symbol, balance = null) => {
  const response = await api.post('/decision', { symbol, balance })
  return response.data
}

// Get batch LLM decisions for multiple symbols
export const getBatchDecisions = async (symbols, balance = null) => {
  const response = await api.post('/decision/batch', { symbols, balance })
  return response.data
}

// Execute trade
export const executeTrade = async (tradeData) => {
  const response = await api.post('/trade', tradeData)
  return response.data
}

// Get trades
export const getTrades = async (limit = 100) => {
  const response = await api.get(`/trades?limit=${limit}`)
  return response.data
}

// Get open positions
export const getOpenPositions = async () => {
  const response = await api.get('/positions')
  return response.data
}

// Get metrics
export const getMetrics = async (days = 30) => {
  const response = await api.get(`/metrics?days=${days}`)
  return response.data
}

// Control bot
export const controlBot = async (action) => {
  const response = await api.post('/control', { action })
  return response.data
}

// Export trades
export const exportTrades = async () => {
  const response = await api.get('/export/trades')
  return response.data
}

// Get AI reasoning logs
export const getAILogs = async (days = 7) => {
  const response = await api.get(`/ai-logs?days=${days}`)
  return response.data
}

// Auto-trading control
export const getAutoTrading = async () => {
  const response = await api.get('/auto-trading')
  return response.data
}

export const setAutoTrading = async (enabled) => {
  const response = await api.post('/auto-trading', null, {
    params: { enabled }
  })
  return response.data
}

export default api

