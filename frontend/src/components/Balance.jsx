import { useState, useEffect } from 'react'
import './Balance.css'

function Balance({ balance }) {
  const [animate, setAnimate] = useState(false)

  useEffect(() => {
    setAnimate(true)
  }, [balance])

  if (!balance) {
    return (
      <div className="balance-card loading">
        <div className="loading-placeholder large"></div>
        <div className="loading-placeholder small"></div>
      </div>
    )
  }

  return (
    <div className="balance-wrapper">
      <div className="balance-glow"></div>
      <div className="balance-card">
        <div className="balance-content">
          <div className="balance-main">
            <div className="balance-header">
              <div className="balance-icon">💰</div>
              <h2 className="balance-label">Portfolio Balance</h2>
            </div>
            <p className={`balance-amount ${animate ? 'animate' : ''}`}>
              ${balance.balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
            <div className="balance-meta">
              <span className="balance-currency">{balance.currency}</span>
              <span className="balance-badge">Available</span>
            </div>
          </div>
          
          <div className="equity-box">
            <div className="equity-label">Total Equity</div>
            <div className="equity-amount">
              ${balance.balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div className="equity-change positive">↑ 0.00%</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Balance

