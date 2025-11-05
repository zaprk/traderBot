import './Controls.css'

function Controls({ onControl, status, autoTrade, onToggleAutoTrade }) {
  const isPaused = status !== 'active'

  return (
    <div className="controls-container">
      <div className="controls-header">
        <div className="controls-icon">⚙️</div>
        <h2 className="controls-title">Bot Controls</h2>
      </div>
      
      <div className="controls-buttons">
        <button
          onClick={() => onControl('resume')}
          disabled={!isPaused}
          className={`control-btn resume ${!isPaused ? 'disabled' : ''}`}
        >
          <span className={`btn-icon ${!isPaused ? '' : 'pulse'}`}>▶</span>
          Resume Trading
        </button>
        
        <button
          onClick={() => onControl('pause')}
          disabled={isPaused}
          className={`control-btn pause ${isPaused ? 'disabled' : ''}`}
        >
          <span className="btn-icon">⏸</span>
          Pause Trading
        </button>
        
        <button
          onClick={() => {
            if (window.confirm('⚠️ Are you sure you want to kill the bot and close all positions?')) {
              onControl('kill')
            }
          }}
          className="control-btn emergency"
        >
          <span className="btn-icon">⏹</span>
          Emergency Stop
        </button>

        <button
          onClick={onToggleAutoTrade}
          disabled={isPaused}
          className={`control-btn auto ${autoTrade ? 'active' : ''} ${isPaused ? 'disabled' : ''}`}
        >
          <span className="btn-icon">{autoTrade ? '🔄' : '🤖'}</span>
          {autoTrade ? 'Auto-Trading ON' : 'Enable Auto-Trading'}
        </button>
      </div>

      <div className="controls-tip">
        <span className="tip-icon">💡</span>
        <span className="tip-text">
          <strong>Tip:</strong> Enable auto-trading to analyze markets automatically every hour.
        </span>
      </div>
    </div>
  )
}

export default Controls

