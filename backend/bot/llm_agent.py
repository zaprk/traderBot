"""
LLM Agent module for DeepSeek V3 integration
Handles prompt building, API calls, and JSON response parsing
"""
import requests
import json
import logging
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)


class LLMAgent:
    """DeepSeek V3 LLM Agent for trading decisions"""
    
    def __init__(self, api_key: str, model: str = "deepseek-reasoner"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.max_retries = 3
    
    def build_prompt(self, symbol: str, balance: float, risk_pct: float, 
                    indicators_multi_tf: Dict[str, Dict]) -> str:
        """
        Build structured prompt from multi-timeframe indicators
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            balance: Current balance in USDT
            risk_pct: Maximum risk per trade as percentage (e.g., 0.02 for 2%)
            indicators_multi_tf: Dictionary mapping timeframe to indicators dict
        
        Returns:
            Formatted prompt string
        """
        prompt_lines = [
            f"SYMBOL: {symbol}",
            f"BALANCE: {balance:.2f} USDT",
            f"MAX_RISK_PER_TRADE: {risk_pct*100:.1f}% (must not recommend trades violating this)",
            "TIMEFRAMES:"
        ]
        
        # Add indicators for each timeframe
        for tf, indicators in indicators_multi_tf.items():
            if not indicators or indicators.get('rsi') is None:
                prompt_lines.append(f"{tf}: insufficient data")
                continue
            
            rsi_val = indicators.get('rsi', 'N/A')
            macd_status = 'hist_pos' if indicators.get('macd_positive') else 'hist_neg'
            ema_20 = indicators.get('ema_20', 'N/A')
            ema_50 = indicators.get('ema_50', 'N/A')
            atr_val = indicators.get('atr', 'N/A')
            last_close = indicators.get('last_close', 'N/A')
            
            ema_relation = ""
            if indicators.get('ema_20_above_50'):
                ema_relation = "EMA20>EMA50"
            elif indicators.get('ema_20_above_50') is False:
                ema_relation = "EMA20<EMA50"
            else:
                ema_relation = "EMA20≈EMA50"
            
            prompt_lines.append(
                f"{tf}: RSI={rsi_val}, MACD={macd_status}, {ema_relation}, "
                f"ATR={atr_val}, last_close={last_close}"
            )
        
        # Add volume analysis (use 1h if available)
        if '1h' in indicators_multi_tf and indicators_multi_tf['1h']:
            vol_change = indicators_multi_tf['1h'].get('volume_change', 0)
            prompt_lines.append(f"VOLUME: 1h {vol_change:+.1f}% vs avg")
        
        prompt_lines.append(
            "STRATEGY: Day trading. Risk per trade cap and ATR-based stop. "
            "Provide JSON with your trading decision."
        )
        
        return "\n".join(prompt_lines)
    
    def get_system_message(self) -> str:
        """Get the system message for the LLM"""
        return """You are a professional crypto day trader with expertise in technical analysis. You will analyze multiple cryptocurrencies simultaneously and decide which ones (if any) present good trading opportunities.

Think through your analysis step-by-step:
1. Analyze each crypto's technical indicators across all timeframes
2. Consider volume, momentum, and trend alignment
3. Compare opportunities - which looks best?
4. Only recommend trades with high confidence (>0.7) and clear edge

STOP-LOSS & TAKE-PROFIT RULES:
- Use ATR (Average True Range) for risk management
- Stop-loss should be 1.5-2x ATR from entry (use the 1h ATR)
- Take-profit should be 2-3x the stop distance (2:1 or 3:1 reward:risk)
- Include the ATR value in your response for validation

IMPORTANT: Be concise in your reasoning. Focus on key insights.

CRITICAL: After your analysis, you MUST output a COMPLETE valid JSON object. Ensure all symbols have entries.

JSON format:
{
  "decisions": {
    "BTC/USDT": {"action": "long"|"short"|"none", "entry_price": float|null, "stop_loss": float|null, "take_profit": float|null, "atr": float|null, "confidence": float, "reason": "string"},
    "ETH/USDT": {...},
    ...
  },
  "summary": "Brief 1-2 sentence overview of market conditions"
}"""
    
    def call_api(self, prompt: str, temperature: float = 0.7) -> Optional[Dict]:
        """
        Call DeepSeek API with retry logic
        
        Args:
            prompt: User prompt containing market data
            temperature: Sampling temperature (default 0.7)
        
        Returns:
            Parsed JSON response or None if failed
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.get_system_message()},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 8000  # Large enough for reasoning + complete JSON
        }
        
        logger.info("=" * 60)
        logger.info(f"🧠 DEEPSEEK API CALL - Model: {self.model}")
        logger.info(f"📝 Prompt (first 300 chars): {prompt[:300]}...")
        logger.info("=" * 60)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📡 Calling DeepSeek API (attempt {attempt + 1}/{self.max_retries})...")
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=120  # 2 minutes for reasoning mode
                )
                
                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                
                result = response.json()
                
                # Extract content from response
                if 'choices' not in result or len(result['choices']) == 0:
                    logger.error("No choices in API response")
                    continue
                
                message = result['choices'][0]['message']
                content = message.get('content', '')
                
                logger.info(f"Content type: {type(content)}, Length: {len(content) if content else 0}")
                logger.info(f"Content preview: {repr(content[:500]) if content else 'EMPTY'}")
                
                # Extract reasoning content if present (deepseek-reasoner specific)
                reasoning_content = None
                if 'reasoning_content' in message:
                    reasoning_content = message['reasoning_content']
                    logger.info(f"DeepSeek reasoning (first 200 chars): {reasoning_content[:200]}")
                    logger.info(f"Reasoning length: {len(reasoning_content)}")
                
                # Log all message keys
                logger.info(f"Message keys: {list(message.keys())}")
                
                # For deepseek-reasoner, the JSON might be at the END of reasoning_content
                # if content is empty
                if not content and reasoning_content:
                    logger.info("Content empty but reasoning present - looking for JSON in reasoning")
                    # Try to extract JSON from the end of reasoning
                    parsed = self.parse_json_response(reasoning_content)
                else:
                    # Parse JSON from content
                    parsed = self.parse_json_response(content)
                
                if not parsed:
                    logger.error(f"=== PARSE FAILED ===")
                    logger.error(f"Full content ({len(content)} chars): {content}")
                    if reasoning_content:
                        logger.error(f"Full reasoning (last 1000 chars): ...{reasoning_content[-1000:]}")
                
                if parsed:
                    # Validate schema
                    if self.validate_response(parsed):
                        logger.info("=" * 60)
                        logger.info("✅ DEEPSEEK RESPONSE RECEIVED & VALIDATED")
                        logger.info(f"Decision: {parsed.get('action', 'N/A')}")
                        logger.info(f"Confidence: {parsed.get('confidence', 0)}")
                        if reasoning_content:
                            logger.info(f"Reasoning length: {len(reasoning_content)} chars")
                        logger.info("=" * 60)
                        # Add full response metadata
                        parsed['_raw_response'] = {
                            'content': content,
                            'reasoning': reasoning_content,
                            'full_result': result
                        }
                        return parsed
                    else:
                        logger.warning("Response validation failed, retrying with stricter prompt")
                        if attempt < self.max_retries - 1:
                            # Modify payload for stricter format
                            payload['messages'].append({
                                "role": "assistant",
                                "content": content
                            })
                            payload['messages'].append({
                                "role": "user",
                                "content": "Please output ONLY the JSON object with no extra text."
                            })
                            time.sleep(1)
                            continue
                
            except requests.RequestException as e:
                logger.error(f"Network error calling DeepSeek API: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            
            except Exception as e:
                logger.error(f"Unexpected error calling LLM: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        logger.error(f"Failed to get valid response after {self.max_retries} attempts")
        return None
    
    def parse_json_response(self, content: str) -> Optional[Dict]:
        """
        Parse JSON from LLM response content
        
        Args:
            content: Raw response content from LLM
        
        Returns:
            Parsed JSON dict or None
        """
        if not content or not content.strip():
            logger.error("Empty content received")
            return None
        
        # Try direct parsing first
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
        
        # Try to extract JSON from markdown code block with json tag
        if "```json" in content.lower():
            try:
                parts = content.lower().split("```json")
                for part in parts[1:]:  # Skip first part (before first ```json)
                    json_str = part.split("```")[0].strip()
                    if json_str:
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            continue
            except (IndexError, json.JSONDecodeError) as e:
                logger.debug(f"Markdown json block parse failed: {e}")
        
        # Try to extract JSON from any code block
        if "```" in content:
            try:
                parts = content.split("```")
                for i, part in enumerate(parts):
                    if i == 0:  # Skip text before first ```
                        continue
                    # Skip language identifier line
                    lines = part.strip().split('\n')
                    # If first line looks like language identifier, skip it
                    if lines[0].strip().lower() in ['json', 'javascript', 'js', '']:
                        json_str = '\n'.join(lines[1:]).strip()
                    else:
                        json_str = part.strip()
                    
                    if json_str and json_str.startswith('{'):
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            continue
            except (IndexError, json.JSONDecodeError) as e:
                logger.debug(f"Code block parse failed: {e}")
        
        # Try to find JSON object in text (looking for nested braces)
        try:
            start_idx = content.find('{')
            if start_idx >= 0:
                # Count braces to find matching closing brace
                brace_count = 0
                for i in range(start_idx, len(content)):
                    if content[i] == '{':
                        brace_count += 1
                    elif content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = content[start_idx:i+1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                break
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Brace matching parse failed: {e}")
        
        logger.error(f"Failed to parse JSON. Content preview: {content[:500]}")
        return None
    
    def validate_response(self, response: Dict) -> bool:
        """
        Validate that response matches expected schema
        
        Args:
            response: Parsed JSON response
        
        Returns:
            True if valid, False otherwise
        """
        # Check if it's a batch response
        if 'decisions' in response:
            if not isinstance(response['decisions'], dict):
                logger.warning("'decisions' should be a dict")
                return False
            
            # Validate each decision
            for symbol, decision in response['decisions'].items():
                if not self._validate_single_decision(decision):
                    logger.warning(f"Invalid decision for {symbol}")
                    return False
            
            return True
        else:
            # Single decision response
            return self._validate_single_decision(response)
    
    def _validate_single_decision(self, decision: Dict) -> bool:
        """Validate a single decision object"""
        required_keys = ['action', 'entry_price', 'stop_loss', 'take_profit', 'confidence', 'reason']
        
        # Check all keys present
        if not all(key in decision for key in required_keys):
            logger.warning(f"Missing required keys. Got: {list(decision.keys())}")
            return False
        
        # Validate action
        if decision['action'] not in ['long', 'short', 'none']:
            logger.warning(f"Invalid action: {decision['action']}")
            return False
        
        # Validate confidence
        try:
            conf = float(decision['confidence'])
            if not 0.0 <= conf <= 1.0:
                logger.warning(f"Confidence out of range: {conf}")
                return False
        except (TypeError, ValueError):
            logger.warning(f"Invalid confidence value: {decision['confidence']}")
            return False
        
        # If action is not 'none', validate prices
        if decision['action'] != 'none':
            for price_key in ['entry_price', 'stop_loss', 'take_profit']:
                if decision[price_key] is None:
                    logger.warning(f"{price_key} is None but action is {decision['action']}")
                    return False
                try:
                    price = float(decision[price_key])
                    if price <= 0:
                        logger.warning(f"Invalid {price_key}: {price}")
                        return False
                except (TypeError, ValueError):
                    logger.warning(f"Invalid {price_key} value: {decision[price_key]}")
                    return False
        
        return True
    
    def build_batch_prompt(self, symbols_data: Dict[str, Dict], balance: float, risk_pct: float, 
                          sentiment_data: Dict[str, Dict] = None, order_flow_data: Dict[str, Dict] = None) -> str:
        """
        Build batch prompt for multiple symbols with sentiment and order flow data
        
        Args:
            symbols_data: Dict mapping symbol to indicators_multi_tf
            balance: Current balance
            risk_pct: Risk percentage per trade
            sentiment_data: Dict mapping symbol to sentiment info (optional)
            order_flow_data: Dict mapping symbol to order flow analysis (optional)
        
        Returns:
            Formatted prompt string
        """
        prompt_lines = [
            f"PORTFOLIO BALANCE: {balance:.2f} USDT",
            f"MAX_RISK_PER_TRADE: {risk_pct*100:.1f}%",
            f"MAX_OPEN_POSITIONS: 3",
            "",
            "MARKET DATA FOR ALL SYMBOLS:",
            "=" * 60
        ]
        
        for symbol, indicators_multi_tf in symbols_data.items():
            prompt_lines.append(f"\n{symbol}:")
            prompt_lines.append("-" * 40)
            
            # Add interpreted indicators for each timeframe
            for tf, indicators in indicators_multi_tf.items():
                if not indicators or indicators.get('rsi') is None:
                    prompt_lines.append(f"  {tf}: insufficient data")
                    continue
                
                # Get human-readable interpretations
                price = indicators.get('last_close', 'N/A')
                atr = indicators.get('atr', 'N/A')
                rsi_interp = indicators.get('rsi_interpretation', 'Unknown')
                trend_interp = indicators.get('trend_interpretation', 'Unknown')
                macd_interp = indicators.get('macd_interpretation', 'Neutral')
                volume_interp = indicators.get('volume_interpretation', 'Normal volume')
                candle_summary = indicators.get('summary', 'No candle data')
                momentum_pct = indicators.get('momentum_pct', 0)
                
                # Build comprehensive summary
                prompt_lines.append(
                    f"  {tf}: ${price} | ATR: ${atr} | {trend_interp} | RSI: {rsi_interp} | {macd_interp}"
                )
                prompt_lines.append(
                    f"      Candle: {candle_summary} (momentum: {momentum_pct:+.2f}%)"
                )
                
                # Add volume on 15m, 30m, and 1h only (avoid repetition)
                if tf in ['15m', '30m', '1h']:
                    prompt_lines.append(f"      Volume: {volume_interp}")
            
            # Add market structure context
            if '1h' in indicators_multi_tf and indicators_multi_tf['1h']:
                structure = indicators_multi_tf['1h'].get('market_structure', 'unknown')
                structure_map = {
                    'HHHL': 'Higher highs & higher lows (uptrend)',
                    'LHLL': 'Lower highs & lower lows (downtrend)',
                    'range': 'Ranging/consolidating'
                }
                structure_desc = structure_map.get(structure, structure)
                prompt_lines.append(f"  Market Structure: {structure_desc}")
            
            # Add sentiment data if available
            if sentiment_data and symbol in sentiment_data:
                sentiment = sentiment_data[symbol]
                sentiment_summary = sentiment.get('summary', 'No sentiment data')
                trend_status = sentiment.get('trend', 'Normal')
                
                emoji = "🔥" if trend_status == "Trending" else "📊"
                prompt_lines.append(f"  {emoji} Sentiment: {sentiment_summary}")
            
            # Add order flow & liquidity data
            if order_flow_data and symbol in order_flow_data:
                order_flow = order_flow_data[symbol]
                
                # Order blocks
                order_blocks = order_flow.get('order_blocks', [])
                if order_blocks:
                    bullish_obs = [ob for ob in order_blocks if 'bullish' in ob.get('category', '')]
                    bearish_obs = [ob for ob in order_blocks if 'bearish' in ob.get('category', '')]
                    
                    if bullish_obs:
                        top_bull = bullish_obs[0]
                        prompt_lines.append(
                            f"  💰 Bullish Order Block: ${top_bull['price_low']:.2f}-${top_bull['price_high']:.2f} "
                            f"(strength: {top_bull['strength']}, vol: {top_bull['volume_ratio']:.1f}x)"
                        )
                    
                    if bearish_obs:
                        top_bear = bearish_obs[0]
                        prompt_lines.append(
                            f"  💰 Bearish Order Block: ${top_bear['price_low']:.2f}-${top_bear['price_high']:.2f} "
                            f"(strength: {top_bear['strength']}, vol: {top_bear['volume_ratio']:.1f}x)"
                        )
                
                # Volume metrics
                volume_delta = order_flow.get('volume_delta', 0)
                absorption = order_flow.get('absorption_detected', False)
                
                volume_pressure = ""
                if volume_delta > 0:
                    volume_pressure = f"Buying pressure (Δ: +{volume_delta:.0f})"
                elif volume_delta < 0:
                    volume_pressure = f"Selling pressure (Δ: {volume_delta:.0f})"
                else:
                    volume_pressure = "Neutral volume"
                
                if absorption:
                    volume_pressure += " | 🔍 INSTITUTIONAL ABSORPTION DETECTED"
                
                prompt_lines.append(f"  📊 Volume Flow: {volume_pressure}")
                
                # Key levels
                nearest_support = order_flow.get('nearest_support')
                nearest_resistance = order_flow.get('nearest_resistance')
                current_price = order_flow.get('current_price')
                
                if nearest_support and nearest_resistance and current_price:
                    support_dist = ((current_price - nearest_support) / current_price) * 100
                    resistance_dist = ((nearest_resistance - current_price) / current_price) * 100
                    prompt_lines.append(
                        f"  💧 Key Levels: Support ${nearest_support:.2f} (-{support_dist:.2f}%), "
                        f"Resistance ${nearest_resistance:.2f} (+{resistance_dist:.2f}%)"
                    )
        
        prompt_lines.append("\n" + "=" * 60)
        prompt_lines.append(
            "\nSTRATEGY: Day trading with 2% risk per trade. ATR-based stops. "
            "Only recommend HIGH CONFIDENCE trades (>0.7). Think through each crypto carefully."
        )
        
        return "\n".join(prompt_lines)
    
    def get_decision(self, symbol: str, balance: float, risk_pct: float, 
                    indicators_multi_tf: Dict[str, Dict]) -> Optional[Dict]:
        """
        Get trading decision from LLM (single symbol - legacy)
        
        Args:
            symbol: Trading pair
            balance: Current balance
            risk_pct: Risk percentage per trade
            indicators_multi_tf: Multi-timeframe indicators
        
        Returns:
            Decision dict with action, prices, confidence, reason, and raw response
        """
        prompt = self.build_prompt(symbol, balance, risk_pct, indicators_multi_tf)
        logger.info(f"Generated prompt for {symbol}")
        logger.debug(f"Prompt:\n{prompt}")
        
        response = self.call_api(prompt)
        
        if response:
            # Add metadata
            response['symbol'] = symbol
            response['prompt'] = prompt
            return response
        
        return None
    
    def get_batch_decisions(self, symbols_data: Dict[str, Dict], balance: float, risk_pct: float, 
                           sentiment_data: Dict[str, Dict] = None, order_flow_data: Dict[str, Dict] = None) -> Optional[Dict]:
        """
        Get trading decisions for multiple symbols in one call
        
        Args:
            symbols_data: Dict mapping symbol to indicators_multi_tf
            balance: Current balance
            risk_pct: Risk percentage per trade
            sentiment_data: Dict mapping symbol to sentiment info (optional)
            order_flow_data: Dict mapping symbol to order flow analysis (optional)
        
        Returns:
            Dict with decisions for each symbol and summary
        """
        prompt = self.build_batch_prompt(symbols_data, balance, risk_pct, sentiment_data, order_flow_data)
        logger.info(f"Generated batch prompt for {len(symbols_data)} symbols")
        logger.debug(f"Prompt:\n{prompt}")
        
        response = self.call_api(prompt, temperature=0.6)
        
        if response and 'decisions' in response:
            logger.info(f"Got batch decisions for {len(response['decisions'])} symbols")
            return response
        
        logger.error("Failed to get valid batch response")
        return None

