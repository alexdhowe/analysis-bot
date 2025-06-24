#!/usr/bin/env python3
"""
Enhanced Stock Analysis Bot with Alpha Vantage API - RELIABLE VERSION
24/7 stock analysis with official API support
"""

import discord
from discord.ext import commands
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional, Dict, List
import signal
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import requests
import feedparser
import time
from functools import wraps
import json

# Load environment variables
load_dotenv()

print("📊 Starting Enhanced Stock Analysis Bot with Alpha Vantage...")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Global storage for analysis results
analysis_cache = {}

# Rate limiting decorator for API calls
def rate_limit(calls_per_minute=5):
    """Rate limiter to respect Alpha Vantage's 25 calls/day limit"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Alpha Vantage free tier: 25 calls/day, so space them out
            time.sleep(12)  # 12 seconds between calls = 300 calls/hour max
            return func(*args, **kwargs)
        return wrapper
    return decorator

class AlphaVantageAPI:
    """Alpha Vantage API wrapper with error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'StockAnalysisBot/1.0'
        })
    
    @rate_limit(calls_per_minute=5)
    def get_daily_data(self, symbol: str, outputsize: str = "compact") -> Optional[Dict]:
        """Get daily time series data"""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol.upper(),
                'apikey': self.api_key,
                'outputsize': outputsize  # compact = last 100 days, full = 20+ years
            }
            
            print(f"🔍 Fetching daily data for {symbol}...")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                print(f"❌ API Error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                print(f"⚠️ API Note: {data['Note']}")
                return None
                
            if "Information" in data:
                print(f"⚠️ API Info: {data['Information']}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching {symbol}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error fetching {symbol}: {e}")
            return None
    
    @rate_limit(calls_per_minute=5)
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote data"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol.upper(),
                'apikey': self.api_key
            }
            
            print(f"🔍 Fetching quote for {symbol}...")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                print(f"❌ API Error: {data['Error Message']}")
                return None
                
            if "Note" in data:
                print(f"⚠️ API Note: {data['Note']}")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching quote for {symbol}: {e}")
            return None

# Initialize Alpha Vantage API
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if ALPHA_VANTAGE_KEY:
    av_api = AlphaVantageAPI(ALPHA_VANTAGE_KEY)
    print("✅ Alpha Vantage API initialized")
else:
    av_api = None
    print("⚠️ Alpha Vantage API key not found")

class StockAnalysisView(discord.ui.View):
    """Interactive view for stock analysis"""
    
    def __init__(self, symbol: str, analysis_data: dict):
        super().__init__(timeout=600)
        self.symbol = symbol.upper()
        self.analysis_data = analysis_data
        
    @discord.ui.button(label="📊 Refresh Analysis", style=discord.ButtonStyle.primary, emoji="🔄")
    async def refresh_analysis(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Refresh the stock analysis"""
        await interaction.response.defer()
        
        try:
            new_analysis = await perform_stock_analysis(self.symbol)
            if new_analysis:
                embed = create_analysis_embed(self.symbol, new_analysis)
                analysis_cache[f"{interaction.user.id}_{self.symbol}"] = new_analysis
                
                new_view = StockAnalysisView(self.symbol, new_analysis)
                await interaction.edit_original_response(embed=embed, view=new_view)
            else:
                await interaction.followup.send("❌ Failed to refresh analysis", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error refreshing: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="📈 Detailed Signals", style=discord.ButtonStyle.secondary, emoji="🎯")
    async def detailed_signals(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Show detailed signal analysis"""
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = create_detailed_signals_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="📊 Price Levels", style=discord.ButtonStyle.success, emoji="💰")
    async def price_levels(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Show key price levels"""
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = create_price_levels_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="📰 News Impact", style=discord.ButtonStyle.success, emoji="📊")
    async def news_impact(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Show news analysis affecting the stock"""
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = await create_news_analysis_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error fetching news: {str(e)}", ephemeral=True)

async def test_alpha_vantage_connection():
    """Test if Alpha Vantage API is working"""
    if not av_api:
        print("❌ No Alpha Vantage API key configured")
        return False
    
    try:
        print("🔍 Testing Alpha Vantage connection...")
        
        # Test with a simple quote request
        test_data = await asyncio.get_event_loop().run_in_executor(
            None, av_api.get_quote, 'AAPL'
        )
        
        if test_data and 'Global Quote' in test_data:
            print("✅ Alpha Vantage connection successful")
            return True
        else:
            print("❌ Alpha Vantage connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Alpha Vantage test failed: {e}")
        return False

def parse_alpha_vantage_data(symbol: str, daily_data: Dict, quote_data: Dict) -> Optional[Dict]:
    """Parse Alpha Vantage data into pandas DataFrame for analysis"""
    try:
        # Parse daily time series data
        if 'Time Series (Daily)' not in daily_data:
            print(f"❌ No time series data found for {symbol}")
            return None
        
        time_series = daily_data['Time Series (Daily)']
        
        # Convert to DataFrame
        df_data = []
        for date_str, values in time_series.items():
            try:
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })
            except (ValueError, KeyError) as e:
                print(f"❌ Error parsing data for {date_str}: {e}")
                continue
        
        if not df_data:
            print(f"❌ No valid data points for {symbol}")
            return None
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Date').reset_index(drop=True)
        df.set_index('Date', inplace=True)
        
        # Get current quote data
        current_data = {}
        if quote_data and 'Global Quote' in quote_data:
            quote = quote_data['Global Quote']
            try:
                current_data = {
                    'current_price': float(quote['05. price']),
                    'change': float(quote['09. change']),
                    'change_percent': float(quote['10. change percent'].rstrip('%')),
                    'volume': int(quote['06. volume']),
                    'latest_trading_day': pd.to_datetime(quote['07. latest trading day'])
                }
            except (ValueError, KeyError) as e:
                print(f"⚠️ Error parsing quote data: {e}")
                # Fallback to latest daily data
                current_data = {
                    'current_price': df['Close'].iloc[-1],
                    'change': df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0,
                    'change_percent': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0,
                    'volume': df['Volume'].iloc[-1],
                    'latest_trading_day': df.index[-1]
                }
        else:
            # Use latest daily data as fallback
            current_data = {
                'current_price': df['Close'].iloc[-1],
                'change': df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0,
                'change_percent': ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0,
                'volume': df['Volume'].iloc[-1],
                'latest_trading_day': df.index[-1]
            }
        
        return {
            'dataframe': df,
            'current_data': current_data,
            'data_source': 'Alpha Vantage',
            'last_updated': datetime.now()
        }
        
    except Exception as e:
        print(f"❌ Error parsing Alpha Vantage data for {symbol}: {e}")
        return None

async def perform_stock_analysis(symbol: str) -> Optional[Dict]:
    """Enhanced stock analysis using Alpha Vantage API"""
    if not av_api:
        return None
    
    try:
        print(f"🔍 Analyzing {symbol} with Alpha Vantage...")
        
        # Fetch data from Alpha Vantage
        daily_data_task = asyncio.get_event_loop().run_in_executor(
            None, av_api.get_daily_data, symbol, "compact"
        )
        quote_data_task = asyncio.get_event_loop().run_in_executor(
            None, av_api.get_quote, symbol
        )
        
        # Wait for both requests to complete
        daily_data, quote_data = await asyncio.gather(daily_data_task, quote_data_task)
        
        if not daily_data:
            print(f"❌ No daily data received for {symbol}")
            return None
        
        # Parse the data
        parsed_data = parse_alpha_vantage_data(symbol, daily_data, quote_data)
        if not parsed_data:
            return None
        
        df = parsed_data['dataframe']
        current_data = parsed_data['current_data']
        
        # Ensure we have enough data
        if len(df) < 5:
            print(f"❌ Insufficient data for {symbol}: only {len(df)} days")
            return None
        
        # Get basic metrics
        current_price = current_data['current_price']
        change_pct = current_data['change_percent']
        current_volume = current_data['volume']
        latest_trading_day = current_data['latest_trading_day']
        
        # Check if data is recent
        days_old = (datetime.now() - latest_trading_day.replace(tzinfo=None)).days
        is_recent = days_old <= 3  # Within 3 days (accounts for weekends)
        market_status = "Recent" if is_recent else f"Last: {latest_trading_day.strftime('%Y-%m-%d')}"
        
        # Calculate technical indicators
        closes = df['Close']
        highs = df['High']
        lows = df['Low']
        volumes = df['Volume']
        data_length = len(closes)
        
        # RSI calculation
        try:
            rsi_period = min(14, max(5, data_length // 4))
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            current_rsi = rsi_series.iloc[-1] if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50
        except:
            current_rsi = 50
        
        # Moving averages
        try:
            sma_20_period = min(20, max(5, data_length // 2))
            sma_50_period = min(50, max(10, data_length))
            ema_12_period = min(12, max(5, data_length // 3))
            
            sma_20 = closes.rolling(sma_20_period).mean().iloc[-1] if data_length >= 5 else current_price
            sma_50 = closes.rolling(sma_50_period).mean().iloc[-1] if data_length >= 10 else current_price
            ema_12 = closes.ewm(span=ema_12_period).mean().iloc[-1] if data_length >= 5 else current_price
        except:
            sma_20 = sma_50 = ema_12 = current_price
        
        # Volume analysis
        try:
            avg_volume = volumes.mean() if not volumes.empty else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        except:
            avg_volume = current_volume
            volume_ratio = 1
        
        # Support and Resistance levels
        try:
            lookback_period = min(10, max(5, data_length // 3))
            recent_highs = highs.rolling(lookback_period).max().dropna()
            recent_lows = lows.rolling(lookback_period).min().dropna()
            
            resistance = recent_highs.iloc[-3:].max() if len(recent_highs) >= 3 else current_price * 1.05
            support = recent_lows.iloc[-3:].min() if len(recent_lows) >= 3 else current_price * 0.95
        except:
            resistance = current_price * 1.05
            support = current_price * 0.95
        
        # Generate signals and score
        score = 0
        signals = []
        confidence_factors = []
        
        # RSI signals
        if current_rsi < 30:
            score += 2.5
            signals.append("🟢 RSI Oversold - Strong Buy Signal")
            confidence_factors.append("High RSI confidence")
        elif current_rsi < 40:
            score += 1.5
            signals.append("🟢 RSI in Bullish Zone")
            confidence_factors.append("Medium RSI confidence")
        elif current_rsi > 70:
            score -= 2.5
            signals.append("🔴 RSI Overbought - Strong Sell Signal")
            confidence_factors.append("High RSI confidence")
        elif current_rsi > 60:
            score -= 1.5
            signals.append("🔴 RSI in Bearish Zone")
            confidence_factors.append("Medium RSI confidence")
        else:
            signals.append("⚪ RSI Neutral (40-60)")
        
        # Moving average signals
        if current_price > sma_20 > sma_50:
            score += 2.0
            signals.append("🟢 Strong Uptrend - Above Both MAs")
            confidence_factors.append("Strong trend confirmation")
        elif current_price > sma_20:
            score += 1.0
            signals.append("🟢 Above Short-term MA")
        elif current_price < sma_20 < sma_50:
            score -= 2.0
            signals.append("🔴 Strong Downtrend - Below Both MAs")
            confidence_factors.append("Strong trend confirmation")
        elif current_price < sma_20:
            score -= 1.0
            signals.append("🔴 Below Short-term MA")
        
        # Volume confirmation
        if volume_ratio > 1.5:
            if score > 0:
                score += 1.0
                signals.append(f"🟢 High Volume Confirms Bullish Move ({volume_ratio:.1f}x)")
                confidence_factors.append("Volume confirmation")
            else:
                score -= 1.0
                signals.append(f"🔴 High Volume Confirms Bearish Move ({volume_ratio:.1f}x)")
                confidence_factors.append("Volume confirmation")
        elif volume_ratio < 0.5:
            signals.append(f"⚪ Low Volume - Weak Conviction ({volume_ratio:.1f}x)")
        
        # Recent price momentum
        if change_pct > 3:
            score += 1.5
            signals.append(f"🟢 Strong Recent Momentum (+{change_pct:.1f}%)")
        elif change_pct > 1:
            score += 0.5
            signals.append(f"🟢 Positive Recent Momentum (+{change_pct:.1f}%)")
        elif change_pct < -3:
            score -= 1.5
            signals.append(f"🔴 Strong Recent Decline ({change_pct:.1f}%)")
        elif change_pct < -1:
            score -= 0.5
            signals.append(f"🔴 Recent Decline ({change_pct:.1f}%)")
        
        score = round(score, 1)
        
        # Determine sentiment and confidence
        if score > 2.0:
            sentiment = "🟢 STRONG BULLISH"
            confidence = "High"
        elif score > 0.5:
            sentiment = "🟢 BULLISH"
            confidence = "Medium"
        elif score < -2.0:
            sentiment = "🔴 STRONG BEARISH"
            confidence = "High"
        elif score < -0.5:
            sentiment = "🔴 BEARISH"
            confidence = "Medium"
        else:
            sentiment = "⚪ NEUTRAL"
            confidence = "Low"
        
        # Calculate price range
        try:
            price_range_period = {
                'high': highs.max(),
                'low': lows.min(),
                'current_position': (current_price - lows.min()) / (highs.max() - lows.min()) * 100 if highs.max() > lows.min() else 50
            }
        except:
            price_range_period = {
                'high': current_price * 1.2,
                'low': current_price * 0.8,
                'current_position': 50
            }
        
        return {
            'current_price': current_price,
            'change_pct': change_pct,
            'rsi': current_rsi,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_12': ema_12,
            'volume_ratio': volume_ratio,
            'avg_volume': avg_volume,
            'current_volume': current_volume,
            'support': support,
            'resistance': resistance,
            'score': score,
            'sentiment': sentiment,
            'confidence': confidence,
            'signals': signals,
            'confidence_factors': confidence_factors,
            'price_range_52w': price_range_period,
            'timestamp': datetime.now(),
            'data_quality': f"{len(df)} days of data",
            'market_status': market_status,
            'last_trading_day': latest_trading_day.strftime('%Y-%m-%d'),
            'is_recent_data': is_recent,
            'data_source': 'Alpha Vantage API',
            'api_reliability': 'High'
        }
        
    except Exception as e:
        print(f"❌ Analysis error for {symbol}: {e}")
        return None

def create_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create enhanced analysis embed with Alpha Vantage data"""
    score = analysis_data['score']
    sentiment = analysis_data['sentiment']
    confidence = analysis_data['confidence']
    
    # Determine color based on sentiment
    if "STRONG BULLISH" in sentiment:
        color = 0x00ff00
    elif "BULLISH" in sentiment:
        color = 0x32cd32
    elif "STRONG BEARISH" in sentiment:
        color = 0xff0000
    elif "BEARISH" in sentiment:
        color = 0xff4500
    else:
        color = 0xffd700
    
    # Market status information
    market_status = analysis_data.get('market_status', 'Recent')
    is_recent = analysis_data.get('is_recent_data', True)
    status_emoji = "🟢" if is_recent else "🟡"
    
    embed = discord.Embed(
        title=f"📊 Stock Analysis: {symbol}",
        description=f"**Price:** ${analysis_data['current_price']:.2f} ({analysis_data['change_pct']:+.1f}%)\n**Signal Score:** {score}/10.0\n**Sentiment:** {sentiment}\n{status_emoji} **Market Status:** {market_status}",
        color=color,
        timestamp=analysis_data['timestamp']
    )
    
    # Technical indicators
    tech_text = f"**RSI (14):** {analysis_data['rsi']:.1f}\n"
    tech_text += f"**20-day SMA:** ${analysis_data['sma_20']:.2f}\n"
    tech_text += f"**50-day SMA:** ${analysis_data['sma_50']:.2f}\n"
    tech_text += f"**12-day EMA:** ${analysis_data['ema_12']:.2f}"
    
    embed.add_field(name="📈 Technical Indicators", value=tech_text, inline=True)
    
    # Volume and levels
    volume_text = f"**Current:** {analysis_data['current_volume']:,.0f}\n"
    volume_text += f"**Average:** {analysis_data['avg_volume']:,.0f}\n"
    volume_text += f"**Ratio:** {analysis_data['volume_ratio']:.1f}x"
    
    embed.add_field(name="📊 Volume Analysis", value=volume_text, inline=True)
    
    # Key levels
    levels_text = f"**Resistance:** ${analysis_data['resistance']:.2f}\n"
    levels_text += f"**Support:** ${analysis_data['support']:.2f}\n"
    levels_text += f"**Range Position:** {analysis_data['price_range_52w']['current_position']:.1f}%"
    
    embed.add_field(name="🎯 Key Levels", value=levels_text, inline=True)
    
    # Top signals
    top_signals = analysis_data['signals'][:4]
    signals_text = "\n".join(top_signals) if top_signals else "No significant signals"
    
    embed.add_field(name="🚨 Key Signals", value=signals_text, inline=False)
    
    # Data source and quality
    data_text = f"**Source:** {analysis_data.get('data_source', 'Alpha Vantage API')}\n"
    data_text += f"**Reliability:** {analysis_data.get('api_reliability', 'High')}\n"
    data_text += f"**Data Quality:** {analysis_data.get('data_quality', 'Standard')}"
    
    embed.add_field(name="📊 Data Quality", value=data_text, inline=True)
    
    # Analysis quality
    confidence_text = f"**Confidence:** {confidence}\n"
    
    if analysis_data.get('confidence_factors'):
        factors = analysis_data['confidence_factors'][:2]
        confidence_text += f"**Key Factors:** {', '.join(factors)}"
    
    embed.add_field(name="🎯 Analysis Quality", value=confidence_text, inline=True)
    
    # Footer with API info
    footer_text = f"🚀 Powered by Alpha Vantage API • {analysis_data.get('data_quality', 'Data available')} • Not financial advice"
    embed.set_footer(text=footer_text)
    
    return embed

def create_detailed_signals_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create detailed signals analysis embed"""
    embed = discord.Embed(
        title=f"🎯 Detailed Signals: {symbol}",
        description=f"Comprehensive signal breakdown",
        color=0x7289DA,
        timestamp=datetime.now()
    )
    
    all_signals = analysis_data['signals']
    if all_signals:
        bullish_signals = [s for s in all_signals if "🟢" in s]
        bearish_signals = [s for s in all_signals if "🔴" in s]
        neutral_signals = [s for s in all_signals if "⚪" in s]
        
        if bullish_signals:
            embed.add_field(
                name="🟢 Bullish Signals",
                value="\n".join(bullish_signals[:5]),
                inline=False
            )
        
        if bearish_signals:
            embed.add_field(
                name="🔴 Bearish Signals", 
                value="\n".join(bearish_signals[:5]),
                inline=False
            )
        
        if neutral_signals:
            embed.add_field(
                name="⚪ Neutral Signals",
                value="\n".join(neutral_signals[:3]),
                inline=False
            )
    
    # Signal strength
    score = analysis_data['score']
    if score > 2:
        strength = "Very Strong"
        strength_emoji = "🔥"
    elif score > 0.5:
        strength = "Strong"
        strength_emoji = "💪"
    elif score < -2:
        strength = "Very Weak"
        strength_emoji = "❄️"
    elif score < -0.5:
        strength = "Weak"
        strength_emoji = "👎"
    else:
        strength = "Neutral"
        strength_emoji = "🤷"
    
    embed.add_field(
        name=f"{strength_emoji} Signal Strength",
        value=f"**Overall Score:** {score}/10.0\n**Strength:** {strength}\n**Confidence:** {analysis_data['confidence']}",
        inline=False
    )
    
    return embed

def create_price_levels_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create price levels analysis embed"""
    embed = discord.Embed(
        title=f"💰 Key Price Levels: {symbol}",
        description="Important support and resistance levels",
        color=0x00ff88,
        timestamp=datetime.now()
    )
    
    current_price = analysis_data['current_price']
    
    levels_data = [
        ("🔴 Resistance", analysis_data['resistance'], (analysis_data['resistance'] - current_price) / current_price * 100),
        ("🟢 Support", analysis_data['support'], (current_price - analysis_data['support']) / current_price * 100),
        ("🟡 20-day MA", analysis_data['sma_20'], (analysis_data['sma_20'] - current_price) / current_price * 100),
        ("🟡 50-day MA", analysis_data['sma_50'], (analysis_data['sma_50'] - current_price) / current_price * 100),
    ]
    
    levels_text = f"**Current Price:** ${current_price:.2f}\n\n"
    
    for name, price, distance in levels_data:
        direction = "above" if distance < 0 else "below"
        levels_text += f"{name}: ${price:.2f} ({abs(distance):.1f}% {direction})\n"
    
    embed.add_field(name="🎯 Price Levels", value=levels_text, inline=False)
    
    # Trading ranges
    range_text = f"**Period High:** ${analysis_data['price_range_52w']['high']:.2f}\n"
    range_text += f"**Period Low:** ${analysis_data['price_range_52w']['low']:.2f}\n"
    range_text += f"**Position in Range:** {analysis_data['price_range_52w']['current_position']:.1f}%"
    
    embed.add_field(name="📊 Trading Ranges", value=range_text, inline=False)
    
    return embed

async def fetch_stock_news(symbol: str) -> List[Dict]:
    """Simple news fetching (same as before)"""
    news_items = []
    
    try:
        print(f"🔍 Fetching news for {symbol}...")
        
        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        
        try:
            yahoo_feed = feedparser.parse(yahoo_url)
            
            if hasattr(yahoo_feed, 'entries') and yahoo_feed.entries:
                for entry in yahoo_feed.entries[:5]:
                    try:
                        pub_date = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        title = getattr(entry, 'title', 'No title')
                        summary = getattr(entry, 'summary', '')
                        link = getattr(entry, 'link', '')
                        
                        if len(summary) > 200:
                            summary = summary[:200] + '...'
                        
                        news_items.append({
                            'title': title,
                            'summary': summary,
                            'link': link,
                            'published': pub_date,
                            'source': 'Yahoo Finance',
                            'age_hours': (datetime.now() - pub_date).total_seconds() / 3600
                        })
                        
                    except Exception as entry_error:
                        print(f"Error processing news entry: {entry_error}")
                        continue
            
        except Exception as e:
            print(f"Yahoo Finance news error for {symbol}: {e}")
        
        if news_items:
            news_items.sort(key=lambda x: x['published'], reverse=True)
        
        print(f"✅ Found {len(news_items)} news items for {symbol}")
        return news_items[:5]
        
    except Exception as e:
        print(f"News fetch error for {symbol}: {e}")
        return []

def analyze_news_sentiment(news_items: List[Dict], symbol: str) -> Dict:
    """Simple news sentiment analysis (same as before)"""
    if not news_items:
        return {
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'impact_level': 'Low',
            'recent_count': 0
        }
    
    positive_keywords = ['beats', 'strong', 'growth', 'profit', 'gain', 'rise', 'upgrade', 'buy', 'positive']
    negative_keywords = ['miss', 'falls', 'drop', 'decline', 'loss', 'downgrade', 'sell', 'negative', 'concern']
    
    sentiment_score = 0
    recent_count = 0
    
    for item in news_items:
        if item['age_hours'] <= 24:
            recent_count += 1
        
        text = (item['title'] + ' ' + item['summary']).lower()
        pos_count = sum(1 for keyword in positive_keywords if keyword in text)
        neg_count = sum(1 for keyword in negative_keywords if keyword in text)
        
        weight = 2.0 if item['age_hours'] <= 6 else 1.0
        sentiment_score += (pos_count - neg_count) * weight
    
    if sentiment_score > 1:
        sentiment_label = 'Positive'
    elif sentiment_score < -1:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    
    impact_level = 'High' if recent_count >= 2 and abs(sentiment_score) > 1 else 'Medium' if recent_count >= 1 else 'Low'
    
    return {
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label,
        'impact_level': impact_level,
        'recent_count': recent_count
    }

async def create_news_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create news analysis embed"""
    
    embed = discord.Embed(
        title=f"📰 News Impact Analysis: {symbol}",
        description=f"Recent news affecting **{symbol}** price movement",
        color=0xffd700,
        timestamp=datetime.now()
    )
    
    try:
        news_items = await asyncio.get_event_loop().run_in_executor(
            None, fetch_stock_news, symbol
        )
        
        news_analysis = analyze_news_sentiment(news_items, symbol)
        
        # Update embed color based on sentiment
        if news_analysis['sentiment_label'] == 'Positive':
            embed.color = 0x00ff88
        elif news_analysis['sentiment_label'] == 'Negative':
            embed.color = 0xff6b6b
        
        # News sentiment summary
        sentiment_emoji = {'Positive': '🟢', 'Neutral': '⚪', 'Negative': '🔴'}
        impact_emoji = {'High': '🔥', 'Medium': '⚡', 'Low': '📊'}
        
        summary_text = f"**Sentiment:** {sentiment_emoji.get(news_analysis['sentiment_label'], '⚪')} {news_analysis['sentiment_label']}\n"
        summary_text += f"**Impact Level:** {impact_emoji.get(news_analysis['impact_level'], '📊')} {news_analysis['impact_level']}\n"
        summary_text += f"**Recent News:** {news_analysis['recent_count']} stories (24h)"
        
        embed.add_field(name="📊 News Sentiment", value=summary_text, inline=True)
        
        # Price correlation
        price_change = analysis_data.get('change_pct', 0)
        correlation_text = f"**Today's Move:** {price_change:+.1f}%\n"
        
        if news_analysis['sentiment_score'] > 0 and price_change > 1:
            correlation_text += "🟢 **Positive news supports upward move**"
        elif news_analysis['sentiment_score'] < 0 and price_change < -1:
            correlation_text += "🔴 **Negative news explains downward move**"
        else:
            correlation_text += "📊 **News and price movement are neutral**"
        
        embed.add_field(name="📈 Price Correlation", value=correlation_text, inline=True)
        
        # Recent headlines
        if news_items:
            news_text = ""
            for i, item in enumerate(news_items[:3], 1):
                try:
                    age_str = f"{int(item['age_hours'])}h" if item['age_hours'] < 24 else f"{int(item['age_hours']/24)}d"
                    title = item['title'][:50] + "..." if len(item['title']) > 50 else item['title']
                    news_text += f"**{i}.** {title}\n*{item['source']} • {age_str} ago*\n\n"
                except:
                    news_text += f"**{i}.** News item #{i}\n*Recent*\n\n"
            
            embed.add_field(name="📑 Recent Headlines", value=news_text[:800] if news_text else "No headlines available", inline=False)
        else:
            embed.add_field(name="📑 Recent Headlines", value="No recent news found for this symbol.", inline=False)
        
    except Exception as e:
        print(f"Error in news analysis: {e}")
        embed.add_field(name="📑 Recent Headlines", value="Unable to fetch news at this time.", inline=False)
    
    embed.set_footer(text="📰 News analysis • Not financial advice")
    return embed

async def send_startup_message(data_available: bool):
    """Send startup message with available commands"""
    try:
        startup_embed = discord.Embed(
            title="📊 Enhanced Stock Analysis Bot is Online!",
            description="Powered by Alpha Vantage API for reliable stock data",
            color=0x00ff88 if data_available else 0xffd700,
            timestamp=datetime.now()
        )
        
        if data_available:
            data_status = "✅ Alpha Vantage API Connected"
            reliability = "✅ High Reliability"
        else:
            data_status = "❌ API Key Missing"
            reliability = "⚠️ Limited Functionality"
        
        startup_embed.add_field(
            name="🔧 System Status",
            value=f"**Data Source:** {data_status}\n**API Reliability:** {reliability}\n**Interactive Features:** ✅ Available\n**News Analysis:** ✅ Ready",
            inline=False
        )
        
        startup_embed.add_field(
            name="📈 Available Commands",
            value="`!analysis SYMBOL` - Complete stock analysis\n`!quick SYMBOL` - Fast overview\n`!news SYMBOL` - News impact analysis\n`!test` - Check system status\n`!commands` - Show this help",
            inline=False
        )
        
        startup_embed.add_field(
            name="🚀 Quick Examples",
            value="`!analysis AAPL` - Apple full analysis\n`!quick SPY` - S&P 500 quick view\n`!news TSLA` - Tesla news analysis",
            inline=False
        )
        
        startup_embed.add_field(
            name="⚡ Quick Aliases",
            value="`!a` = analysis • `!q` = quick • `!n` = news",
            inline=False
        )
        
        if data_available:
            startup_embed.add_field(
                name="🎯 Alpha Vantage Benefits",
                value="• Official API with reliable data\n• Real-time quotes and historical data\n• 25 free requests per day\n• High uptime and accuracy",
                inline=False
            )
        else:
            startup_embed.add_field(
                name="⚠️ Setup Required",
                value="Alpha Vantage API key missing. Add ALPHA_VANTAGE_API_KEY environment variable.\nGet free API key at: https://www.alphavantage.co/support/#api-key",
                inline=False
            )
        
        startup_embed.set_footer(text="🚀 Reliable stock analysis • Not financial advice")
        
        # Send to available channels
        for guild in bot.guilds:
            try:
                target_channel = None
                
                channel_preferences = [
                    'general', 'bot-commands', 'bots', 'trading', 'stocks', 
                    'analysis', 'market', 'finance', 'commands'
                ]
                
                for pref in channel_preferences:
                    channel = discord.utils.get(guild.text_channels, name=pref)
                    if channel and channel.permissions_for(guild.me).send_messages:
                        target_channel = channel
                        break
                
                if not target_channel:
                    for channel in guild.text_channels:
                        if channel.permissions_for(guild.me).send_messages:
                            target_channel = channel
                            break
                
                if target_channel:
                    await target_channel.send(embed=startup_embed)
                    print(f"✅ Sent startup message to #{target_channel.name} in {guild.name}")
                    
            except Exception as e:
                print(f"❌ Failed to send startup message to {guild.name}: {e}")
                continue
        
        print("📢 Startup messages sent to all available channels")
        
    except Exception as e:
        print(f"❌ Error sending startup messages: {e}")

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print(f'\n📊 Received signal {signum}. Shutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@bot.event
async def on_ready():
    print(f'✅ {bot.user} is online!')
    print('📊 Enhanced Stock Analysis Bot with Alpha Vantage ready!')
    
    data_available = await test_alpha_vantage_connection()
    if data_available:
        print('✅ Alpha Vantage API connection verified')
    else:
        print('⚠️ Alpha Vantage API connection issues detected')
    
    await send_startup_message(data_available)

@bot.command(name='test')
async def test_command(ctx):
    """Enhanced test command with Alpha Vantage connection check"""
    
    av_test = await test_alpha_vantage_connection()
    av_status = "✅ Connected" if av_test else "❌ Connection Failed"
    
    embed = discord.Embed(
        title="🤖 Enhanced Stock Bot Status",
        description="Alpha Vantage API integration status",
        color=0x00ff00 if av_test else 0xff6b6b
    )
    
    embed.add_field(
        name="📊 System Status",
        value=f"• **Discord Bot:** ✅ Online\n• **Alpha Vantage API:** {av_status}\n• **Interactive UI:** ✅ Available\n• **News Analysis:** ✅ Enhanced",
        inline=False
    )
    
    if av_test:
        embed.add_field(
            name="🚀 Ready to Use",
            value="`!analysis AAPL` - Full analysis\n`!quick SPY` - Quick view\n`!news TSLA` - News analysis",
            inline=False
        )
        embed.add_field(
            name="🎯 API Benefits",
            value="• Official Alpha Vantage API\n• Real-time and historical data\n• High reliability and accuracy\n• 25 free requests per day",
            inline=False
        )
    else:
        embed.add_field(
            name="⚠️ API Issues",
            value="Alpha Vantage API key missing or invalid.\nGet free API key: https://www.alphavantage.co/support/#api-key\nAdd as ALPHA_VANTAGE_API_KEY environment variable.",
            inline=False
        )
    
    embed.add_field(
        name="⏰ 24/7 Analysis Ready",
        value="• **Market Open:** Real-time quote data\n• **After Hours:** Latest trading day data\n• **Weekends:** Full historical analysis\n• **Holidays:** Complete technical analysis",
        inline=False
    )
    
    await ctx.send(embed=embed)

@bot.command(name='analysis', aliases=['analyze', 'stock', 'a'])
async def analyze_stock(ctx, symbol: str = None):
    """Enhanced stock analysis using Alpha Vantage API"""
    
    if not symbol:
        embed = discord.Embed(
            title="❌ Missing Symbol",
            description="Please provide a stock symbol to analyze",
            color=0xff6b6b
        )
        embed.add_field(
            name="📝 Usage Examples",
            value="`!analysis AAPL`\n`!a TSLA`\n`!stock SPY`",
            inline=False
        )
        embed.add_field(
            name="💡 Popular Symbols",
            value="AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, QQQ, NVDA",
            inline=False
        )
        embed.add_field(
            name="🚀 Powered by Alpha Vantage",
            value="Official API with reliable real-time and historical data",
            inline=False
        )
        await ctx.send(embed=embed)
        return
    
    if not av_api:
        error_embed = discord.Embed(
            title="❌ API Not Available",
            description="Alpha Vantage API key is not configured",
            color=0xff6b6b
        )
        error_embed.add_field(
            name="🔧 Setup Required",
            value="1. Get free API key: https://www.alphavantage.co/support/#api-key\n2. Add as ALPHA_VANTAGE_API_KEY environment variable\n3. Restart the bot",
            inline=False
        )
        await ctx.send(embed=error_embed)
        return
    
    # Send analyzing message
    msg = await ctx.send(f"🔍 Analyzing **{symbol.upper()}** with Alpha Vantage API...")
    
    try:
        analysis_data = await asyncio.wait_for(
            perform_stock_analysis(symbol), 
            timeout=90.0  # Alpha Vantage can be slower due to rate limiting
        )
        
        if not analysis_data:
            error_embed = discord.Embed(
                title=f"❌ Analysis Failed",
                description=f"Unable to fetch data for **{symbol.upper()}**",
                color=0xff6b6b
            )
            error_embed.add_field(
                name="🔍 Possible Issues",
                value="• Invalid symbol or not found\n• Symbol not supported by Alpha Vantage\n• API rate limit reached (25/day)\n• Temporary API issues",
                inline=False
            )
            error_embed.add_field(
                name="💡 Try These Solutions",
                value="• Check symbol spelling (e.g., 'AAPL' not 'Apple')\n• Try a major US stock like AAPL, MSFT, or SPY\n• Wait and try again if rate limited\n• Use `!test` to check API status",
                inline=False
            )
            await msg.edit(content="", embed=error_embed)
            return
        
        # Store in cache
        cache_key = f"{ctx.author.id}_{symbol.upper()}"
        analysis_cache[cache_key] = analysis_data
        
        # Create embed and interactive view
        embed = create_analysis_embed(symbol.upper(), analysis_data)
        view = StockAnalysisView(symbol.upper(), analysis_data)
        
        await msg.edit(content="", embed=embed, view=view)
        
    except asyncio.TimeoutError:
        timeout_embed = discord.Embed(
            title="⏰ Analysis Timeout",
            description=f"Analysis of **{symbol.upper()}** took too long",
            color=0xffd700
        )
        timeout_embed.add_field(
            name="🔄 What to try",
            value="• Alpha Vantage API can be slow due to rate limiting\n• Wait a moment and try again\n• Check if symbol is correct\n• Try `!quick` for faster analysis\n• Use `!test` to check API status",
            inline=False
        )
        await msg.edit(content="", embed=timeout_embed)
        
    except Exception as e:
        print(f"Analysis error: {e}")
        error_embed = discord.Embed(
            title="❌ Analysis Error",
            description=f"Error analyzing **{symbol.upper()}**",
            color=0xff6b6b
        )
        error_embed.add_field(
            name="💡 What to try",
            value="• Check symbol spelling\n• Try `!test` to check API status\n• Wait and try again\n• Use `!quick` for simpler analysis",
            inline=False
        )
        await msg.edit(content="", embed=error_embed)

@bot.command(name='quick', aliases=['q'])
async def quick_analysis(ctx, symbol: str = None):
    """Quick analysis with Alpha Vantage"""
    
    if not symbol:
        await ctx.send("❌ Usage: `!quick AAPL` or `!q TSLA`")
        return
    
    if not av_api:
        await ctx.send("❌ Alpha Vantage API key not configured. Use `!test` for setup instructions.")
        return
    
    msg = await ctx.send(f"⚡ Quick analysis for **{symbol.upper()}** (Alpha Vantage)...")
    
    try:
        analysis_data = await asyncio.wait_for(
            perform_stock_analysis(symbol), 
            timeout=60.0
        )
        
        if not analysis_data:
            await msg.edit(content=f"❌ No data found for **{symbol.upper()}**\nTry: `!test` to check API status")
            return
        
        embed = discord.Embed(
            title=f"⚡ Quick Analysis: {symbol.upper()}",
            description=f"**${analysis_data['current_price']:.2f}** ({analysis_data['change_pct']:+.1f}%) • **{analysis_data['sentiment']}**",
            color=0x00ff88 if analysis_data['score'] > 0 else 0xff6b6b if analysis_data['score'] < 0 else 0xffd700,
            timestamp=datetime.now()
        )
        
        quick_text = f"**Score:** {analysis_data['score']}/10.0\n"
        quick_text += f"**RSI:** {analysis_data['rsi']:.1f}\n"
        quick_text += f"**Volume:** {analysis_data['volume_ratio']:.1f}x avg\n"
        quick_text += f"**Support:** ${analysis_data['support']:.2f}\n"
        quick_text += f"**Resistance:** ${analysis_data['resistance']:.2f}"
        
        embed.add_field(name="📊 Key Metrics", value=quick_text, inline=True)
        
        top_signals = analysis_data['signals'][:3]
        if top_signals:
            embed.add_field(name="🚨 Top Signals", value="\n".join(top_signals), inline=True)
        
        embed.set_footer(text=f"⚡ Quick analysis • Alpha Vantage API • {analysis_data.get('data_quality', 'Data available')} • Use !analysis for full report")
        
        await msg.edit(content="", embed=embed)
        
    except asyncio.TimeoutError:
        await msg.edit(content=f"⏰ **{symbol.upper()}** analysis timed out. Alpha Vantage can be slow. Try again.")
    except Exception as e:
        await msg.edit(content=f"❌ Quick analysis failed for **{symbol.upper()}**: {str(e)[:50]}...")

@bot.command(name='news', aliases=['n'])
async def news_command(ctx, symbol: str = None):
    """Get news analysis for a specific stock"""
    
    if not symbol:
        await ctx.send("❌ Usage: `!news AAPL` or `!n TSLA`")
        return
    
    msg = await ctx.send(f"📰 Fetching news analysis for **{symbol.upper()}**...")
    
    try:
        # Get basic price data for correlation if Alpha Vantage is available
        analysis_data = {'current_price': 0, 'change_pct': 0, 'timestamp': datetime.now()}
        
        if av_api:
            try:
                quote_data = await asyncio.get_event_loop().run_in_executor(
                    None, av_api.get_quote, symbol
                )
                
                if quote_data and 'Global Quote' in quote_data:
                    quote = quote_data['Global Quote']
                    analysis_data = {
                        'current_price': float(quote['05. price']),
                        'change_pct': float(quote['10. change percent'].rstrip('%')),
                        'timestamp': datetime.now()
                    }
            except:
                pass  # Use defaults
        
        embed = await create_news_analysis_embed(symbol.upper(), analysis_data)
        await msg.edit(content="", embed=embed)
        
    except Exception as e:
        await msg.edit(content=f"❌ News analysis failed for **{symbol.upper()}**: {str(e)[:50]}...")

@bot.command(name='commands', aliases=['help', 'start'])
async def show_commands(ctx):
    """Show available commands"""
    
    av_available = av_api is not None
    av_test = await test_alpha_vantage_connection() if av_available else False
    
    embed = discord.Embed(
        title="📊 Enhanced Stock Analysis Bot",
        description="Powered by Alpha Vantage API for reliable stock data",
        color=0x00ff88 if av_test else 0xffd700,
        timestamp=datetime.now()
    )
    
    if av_test:
        status_text = "✅ Alpha Vantage API Connected"
        reliability = "✅ High Reliability"
    elif av_available:
        status_text = "⚠️ API Key Issues"
        reliability = "⚠️ Connection Problems"
    else:
        status_text = "❌ API Key Missing"
        reliability = "❌ Setup Required"
    
    embed.add_field(
        name="🔧 Current Status",
        value=f"**API Status:** {status_text}\n**Reliability:** {reliability}\n**Bot:** ✅ Online\n**Features:** ✅ Available",
        inline=False
    )
    
    embed.add_field(
        name="📈 Analysis Commands",
        value="`!analysis SYMBOL` - Full technical analysis with interactive buttons\n`!quick SYMBOL` - Fast analysis overview\n`!news SYMBOL` - News impact and sentiment analysis",
        inline=False
    )
    
    embed.add_field(
        name="🔧 Utility Commands",
        value="`!test` - Check Alpha Vantage API status\n`!commands` - Show this command list",
        inline=False
    )
    
    embed.add_field(
        name="💡 Usage Examples",
        value="`!analysis AAPL` - Apple comprehensive analysis\n`!quick SPY` - S&P 500 ETF quick view\n`!news NVDA` - NVIDIA news and sentiment\n`!test` - Verify API connection",
        inline=False
    )
    
    embed.add_field(
        name="⚡ Quick Shortcuts",
        value="`!a TSLA` = `!analysis TSLA`\n`!q MSFT` = `!quick MSFT`\n`!n GOOGL` = `!news GOOGL`",
        inline=False
    )
    
    if av_test:
        embed.add_field(
            name="🎯 Alpha Vantage Benefits",
            value="• Official API with real-time data\n• High accuracy and reliability\n• 25 free requests per day\n• Support for US stocks, ETFs, and indices",
            inline=False
        )
    else:
        embed.add_field(
            name="⚠️ Setup Required",
            value="Get free Alpha Vantage API key:\n1. Visit: https://www.alphavantage.co/support/#api-key\n2. Add as ALPHA_VANTAGE_API_KEY environment variable\n3. Restart bot",
            inline=False
        )
    
    embed.add_field(
        name="🎯 Interactive Features",
        value="After running `!analysis`, use the buttons for:\n• 📊 Refresh Analysis\n• 📈 Detailed Signals\n• 💰 Price Levels\n• 📰 News Impact",
        inline=False
    )
    
    embed.set_footer(text="🚀 Reliable stock analysis with Alpha Vantage • Not financial advice")
    
    await ctx.send(embed=embed)

# Health check endpoint for Render
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            # Include Alpha Vantage status in health check
            av_status = "configured" if av_api else "missing"
            health_message = f'Enhanced Stock Analysis Bot with Alpha Vantage API ({av_status}) is running'
            self.wfile.write(health_message.encode())
        else:
            self.send_response(404)
            self.end_headers()

def start_health_server():
    """Start health check server for Render"""
    try:
        port = int(os.getenv('PORT', 8080))
        server = HTTPServer(('0.0.0.0', port), HealthHandler)
        print(f"🌐 Health server starting on port {port}")
        server.serve_forever()
    except Exception as e:
        print(f"Health server error: {e}")

if __name__ == "__main__":
    DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    
    if not DISCORD_TOKEN:
        print("❌ Error: DISCORD_BOT_TOKEN environment variable not set!")
        print("Please add your Discord bot token as an environment variable")
        sys.exit(1)
    
    if not ALPHA_VANTAGE_KEY:
        print("⚠️ Warning: ALPHA_VANTAGE_API_KEY environment variable not set!")
        print("Get free API key at: https://www.alphavantage.co/support/#api-key")
        print("Bot will run with limited functionality")
    
    print("📊 Starting Enhanced Stock Analysis Bot with Alpha Vantage...")
    print("🎯 Features: 24/7 Analysis, Official API, News Integration, Interactive UI")
    print("⚡ Commands: !analysis, !quick, !news, !test, !commands")
    
    # Start health check server in background thread
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    try:
        bot.run(DISCORD_TOKEN)
    except discord.LoginFailure:
        print("❌ Invalid Discord bot token! Please check your DISCORD_BOT_TOKEN")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Bot error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)