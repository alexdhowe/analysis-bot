#!/usr/bin/env python3
"""
Enhanced Stock Analysis Bot with Polygon.io API - COMPLETE FIXED VERSION WITH REAL-TIME PRICES
24/7 stock analysis with professional-grade API
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

print("📊 Starting Enhanced Stock Analysis Bot with Polygon.io...", flush=True)
print(f"🐍 Python version: {sys.version}", flush=True)
print(f"🤖 Discord.py version: {discord.__version__}", flush=True)

# Debug environment variables (don't print the actual values for security)
print(f"🔑 DISCORD_TOKEN present: {'Yes' if os.getenv('DISCORD_TOKEN') else 'No'}", flush=True)
print(f"📈 POLYGON_API_KEY present: {'Yes' if os.getenv('POLYGON_API_KEY') else 'No'}", flush=True)
print(f"🌐 PORT: {os.getenv('PORT', '10000')}", flush=True)

# Bot setup - DISABLE DEFAULT HELP COMMAND
intents = discord.Intents.default()
intents.message_content = True
print("🎯 Bot intents configured", flush=True)

bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)
print("🤖 Bot instance created", flush=True)

# Global storage for analysis results
analysis_cache = {}

def rate_limit(calls_per_minute=5):
    """Rate limiter for Polygon.io's free tier"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(1)  # Conservative rate limiting
            return func(*args, **kwargs)
        return wrapper
    return decorator

class PolygonAPI:
    """Polygon.io API wrapper - COMPLETE FIXED VERSION WITH REAL-TIME QUOTES"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'StockAnalysisBot/1.0'
        })
    
    @rate_limit(calls_per_minute=5)
    def get_current_quote(self, symbol: str) -> Optional[Dict]:
        """Get current/real-time quote data"""
        try:
            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol.upper()}"
            
            params = {
                'apikey': self.api_key
            }
            
            print(f"🔍 Fetching current quote for {symbol}...", flush=True)
            print(f"🌐 URL: {url}", flush=True)
            
            response = self.session.get(url, params=params, timeout=30)
            
            print(f"📊 Response status: {response.status_code}", flush=True)
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}: {response.text}", flush=True)
                return None
            
            data = response.json()
            
            # Handle both 'OK' and 'DELAYED' status
            if data.get('status') in ['OK', 'DELAYED'] and 'results' in data and len(data['results']) > 0:
                print(f"✅ Successfully fetched current quote for {symbol}", flush=True)
                result = data['results'][0]
                print(f"💰 Current quote data: {result}", flush=True)
                return result
            else:
                print(f"❌ No current quote data for {symbol}: {data}", flush=True)
                return None
                
        except Exception as e:
            print(f"❌ Error fetching current quote for {symbol}: {e}", flush=True)
            return None

    @rate_limit(calls_per_minute=5)
    def get_previous_close(self, symbol: str) -> Optional[Dict]:
        """Get previous trading day's data - FIXED VERSION"""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/prev"
            
            # Fixed: Use proper parameter name and format
            params = {
                'apikey': self.api_key,
                'adjusted': 'true'
            }
            
            print(f"🔍 Fetching previous close for {symbol}...", flush=True)
            print(f"🌐 URL: {url}", flush=True)
            print(f"🔑 Using API key: {self.api_key[:8]}...{self.api_key[-4:]}", flush=True)
            
            response = self.session.get(url, params=params, timeout=30)
            
            print(f"📊 Response status: {response.status_code}", flush=True)
            print(f"📊 Response headers: {dict(response.headers)}", flush=True)
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}: {response.text}", flush=True)
                return None
            
            data = response.json()
            print(f"📊 Response data keys: {list(data.keys())}", flush=True)
            
            # Handle both 'OK' and 'DELAYED' status (both are valid for free tier)
            if data.get('status') in ['OK', 'DELAYED'] and 'results' in data and len(data['results']) > 0:
                print(f"✅ Successfully fetched previous close for {symbol}", flush=True)
                result = data['results'][0]
                print(f"💰 Price data: {result}", flush=True)
                return result
            else:
                print(f"❌ No data in response for {symbol}: {data}", flush=True)
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching previous close for {symbol}: {e}", flush=True)
            return None
        except Exception as e:
            print(f"❌ Unexpected error fetching previous close for {symbol}: {e}", flush=True)
            return None

    @rate_limit(calls_per_minute=5)
    def get_historical_data(self, symbol: str, days: int = 90) -> Optional[List[Dict]]:
        """Get historical aggregated data - FIXED VERSION"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start_date}/{end_date}"
            
            # Fixed: Use proper parameter format
            params = {
                'apikey': self.api_key,
                'adjusted': 'true'
            }
            
            print(f"🔍 Fetching {days} days of historical data for {symbol}...", flush=True)
            print(f"🌐 URL: {url}", flush=True)
            
            response = self.session.get(url, params=params, timeout=30)
            
            print(f"📊 Response status: {response.status_code}", flush=True)
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}: {response.text}", flush=True)
                return None
            
            data = response.json()
            
            # Handle both 'OK' and 'DELAYED' status (both are valid)
            if data.get('status') in ['OK', 'DELAYED'] and 'results' in data:
                results = data['results']
                print(f"✅ Successfully fetched {len(results)} days of historical data for {symbol}", flush=True)
                return results
            else:
                print(f"❌ No historical data for {symbol}: {data}", flush=True)
                return None
                
        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}", flush=True)
            return None

# Initialize Polygon API
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if POLYGON_API_KEY:
    polygon_api = PolygonAPI(POLYGON_API_KEY)
    print("✅ Polygon.io API initialized", flush=True)
else:
    polygon_api = None
    print("⚠️ Polygon.io API key not found", flush=True)

class StockAnalysisView(discord.ui.View):
    """Interactive view for stock analysis - FIXED VERSION"""
    
    def __init__(self, symbol: str, analysis_data: dict):
        super().__init__(timeout=300)  # Reduced timeout to 5 minutes
        self.symbol = symbol.upper()
        self.analysis_data = analysis_data
        print(f"🎮 Created view for {self.symbol}", flush=True)
        
    async def on_timeout(self):
        """Called when the view times out"""
        print(f"⏰ View timed out for {self.symbol}", flush=True)
        # Disable all buttons
        for item in self.children:
            item.disabled = True
        
    @discord.ui.button(label="📊 Refresh Analysis", style=discord.ButtonStyle.primary, emoji="🔄")
    async def refresh_analysis(self, interaction: discord.Interaction, button: discord.ui.Button):
        print(f"🔄 Refresh button clicked by {interaction.user} for {self.symbol}", flush=True)
        await interaction.response.defer()
        
        try:
            new_analysis = await perform_stock_analysis(self.symbol)
            if new_analysis:
                embed = create_analysis_embed(self.symbol, new_analysis)
                analysis_cache[f"{interaction.user.id}_{self.symbol}"] = new_analysis
                
                new_view = StockAnalysisView(self.symbol, new_analysis)
                await interaction.edit_original_response(embed=embed, view=new_view)
                print(f"✅ Successfully refreshed analysis for {self.symbol}", flush=True)
            else:
                await interaction.followup.send("❌ Failed to refresh analysis", ephemeral=True)
        except Exception as e:
            print(f"❌ Error in refresh: {e}", flush=True)
            await interaction.followup.send(f"❌ Error refreshing: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="📈 Detailed Signals", style=discord.ButtonStyle.secondary, emoji="🎯")
    async def detailed_signals(self, interaction: discord.Interaction, button: discord.ui.Button):
        print(f"🎯 Detailed signals button clicked by {interaction.user} for {self.symbol}", flush=True)
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = create_detailed_signals_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
            print(f"✅ Sent detailed signals for {self.symbol}", flush=True)
        except Exception as e:
            print(f"❌ Error in detailed signals: {e}", flush=True)
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="📊 Price Levels", style=discord.ButtonStyle.success, emoji="💰")
    async def price_levels(self, interaction: discord.Interaction, button: discord.ui.Button):
        print(f"💰 Price levels button clicked by {interaction.user} for {self.symbol}", flush=True)
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = create_price_levels_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
            print(f"✅ Sent price levels for {self.symbol}", flush=True)
        except Exception as e:
            print(f"❌ Error in price levels: {e}", flush=True)
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="📰 News Impact", style=discord.ButtonStyle.success, emoji="📊")
    async def news_impact(self, interaction: discord.Interaction, button: discord.ui.Button):
        print(f"📰 News impact button clicked by {interaction.user} for {self.symbol}", flush=True)
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = await create_news_analysis_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
            print(f"✅ Sent news analysis for {self.symbol}", flush=True)
        except Exception as e:
            print(f"❌ Error in news impact: {e}", flush=True)
            await interaction.followup.send(f"❌ Error fetching news: {str(e)}", ephemeral=True)

async def test_polygon_connection():
    """Test if Polygon.io API is working"""
    if not polygon_api:
        print("❌ No Polygon.io API key configured", flush=True)
        return False
    
    try:
        print("🔍 Testing Polygon.io connection...", flush=True)
        
        test_data = await asyncio.get_event_loop().run_in_executor(
            None, polygon_api.get_previous_close, 'AAPL'
        )
        
        if test_data and 'c' in test_data:
            print("✅ Polygon.io connection successful", flush=True)
            return True
        else:
            print("❌ Polygon.io connection failed - no data returned", flush=True)
            return False
            
    except Exception as e:
        print(f"❌ Polygon.io test failed: {e}", flush=True)
        return False

def parse_polygon_data(symbol: str, prev_close_data: Dict, historical_data: List[Dict], current_quote_data: Optional[Dict] = None) -> Optional[Dict]:
    """Parse Polygon.io data into analysis format with real-time pricing"""
    try:
        if not prev_close_data or not historical_data:
            print(f"❌ Missing data for {symbol}", flush=True)
            return None
        
        # Convert historical data to DataFrame
        df_data = []
        for day in historical_data:
            try:
                date = pd.to_datetime(day['t'], unit='ms')
                df_data.append({
                    'Date': date,
                    'Open': float(day['o']),
                    'High': float(day['h']),
                    'Low': float(day['l']),
                    'Close': float(day['c']),
                    'Volume': int(day['v'])
                })
            except (ValueError, KeyError, TypeError):
                continue
        
        if not df_data:
            print(f"❌ No valid historical data for {symbol}", flush=True)
            return None
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Date').reset_index(drop=True)
        df.set_index('Date', inplace=True)
        
        # Parse current quote data if available
        current_price = float(prev_close_data['c'])  # Fallback to previous close
        current_price_label = "Previous Close"
        current_change = 0
        current_change_pct = 0
        
        # Try to get real-time price from current quote
        if current_quote_data and 'value' in current_quote_data:
            try:
                real_time_price = float(current_quote_data['value'])
                if real_time_price > 0:
                    current_price = real_time_price
                    current_price_label = "Current Price"
                    # Calculate change from previous close
                    prev_close_price = float(prev_close_data['c'])
                    current_change = current_price - prev_close_price
                    current_change_pct = (current_change / prev_close_price * 100) if prev_close_price > 0 else 0
                    print(f"✅ Using real-time price: ${current_price:.2f}", flush=True)
            except (ValueError, TypeError) as e:
                print(f"⚠️ Could not parse real-time price, using previous close: {e}", flush=True)
        
        # If no real-time data, fall back to previous close change
        if current_change == 0:
            prev_price = float(prev_close_data['o'])
            current_change = current_price - prev_price
            current_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        current_data = {
            'current_price': current_price,
            'current_price_label': current_price_label,
            'change': current_change,
            'change_percent': current_change_pct,
            'volume': int(prev_close_data.get('v', 0)),
            'latest_trading_day': pd.to_datetime(prev_close_data['t'], unit='ms'),
            'previous_close': float(prev_close_data['c']),
            'current_quote_data': current_quote_data
        }
        
        return {
            'dataframe': df,
            'current_data': current_data,
            'data_source': 'Polygon.io',
            'last_updated': datetime.now()
        }
        
    except Exception as e:
        print(f"❌ Error parsing Polygon.io data for {symbol}: {e}", flush=True)
        return None

async def perform_stock_analysis(symbol: str) -> Optional[Dict]:
    """Enhanced stock analysis using Polygon.io API with real-time pricing"""
    if not polygon_api:
        return None
    
    try:
        print(f"🔍 Analyzing {symbol} with Polygon.io...", flush=True)
        
        # Fetch both current quote and historical data from Polygon.io
        current_quote_task = asyncio.get_event_loop().run_in_executor(
            None, polygon_api.get_current_quote, symbol
        )
        prev_close_task = asyncio.get_event_loop().run_in_executor(
            None, polygon_api.get_previous_close, symbol
        )
        historical_task = asyncio.get_event_loop().run_in_executor(
            None, polygon_api.get_historical_data, symbol, 90
        )
        
        current_quote_data, prev_close_data, historical_data = await asyncio.gather(
            current_quote_task, prev_close_task, historical_task
        )
        
        if not prev_close_data:
            print(f"❌ No quote data received for {symbol}", flush=True)
            return None
        
        parsed_data = parse_polygon_data(symbol, prev_close_data, historical_data or [], current_quote_data)
        if not parsed_data:
            return None
        
        df = parsed_data['dataframe']
        current_data = parsed_data['current_data']
        
        if len(df) < 5:
            print(f"❌ Insufficient data for {symbol}: only {len(df)} days", flush=True)
            return None
        
        # Get basic metrics
        current_price = current_data['current_price']
        current_price_label = current_data['current_price_label']
        change_pct = current_data['change_percent']
        current_volume = current_data['volume']
        latest_trading_day = current_data['latest_trading_day']
        previous_close = current_data['previous_close']
        
        # Check if data is recent
        days_old = (datetime.now() - latest_trading_day.replace(tzinfo=None)).days
        is_recent = days_old <= 3
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
            'current_price_label': current_price_label,
            'previous_close': previous_close,
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
            'data_source': 'Polygon.io API',
            'api_reliability': 'Professional'
        }
        
    except Exception as e:
        print(f"❌ Analysis error for {symbol}: {e}", flush=True)
        return None

def create_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create enhanced analysis embed with real-time pricing"""
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
    
    market_status = analysis_data.get('market_status', 'Recent')
    is_recent = analysis_data.get('is_recent_data', True)
    status_emoji = "🟢" if is_recent else "🟡"
    
    # Create price display
    price_label = analysis_data.get('current_price_label', 'Current Price')
    price_display = f"**{price_label}:** ${analysis_data['current_price']:.2f} ({analysis_data['change_pct']:+.1f}%)"
    
    # Add previous close if we have real-time data
    if price_label == "Current Price" and 'previous_close' in analysis_data:
        price_display += f"\n**Previous Close:** ${analysis_data['previous_close']:.2f}"
    
    embed = discord.Embed(
        title=f"📊 Stock Analysis: {symbol}",
        description=f"{price_display}\n**Signal Score:** {score}/10.0\n**Sentiment:** {sentiment}\n{status_emoji} **Data Status:** {market_status}",
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
    data_text = f"**Source:** {analysis_data.get('data_source', 'Polygon.io API')}\n"
    data_text += f"**Reliability:** {analysis_data.get('api_reliability', 'Professional')}\n"
    data_text += f"**Data Quality:** {analysis_data.get('data_quality', 'Standard')}"
    
    embed.add_field(name="📊 Data Quality", value=data_text, inline=True)
    
    # Analysis quality
    confidence_text = f"**Confidence:** {confidence}\n"
    
    if analysis_data.get('confidence_factors'):
        factors = analysis_data['confidence_factors'][:2]
        confidence_text += f"**Key Factors:** {', '.join(factors)}"
    
    embed.add_field(name="🎯 Analysis Quality", value=confidence_text, inline=True)
    
    footer_text = f"🚀 Powered by Polygon.io API • {analysis_data.get('data_quality', 'Data available')} • Not financial advice"
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
    
    range_text = f"**Period High:** ${analysis_data['price_range_52w']['high']:.2f}\n"
    range_text += f"**Period Low:** ${analysis_data['price_range_52w']['low']:.2f}\n"
    range_text += f"**Position in Range:** {analysis_data['price_range_52w']['current_position']:.1f}%"
    
    embed.add_field(name="📊 Trading Ranges", value=range_text, inline=False)
    
    return embed

async def fetch_stock_news(symbol: str) -> List[Dict]:
    """Simple news fetching using Yahoo Finance RSS"""
    news_items = []
    
    try:
        print(f"🔍 Fetching news for {symbol}...", flush=True)
        
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
                        
                    except Exception as e:
                        print(f"⚠️ Error parsing news entry: {e}", flush=True)
                        continue
            
        except Exception as e:
            print(f"❌ Error fetching Yahoo Finance news: {e}", flush=True)
    
    except Exception as e:
        print(f"❌ Error in news fetching: {e}", flush=True)
    
    return news_items

async def create_news_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create news analysis embed"""
    embed = discord.Embed(
        title=f"📰 Recent News: {symbol}",
        description="Latest news and market sentiment",
        color=0x1f8b4c,
        timestamp=datetime.now()
    )
    
    try:
        news_items = await fetch_stock_news(symbol)
        
        if news_items:
            for i, item in enumerate(news_items[:3]):
                age_str = f"{item['age_hours']:.1f}h ago" if item['age_hours'] < 24 else f"{item['age_hours']/24:.1f}d ago"
                
                news_text = f"**{item['title']}**\n"
                if item['summary']:
                    news_text += f"{item['summary']}\n"
                news_text += f"*{item['source']} - {age_str}*"
                
                embed.add_field(
                    name=f"📄 Article {i+1}",
                    value=news_text,
                    inline=False
                )
        else:
            embed.add_field(
                name="📭 No Recent News",
                value="No recent news articles found for this symbol",
                inline=False
            )
    
    except Exception as e:
        embed.add_field(
            name="❌ News Unavailable",
            value=f"Could not fetch news: {str(e)}",
            inline=False
        )
    
    return embed

# Health check server for Render
class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health checks"""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        
        # More detailed health status
        status = f"Bot Status: {'Online' if bot.is_ready() else 'Starting'}\n"
        status += f"Guilds: {len(bot.guilds) if bot.is_ready() else 'Not ready'}\n"
        status += f"API: {'Available' if polygon_api else 'Not configured'}\n"
        status += f"Time: {datetime.now().isoformat()}"
        
        self.wfile.write(status.encode())
    
    def log_message(self, format, *args):
        # Suppress HTTP logs
        pass

def run_health_server():
    """Run health check server in background"""
    port = int(os.getenv('PORT', 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"✅ Health check server starting on port {port}", flush=True)
    try:
        server.serve_forever()
    except Exception as e:
        print(f"❌ Health server error: {e}", flush=True)

# Bot events and commands
@bot.event
async def on_ready():
    """Bot startup event"""
    print("=" * 60, flush=True)
    print(f"✅ Bot successfully logged in!", flush=True)
    print(f"🤖 Bot Name: {bot.user}", flush=True)
    print(f"📊 Bot ID: {bot.user.id}", flush=True)
    print(f"🌐 Connected to {len(bot.guilds)} guilds", flush=True)
    
    if bot.guilds:
        print("🏠 Guilds:", flush=True)
        for guild in bot.guilds:
            print(f"   - {guild.name} (ID: {guild.id})", flush=True)
    
    print("=" * 60, flush=True)
    
    # Test Polygon.io connection
    if polygon_api:
        print("🔍 Testing Polygon.io API...", flush=True)
        try:
            test_data = await asyncio.get_event_loop().run_in_executor(
                None, polygon_api.get_previous_close, 'AAPL'
            )
            if test_data and 'c' in test_data:
                print("✅ Polygon.io API is working correctly", flush=True)
            else:
                print("❌ Polygon.io API test failed - no data returned", flush=True)
        except Exception as e:
            print(f"❌ Polygon.io test failed: {e}", flush=True)
    else:
        print("⚠️ Polygon.io API not configured", flush=True)
    
    print("=" * 60, flush=True)
    print("🚀 Bot is ready to analyze stocks!", flush=True)
    print("💡 Try these commands:", flush=True)
    print("   !help - Show help", flush=True)
    print("   !stock AAPL - Analyze Apple", flush=True)
    print("   !ping - Test bot response", flush=True)
    print("=" * 60, flush=True)

@bot.event
async def on_guild_join(guild):
    """When bot joins a new guild"""
    print(f"📥 Joined guild: {guild.name} (ID: {guild.id})", flush=True)

@bot.event
async def on_message(message):
    """Process all messages - SIMPLE VERSION"""
    # Ignore bot messages
    if message.author.bot:
        return
    
    # Log message for debugging
    try:
        print(f"💬 Message from {message.author} in {message.guild.name if message.guild else 'DM'}", flush=True)
    except:
        print(f"💬 Message from {message.author}", flush=True)
    
    # Process commands
    await bot.process_commands(message)

@bot.command(name='ping')
async def ping_command(ctx):
    """Simple ping test"""
    print(f"🏓 Ping command received from {ctx.author}", flush=True)
    latency = round(bot.latency * 1000)
    await ctx.send(f"🏓 Pong! Latency: {latency}ms")

@bot.command(name='test')
async def test_command(ctx):
    """Test command"""
    print(f"🧪 Test command received from {ctx.author}", flush=True)
    embed = discord.Embed(
        title="🧪 Bot Test",
        description="Bot is working correctly!",
        color=0x00ff00
    )
    embed.add_field(name="Status", value="✅ Online", inline=True)
    embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="Guilds", value=len(bot.guilds), inline=True)
    
    await ctx.send(embed=embed)

# Global set to track recent command executions
recent_commands = {}

@bot.command(name='stock', aliases=['s', 'analyze'])
async def stock_command(ctx, symbol: str = None):
    """Main stock analysis command - SIMPLIFIED VERSION"""
    print(f"📊 Stock command received from {ctx.author} for symbol: {symbol}", flush=True)
    
    if not symbol:
        await ctx.send("❌ Please provide a stock symbol! Example: `!stock AAPL`")
        return
    
    # Check if API is configured
    if not polygon_api:
        await ctx.send("❌ Stock analysis is not available - Polygon.io API key not configured")
        return
    
    # Simple duplicate prevention using user ID and symbol
    user_symbol_key = f"{ctx.author.id}_{symbol.upper()}"
    current_time = datetime.now()
    
    # Check if same user requested same symbol within last 30 seconds
    if user_symbol_key in recent_commands:
        last_time = recent_commands[user_symbol_key]
        if (current_time - last_time).total_seconds() < 30:
            print(f"⚠️ Duplicate request blocked for {symbol} by {ctx.author}", flush=True)
            await ctx.send(f"⚠️ Please wait a moment before analyzing {symbol.upper()} again.")
            return
    
    # Update recent commands
    recent_commands[user_symbol_key] = current_time
    
    # Clean up old entries (keep only last 50)
    if len(recent_commands) > 50:
        # Remove entries older than 5 minutes
        cutoff = current_time - timedelta(minutes=5)
        recent_commands.clear()  # Simple cleanup
    
    # Send initial message
    loading_msg = await ctx.send(f"🔍 Analyzing **{symbol.upper()}**... Please wait...")
    
    try:
        # Perform full analysis
        print(f"🔍 Starting full analysis for {symbol}", flush=True)
        
        analysis_data = await perform_stock_analysis(symbol)
        
        if analysis_data:
            # Create embed and view
            embed = create_analysis_embed(symbol, analysis_data)
            view = StockAnalysisView(symbol, analysis_data)
            
            # Store in cache with timestamp
            analysis_cache[user_symbol_key] = analysis_data
            
            # Edit message with results
            await loading_msg.edit(content=None, embed=embed, view=view)
            print(f"✅ Analysis complete for {symbol}, sent to {ctx.author}", flush=True)
        else:
            await loading_msg.edit(
                content=f"❌ Could not analyze **{symbol.upper()}**. Please check:\n"
                       f"• Is this a valid US stock symbol?\n"
                       f"• Is the market open or was it open recently?\n"
                       f"• Try again in a moment."
            )
    
    except Exception as e:
        print(f"❌ Error in stock command: {e}", flush=True)
        await loading_msg.edit(
            content=f"❌ An error occurred while analyzing {symbol}: {str(e)}"
        )

@bot.command(name='help')
async def help_command(ctx):
    """Help command"""
    print(f"❓ Help command received from {ctx.author}", flush=True)
    
    embed = discord.Embed(
        title="📊 Stock Analysis Bot Help",
        description="Professional stock analysis powered by Polygon.io",
        color=0x7289DA
    )
    
    embed.add_field(
        name="🧪 Test Commands",
        value=(
            "`!ping` - Test bot response\n"
            "`!test` - Show bot status\n"
            "`!help` - Show this help message"
        ),
        inline=False
    )
    
    embed.add_field(
        name="📈 Stock Commands",
        value=(
            "`!stock <symbol>` - Analyze a stock (aliases: `!s`, `!analyze`)\n"
        ),
        inline=False
    )
    
    embed.add_field(
        name="🎯 Examples",
        value=(
            "`!ping` - Test the bot\n"
            "`!stock AAPL` - Analyze Apple Inc.\n"
            "`!s MSFT` - Analyze Microsoft"
        ),
        inline=False
    )
    
    embed.set_footer(text="Powered by Polygon.io • Not financial advice")
    
    await ctx.send(embed=embed)

# Error handler
@bot.event
async def on_command_error(ctx, error):
    """Global error handler - FIXED VERSION"""
    print(f"❌ Command error: {error}", flush=True)
    
    if isinstance(error, commands.CommandNotFound):
        # Don't respond to unknown commands to prevent spam
        return
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ Missing required argument. Use `!help` for command usage.")
    else:
        await ctx.send(f"❌ An error occurred: {str(error)}")

# Add interaction error handler
@bot.event
async def on_interaction(interaction):
    """Debug all interactions"""
    print(f"🎮 Interaction received: {interaction.type} from {interaction.user}", flush=True)

@bot.event  
async def on_error(event, *args, **kwargs):
    """Global error handler"""
    import traceback
    print(f"❌ Bot error in {event}:", flush=True)
    print(f"Args: {args}", flush=True)
    print(f"Traceback: {traceback.format_exc()}", flush=True)

# Graceful shutdown
def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutdown signal received...", flush=True)
    asyncio.create_task(bot.close())
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main execution
if __name__ == "__main__":
    # Force unbuffered output for Render
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("🚀 Starting bot initialization...", flush=True)
    
    # Start health check server
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    print("✅ Health check server thread started", flush=True)
    
    # Give health server a moment to start
    time.sleep(2)
    
    # Get Discord token
    TOKEN = os.getenv('DISCORD_TOKEN')
    
    if not TOKEN:
        print("❌ ERROR: DISCORD_TOKEN not found in environment variables!", flush=True)
        print("Please set your Discord bot token in Render environment variables", flush=True)
        print(f"Environment variables found: {list(os.environ.keys())}", flush=True)
        sys.exit(1)
    
    if not POLYGON_API_KEY:
        print("⚠️ WARNING: POLYGON_API_KEY not found - stock analysis will be limited!", flush=True)
    
    print("🔑 Discord token found, attempting to login...", flush=True)
    
    # Run the bot
    try:
        bot.run(TOKEN, log_handler=None)  # Disable discord.py's default logging
    except discord.LoginFailure:
        print("❌ ERROR: Invalid Discord token!", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}", flush=True)
        sys.exit(1)