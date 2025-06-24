#!/usr/bin/env python3
"""
Enhanced Stock Analysis Bot with Polygon.io API - CLEAN WORKING VERSION
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

print("📊 Starting Enhanced Stock Analysis Bot with Polygon.io...")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

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
    """Polygon.io API wrapper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'StockAnalysisBot/1.0'
        })
    
    @rate_limit(calls_per_minute=5)
    def get_previous_close(self, symbol: str) -> Optional[Dict]:
        """Get previous trading day's data"""
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/prev"
            params = {'apikey': self.api_key, 'adjusted': 'true'}
            
            print(f"🔍 Fetching previous close for {symbol}...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data and len(data['results']) > 0:
                print(f"✅ Successfully fetched previous close for {symbol}")
                return data['results'][0]
            else:
                print(f"❌ No previous close data for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching previous close for {symbol}: {e}")
            return None
    
    @rate_limit(calls_per_minute=5)
    def get_historical_data(self, symbol: str, days: int = 90) -> Optional[List[Dict]]:
        """Get historical aggregated data"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start_date}/{end_date}"
            params = {'apikey': self.api_key, 'adjusted': 'true'}
            
            print(f"🔍 Fetching {days} days of historical data for {symbol}...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                results = data['results']
                print(f"✅ Successfully fetched {len(results)} days of historical data for {symbol}")
                return results
            else:
                print(f"❌ No historical data for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}")
            return None

# Initialize Polygon API
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if POLYGON_API_KEY:
    polygon_api = PolygonAPI(POLYGON_API_KEY)
    print("✅ Polygon.io API initialized")
else:
    polygon_api = None
    print("⚠️ Polygon.io API key not found")

class StockAnalysisView(discord.ui.View):
    """Interactive view for stock analysis"""
    
    def __init__(self, symbol: str, analysis_data: dict):
        super().__init__(timeout=600)
        self.symbol = symbol.upper()
        self.analysis_data = analysis_data
        
    @discord.ui.button(label="📊 Refresh Analysis", style=discord.ButtonStyle.primary, emoji="🔄")
    async def refresh_analysis(self, interaction: discord.Interaction, button: discord.ui.Button):
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
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = create_detailed_signals_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="📊 Price Levels", style=discord.ButtonStyle.success, emoji="💰")
    async def price_levels(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = create_price_levels_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)
    
    @discord.ui.button(label="📰 News Impact", style=discord.ButtonStyle.success, emoji="📊")
    async def news_impact(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        
        try:
            embed = await create_news_analysis_embed(self.symbol, self.analysis_data)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error fetching news: {str(e)}", ephemeral=True)

async def test_polygon_connection():
    """Test if Polygon.io API is working"""
    if not polygon_api:
        print("❌ No Polygon.io API key configured")
        return False
    
    try:
        print("🔍 Testing Polygon.io connection...")
        
        test_data = await asyncio.get_event_loop().run_in_executor(
            None, polygon_api.get_previous_close, 'AAPL'
        )
        
        if test_data and 'c' in test_data:
            print("✅ Polygon.io connection successful")
            return True
        else:
            print("❌ Polygon.io connection failed - no data returned")
            return False
            
    except Exception as e:
        print(f"❌ Polygon.io test failed: {e}")
        return False

def parse_polygon_data(symbol: str, prev_close_data: Dict, historical_data: List[Dict]) -> Optional[Dict]:
    """Parse Polygon.io data into analysis format"""
    try:
        if not prev_close_data or not historical_data:
            print(f"❌ Missing data for {symbol}")
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
            print(f"❌ No valid historical data for {symbol}")
            return None
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Date').reset_index(drop=True)
        df.set_index('Date', inplace=True)
        
        # Parse current quote data
        current_price = float(prev_close_data['c'])
        prev_price = float(prev_close_data['o'])
        
        current_data = {
            'current_price': current_price,
            'change': current_price - prev_price,
            'change_percent': ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0,
            'volume': int(prev_close_data.get('v', 0)),
            'latest_trading_day': pd.to_datetime(prev_close_data['t'], unit='ms')
        }
        
        return {
            'dataframe': df,
            'current_data': current_data,
            'data_source': 'Polygon.io',
            'last_updated': datetime.now()
        }
        
    except Exception as e:
        print(f"❌ Error parsing Polygon.io data for {symbol}: {e}")
        return None

async def perform_stock_analysis(symbol: str) -> Optional[Dict]:
    """Enhanced stock analysis using Polygon.io API"""
    if not polygon_api:
        return None
    
    try:
        print(f"🔍 Analyzing {symbol} with Polygon.io...")
        
        # Fetch data from Polygon.io
        prev_close_task = asyncio.get_event_loop().run_in_executor(
            None, polygon_api.get_previous_close, symbol
        )
        historical_task = asyncio.get_event_loop().run_in_executor(
            None, polygon_api.get_historical_data, symbol, 90
        )
        
        prev_close_data, historical_data = await asyncio.gather(prev_close_task, historical_task)
        
        if not prev_close_data:
            print(f"❌ No quote data received for {symbol}")
            return None
        
        parsed_data = parse_polygon_data(symbol, prev_close_data, historical_data or [])
        if not parsed_data:
            return None
        
        df = parsed_data['dataframe']
        current_data = parsed_data['current_data']
        
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
        print(f"❌ Analysis error for {symbol}: {e}")
        return None

def create_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create enhanced analysis embed with Polygon.io data"""
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
                        