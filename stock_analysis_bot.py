#!/usr/bin/env python3
"""
Enhanced Stock Analysis Bot with News Analysis - FIXED VERSION
Improved data fetching and error handling
"""

import discord
from discord.ext import commands
import yfinance as yf
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
import re
from urllib.parse import quote
import ssl
import certifi

# Load environment variables
load_dotenv()

print("📊 Starting Enhanced Stock Analysis Bot...")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Global storage for analysis results
analysis_cache = {}

# Fix SSL issues for data fetching
ssl_context = ssl.create_default_context(cafile=certifi.where())

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

async def test_stock_data_connection():
    """Test if we can fetch stock data"""
    try:
        print("🔍 Testing stock data connection...")
        
        # Test with reliable stocks that should always work
        test_symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'GOOGL']
        
        for symbol in test_symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Use shorter timeout and simpler request
                hist = ticker.history(period="2d", timeout=10)
                
                if not hist.empty and len(hist) > 0:
                    print(f"✅ Successfully fetched data for {symbol}")
                    return True
                else:
                    print(f"⚠️ Empty data for {symbol}")
                    
            except Exception as e:
                print(f"❌ Failed to fetch {symbol}: {e}")
                continue
        
        print("❌ All test symbols failed")
        return False
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

async def perform_stock_analysis(symbol: str) -> Optional[Dict]:
    """Enhanced stock analysis that works when market is closed"""
    try:
        print(f"🔍 Analyzing {symbol}...")
        
        # Try multiple data periods to ensure we get data even when market is closed
        ticker = yf.Ticker(symbol.upper())
        
        # Try different data periods in order of preference
        data_periods = ["90d", "30d", "1y", "6mo", "3mo", "1mo", "5d"]
        hist = pd.DataFrame()
        
        for period in data_periods:
            try:
                print(f"  Trying {period} data for {symbol}...")
                hist = ticker.history(period=period, timeout=30)
                if not hist.empty and len(hist) >= 5:  # Need at least 5 days for analysis
                    print(f"  ✅ Got {len(hist)} days from {period} period")
                    break
                else:
                    print(f"  ⚠️ Only got {len(hist)} days from {period}")
            except Exception as e:
                print(f"  ❌ {period} failed: {e}")
                continue
        
        if hist.empty:
            print(f"❌ No data found for {symbol} in any period")
            return None
        
        print(f"✅ Successfully got {len(hist)} days of data for {symbol}")
        
        # Get the most recent trading day data (not necessarily today)
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
        
        # Get market status info
        last_trading_day = hist.index[-1].strftime('%Y-%m-%d')
        is_recent = (datetime.now() - hist.index[-1]).days <= 1
        market_status = "Recent" if is_recent else f"Last trading: {last_trading_day}"
        
        # Calculate technical indicators with flexible periods
        closes = hist['Close']
        highs = hist['High']
        lows = hist['Low']
        volumes = hist['Volume']
        data_length = len(closes)
        
        # RSI calculation (adaptive period)
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
        
        # Moving averages (adaptive periods)
        try:
            sma_20_period = min(20, max(5, data_length // 2))
            sma_50_period = min(50, max(10, data_length))
            ema_12_period = min(12, max(5, data_length // 3))
            
            sma_20 = closes.rolling(sma_20_period).mean().iloc[-1] if data_length >= 5 else current_price
            sma_50 = closes.rolling(sma_50_period).mean().iloc[-1] if data_length >= 10 else current_price
            ema_12 = closes.ewm(span=ema_12_period).mean().iloc[-1] if data_length >= 5 else current_price
        except:
            sma_20 = sma_50 = ema_12 = current_price
        
        # Bollinger Bands (adaptive)
        try:
            bb_period = min(20, max(10, data_length // 2))
            bb_middle = closes.rolling(bb_period).mean().iloc[-1] if data_length >= 10 else current_price
            bb_std_dev = closes.rolling(bb_period).std().iloc[-1] if data_length >= 10 else current_price * 0.02
            bb_upper = bb_middle + (bb_std_dev * 2)
            bb_lower = bb_middle - (bb_std_dev * 2)
        except:
            bb_middle = current_price
            bb_upper = current_price * 1.04
            bb_lower = current_price * 0.96
        
        # MACD (adaptive)
        try:
            ema_12_period = min(12, max(5, data_length // 4))
            ema_26_period = min(26, max(10, data_length // 2))
            
            ema_12_series = closes.ewm(span=ema_12_period).mean()
            ema_26_series = closes.ewm(span=ema_26_period).mean()
            macd_line = ema_12_series - ema_26_series
            macd_signal = macd_line.ewm(span=9).mean()
            macd_current = macd_line.iloc[-1] if not macd_line.empty else 0
            macd_signal_current = macd_signal.iloc[-1] if not macd_signal.empty else 0
        except:
            macd_current = macd_signal_current = 0
        
        # Volume analysis
        try:
            avg_volume = volumes.mean() if not volumes.empty else 1000000
            current_volume = volumes.iloc[-1] if not volumes.empty else 1000000
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        except:
            avg_volume = current_volume = 1000000
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
        
        # Volatility calculation
        try:
            volatility_period = min(14, max(5, data_length // 3))
            high_low = highs - lows
            volatility = high_low.rolling(volatility_period).mean().iloc[-1] if data_length >= 5 else current_price * 0.02
        except:
            volatility = current_price * 0.02
        
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
        
        # MACD signals
        if abs(macd_current) > 0.01:  # Only if MACD is meaningful
            if macd_current > macd_signal_current and macd_current > 0:
                score += 1.0
                signals.append("🟢 MACD Bullish Crossover")
            elif macd_current < macd_signal_current and macd_current < 0:
                score -= 1.0
                signals.append("🔴 MACD Bearish Crossover")
        
        # Bollinger Bands signals
        if current_price < bb_lower:
            score += 1.5
            signals.append("🟢 Below Lower Bollinger Band - Oversold")
        elif current_price > bb_upper:
            score -= 1.5
            signals.append("🔴 Above Upper Bollinger Band - Overbought")
        
        # Volume confirmation (only if volume is meaningful)
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
        
        # Support/Resistance proximity
        distance_to_resistance = (resistance - current_price) / current_price
        distance_to_support = (current_price - support) / current_price
        
        if distance_to_resistance < 0.02:  # Within 2% of resistance
            score -= 0.8
            signals.append(f"🔴 Near Resistance Level (${resistance:.2f})")
        elif distance_to_support < 0.02:  # Within 2% of support
            score += 0.8
            signals.append(f"🟢 Near Support Level (${support:.2f})")
        
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
            price_range_52w = {
                'high': highs.max(),
                'low': lows.min(),
                'current_position': (current_price - lows.min()) / (highs.max() - lows.min()) * 100 if highs.max() > lows.min() else 50
            }
        except:
            price_range_52w = {
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
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'macd': macd_current,
            'macd_signal': macd_signal_current,
            'volume_ratio': volume_ratio,
            'avg_volume': avg_volume,
            'current_volume': current_volume,
            'support': support,
            'resistance': resistance,
            'volatility': volatility,
            'score': score,
            'sentiment': sentiment,
            'confidence': confidence,
            'signals': signals,
            'confidence_factors': confidence_factors,
            'price_range_52w': price_range_52w,
            'timestamp': datetime.now(),
            'data_quality': f"{len(hist)} days of data",
            'market_status': market_status,
            'last_trading_day': last_trading_day,
            'is_recent_data': is_recent
        }
        
    except Exception as e:
        print(f"❌ Analysis error for {symbol}: {e}")
        return None

def create_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create enhanced analysis embed that works when market is closed"""
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
    
    # Volume and momentum
    volume_text = f"**Current:** {analysis_data['current_volume']:,.0f}\n"
    volume_text += f"**Average:** {analysis_data['avg_volume']:,.0f}\n"
    volume_text += f"**Ratio:** {analysis_data['volume_ratio']:.1f}x"
    
    # Add Bollinger Bands if available
    if 'bb_upper' in analysis_data:
        volume_text += f"\n**BB Upper:** ${analysis_data['bb_upper']:.2f}"
    
    embed.add_field(name="📊 Volume & Bands", value=volume_text, inline=True)
    
    # Key levels
    levels_text = f"**Resistance:** ${analysis_data['resistance']:.2f}\n"
    levels_text += f"**Support:** ${analysis_data['support']:.2f}\n"
    levels_text += f"**Range Position:** {analysis_data['price_range_52w']['current_position']:.1f}%"
    
    # Add MACD if meaningful
    if 'macd' in analysis_data and abs(analysis_data['macd']) > 0.01:
        levels_text += f"\n**MACD:** {analysis_data['macd']:.3f}"
    
    embed.add_field(name="🎯 Key Levels", value=levels_text, inline=True)
    
    # Top signals
    top_signals = analysis_data['signals'][:4]
    signals_text = "\n".join(top_signals) if top_signals else "No significant signals"
    
    embed.add_field(name="🚨 Key Signals", value=signals_text, inline=False)
    
    # Confidence and data quality
    confidence_text = f"**Analysis Confidence:** {confidence}\n"
    confidence_text += f"**Data Quality:** {analysis_data.get('data_quality', 'Standard')}\n"
    
    if analysis_data.get('confidence_factors'):
        factors = analysis_data['confidence_factors'][:2]  # Top 2 factors
        confidence_text += f"**Key Factors:** {', '.join(factors)}"
    
    embed.add_field(name="📊 Analysis Quality", value=confidence_text, inline=True)
    
    # 52-week positioning
    range_52w = analysis_data['price_range_52w']
    range_text = f"**52W High:** ${range_52w['high']:.2f}\n"
    range_text += f"**52W Low:** ${range_52w['low']:.2f}\n"
    range_text += f"**Current Position:** {range_52w['current_position']:.1f}% of range"
    
    embed.add_field(name="📅 52-Week Range", value=range_text, inline=True)
    
    # Market timing note
    if not is_recent:
        embed.add_field(
            name="⏰ Market Timing Note",
            value=f"Analysis based on last trading day: {analysis_data.get('last_trading_day', 'Recent')}\nTechnical analysis remains valid during market hours.",
            inline=False
        )
    
    # Footer with enhanced info
    footer_text = f"💡 Click buttons for detailed analysis • {analysis_data.get('data_quality', 'Data available')}"
    if not is_recent:
        footer_text += " • Market closed data"
    footer_text += " • Not financial advice"
    
    embed.set_footer(text=footer_text)
    
    return embedanalysis_data['current_volume']:,.0f}\n"
    volume_text += f"**Average:** {analysis_data['avg_volume']:,.0f}\n"
    volume_text += f"**Ratio:** {analysis_data['volume_ratio']:.1f}x"
    
    embed.add_field(name="📊 Volume", value=volume_text, inline=True)
    
    # Key levels
    levels_text = f"**Resistance:** ${analysis_data['resistance']:.2f}\n"
    levels_text += f"**Support:** ${analysis_data['support']:.2f}\n"
    levels_text += f"**Position:** {analysis_data['price_range_52w']['current_position']:.1f}%"
    
    embed.add_field(name="🎯 Key Levels", value=levels_text, inline=True)
    
    # Top signals
    top_signals = analysis_data['signals'][:4]
    signals_text = "\n".join(top_signals) if top_signals else "No significant signals"
    
    embed.add_field(name="🚨 Key Signals", value=signals_text, inline=False)
    
    # Data quality indicator
    embed.set_footer(text=f"💡 Click buttons for detailed analysis • {analysis_data.get('data_quality', 'Data available')} • Not financial advice")
    
    return embed

def create_detailed_signals_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create detailed signals analysis embed"""
    embed = discord.Embed(
        title=f"🎯 Detailed Signals: {symbol}",
        description=f"Comprehensive signal breakdown",
        color=0x7289DA,
        timestamp=datetime.now()
    )
    
    # All signals
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
    
    embed.set_footer(text="📊 Detailed technical signal analysis")
    
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
    
    # Key levels
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
    range_text = f"**52W High:** ${analysis_data['price_range_52w']['high']:.2f}\n"
    range_text += f"**52W Low:** ${analysis_data['price_range_52w']['low']:.2f}\n"
    range_text += f"**Position in Range:** {analysis_data['price_range_52w']['current_position']:.1f}%"
    
    embed.add_field(name="📊 Trading Ranges", value=range_text, inline=False)
    
    embed.set_footer(text="💡 Use these levels for entry/exit planning")
    
    return embed

async def fetch_stock_news(symbol: str, company_name: str = None) -> List[Dict]:
    """Fixed news fetching with better error handling"""
    news_items = []
    
    try:
        print(f"🔍 Fetching news for {symbol}...")
        
        # Only use Yahoo Finance RSS (most reliable)
        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        
        try:
            # Add timeout to prevent hanging
            import urllib.request
            from urllib.error import URLError
            
            # Simple request with timeout
            req = urllib.request.Request(yahoo_url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                feed_data = response.read()
            
            # Parse the feed
            yahoo_feed = feedparser.parse(feed_data)
            
            if hasattr(yahoo_feed, 'entries') and yahoo_feed.entries:
                for entry in yahoo_feed.entries[:5]:
                    try:
                        pub_date = datetime.now()  # Fallback date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        
                        title = getattr(entry, 'title', 'No title')
                        summary = getattr(entry, 'summary', '')
                        link = getattr(entry, 'link', '')
                        
                        # Truncate summary
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
            else:
                print(f"No news entries found for {symbol}")
                
        except Exception as e:
            print(f"Yahoo Finance news error for {symbol}: {e}")
        
        # Sort by publication date if we have items
        if news_items:
            news_items.sort(key=lambda x: x['published'], reverse=True)
        
        print(f"✅ Found {len(news_items)} news items for {symbol}")
        return news_items[:5]  # Return max 5 items
        
    except Exception as e:
        print(f"News fetch error for {symbol}: {e}")
        return []

def analyze_news_sentiment(news_items: List[Dict], symbol: str) -> Dict:
    """Simple news sentiment analysis"""
    if not news_items:
        return {
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'impact_level': 'Low',
            'key_themes': [],
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
    
    # Determine sentiment label
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
        'key_themes': ['earnings', 'analyst'] if sentiment_score != 0 else [],
        'recent_count': recent_count
    }

async def create_news_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create news analysis embed with better error handling"""
    
    # Create default embed first
    embed = discord.Embed(
        title=f"📰 News Impact Analysis: {symbol}",
        description=f"Recent news affecting **{symbol}** price movement",
        color=0xffd700,
        timestamp=datetime.now()
    )
    
    try:
        # Fetch news with error handling
        news_items = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_stock_news(symbol)
        )
        
        # Handle the case where fetch_stock_news returns a coroutine
        if hasattr(news_items, '__await__'):
            news_items = await news_items
        
        # Ensure we have a list
        if not isinstance(news_items, list):
            news_items = []
            
    except Exception as e:
        print(f"Error in news fetch: {e}")
        news_items = []
    
    # Analyze sentiment
    try:
        news_analysis = analyze_news_sentiment(news_items, symbol)
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        news_analysis = {
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'impact_level': 'Low',
            'key_themes': [],
            'recent_count': 0
        }
    
    # Update embed color based on sentiment
    if news_analysis['sentiment_label'] == 'Positive':
        embed.color = 0x00ff88
    elif news_analysis['sentiment_label'] == 'Negative':
        embed.color = 0xff6b6b
    else:
        embed.color = 0xffd700
    
    # News sentiment summary
    sentiment_emoji = {'Positive': '🟢', 'Neutral': '⚪', 'Negative': '🔴'}
    impact_emoji = {'High': '🔥', 'Medium': '⚡', 'Low': '📊'}
    
    summary_text = f"**Sentiment:** {sentiment_emoji.get(news_analysis['sentiment_label'], '⚪')} {news_analysis['sentiment_label']}\n"
    summary_text += f"**Impact Level:** {impact_emoji.get(news_analysis['impact_level'], '📊')} {news_analysis['impact_level']}\n"
    summary_text += f"**Recent News:** {news_analysis['recent_count']} stories (24h)"
    
    embed.add_field(name="📊 News Sentiment", value=summary_text, inline=True)
    
    # Price correlation (if we have price data)
    try:
        price_change = analysis_data.get('change_pct', 0)
        correlation_text = f"**Today's Move:** {price_change:+.1f}%\n"
        
        if news_analysis['sentiment_score'] > 0 and price_change > 1:
            correlation_text += "🟢 **Positive news supports upward move**"
        elif news_analysis['sentiment_score'] < 0 and price_change < -1:
            correlation_text += "🔴 **Negative news explains downward move**"
        else:
            correlation_text += "📊 **News and price movement are neutral**"
        
        embed.add_field(name="📈 Price Correlation", value=correlation_text, inline=True)
    except:
        embed.add_field(name="📈 Price Correlation", value="Price data unavailable", inline=True)
    
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
    
    embed.set_footer(text="📰 News analysis • Not financial advice")
    
    return embed

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print(f'\n📊 Received signal {signum}. Shutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@bot.event
async def on_ready():
    print(f'✅ {bot.user} is online!')
    print('📊 Enhanced Stock Analysis Bot ready!')
    
    # Test data connection on startup
    data_available = await test_stock_data_connection()
    if data_available:
        print('✅ Stock data connection verified')
    else:
        print('⚠️ Stock data connection issues detected')
    
    # Send startup message to all channels where bot has permissions
    await send_startup_message(data_available)

async def send_startup_message(data_available: bool):
    """Send startup message with available commands to appropriate channels"""
    try:
        startup_embed = discord.Embed(
            title="📊 Stock Analysis Bot is Online!",
            description="Enhanced stock analysis with technical indicators and news",
            color=0x00ff88 if data_available else 0xffd700,
            timestamp=datetime.now()
        )
        
        # Add status information
        data_status = "✅ Working" if data_available else "⚠️ Limited"
        startup_embed.add_field(
            name="🔧 System Status",
            value=f"**Data Feed:** {data_status}\n**Interactive Features:** ✅ Available\n**News Analysis:** ✅ Ready",
            inline=False
        )
        
        # Add available commands
        startup_embed.add_field(
            name="📈 Available Commands",
            value="`!analysis SYMBOL` - Complete stock analysis\n`!quick SYMBOL` - Fast overview\n`!news SYMBOL` - News impact analysis\n`!test` - Check system status\n`!info` - Show detailed help",
            inline=False
        )
        
        # Add usage examples
        startup_embed.add_field(
            name="🚀 Quick Examples",
            value="`!analysis AAPL` - Apple full analysis\n`!quick SPY` - S&P 500 quick view\n`!news TSLA` - Tesla news analysis",
            inline=False
        )
        
        # Add aliases
        startup_embed.add_field(
            name="⚡ Quick Aliases",
            value="`!a` = analysis • `!q` = quick • `!n` = news",
            inline=False
        )
        
        if not data_available:
            startup_embed.add_field(
                name="⚠️ Data Issues Detected",
                value="Some features may be limited. Use `!test` to check status or try again later.",
                inline=False
            )
        
        startup_embed.set_footer(text="💡 Type any command to get started • Not financial advice")
        
        # Send to channels where bot can post
        for guild in bot.guilds:
            try:
                # Look for general channels first
                target_channel = None
                
                # Priority order for channel selection
                channel_preferences = [
                    'general', 'bot-commands', 'bots', 'trading', 'stocks', 
                    'analysis', 'market', 'finance', 'commands'
                ]
                
                # Try to find a preferred channel
                for pref in channel_preferences:
                    channel = discord.utils.get(guild.text_channels, name=pref)
                    if channel and channel.permissions_for(guild.me).send_messages:
                        target_channel = channel
                        break
                
                # If no preferred channel found, use first available channel
                if not target_channel:
                    for channel in guild.text_channels:
                        if channel.permissions_for(guild.me).send_messages:
                            target_channel = channel
                            break
                
                # Send startup message
                if target_channel:
                    await target_channel.send(embed=startup_embed)
                    print(f"✅ Sent startup message to #{target_channel.name} in {guild.name}")
                else:
                    print(f"⚠️ No suitable channel found in {guild.name}")
                    
            except Exception as e:
                print(f"❌ Failed to send startup message to {guild.name}: {e}")
                continue
        
        print("📢 Startup messages sent to all available channels")
        
    except Exception as e:
        print(f"❌ Error sending startup messages: {e}")

# Add a new command to manually show the help
@bot.command(name='commands', aliases=['start'])
async def show_commands(ctx):
    """Show available commands (can be triggered manually)"""
    
    # Test current data status
    data_available = await test_stock_data_connection()
    
    embed = discord.Embed(
        title="📊 Stock Analysis Bot Commands",
        description="All available commands and features",
        color=0x00ff88 if data_available else 0xffd700,
        timestamp=datetime.now()
    )
    
    # System status
    data_status = "✅ Working" if data_available else "⚠️ Limited"
    embed.add_field(
        name="🔧 Current Status",
        value=f"**Data Feed:** {data_status}\n**Bot:** ✅ Online\n**Features:** ✅ Available",
        inline=False
    )
    
    # Main commands
    embed.add_field(
        name="📈 Analysis Commands",
        value="`!analysis SYMBOL` - Full technical analysis with interactive buttons\n`!quick SYMBOL` - Fast analysis overview\n`!news SYMBOL` - News impact and sentiment analysis",
        inline=False
    )
    
    # Utility commands
    embed.add_field(
        name="🔧 Utility Commands",
        value="`!test` - Check system status and data connection\n`!info` - Detailed help and feature list\n`!commands` - Show this command list",
        inline=False
    )
    
    # Examples with real symbols
    embed.add_field(
        name="💡 Usage Examples",
        value="`!analysis AAPL` - Apple comprehensive analysis\n`!quick SPY` - S&P 500 ETF quick view\n`!news NVDA` - NVIDIA news and sentiment\n`!test` - Verify bot is working properly",
        inline=False
    )
    
    # Shortcuts
    embed.add_field(
        name="⚡ Quick Shortcuts",
        value="`!a TSLA` = `!analysis TSLA`\n`!q MSFT` = `!quick MSFT`\n`!n GOOGL` = `!news GOOGL`",
        inline=False
    )
    
    # Interactive features
    embed.add_field(
        name="🎯 Interactive Features",
        value="After running `!analysis`, use the buttons for:\n• 📊 Refresh Analysis\n• 📈 Detailed Signals\n• 💰 Price Levels\n• 📰 News Impact",
        inline=False
    )
    
    if not data_available:
        embed.add_field(
            name="⚠️ Current Limitations",
            value="Data connection issues detected. Some features may be limited. Try `!test` for diagnostics.",
            inline=False
        )
    
    embed.set_footer(text="🚀 Ready to analyze stocks • Not financial advice • Do your research")
    
    await ctx.send(embed=embed)

@bot.command(name='test')
async def test_command(ctx):
    """Enhanced test command with data connection check"""
    
    # Test data fetching
    data_test = await test_stock_data_connection()
    data_status = "✅ Working" if data_test else "❌ Issues detected"
    
    embed = discord.Embed(
        title="🤖 Enhanced Stock Bot Status",
        description="System status check",
        color=0x00ff00 if data_test else 0xff6b6b
    )
    
    embed.add_field(
        name="📊 System Status",
        value=f"• **Discord Bot:** ✅ Online\n• **Data Feed:** {data_status}\n• **Interactive UI:** ✅ Available\n• **News Analysis:** ✅ Enhanced",
        inline=False
    )
    
    if data_test:
        embed.add_field(
            name="🚀 Ready to Use",
            value="`!analysis AAPL` - Full analysis\n`!quick SPY` - Quick view\n`!news TSLA` - News analysis",
            inline=False
        )
    else:
        embed.add_field(
            name="⚠️ Data Issues",
            value="Some features may be limited. Trying backup data sources.",
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name='analysis', aliases=['analyze', 'stock', 'a'])
async def analyze_stock(ctx, symbol: str = None):
    """Enhanced stock analysis that works 24/7"""
    
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
            name="⏰ Works 24/7",
            value="Analysis works even when markets are closed using latest trading data",
            inline=False
        )
        await ctx.send(embed=embed)
        return
    
    # Send analyzing message
    current_time = datetime.now()
    is_weekend = current_time.weekday() >= 5
    is_after_hours = current_time.hour < 9 or current_time.hour >= 16
    
    if is_weekend or is_after_hours:
        status_msg = "🔍 Analyzing **{}** (using latest trading data)...".format(symbol.upper())
    else:
        status_msg = "🔍 Performing live analysis on **{}**...".format(symbol.upper())
    
    msg = await ctx.send(status_msg)
    
    try:
        # Perform analysis with extended timeout for after-hours
        timeout_duration = 60.0 if (is_weekend or is_after_hours) else 45.0
        
        analysis_data = await asyncio.wait_for(
            perform_stock_analysis(symbol), 
            timeout=timeout_duration
        )
        
        if not analysis_data:
            error_embed = discord.Embed(
                title=f"❌ Analysis Failed",
                description=f"Unable to fetch data for **{symbol.upper()}**",
                color=0xff6b6b
            )
            error_embed.add_field(
                name="🔍 Possible Issues",
                value="• Invalid or delisted symbol\n• Symbol not found on major exchanges\n• Temporary data provider issues\n• Very new IPO (insufficient data)",
                inline=False
            )
            error_embed.add_field(
                name="💡 Try These Solutions",
                value="• Check symbol spelling (e.g., 'AAPL' not 'Apple')\n• Try a major stock like AAPL, MSFT, or SPY\n• Wait a moment and try again\n• Use `!test` to check system status",
                inline=False
            )
            error_embed.add_field(
                name="⏰ Market Hours Info",
                value="Analysis works 24/7 using the latest available trading data, even when markets are closed.",
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
        
        # Send options info if available
        try:
            await send_basic_options_info(ctx, symbol.upper(), analysis_data)
        except Exception as e:
            print(f"Options analysis error: {e}")
            # Don't fail the whole command if options fail
            pass
        
    except asyncio.TimeoutError:
        timeout_embed = discord.Embed(
            title="⏰ Analysis Timeout",
            description=f"Analysis of **{symbol.upper()}** took too long",
            color=0xffd700
        )
        timeout_embed.add_field(
            name="🔄 What to try",
            value="• Wait a moment and try again\n• Check if symbol is correct\n• Try a simpler command like `!quick`\n• Use `!test` to check system status",
            inline=False
        )
        timeout_embed.add_field(
            name="⏰ Market Hours",
            value="Analysis may take longer when markets are closed as we fetch historical data.",
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
            name="🔧 Error Details",
            value=f"```{str(e)[:100]}...```",
            inline=False
        )
        error_embed.add_field(
            name="💡 What to try",
            value="• Check symbol spelling\n• Try `!test` to check system status\n• Try again in a moment\n• Use `!quick` for faster analysis",
            inline=False
        )
        await msg.edit(content="", embed=error_embed)

async def send_basic_options_info(ctx, symbol: str, analysis_data: dict):
    """Send basic options information without complex calculations"""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        if not expirations:
            return  # No options available, just skip
        
        # Simple options info embed
        embed = discord.Embed(
            title=f"📊 Options Info: {symbol}",
            description=f"Basic options information for **{symbol}**",
            color=0x7289DA,
            timestamp=datetime.now()
        )
        
        current_price = analysis_data['current_price']
        sentiment = analysis_data['sentiment']
        
        embed.add_field(
            name="💰 Current Context",
            value=f"**Price:** ${current_price:.2f}\n**Sentiment:** {sentiment}",
            inline=True
        )
        
        # Available expirations
        next_exps = expirations[:3] if len(expirations) >= 3 else expirations
        exp_text = "\n".join([f"• {exp}" for exp in next_exps])
        
        embed.add_field(
            name="📅 Available Expirations",
            value=exp_text,
            inline=True
        )
        
        # Basic strategy suggestions based on sentiment
        if "BULLISH" in sentiment:
            strategy_text = "🟢 **Bullish Strategies:**\n• Long calls\n• Bull call spreads\n• Cash-secured puts"
        elif "BEARISH" in sentiment:
            strategy_text = "🔴 **Bearish Strategies:**\n• Long puts\n• Bear put spreads\n• Covered calls"
        else:
            strategy_text = "⚪ **Neutral Strategies:**\n• Iron condors\n• Straddles\n• Covered calls"
        
        embed.add_field(name="🎯 Strategy Ideas", value=strategy_text, inline=False)
        
        embed.set_footer(text="📊 Basic options info • Not financial advice • Do your research")
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        print(f"Basic options info error: {e}")
        # Fail silently - options info is supplementary

@bot.command(name='quick', aliases=['q'])
async def quick_analysis(ctx, symbol: str = None):
    """Quick analysis with faster response"""
    
    if not symbol:
        await ctx.send("❌ Usage: `!quick AAPL` or `!q TSLA`")
        return
    
    msg = await ctx.send(f"⚡ Quick analysis for **{symbol.upper()}**...")
    
    try:
        # Use shorter timeout for quick analysis
        analysis_data = await asyncio.wait_for(
            perform_stock_analysis(symbol), 
            timeout=20.0
        )
        
        if not analysis_data:
            await msg.edit(content=f"❌ No data found for **{symbol.upper()}**\nTry: `!test` to check system status")
            return
        
        # Create simplified embed
        embed = discord.Embed(
            title=f"⚡ Quick Analysis: {symbol.upper()}",
            description=f"**${analysis_data['current_price']:.2f}** ({analysis_data['change_pct']:+.1f}%) • **{analysis_data['sentiment']}**",
            color=0x00ff88 if analysis_data['score'] > 0 else 0xff6b6b if analysis_data['score'] < 0 else 0xffd700,
            timestamp=datetime.now()
        )
        
        # Key metrics only
        quick_text = f"**Score:** {analysis_data['score']}/10.0\n"
        quick_text += f"**RSI:** {analysis_data['rsi']:.1f}\n"
        quick_text += f"**Volume:** {analysis_data['volume_ratio']:.1f}x avg\n"
        quick_text += f"**Support:** ${analysis_data['support']:.2f}\n"
        quick_text += f"**Resistance:** ${analysis_data['resistance']:.2f}"
        
        embed.add_field(name="📊 Key Metrics", value=quick_text, inline=True)
        
        # Top 3 signals
        top_signals = analysis_data['signals'][:3]
        if top_signals:
            embed.add_field(name="🚨 Top Signals", value="\n".join(top_signals), inline=True)
        
        embed.set_footer(text=f"⚡ Quick analysis • {analysis_data.get('data_quality', 'Data available')} • Use !analysis for full report")
        
        await msg.edit(content="", embed=embed)
        
    except asyncio.TimeoutError:
        await msg.edit(content=f"⏰ **{symbol.upper()}** analysis timed out. Try `!test` to check system status.")
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
        # Get basic price data for correlation
        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period="2d", timeout=15)
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
                
                analysis_data = {
                    'current_price': current_price,
                    'change_pct': change_pct,
                    'timestamp': datetime.now()
                }
            else:
                raise Exception("No price data")
                
        except:
            # Fallback if price data fails
            analysis_data = {
                'current_price': 0,
                'change_pct': 0,
                'timestamp': datetime.now()
            }
        
        # Create news analysis embed
        embed = await create_news_analysis_embed(symbol.upper(), analysis_data)
        await msg.edit(content="", embed=embed)
        
    except Exception as e:
        await msg.edit(content=f"❌ News analysis failed for **{symbol.upper()}**: {str(e)[:50]}...")

@bot.command(name='info')
async def info_command(ctx):
    """Show comprehensive help information"""
    embed = discord.Embed(
        title="📊 Enhanced Stock Analysis Bot",
        description="Professional-grade stock analysis with news integration!",
        color=0x0099ff
    )
    
    embed.add_field(
        name="📈 Analysis Commands",
        value="`!analysis SYMBOL` - Full interactive analysis\n`!quick SYMBOL` - Fast overview\n`!news SYMBOL` - News impact analysis\n`!test` - System status check",
        inline=False
    )
    
    embed.add_field(
        name="🎯 Interactive Features",
        value="• **Refresh Analysis** - Update with latest data\n• **Detailed Signals** - Comprehensive breakdown\n• **Price Levels** - Key support/resistance\n• **News Impact** - Recent news affecting price",
        inline=False
    )
    
    embed.add_field(
        name="📊 Technical Analysis",
        value="• RSI (14-period)\n• Moving Averages (20, 50-day)\n• Volume Analysis\n• Support/Resistance\n• Price momentum signals",
        inline=True
    )
    
    embed.add_field(
        name="📰 News Analysis",
        value="• Recent headlines\n• Sentiment analysis\n• Price correlation\n• Impact assessment",
        inline=True
    )
    
    embed.add_field(
        name="🚀 Usage Examples",
        value="`!analysis AAPL` - Apple full analysis\n`!quick TSLA` - Tesla quick view\n`!news NVDA` - Nvidia news analysis\n`!test` - Check if system working",
        inline=False
    )
    
    embed.add_field(
        name="⚡ Aliases",
        value="`!a` = `!analysis`\n`!q` = `!quick`\n`!n` = `!news`",
        inline=False
    )
    
    embed.set_footer(text="⚠️ Not financial advice • Educational purposes only • Do your own research")
    
    await ctx.send(embed=embed)

# Health check endpoint for Render
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Enhanced Stock Analysis Bot is running')
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
    # Get Discord token
    DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    
    if not DISCORD_TOKEN:
        print("❌ Error: DISCORD_BOT_TOKEN environment variable not set!")
        print("Please add your Discord bot token as an environment variable")
        sys.exit(1)
    
    print("📊 Starting Enhanced Stock Analysis Bot...")
    print("🎯 Features: Enhanced TA, News Analysis, Better Error Handling")
    print("⚡ Commands: !analysis, !quick, !news, !test")
    
    # Start health check server in background thread
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    try:
        # Run the bot
        bot.run(DISCORD_TOKEN)
    except discord.LoginFailure:
        print("❌ Invalid Discord bot token! Please check your DISCORD_BOT_TOKEN")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Bot error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)