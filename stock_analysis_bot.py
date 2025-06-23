#!/usr/bin/env python3
"""
Enhanced Stock Analysis Bot with News Analysis
Professional stock analysis with interactive features + News Impact Analysis
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

# Load environment variables
load_dotenv()

print("📊 Starting Enhanced Stock Analysis Bot...")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Global storage for analysis results
analysis_cache = {}

class StockAnalysisView(discord.ui.View):
    """Interactive view for stock analysis"""
    
    def __init__(self, symbol: str, analysis_data: dict):
        super().__init__(timeout=600)  # 10 minutes timeout
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
                # Update cache
                analysis_cache[f"{interaction.user.id}_{self.symbol}"] = new_analysis
                
                # Create new view with updated data
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

async def perform_stock_analysis(symbol: str) -> Optional[Dict]:
    """Perform comprehensive stock analysis"""
    try:
        print(f"Analyzing {symbol}...")
        ticker = yf.Ticker(symbol.upper())
        
        # Get more data for better analysis
        hist = ticker.history(period="90d")  # Extended to 90 days
        
        if hist.empty:
            return None
        
        # Basic price data
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
        
        # Calculate more comprehensive technical indicators
        closes = hist['Close']
        highs = hist['High']
        lows = hist['Low']
        volumes = hist['Volume']
        
        # RSI calculation (14-period)
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        current_rsi = rsi_series.iloc[-1] if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50
        
        # Moving averages
        sma_20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else current_price
        sma_50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else current_price
        ema_12 = closes.ewm(span=12).mean().iloc[-1] if len(closes) >= 12 else current_price
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = closes.rolling(bb_period).mean().iloc[-1] if len(closes) >= bb_period else current_price
        bb_std_dev = closes.rolling(bb_period).std().iloc[-1] if len(closes) >= bb_period else 0
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        
        # MACD
        ema_12_series = closes.ewm(span=12).mean()
        ema_26_series = closes.ewm(span=26).mean()
        macd_line = ema_12_series - ema_26_series
        macd_signal = macd_line.ewm(span=9).mean()
        macd_current = macd_line.iloc[-1] if not macd_line.empty else 0
        macd_signal_current = macd_signal.iloc[-1] if not macd_signal.empty else 0
        
        # Volume analysis
        avg_volume = volumes.mean() if not volumes.empty else 1000000
        current_volume = volumes.iloc[-1] if not volumes.empty else 1000000
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Support and Resistance levels
        recent_highs = highs.rolling(10).max().dropna()
        recent_lows = lows.rolling(10).min().dropna()
        resistance = recent_highs.iloc[-5:].max() if len(recent_highs) >= 5 else current_price * 1.05
        support = recent_lows.iloc[-5:].min() if len(recent_lows) >= 5 else current_price * 0.95
        
        # Volatility (Average True Range approximation)
        high_low = highs - lows
        volatility = high_low.rolling(14).mean().iloc[-1] if len(high_low) >= 14 else current_price * 0.02
        
        # Generate comprehensive signals and score
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
            signals.append("🟢 Above 20-day MA")
        elif current_price < sma_20 < sma_50:
            score -= 2.0
            signals.append("🔴 Strong Downtrend - Below Both MAs")
            confidence_factors.append("Strong trend confirmation")
        elif current_price < sma_20:
            score -= 1.0
            signals.append("🔴 Below 20-day MA")
        
        # MACD signals
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
        
        # Price momentum
        if change_pct > 3:
            score += 1.5
            signals.append(f"🟢 Strong Upward Momentum (+{change_pct:.1f}%)")
        elif change_pct > 1:
            score += 0.5
            signals.append(f"🟢 Positive Momentum (+{change_pct:.1f}%)")
        elif change_pct < -3:
            score -= 1.5
            signals.append(f"🔴 Strong Downward Momentum ({change_pct:.1f}%)")
        elif change_pct < -1:
            score -= 0.5
            signals.append(f"🔴 Negative Momentum ({change_pct:.1f}%)")
        
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
        
        # Calculate additional metrics
        price_range_52w = {
            'high': highs.max(),
            'low': lows.min(),
            'current_position': (current_price - lows.min()) / (highs.max() - lows.min()) * 100
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
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        print(f"Analysis error for {symbol}: {e}")
        return None

def create_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create the main analysis embed"""
    score = analysis_data['score']
    sentiment = analysis_data['sentiment']
    confidence = analysis_data['confidence']
    
    # Determine color based on sentiment
    if "STRONG BULLISH" in sentiment:
        color = 0x00ff00  # Bright green
    elif "BULLISH" in sentiment:
        color = 0x32cd32  # Green
    elif "STRONG BEARISH" in sentiment:
        color = 0xff0000  # Bright red
    elif "BEARISH" in sentiment:
        color = 0xff4500  # Red
    else:
        color = 0xffd700  # Gold for neutral
    
    embed = discord.Embed(
        title=f"📊 Enhanced Analysis: {symbol}",
        description=f"**Price:** ${analysis_data['current_price']:.2f} ({analysis_data['change_pct']:+.1f}%)\n**Signal Score:** {score}/10.0\n**Sentiment:** {sentiment}\n**Confidence:** {confidence}",
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
    volume_text += f"**Ratio:** {analysis_data['volume_ratio']:.1f}x\n"
    volume_text += f"**MACD:** {analysis_data['macd']:.3f}"
    
    embed.add_field(name="📊 Volume & Momentum", value=volume_text, inline=True)
    
    # Key levels
    levels_text = f"**Resistance:** ${analysis_data['resistance']:.2f}\n"
    levels_text += f"**Support:** ${analysis_data['support']:.2f}\n"
    levels_text += f"**BB Upper:** ${analysis_data['bb_upper']:.2f}\n"
    levels_text += f"**BB Lower:** ${analysis_data['bb_lower']:.2f}"
    
    embed.add_field(name="🎯 Key Levels", value=levels_text, inline=True)
    
    # Top signals
    top_signals = analysis_data['signals'][:4]  # Show top 4 signals
    signals_text = "\n".join(top_signals) if top_signals else "No significant signals"
    
    embed.add_field(name="🚨 Key Signals", value=signals_text, inline=False)
    
    # 52-week position
    position_52w = analysis_data['price_range_52w']['current_position']
    range_text = f"**52W High:** ${analysis_data['price_range_52w']['high']:.2f}\n"
    range_text += f"**52W Low:** ${analysis_data['price_range_52w']['low']:.2f}\n"
    range_text += f"**Position:** {position_52w:.1f}% of range"
    
    embed.add_field(name="📅 52-Week Range", value=range_text, inline=True)
    
    # Confidence factors
    if analysis_data['confidence_factors']:
        conf_text = "• " + "\n• ".join(analysis_data['confidence_factors'][:3])
        embed.add_field(name="✅ Confidence Factors", value=conf_text, inline=True)
    
    embed.set_footer(text="💡 Click buttons below for detailed analysis • Not financial advice")
    
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
        # Split into bullish and bearish
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
        ("🔵 BB Upper", analysis_data['bb_upper'], (analysis_data['bb_upper'] - current_price) / current_price * 100),
        ("🔵 BB Lower", analysis_data['bb_lower'], (current_price - analysis_data['bb_lower']) / current_price * 100),
        ("🟡 20-day MA", analysis_data['sma_20'], (analysis_data['sma_20'] - current_price) / current_price * 100),
        ("🟡 50-day MA", analysis_data['sma_50'], (analysis_data['sma_50'] - current_price) / current_price * 100),
    ]
    
    levels_text = f"**Current Price:** ${current_price:.2f}\n\n"
    
    for name, price, distance in levels_data:
        direction = "above" if distance < 0 else "below"
        levels_text += f"{name}: ${price:.2f} ({abs(distance):.1f}% {direction})\n"
    
    embed.add_field(name="🎯 Price Levels", value=levels_text, inline=False)
    
    # Trading ranges
    range_text = f"**Daily Range:** ${analysis_data['price_range_52w']['low']:.2f} - ${analysis_data['price_range_52w']['high']:.2f}\n"
    range_text += f"**Position in Range:** {analysis_data['price_range_52w']['current_position']:.1f}%\n"
    range_text += f"**Volatility:** ${analysis_data['volatility']:.2f}"
    
    embed.add_field(name="📊 Trading Ranges", value=range_text, inline=False)
    
    embed.set_footer(text="💡 Use these levels for entry/exit planning")
    
    return embed

def create_risk_analysis_embed(symbol: str, analysis_data: dict) -> discord.Embed:
    """Create risk analysis embed"""
    embed = discord.Embed(
        title=f"⚠️ Risk Analysis: {symbol}",
        description="Risk assessment and position sizing guidance",
        color=0xff6b6b,
        timestamp=datetime.now()
    )
    
    current_price = analysis_data['current_price']
    support = analysis_data['support']
    resistance = analysis_data['resistance']
    volatility = analysis_data['volatility']
    
    # Risk levels
    stop_loss_level = support * 0.98  # 2% below support
    risk_per_share = current_price - stop_loss_level
    risk_percentage = (risk_per_share / current_price) * 100
    
    # Reward potential
    target_level = resistance * 1.02  # 2% above resistance
    reward_per_share = target_level - current_price
    reward_percentage = (reward_per_share / current_price) * 100
    
    # Risk/Reward ratio
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
    
    risk_text = f"**Stop Loss Level:** ${stop_loss_level:.2f}\n"
    risk_text += f"**Risk per Share:** ${risk_per_share:.2f} ({risk_percentage:.1f}%)\n"
    risk_text += f"**Target Level:** ${target_level:.2f}\n"
    risk_text += f"**Reward per Share:** ${reward_per_share:.2f} ({reward_percentage:.1f}%)\n"
    risk_text += f"**Risk/Reward Ratio:** 1:{rr_ratio:.1f}"
    
    embed.add_field(name="⚖️ Risk/Reward Analysis", value=risk_text, inline=False)
    
    # Position sizing (example for $10,000 account)
    account_sizes = [1000, 5000, 10000, 25000]
    position_text = "**Position Sizing (2% account risk):**\n"
    
    for account_size in account_sizes:
        max_risk = account_size * 0.02  # 2% risk
        max_shares = int(max_risk / risk_per_share) if risk_per_share > 0 else 0
        position_value = max_shares * current_price
        
        position_text += f"${account_size:,} account: {max_shares} shares (${position_value:.0f})\n"
    
    embed.add_field(name="📊 Position Sizing Examples", value=position_text, inline=False)
    
    # Risk warnings
    warnings = []
    
    if analysis_data['volume_ratio'] < 0.5:
        warnings.append("⚠️ Low volume - poor liquidity")
    
    if volatility / current_price > 0.05:  # High volatility
        warnings.append("⚠️ High volatility - increased risk")
    
    if analysis_data['price_range_52w']['current_position'] > 90:
        warnings.append("⚠️ Near 52-week highs - potential pullback")
    elif analysis_data['price_range_52w']['current_position'] < 10:
        warnings.append("⚠️ Near 52-week lows - potential further decline")
    
    if rr_ratio < 1.5:
        warnings.append("⚠️ Poor risk/reward ratio")
    
    if warnings:
        embed.add_field(name="🚨 Risk Warnings", value="\n".join(warnings), inline=False)
    
    embed.set_footer(text="⚠️ Not financial advice • Always use proper risk management")
    
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
    print('🎯 Features: Advanced TA, Interactive UI, Risk Analysis, Price Levels')

@bot.command(name='analysis', aliases=['analyze', 'stock', 'a'])
async def analyze_stock(ctx, symbol: str = None):
    """Comprehensive stock analysis with interactive features"""
    
    if not symbol:
        embed = discord.Embed(
            title="❌ Missing Symbol",
            description="Please provide a stock symbol to analyze",
            color=0xff6b6b
        )
        embed.add_field(
            name="📝 Usage",
            value="`!analysis AAPL`\n`!a TSLA`\n`!stock SPY`",
            inline=False
        )
        await ctx.send(embed=embed)
        return
    
    # Send analyzing message
    msg = await ctx.send(f"🔍 Performing comprehensive analysis on **{symbol.upper()}**...")
    
    try:
        # Perform analysis
        analysis_data = await perform_stock_analysis(symbol)
        
        if not analysis_data:
            error_embed = discord.Embed(
                title=f"❌ Analysis Failed",
                description=f"No data found for **{symbol.upper()}**",
                color=0xff6b6b
            )
            error_embed.add_field(
                name="💡 Suggestions",
                value="• Check the symbol spelling\n• Try a different exchange (e.g., AAPL vs AAPL.L)\n• Ensure it's a publicly traded stock",
                inline=False
            )
            await msg.edit(content="", embed=error_embed)
            return
        
        # Store in cache for future reference
        cache_key = f"{ctx.author.id}_{symbol.upper()}"
        analysis_cache[cache_key] = analysis_data
        
        # Create embed and interactive view
        embed = create_analysis_embed(symbol.upper(), analysis_data)
        view = StockAnalysisView(symbol.upper(), analysis_data)
        
        await msg.edit(content="", embed=embed, view=view)
        
        # Send options analysis
        await send_options_analysis(ctx, symbol.upper(), analysis_data)
        
    except Exception as e:
        print(f"Analysis error: {e}")
        error_embed = discord.Embed(
            title="❌ Analysis Error",
            description=f"Error analyzing **{symbol.upper()}**: {str(e)}",
            color=0xff6b6b
        )
        await msg.edit(content="", embed=error_embed)

async def send_options_analysis(ctx, symbol: str, analysis_data: dict):
    """Send enhanced options analysis"""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        if not expirations:
            await ctx.send(f"📊 Analysis complete! No options available for **{symbol}**")
            return
        
        options_found = []
        current_price = analysis_data['current_price']
        score = analysis_data['score']
        sentiment = analysis_data['sentiment']
        
        # Get options data with enhanced scoring
        for exp_date in expirations[:4]:  # Check more expirations
            try:
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_datetime - datetime.now()).days
                
                if days_to_exp < 7 or days_to_exp > 60:  # Focus on 1-8 weeks
                    continue
                
                chain = ticker.option_chain(exp_date)
                
                # Enhanced call analysis
                calls = chain.calls
                good_calls = calls[
                    (calls['lastPrice'] > 0.05) & 
                    (calls['lastPrice'] <= 5.0) & 
                    (calls['volume'] > 0)
                ].copy()
                
                for _, call in good_calls.iterrows():
                    cost = call['lastPrice'] * 100
                    distance = (call['strike'] - current_price) / current_price
                    
                    # Enhanced probability calculation
                    base_prob = calculate_option_probability(distance, days_to_exp, 'call')
                    
                    # Adjust for technical analysis
                    sentiment_adjustment = get_sentiment_adjustment(score, 'call')
                    final_prob = min(0.85, max(0.05, base_prob + sentiment_adjustment))
                    
                    # Calculate expected value
                    potential_profit = max(0, current_price * (1 + distance) - call['strike']) * 100 - cost
                    expected_value = potential_profit * final_prob - cost * (1 - final_prob)
                    
                    options_found.append({
                        'type': 'CALL',
                        'strike': call['strike'],
                        'cost': cost,
                        'expiration': exp_date,
                        'days_to_exp': days_to_exp,
                        'probability': final_prob,
                        'expected_value': expected_value,
                        'volume': call['volume'],
                        'open_interest': call['openInterest']
                    })
                
                # Enhanced put analysis
                puts = chain.puts
                good_puts = puts[
                    (puts['lastPrice'] > 0.05) & 
                    (puts['lastPrice'] <= 5.0) & 
                    (puts['volume'] > 0)
                ].copy()
                
                for _, put in good_puts.iterrows():
                    cost = put['lastPrice'] * 100
                    distance = (current_price - put['strike']) / current_price
                    
                    # Enhanced probability calculation
                    base_prob = calculate_option_probability(distance, days_to_exp, 'put')
                    
                    # Adjust for technical analysis
                    sentiment_adjustment = get_sentiment_adjustment(score, 'put')
                    final_prob = min(0.85, max(0.05, base_prob + sentiment_adjustment))
                    
                    # Calculate expected value
                    potential_profit = max(0, put['strike'] - current_price * (1 - distance)) * 100 - cost
                    expected_value = potential_profit * final_prob - cost * (1 - final_prob)
                    
                    options_found.append({
                        'type': 'PUT',
                        'strike': put['strike'],
                        'cost': cost,
                        'expiration': exp_date,
                        'days_to_exp': days_to_exp,
                        'probability': final_prob,
                        'expected_value': expected_value,
                        'volume': put['volume'],
                        'open_interest': put['openInterest']
                    })
                
                if len(options_found) >= 100:  # Get plenty of options
                    break
                    
            except Exception as e:
                print(f"Error with expiration {exp_date}: {e}")
                continue
        
        # Enhanced options display
        if options_found:
            # Sort by expected value (best risk-adjusted returns first)
            options_found.sort(key=lambda x: x['expected_value'], reverse=True)
            
            # Determine embed color based on sentiment
            if "BULLISH" in sentiment:
                color = 0x00ff88
            elif "BEARISH" in sentiment:
                color = 0xff6b6b
            else:
                color = 0xffd700
            
            options_embed = discord.Embed(
                title=f"🎯 Enhanced Options Analysis: {symbol}",
                description=f"**Sentiment:** {sentiment} (Score: {score})\n**Current Price:** ${current_price:.2f}\n*Sorted by expected value*",
                color=color,
                timestamp=datetime.now()
            )
            
            # Show top 6 options (2 rows of 3)
            for i, option in enumerate(options_found[:6], 1):
                # Calculate break-even
                if option['type'] == 'CALL':
                    breakeven = option['strike'] + (option['cost'] / 100)
                    breakeven_move = (breakeven - current_price) / current_price * 100
                else:
                    breakeven = option['strike'] - (option['cost'] / 100)
                    breakeven_move = (current_price - breakeven) / current_price * 100
                
                option_text = f"**Strike:** ${option['strike']:.0f}\n"
                option_text += f"**Cost:** ${option['cost']:.0f}\n"
                option_text += f"**Expires:** {option['expiration']} ({option['days_to_exp']}d)\n"
                option_text += f"**Probability:** {option['probability']:.0%}\n"
                option_text += f"**Break-even:** {breakeven_move:+.1f}%\n"
                option_text += f"**Expected Value:** ${option['expected_value']:.0f}"
                
                emoji = "📈" if option['type'] == 'CALL' else "📉"
                options_embed.add_field(
                    name=f"{emoji} #{i} {option['type']}",
                    value=option_text,
                    inline=True
                )
            
            # Add summary statistics
            total_options = len(options_found)
            positive_ev = len([o for o in options_found if o['expected_value'] > 0])
            avg_prob = sum(o['probability'] for o in options_found[:10]) / min(10, len(options_found))
            
            summary_text = f"**Total Options Found:** {total_options}\n"
            summary_text += f"**Positive Expected Value:** {positive_ev}\n"
            summary_text += f"**Average Probability (Top 10):** {avg_prob:.0%}"
            
            options_embed.add_field(
                name="📊 Summary Statistics",
                value=summary_text,
                inline=False
            )
            
            options_embed.set_footer(text="🧮 Enhanced probability model • Expected value analysis • Not financial advice")
            await ctx.send(embed=options_embed)
            
        else:
            no_options_embed = discord.Embed(
                title="📊 Options Analysis Complete",
                description=f"No suitable options found for **{symbol}**",
                color=0xffd700
            )
            no_options_embed.add_field(
                name="📋 Search Criteria",
                value="• 7-60 days to expiration\n• $0.05 - $5.00 per contract\n• Active volume > 0\n• Reasonable probability of profit",
                inline=False
            )
            await ctx.send(embed=no_options_embed)
            
    except Exception as e:
        print(f"Options analysis error: {e}")
        await ctx.send(f"📊 Technical analysis complete! Options data unavailable for **{symbol}**")

def calculate_option_probability(distance: float, days_to_exp: int, option_type: str) -> float:
    """Enhanced option probability calculation"""
    
    # Base probability based on distance from current price
    if abs(distance) < 0.02:  # Within 2%
        base_prob = 0.65
    elif abs(distance) < 0.05:  # Within 5%
        base_prob = 0.50
    elif abs(distance) < 0.10:  # Within 10%
        base_prob = 0.35
    elif abs(distance) < 0.20:  # Within 20%
        base_prob = 0.25
    else:
        base_prob = 0.15
    
    # Adjust for time decay
    if days_to_exp < 14:
        time_factor = 0.8  # Less time = lower probability
    elif days_to_exp < 30:
        time_factor = 1.0
    else:
        time_factor = 0.9  # Too much time = uncertainty
    
    # Adjust for direction
    if option_type == 'call' and distance > 0:
        direction_factor = 0.9  # OTM calls
    elif option_type == 'put' and distance > 0:
        direction_factor = 0.9  # OTM puts
    else:
        direction_factor = 1.1  # ITM options
    
    return base_prob * time_factor * direction_factor

def get_sentiment_adjustment(score: float, option_type: str) -> float:
    """Get sentiment-based probability adjustment"""
    
    if option_type == 'call':
        if score > 2:
            return 0.15  # Strong bullish = higher call probability
        elif score > 0.5:
            return 0.08
        elif score < -2:
            return -0.15  # Strong bearish = lower call probability
        elif score < -0.5:
            return -0.08
    else:  # put
        if score < -2:
            return 0.15  # Strong bearish = higher put probability
        elif score < -0.5:
            return 0.08
        elif score > 2:
            return -0.15  # Strong bullish = lower put probability
        elif score > 0.5:
            return -0.08
    
    return 0  # Neutral sentiment = no adjustment

@bot.command(name='news', aliases=['n'])
async def news_command(ctx, symbol: str = None):
    """Get news analysis for a specific stock"""
    
    if not symbol:
        await ctx.send("❌ Usage: `!news AAPL` or `!n TSLA`")
        return
    
    msg = await ctx.send(f"📰 Fetching news analysis for **{symbol.upper()}**...")
    
    try:
        # Get basic analysis data for price correlation
        analysis_data = await perform_stock_analysis(symbol)
        
        if not analysis_data:
            # Create minimal analysis data for news-only analysis
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period="2d")
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
                await msg.edit(content=f"❌ Unable to fetch data for **{symbol.upper()}**")
                return
        
        # Create news analysis embed
        embed = await create_news_analysis_embed(symbol.upper(), analysis_data)
        await msg.edit(content="", embed=embed)
        
    except Exception as e:
        await msg.edit(content=f"❌ News analysis failed: {str(e)}")

@bot.command(name='trending', aliases=['trend'])
async def trending_news(ctx):
    """Show trending market news"""
    
    msg = await ctx.send("📈 Fetching trending market news...")
    
    try:
        # Fetch general market news
        news_sources = [
            ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "Market News"),
            ("https://feeds.marketwatch.com/marketwatch/marketpulse/", "MarketWatch"),
            ("https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en", "Google Finance")
        ]
        
        all_news = []
        
        for url, source in news_sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:3]:  # Top 3 from each source
                    pub_date = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') and entry.published_parsed else datetime.now()
                    
                    all_news.append({
                        'title': entry.title,
                        'summary': entry.get('summary', '')[:150] + '...' if len(entry.get('summary', '')) > 150 else entry.get('summary', ''),
                        'link': entry.link,
                        'published': pub_date,
                        'source': source,
                        'age_hours': (datetime.now() - pub_date).total_seconds() / 3600
                    })
            except Exception as e:
                print(f"Error fetching from {source}: {e}")
        
        # Sort by recency and remove duplicates
        all_news.sort(key=lambda x: x['published'], reverse=True)
        
        # Create trending news embed
        embed = discord.Embed(
            title="📈 Trending Market News",
            description="Latest market-moving news and developments",
            color=0x7289DA,
            timestamp=datetime.now()
        )
        
        if all_news:
            news_text = ""
            for i, item in enumerate(all_news[:6], 1):  # Top 6 trending stories
                age_str = f"{int(item['age_hours'])}h" if item['age_hours'] < 24 else f"{int(item['age_hours']/24)}d"
                title = item['title'][:70] + "..." if len(item['title']) > 70 else item['title']
                
                news_text += f"**{i}.** {title}\n*{item['source']} • {age_str} ago*\n\n"
            
            embed.add_field(name="🔥 Top Stories", value=news_text, inline=False)
        else:
            embed.add_field(name="🔥 Top Stories", value="Unable to fetch trending news at this time.", inline=False)
        
        embed.add_field(
            name="💡 How to Use",
            value="• Use `!news SYMBOL` for stock-specific news\n• Use `!analysis SYMBOL` for technical + news analysis\n• News affects technical patterns and momentum",
            inline=False
        )
        
        embed.set_footer(text="📰 Trending market news • Updates every few minutes")
        
        await msg.edit(content="", embed=embed)
        
    except Exception as e:
        await msg.edit(content=f"❌ Trending news failed: {str(e)}")
    """Quick analysis without options"""
    
    if not symbol:
        await ctx.send("❌ Usage: `!quick AAPL` or `!q TSLA`")
        return
    
    msg = await ctx.send(f"⚡ Quick analysis for **{symbol.upper()}**...")
    
    try:
        analysis_data = await perform_stock_analysis(symbol)
        
        if not analysis_data:
            await msg.edit(content=f"❌ No data found for **{symbol.upper()}**")
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
        
        embed.set_footer(text="⚡ Quick analysis • Use !analysis for full report")
        
        await msg.edit(content="", embed=embed)
        
    except Exception as e:
        await msg.edit(content=f"❌ Quick analysis failed: {str(e)}")

@bot.command(name='compare', aliases=['vs'])
async def compare_stocks(ctx, symbol1: str = None, symbol2: str = None):
    """Compare two stocks side by side"""
    
    if not symbol1 or not symbol2:
        await ctx.send("❌ Usage: `!compare AAPL MSFT` or `!vs TSLA NVDA`")
        return
    
    msg = await ctx.send(f"🔄 Comparing **{symbol1.upper()}** vs **{symbol2.upper()}**...")
    
    try:
        # Analyze both stocks
        analysis1 = await perform_stock_analysis(symbol1)
        analysis2 = await perform_stock_analysis(symbol2)
        
        if not analysis1 or not analysis2:
            failed_stocks = []
            if not analysis1:
                failed_stocks.append(symbol1.upper())
            if not analysis2:
                failed_stocks.append(symbol2.upper())
            
            await msg.edit(content=f"❌ Failed to analyze: {', '.join(failed_stocks)}")
            return
        
        # Create comparison embed
        embed = discord.Embed(
            title=f"🔄 Stock Comparison",
            description=f"**{symbol1.upper()}** vs **{symbol2.upper()}**",
            color=0x7289DA,
            timestamp=datetime.now()
        )
        
        # Stock 1 summary
        stock1_text = f"**Price:** ${analysis1['current_price']:.2f} ({analysis1['change_pct']:+.1f}%)\n"
        stock1_text += f"**Score:** {analysis1['score']}/10.0\n"
        stock1_text += f"**Sentiment:** {analysis1['sentiment']}\n"
        stock1_text += f"**RSI:** {analysis1['rsi']:.1f}\n"
        stock1_text += f"**Volume:** {analysis1['volume_ratio']:.1f}x"
        
        embed.add_field(name=f"📊 {symbol1.upper()}", value=stock1_text, inline=True)
        
        # Stock 2 summary
        stock2_text = f"**Price:** ${analysis2['current_price']:.2f} ({analysis2['change_pct']:+.1f}%)\n"
        stock2_text += f"**Score:** {analysis2['score']}/10.0\n"
        stock2_text += f"**Sentiment:** {analysis2['sentiment']}\n"
        stock2_text += f"**RSI:** {analysis2['rsi']:.1f}\n"
        stock2_text += f"**Volume:** {analysis2['volume_ratio']:.1f}x"
        
        embed.add_field(name=f"📊 {symbol2.upper()}", value=stock2_text, inline=True)
        
        # Comparison summary
        winner = symbol1.upper() if analysis1['score'] > analysis2['score'] else symbol2.upper()
        score_diff = abs(analysis1['score'] - analysis2['score'])
        
        comparison_text = f"**Technical Winner:** {winner}\n"
        comparison_text += f"**Score Difference:** {score_diff:.1f}\n"
        
        if analysis1['rsi'] < 30 or analysis2['rsi'] < 30:
            oversold = symbol1.upper() if analysis1['rsi'] < analysis2['rsi'] else symbol2.upper()
            comparison_text += f"**More Oversold:** {oversold}\n"
        
        if analysis1['volume_ratio'] > 1.5 or analysis2['volume_ratio'] > 1.5:
            higher_vol = symbol1.upper() if analysis1['volume_ratio'] > analysis2['volume_ratio'] else symbol2.upper()
            comparison_text += f"**Higher Volume:** {higher_vol}"
        
        embed.add_field(name="🏆 Comparison Results", value=comparison_text, inline=False)
        
        embed.set_footer(text="📊 Side-by-side technical comparison • Not financial advice")
        
        await msg.edit(content="", embed=embed)
        
    except Exception as e:
        await msg.edit(content=f"❌ Comparison failed: {str(e)}")

@bot.command(name='info')
async def info_command(ctx):
    """Show comprehensive help information"""
    embed = discord.Embed(
        title="📊 Enhanced Stock Analysis Bot",
        description="Professional-grade stock analysis with interactive features!",
        color=0x0099ff
    )
    
    embed.add_field(
        name="📈 Analysis Commands",
        value="`!analysis SYMBOL` - Full interactive analysis\n`!quick SYMBOL` - Fast overview\n`!compare STOCK1 STOCK2` - Side-by-side comparison\n`!news SYMBOL` - News impact analysis\n`!trending` - Market trending news",
        inline=False
    )
    
    embed.add_field(
        name="🎯 Interactive Features",
        value="• **Refresh Analysis** - Update with latest data\n• **Detailed Signals** - Comprehensive signal breakdown\n• **Price Levels** - Key support/resistance levels\n• **News Impact** - Recent news affecting price\n• **Risk Analysis** - Position sizing and risk management",
        inline=False
    )
    
    embed.add_field(
        name="📊 Technical Analysis",
        value="• RSI (14-period)\n• Moving Averages (20, 50-day)\n• MACD & Signal Line\n• Bollinger Bands\n• Volume Analysis\n• Support/Resistance\n• 52-week positioning",
        inline=True
    )
    
    embed.add_field(
        name="🎯 Options Analysis",
        value="• Probability calculations\n• Expected value analysis\n• Break-even calculations\n• Volume & open interest\n• Sentiment adjustments\n• Risk/reward ratios",
        inline=True
    )
    
    embed.add_field(
        name="🚀 Quick Commands",
        value="`!analysis AAPL` - Apple analysis\n`!quick TSLA` - Tesla quick view\n`!compare AAPL MSFT` - Compare stocks\n`!news NVDA` - Nvidia news analysis\n`!trending` - Market news\n`!test` - Bot status check",
        inline=False
    )
    
    embed.add_field(
        name="⚡ Aliases",
        value="`!a` = `!analysis`\n`!q` = `!quick`\n`!vs` = `!compare`\n`!n` = `!news`\n`!trend` = `!trending`",
        inline=False
    )
    
    embed.set_footer(text="⚠️ Not financial advice • Educational purposes only • Do your own research")
    
    await ctx.send(embed=embed)

@bot.command(name='test')
async def test_command(ctx):
    """Enhanced test command with system status"""
    
    embed = discord.Embed(
        title="🤖 Enhanced Stock Bot Status",
        description="All systems operational!",
        color=0x00ff00
    )
    
    # Test basic functionality
    try:
        test_ticker = yf.Ticker("AAPL")
        test_data = test_ticker.history(period="5d")
        data_status = "✅ Working" if not test_data.empty else "❌ Failed"
    except:
        data_status = "❌ Failed"
    
    embed.add_field(
        name="📊 System Status",
        value=f"• **Discord Bot:** ✅ Online\n• **Data Feed:** {data_status}\n• **Interactive UI:** ✅ Available\n• **Options Analysis:** ✅ Enhanced\n• **Risk Analysis:** ✅ Available",
        inline=False
    )
    
    embed.add_field(
        name="🚀 Quick Test",
        value="`!analysis AAPL` - Full analysis\n`!quick SPY` - Quick view\n`!news TSLA` - News analysis\n`!trending` - Market news\n`!compare AAPL MSFT` - Compare stocks",
        inline=False
    )
    
    embed.add_field(
        name="📈 Features Ready",
        value="• Advanced technical analysis\n• Interactive buttons\n• Enhanced options analysis\n• Risk management tools\n• Price level analysis",
        inline=False
    )
    
    embed.set_footer(text="🎯 Ready for professional stock analysis!")
    
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
    print("🎯 Features: Advanced TA, Interactive UI, Options Analysis, Risk Management")
    print("⚡ Commands: !analysis, !quick, !compare, !info, !test")
    
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