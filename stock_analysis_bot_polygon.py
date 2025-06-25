#!/usr/bin/env python3
"""
Enhanced Stock Analysis Bot with Polygon.io API - DEBUG VERSION
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
    print(f"✅ Health check server starting on port {port}")
    try:
        server.serve_forever()
    except Exception as e:
        print(f"❌ Health server error: {e}")

# Bot events and commands
@bot.event
async def on_ready():
    """Bot startup event"""
    print("=" * 60)
    print(f"✅ Bot successfully logged in!")
    print(f"🤖 Bot Name: {bot.user}")
    print(f"📊 Bot ID: {bot.user.id}")
    print(f"🌐 Connected to {len(bot.guilds)} guilds")
    
    if bot.guilds:
        print("🏠 Guilds:")
        for guild in bot.guilds:
            print(f"   - {guild.name} (ID: {guild.id})")
    
    print("=" * 60)
    
    # Test Polygon.io connection
    if polygon_api:
        print("🔍 Testing Polygon.io API...")
        try:
            test_data = await asyncio.get_event_loop().run_in_executor(
                None, polygon_api.get_previous_close, 'AAPL'
            )
            if test_data and 'c' in test_data:
                print("✅ Polygon.io API is working correctly")
            else:
                print("❌ Polygon.io API test failed - no data returned")
        except Exception as e:
            print(f"❌ Polygon.io test failed: {e}")
    else:
        print("⚠️ Polygon.io API not configured")
    
    print("=" * 60)
    print("🚀 Bot is ready to analyze stocks!")
    print("💡 Try these commands:")
    print("   !help - Show help")
    print("   !stock AAPL - Analyze Apple")
    print("   !ping - Test bot response")
    print("=" * 60)

@bot.event
async def on_guild_join(guild):
    """When bot joins a new guild"""
    print(f"📥 Joined guild: {guild.name} (ID: {guild.id})")

@bot.event
async def on_message(message):
    """Process all messages for debugging"""
    if message.author == bot.user:
        return
    
    # Log all messages for debugging (without content for privacy)
    print(f"💬 Message from {message.author} in {message.guild.name if message.guild else 'DM'}")
    
    # Process commands
    await bot.process_commands(message)

@bot.command(name='ping')
async def ping_command(ctx):
    """Simple ping test"""
    print(f"🏓 Ping command received from {ctx.author}")
    latency = round(bot.latency * 1000)
    await ctx.send(f"🏓 Pong! Latency: {latency}ms")

@bot.command(name='test')
async def test_command(ctx):
    """Test command"""
    print(f"🧪 Test command received from {ctx.author}")
    embed = discord.Embed(
        title="🧪 Bot Test",
        description="Bot is working correctly!",
        color=0x00ff00
    )
    embed.add_field(name="Status", value="✅ Online", inline=True)
    embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="Guilds", value=len(bot.guilds), inline=True)
    
    await ctx.send(embed=embed)

@bot.command(name='stock', aliases=['s', 'analyze'])
async def stock_command(ctx, symbol: str = None):
    """Main stock analysis command - simplified for testing"""
    print(f"📊 Stock command received from {ctx.author} for symbol: {symbol}")
    
    if not symbol:
        await ctx.send("❌ Please provide a stock symbol! Example: `!stock AAPL`")
        return
    
    # Check if API is configured
    if not polygon_api:
        await ctx.send("❌ Stock analysis is not available - Polygon.io API key not configured")
        return
    
    # Send initial message
    loading_msg = await ctx.send(f"🔍 Analyzing **{symbol.upper()}**... Please wait...")
    
    try:
        # Simple test - just get previous close
        print(f"🔍 Testing API for {symbol}")
        
        prev_close_data = await asyncio.get_event_loop().run_in_executor(
            None, polygon_api.get_previous_close, symbol
        )
        
        if prev_close_data and 'c' in prev_close_data:
            price = prev_close_data['c']
            volume = prev_close_data.get('v', 'N/A')
            
            embed = discord.Embed(
                title=f"📊 Basic Quote: {symbol.upper()}",
                description=f"Quick test of Polygon.io API",
                color=0x00ff00
            )
            
            embed.add_field(name="Price", value=f"${price:.2f}", inline=True)
            embed.add_field(name="Volume", value=f"{volume:,.0f}" if volume != 'N/A' else 'N/A', inline=True)
            embed.add_field(name="Status", value="✅ API Working", inline=True)
            
            await loading_msg.edit(content=None, embed=embed)
        else:
            await loading_msg.edit(
                content=f"❌ Could not get data for **{symbol.upper()}**. Please check if this is a valid US stock symbol."
            )
    
    except Exception as e:
        print(f"❌ Error in stock command: {e}")
        await loading_msg.edit(
            content=f"❌ An error occurred while analyzing {symbol}: {str(e)}"
        )

@bot.command(name='help')
async def help_command(ctx):
    """Help command"""
    print(f"❓ Help command received from {ctx.author}")
    
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
    """Global error handler"""
    print(f"❌ Command error: {error}")
    
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f"❌ Command not found. Use `!help` to see available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ Missing required argument. Use `!help` for command usage.")
    else:
        await ctx.send(f"❌ An error occurred: {str(error)}")

# Graceful shutdown
def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutdown signal received...")
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