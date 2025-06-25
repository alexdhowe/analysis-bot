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
            
            print(f"🔍 Fetching previous close for {symbol}...")
            print(f"🌐 URL: {url}")
            print(f"🔑 Using API key: {self.api_key[:8]}...{self.api_key[-4:]}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            print(f"📊 Response status: {response.status_code}")
            print(f"📊 Response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            print(f"📊 Response data keys: {list(data.keys())}")
            
            if data.get('status') == 'OK' and 'results' in data and len(data['results']) > 0:
                print(f"✅ Successfully fetched previous close for {symbol}")
                result = data['results'][0]
                print(f"💰 Price data: {result}")
                return result
            else:
                print(f"❌ No data in response for {symbol}: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching previous close for {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error fetching previous close for {symbol}: {e}")
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
            
            print(f"🔍 Fetching {days} days of historical data for {symbol}...")
            print(f"🌐 URL: {url}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            print(f"📊 Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                results = data['results']
                print(f"✅ Successfully fetched {len(results)} days of historical data for {symbol}")
                return results
            else:
                print(f"❌ No historical data for {symbol}: {data}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}")
            return None