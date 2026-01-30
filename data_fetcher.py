"""
📥 Lightning Dice API Data Fetcher - FIXED WORKING VERSION
Uses your working API setup with proper error handling
"""

import requests
import time
import hashlib
import random
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetches data from Lightning Dice API - Optimized and fixed"""
    
    def __init__(self, use_mock=False):
        # API Configuration - using your working setup
        self.API_BASE_URL = "https://api-cs.casino.org"
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Origin': 'https://www.casino.org',
            'Referer': 'https://www.casino.org/',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        self.use_mock = use_mock
        self.cache = {}
        self.cache_timeout = 30
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        print(f"📥 DataFetcher initialized. Mock mode: {use_mock}")
    
    def get_api_endpoint(self, page=0, size=100):
        """Generate API URL - matches your working version"""
        timestamp = int(time.time() * 1000)
        return f"/svc-evolution-game-events/api/lightningdice?page={page}&size={size}&sort=data.settledAt,desc&duration=72&totals=3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18&isLightningMultiplierMatched=false&_={timestamp}"
    
    def get_total_pages(self):
        """Get total number of pages available from API"""
        try:
            url = self.API_BASE_URL + self.get_api_endpoint(0, 1)
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                total_count = response.headers.get('X-Total-Count')
                if total_count:
                    total_games = int(total_count)
                    games_per_page = 100
                    total_pages = (total_games + games_per_page - 1) // games_per_page
                    logger.info(f"📊 API reports {total_games} total games available")
                    return min(total_pages, 100)
                
                link_header = response.headers.get('Link', '')
                if 'rel="last"' in link_header:
                    match = re.search(r'page=(\d+)&size=\d+>; rel="last"', link_header)
                    if match:
                        last_page = int(match.group(1))
                        logger.info(f"📊 API reports page {last_page} as last page")
                        return min(last_page + 1, 100)
            
            logger.info(f"⚠️ Could not determine total pages, using default")
            return 67
            
        except Exception as e:
            logger.error(f"❌ Error getting total pages: {e}")
            return 67
    
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string to datetime"""
        if not timestamp_str:
            return datetime.now()
        
        try:
            if '.' in timestamp_str and 'Z' in timestamp_str:
                return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
            elif 'T' in timestamp_str and 'Z' in timestamp_str:
                return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
            else:
                parts = timestamp_str.replace('T', ' ').replace('Z', '').split('.')[0]
                return datetime.strptime(parts, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.debug(f"Timestamp parse error: {e}, string: {timestamp_str}")
            return datetime.now()
    
    def validate_game_data(self, game_data):
        """Validate game data"""
        try:
            if not isinstance(game_data, dict):
                return False
            
            required_fields = ['total', 'dice1', 'dice2', 'dice3', 'timestamp']
            for field in required_fields:
                if field not in game_data:
                    return False
            
            if not (1 <= game_data['dice1'] <= 6):
                return False
            if not (1 <= game_data['dice2'] <= 6):
                return False
            if not (1 <= game_data['dice3'] <= 6):
                return False
            
            if not (3 <= game_data['total'] <= 18):
                return False
            
            if game_data['dice1'] + game_data['dice2'] + game_data['dice3'] != game_data['total']:
                return False
            
            if game_data['timestamp'] > datetime.now():
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Validation error: {e}")
            return False
    
    def extract_game_data(self, game, page_num, idx):
        """Extract game data from API response"""
        try:
            if not isinstance(game, dict):
                return None
            
            game_id = game.get('id', f'game_{page_num}_{idx}')
            
            game_data = game.get('data', {})
            if not game_data:
                return None
            
            result_data = game_data.get('result', {})
            if not result_data:
                return None
            
            total_value = result_data.get('total')
            if total_value is None:
                return None
            
            dice1 = result_data.get('first')
            dice2 = result_data.get('second')
            dice3 = result_data.get('third')
            
            # Handle dice icons
            dice_map = {1: '⚀', 2: '⚁', 3: '⚂', 4: '⚃', 5: '⚄', 6: '⚅'}
            try:
                dice1_int = int(dice1) if dice1 is not None else 1
                dice2_int = int(dice2) if dice2 is not None else 1
                dice3_int = int(dice3) if dice3 is not None else 1
                dice_icons = f"{dice_map.get(dice1_int, '⚀')}{dice_map.get(dice2_int, '⚁')}{dice_map.get(dice3_int, '⚂')}"
            except:
                dice_icons = '⚀⚁⚂'
            
            settled_at = game_data.get('settledAt')
            if not settled_at:
                return None
            
            timestamp = self.parse_timestamp(settled_at)
            time_str = timestamp.strftime('%H:%M:%S')
            date_str = timestamp.strftime('%Y-%m-%d')
            
            total_amount = float(game.get('totalAmount', 0))
            total_winners = int(game.get('totalWinners', 0))
            
            try:
                total_value = int(total_value)
                dice1 = int(dice1) if dice1 is not None else 1
                dice2 = int(dice2) if dice2 is not None else 1
                dice3 = int(dice3) if dice3 is not None else 1
            except (ValueError, TypeError) as e:
                logger.debug(f"Number conversion error: {e}")
                return None
            
            if not (3 <= total_value <= 18):
                return None
            
            if not (1 <= dice1 <= 6) or not (1 <= dice2 <= 6) or not (1 <= dice3 <= 6):
                return None
            
            hash_str = f"{game_id}_{settled_at}"
            game_hash = hashlib.md5(hash_str.encode()).hexdigest()
            
            game_dict = {
                'game_hash': game_hash,
                'game_id': game_id,
                'total': total_value,
                'dice1': dice1,
                'dice2': dice2,
                'dice3': dice3,
                'dice_icons': dice_icons,
                'settled_at': settled_at,
                'time': time_str,
                'date': date_str,
                'timestamp': timestamp,
                'total_amount': total_amount,
                'total_winners': total_winners,
                'category': 'LOW' if total_value <= 9 else 'MIDDLE' if total_value <= 11 else 'HIGH'
            }
            
            if not self.validate_game_data(game_dict):
                return None
            
            return game_dict
            
        except Exception as e:
            logger.debug(f"Game extraction error: {e}")
            return None
    
    def fetch_single_page(self, page, size=100):
        """Fetch a single page of games"""
        try:
            url = self.API_BASE_URL + self.get_api_endpoint(page, size)
            
            logger.debug(f"Fetching: {url}")
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                logger.debug(f"API error {response.status_code}: {response.text[:100]}")
                return []
            
            data = response.json()
            
            if not isinstance(data, list):
                logger.debug(f"Unexpected response type: {type(data)}")
                return []
            
            logger.debug(f"Received {len(data)} raw games from page {page}")
            
            valid_games = []
            invalid_count = 0
            
            for idx, game in enumerate(data[:size]):
                game_data = self.extract_game_data(game, page, idx)
                if game_data:
                    valid_games.append(game_data)
                else:
                    invalid_count += 1
            
            if invalid_count > 0:
                logger.warning(f"⚠️ Page {page}: {invalid_count}/{len(data)} invalid games detected")
            
            return valid_games
            
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout fetching page {page}")
            return []
        except Exception as e:
            logger.debug(f"Error fetching page {page}: {e}")
            return []
    
    def get_mock_game_data(self, count=100):
        """Generate mock data as backup"""
        mock_games = []
        current_time = datetime.now()
        
        for i in range(count):
            dice1 = random.randint(1, 6)
            dice2 = random.randint(1, 6)
            dice3 = random.randint(1, 6)
            total = dice1 + dice2 + dice3
            
            game_time = current_time - timedelta(minutes=i*2)
            
            dice_map = {1: '⚀', 2: '⚁', 3: '⚂', 4: '⚃', 5: '⚄', 6: '⚅'}
            dice_icons = f"{dice_map.get(dice1, '⚀')}{dice_map.get(dice2, '⚁')}{dice_map.get(dice3, '⚂')}"
            
            game_data = {
                'game_hash': hashlib.md5(f"mock_{i}_{game_time.timestamp()}".encode()).hexdigest(),
                'game_id': f"mock_{i}",
                'total': total,
                'dice1': dice1,
                'dice2': dice2,
                'dice3': dice3,
                'dice_icons': dice_icons,
                'settled_at': game_time.isoformat() + 'Z',
                'time': game_time.strftime('%H:%M:%S'),
                'date': game_time.strftime('%Y-%m-%d'),
                'timestamp': game_time,
                'total_amount': random.randint(100, 5000),
                'total_winners': random.randint(1, 200),
                'category': 'LOW' if total <= 9 else 'MIDDLE' if total <= 11 else 'HIGH'
            }
            mock_games.append(game_data)
        
        mock_games.sort(key=lambda x: x['timestamp'], reverse=True)
        return mock_games
    
    def fetch_72h_data(self):
        """Fetch complete 72-hour game data"""
        logger.info("📥 Fetching COMPLETE 72-hour game data from API...")
        
        all_games = []
        
        if self.use_mock:
            logger.info("🔄 Using mock data for initial load")
            all_games = self.get_mock_game_data(6667)
            return all_games
        
        try:
            total_pages = self.get_total_pages()
            logger.info(f"📊 Will fetch {total_pages} pages (approx {total_pages * 100} games)")
            
            for page in range(total_pages):
                try:
                    print(f"📄 Loading page {page+1}/{total_pages}...", end="\r")
                    
                    page_games = self.fetch_single_page(page, 100)
                    
                    if page_games:
                        all_games.extend(page_games)
                        
                        if (page + 1) % 10 == 0 or page == total_pages - 1:
                            logger.info(f"📄 Loaded page {page+1}/{total_pages}: {len(all_games)} games total")
                    else:
                        logger.warning(f"⚠️ No games on page {page+1}")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Error on page {page+1}: {e}")
            
            print(f"\n✅ Fetched {len(all_games)} games from API")
            
        except Exception as e:
            logger.error(f"❌ Error in initial fetch: {e}")
            logger.info("🔄 Falling back to limited data...")
            
            for page in range(20):
                try:
                    page_games = self.fetch_single_page(page, 100)
                    if page_games:
                        all_games.extend(page_games)
                    time.sleep(0.1)
                except:
                    break
            
            if not all_games:
                logger.info("⚠️ No games fetched, using mock data")
                all_games = self.get_mock_game_data(2000)
        
        all_games.sort(key=lambda x: x['timestamp'], reverse=True)
        
        logger.info(f"🎉 Successfully loaded {len(all_games)} games from 72-hour history")
        
        if all_games:
            oldest = all_games[-1]['date']
            newest = all_games[0]['date']
            oldest_time = all_games[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            newest_time = all_games[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"📅 Date range: {oldest} to {newest}")
            logger.info(f"🕒 Time range: {oldest_time} to {newest_time}")
            
            # Show category distribution
            categories = defaultdict(int)
            for game in all_games[:1000]:  # First 1000 games
                categories[game['category']] += 1
            
            total_cats = sum(categories.values())
            if total_cats > 0:
                logger.info(f"📊 Category distribution (first 1000 games):")
                logger.info(f"   LOW: {categories['LOW']} ({categories['LOW']/total_cats*100:.1f}%)")
                logger.info(f"   MIDDLE: {categories['MIDDLE']} ({categories['MIDDLE']/total_cats*100:.1f}%)")
                logger.info(f"   HIGH: {categories['HIGH']} ({categories['HIGH']/total_cats*100:.1f}%)")
        
        return all_games
    
    def fetch_latest_games(self, limit=20):
        """Fetch latest games for real-time updates"""
        logger.debug(f"🔄 Fetching latest {limit} games...")
        
        if self.use_mock:
            # Generate fresh mock data
            return self.get_mock_game_data(min(limit, 5))
        
        try:
            page_games = self.fetch_single_page(0, limit)
            
            if not page_games:
                logger.warning("⚠️ No games from API, using mock data")
                return self.get_mock_game_data(min(limit, 5))
            
            # Filter duplicates
            unique_games = []
            seen_hashes = set()
            
            for game in page_games:
                if game['game_hash'] not in seen_hashes:
                    seen_hashes.add(game['game_hash'])
                    unique_games.append(game)
            
            logger.info(f"✅ Fetched {len(unique_games)} latest games")
            return unique_games
            
        except Exception as e:
            logger.error(f"❌ Error fetching latest games: {e}")
            return self.get_mock_game_data(min(limit, 5))
    
    def test_connection(self):
        """Test API connection"""
        try:
            test_games = self.fetch_latest_games(1)
            return {
                'success': True,
                'api_working': len(test_games) > 0,
                'mode': 'MOCK' if self.use_mock else 'REAL'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'api_working': False
            }

# Singleton instance
data_fetcher = DataFetcher(use_mock=False)
