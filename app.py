from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import json
import time
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import hashlib
import random
import re
import logging

# ✅ AI সিস্টেম import করুন
from prediction_system import IntelligentPredictionSystem

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Configuration
API_BASE_URL = "https://api-cs.casino.org"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Encoding': 'gzip, deflate',
    'Origin': 'https://www.casino.org',
    'Referer': 'https://www.casino.org/',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
}

# Configuration
USE_MOCK_DATA = False
FETCH_INTERVAL = 3
MAX_GAMES_TO_KEEP = 50000
ENABLE_DEBUG = True

# Global variables
all_games_data = []
last_fetch_time = None
is_fetching = False
DATA_LOCK = threading.Lock()
fetcher_started = False
system_start_time = datetime.now()

# ✅ Global prediction system instance
prediction_system = IntelligentPredictionSystem(memory_hours=72)

def initialize_prediction_system(initial_games):
    """Initialize prediction system with initial data"""
    logging.info("🧠 Initializing prediction system...")
    
    for game in initial_games:
        prediction_system.add_game_data(game)
    
    logging.info(f"✅ Prediction system ready! {len(initial_games)} games loaded")
    stats = prediction_system.get_system_stats()
    logging.info(f"📊 Distribution: {stats['class_distribution']}")
    
    # Make an initial prediction
    initial_prediction = prediction_system.predict_next_game()
    logging.info(f"🤖 Initial prediction: {initial_prediction['prediction']} with {initial_prediction['confidence']*100:.1f}% confidence")

def get_live_prediction():
    """Get live prediction"""
    return prediction_system.predict_next_game()

def update_with_new_game(game_data):
    """Update with new game"""
    prediction_system.add_game_data(game_data)
    
    # Update accuracy for pending predictions
    actual_category = prediction_system._get_category(game_data['total'])
    result = prediction_system.update_accuracy(
        actual_category, 
        game_data.get('game_id'),
        game_data.get('timestamp')
    )
    
    # If no pending predictions, make a new one
    if result is None:
        # Make new prediction after game
        new_prediction = prediction_system.predict_next_game()
        logging.info(f"🎯 New prediction after game {game_data['game_id']}: {new_prediction['prediction']}")
    
    return actual_category

def get_prediction_stats():
    """Get prediction statistics"""
    return prediction_system.get_system_stats()

# ==================== ORIGINAL FUNCTIONS ====================

def log_debug(message):
    """Debug logging"""
    if ENABLE_DEBUG:
        logging.debug(f"🔍 {message}")

def get_api_endpoint(page=0, size=100):
    """API endpoint creation for 72-hour data with cache busting"""
    timestamp = int(time.time() * 1000)
    return f"/svc-evolution-game-events/api/lightningdice?page={page}&size={size}&sort=data.settledAt,desc&duration=72&totals=3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18&isLightningMultiplierMatched=false&_={timestamp}"

def get_total_pages():
    """Get total number of pages available from API"""
    try:
        url = API_BASE_URL + get_api_endpoint(0, 1)
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            total_count = response.headers.get('X-Total-Count')
            if total_count:
                total_games = int(total_count)
                games_per_page = 100
                total_pages = (total_games + games_per_page - 1) // games_per_page
                logging.info(f"📊 API reports {total_games} total games available")
                return min(total_pages, 100)
            
            link_header = response.headers.get('Link', '')
            if 'rel="last"' in link_header:
                match = re.search(r'page=(\d+)&size=\d+>; rel="last"', link_header)
                if match:
                    last_page = int(match.group(1))
                    logging.info(f"📊 API reports page {last_page} as last page")
                    return min(last_page + 1, 100)
        
        logging.info(f"⚠️ Could not determine total pages, using default")
        return 67
        
    except Exception as e:
        logging.error(f"❌ Error getting total pages: {e}")
        return 67

def parse_timestamp(timestamp_str):
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
        log_debug(f"Timestamp parse error: {e}, string: {timestamp_str}")
        return datetime.now()

def validate_game_data(game_data):
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
        log_debug(f"Validation error: {e}")
        return False

def extract_game_data(game, page_num, idx):
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
        
        timestamp = parse_timestamp(settled_at)
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
            log_debug(f"Number conversion error: {e}")
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
            'total_winners': total_winners
        }
        
        if not validate_game_data(game_dict):
            return None
        
        return game_dict
        
    except Exception as e:
        log_debug(f"Game extraction error: {e}")
        return None

def fetch_single_page(page, size):
    """Fetch a single page of games"""
    try:
        url = API_BASE_URL + get_api_endpoint(page, size)
        
        log_debug(f"Fetching: {url}")
        
        response = requests.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code != 200:
            log_debug(f"API error {response.status_code}: {response.text[:100]}")
            return []
        
        data = response.json()
        
        if not isinstance(data, list):
            log_debug(f"Unexpected response type: {type(data)}")
            return []
        
        log_debug(f"Received {len(data)} raw games from page {page}")
        
        valid_games = []
        invalid_count = 0
        
        for idx, game in enumerate(data[:size]):
            game_data = extract_game_data(game, page, idx)
            if game_data:
                valid_games.append(game_data)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            logging.warning(f"⚠️ Page {page}: {invalid_count}/{len(data)} invalid games detected")
        
        return valid_games
        
    except requests.exceptions.Timeout:
        log_debug(f"Timeout fetching page {page}")
        return []
    except Exception as e:
        log_debug(f"Error fetching page {page}: {e}")
        return []

def get_mock_game_data(count=100):
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
            'total_winners': random.randint(1, 200)
        }
        mock_games.append(game_data)
    
    mock_games.sort(key=lambda x: x['timestamp'], reverse=True)
    return mock_games

def fetch_initial_data():
    """Fetch ALL 72-hour game data on startup"""
    logging.info("📥 Fetching COMPLETE 72-hour game data from API...")
    
    all_games = []
    
    if USE_MOCK_DATA:
        logging.info("🔄 Using mock data for initial load")
        all_games = get_mock_game_data(6667)
        return all_games
    
    try:
        total_pages = get_total_pages()
        logging.info(f"📊 Will fetch {total_pages} pages (approx {total_pages * 100} games)")
        
        for page in range(total_pages):
            try:
                print(f"📄 Loading page {page+1}/{total_pages}...", end="\r")
                
                page_games = fetch_single_page(page, 100)
                
                if page_games:
                    all_games.extend(page_games)
                    
                    if (page + 1) % 10 == 0 or page == total_pages - 1:
                        logging.info(f"📄 Loaded page {page+1}/{total_pages}: {len(all_games)} games total")
                else:
                    logging.warning(f"⚠️ No games on page {page+1}")
                
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"❌ Error on page {page+1}: {e}")
        
        print(f"\n✅ Fetched {len(all_games)} games from API")
        
    except Exception as e:
        logging.error(f"❌ Error in initial fetch: {e}")
        logging.info("🔄 Falling back to limited data...")
        
        for page in range(20):
            try:
                page_games = fetch_single_page(page, 100)
                if page_games:
                    all_games.extend(page_games)
                time.sleep(0.1)
            except:
                break
        
        if not all_games:
            logging.info("⚠️ No games fetched, using mock data")
            all_games = get_mock_game_data(2000)
    
    all_games.sort(key=lambda x: x['timestamp'], reverse=True)
    
    logging.info(f"🎉 Successfully loaded {len(all_games)} games from 72-hour history")
    
    if all_games:
        oldest = all_games[-1]['date']
        newest = all_games[0]['date']
        oldest_time = all_games[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        newest_time = all_games[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"📅 Date range: {oldest} to {newest}")
        logging.info(f"🕒 Time range: {oldest_time} to {newest_time}")
        
        totals = [g['total'] for g in all_games]
        logging.info(f"📈 Total range: {min(totals)}-{max(totals)}, Average: {sum(totals)/len(totals):.2f}")
    
    return all_games

def fetch_new_games():
    """Fetch only new games"""
    global all_games_data, last_fetch_time, is_fetching
    
    if is_fetching:
        return []
    
    is_fetching = True
    new_games = []
    
    try:
        log_debug("Checking for new games...")
        
        start_time = time.time()
        page_games = fetch_single_page(0, 50)
        
        if USE_MOCK_DATA and not page_games:
            current_time = datetime.now()
            dice1 = random.randint(1, 6)
            dice2 = random.randint(1, 6)
            dice3 = random.randint(1, 6)
            total = dice1 + dice2 + dice3
            
            dice_map = {1: '⚀', 2: '⚁', 3: '⚂', 4: '⚃', 5: '⚄', 6: '⚅'}
            dice_icons = f"{dice_map.get(dice1, '⚀')}{dice_map.get(dice2, '⚁')}{dice_map.get(dice3, '⚂')}"
            
            mock_game = {
                'game_hash': hashlib.md5(f"new_mock_{current_time.timestamp()}".encode()).hexdigest(),
                'game_id': f"new_mock_{int(current_time.timestamp())}",
                'total': total,
                'dice1': dice1,
                'dice2': dice2,
                'dice3': dice3,
                'dice_icons': dice_icons,
                'settled_at': current_time.isoformat() + 'Z',
                'time': current_time.strftime('%H:%M:%S'),
                'date': current_time.strftime('%Y-%m-%d'),
                'timestamp': current_time,
                'total_amount': random.randint(100, 1000),
                'total_winners': random.randint(1, 50)
            }
            page_games = [mock_game]
        
        fetch_time = time.time() - start_time
        
        if not page_games:
            log_debug("No new games found")
            is_fetching = False
            return []
        
        log_debug(f"Fetched {len(page_games)} games in {fetch_time:.2f}s")
        
        with DATA_LOCK:
            existing_hashes = set(item['game_hash'] for item in all_games_data)
            
            for game in page_games:
                if game['game_hash'] not in existing_hashes:
                    new_games.append(game)
            
            if new_games:
                log_debug(f"Found {len(new_games)} new games!")
                
                # Add new games to the beginning
                all_games_data = new_games + all_games_data
                
                # Limit total games
                if len(all_games_data) > MAX_GAMES_TO_KEEP:
                    all_games_data = all_games_data[:MAX_GAMES_TO_KEEP]
                
                # Keep sorted by timestamp
                all_games_data.sort(key=lambda x: x['timestamp'], reverse=True)
                
                last_fetch_time = datetime.now()
                
                # 🧠 Update prediction system with new games
                for game in new_games:
                    update_with_new_game(game)
                
                for i, game in enumerate(new_games[:3]):
                    log_debug(f"  🎲 New {i+1}: Total={game['total']}, Dice={game['dice1']},{game['dice2']},{game['dice3']}, Time={game['time']}")
        
    except Exception as e:
        logging.error(f"Error in fetch_new_games: {e}")
    
    is_fetching = False
    return new_games

def background_fetcher():
    """Background fetcher thread"""
    logging.info("🔄 Starting background fetcher...")
    
    fetch_count = 0
    
    while True:
        try:
            fetch_count += 1
            new_games = fetch_new_games()
            
            current_time = datetime.now().strftime('%H:%M:%S')
            with DATA_LOCK:
                game_count = len(all_games_data)
            
            if fetch_count % 10 == 0:
                logging.info(f"📊 [{current_time}] Total games: {game_count}, Fetch #{fetch_count}")
            
            time.sleep(FETCH_INTERVAL)
            
        except Exception as e:
            logging.error(f"❌ Background fetcher error: {e}")
            time.sleep(10)

def start_background_fetcher():
    """Start the background fetcher"""
    global fetcher_started
    
    if not fetcher_started:
        with DATA_LOCK:
            if not fetcher_started:
                logging.info(f"\n⚡ Starting background fetcher (checking every {FETCH_INTERVAL} seconds)")
                threading.Thread(target=background_fetcher, daemon=True).start()
                fetcher_started = True

# ==================== API ROUTES ====================

@app.route('/')
def index():
    """Main page"""
    start_background_fetcher()
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API status check"""
    start_background_fetcher()
    
    with DATA_LOCK:
        total_games = len(all_games_data)
        latest_game = all_games_data[0] if all_games_data else None
    
    uptime_seconds = (datetime.now() - system_start_time).total_seconds()
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"
    
    latest_info = None
    if latest_game:
        latest_info = {
            'total': latest_game['total'],
            'dice': [latest_game['dice1'], latest_game['dice2'], latest_game['dice3']],
            'dice_icons': latest_game['dice_icons'],
            'time': latest_game['time'],
            'date': latest_game['date']
        }
    
    # Get AI stats
    ai_stats = get_prediction_stats()
    
    return jsonify({
        'success': True,
        'status': 'online',
        'server': 'Lightning Dice Tracker - AI Prediction System',
        'total_games': total_games,
        'latest_game': latest_info,
        'last_update': last_fetch_time.strftime('%H:%M:%S') if last_fetch_time else '--:--:--',
        'server_time': datetime.now().strftime('%H:%M:%S'),
        'uptime': uptime_str,
        'fetch_interval': f'{FETCH_INTERVAL} seconds',
        'max_capacity': MAX_GAMES_TO_KEEP,
        'memory_usage': f"{total_games}/{MAX_GAMES_TO_KEEP}",
        'api_mode': 'MOCK' if USE_MOCK_DATA else 'REAL',
        'connection_status': 'online',
        'data_range': '72 hours (complete)',
        'ai_status': ai_stats['system_status'],
        'ai_accuracy': ai_stats['accuracy_percentage']
    })

@app.route('/api/get-games')
def get_games():
    """Get all games data"""
    start_background_fetcher()
    
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    
    with DATA_LOCK:
        total_games = len(all_games_data)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        if start_idx >= total_games:
            page_games = []
        else:
            page_games = all_games_data[start_idx:end_idx]
    
    distribution = defaultdict(int)
    with DATA_LOCK:
        for item in all_games_data:
            try:
                if 3 <= item['total'] <= 18:
                    distribution[item['total']] += 1
            except:
                continue
    
    return jsonify({
        'success': True,
        'games': page_games,
        'total_count': len(page_games),
        'total_in_memory': total_games,
        'distribution': dict(sorted(distribution.items())),
        'last_fetch': last_fetch_time.strftime('%H:%M:%S') if last_fetch_time else '--:--:--',
        'next_check_in': FETCH_INTERVAL
    })

@app.route('/api/get-stats')
def get_stats():
    """Get detailed stats"""
    start_background_fetcher()
    
    with DATA_LOCK:
        total_games = len(all_games_data)
        valid_games = [g for g in all_games_data if 3 <= g['total'] <= 18]
    
    stats = {
        'total_games': total_games,
        'valid_games': len(valid_games),
        'invalid_games': total_games - len(valid_games),
        'fetch_interval': f'{FETCH_INTERVAL} seconds',
        'max_games': MAX_GAMES_TO_KEEP,
        'storage_used': f"{total_games}/{MAX_GAMES_TO_KEEP}",
        'last_update': last_fetch_time.strftime('%Y-%m-%d %H:%M:%S') if last_fetch_time else None,
        'api_mode': 'MOCK' if USE_MOCK_DATA else 'REAL',
        'data_range': '72 hours (complete)',
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if valid_games:
        totals = [g['total'] for g in valid_games]
        stats.update({
            'min_total': min(totals),
            'max_total': max(totals),
            'avg_total': round(sum(totals) / len(valid_games), 2),
            'date_range': f"{all_games_data[-1]['date']} to {all_games_data[0]['date']}" if all_games_data else "N/A"
        })
    
    distribution = defaultdict(int)
    with DATA_LOCK:
        for item in all_games_data:
            try:
                if 3 <= item['total'] <= 18:
                    distribution[item['total']] += 1
            except:
                continue
    
    if distribution:
        most_common = max(distribution.items(), key=lambda x: x[1])
        stats['most_common_total'] = most_common[0]
        stats['most_common_count'] = most_common[1]
    
    return jsonify({
        'success': True,
        'stats': stats,
        'distribution': dict(sorted(distribution.items()))
    })

@app.route('/api/predict-next')
def predict_next():
    """Get prediction for next game"""
    start_background_fetcher()
    
    try:
        prediction = get_live_prediction()
        stats = get_prediction_stats()
        
        with DATA_LOCK:
            latest_game = all_games_data[0] if all_games_data else None
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'stats': stats,
            'latest_game': latest_game,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'prediction_id': prediction['prediction_id']
        })
    except Exception as e:
        logging.error(f"Error in predict-next: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'prediction': {
                'prediction': 'LOW',
                'confidence': 0.5,
                'reason': 'System initializing...'
            }
        })

@app.route('/api/prediction-stats')
def prediction_statistics():
    """Get prediction system statistics"""
    start_background_fetcher()
    
    stats = get_prediction_stats()
    
    # Calculate total summary
    all_predictions = prediction_system.predictions_history
    evaluated_predictions = [p for p in all_predictions if p.get('status') in ['correct', 'incorrect']]
    
    total_predictions = len(all_predictions)
    total_evaluated = len(evaluated_predictions)
    total_correct = sum(1 for p in evaluated_predictions if p.get('is_correct', False))
    total_incorrect = total_evaluated - total_correct
    overall_accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
    
    return jsonify({
        'success': True,
        'stats': stats,
        'total_summary': {
            'all_predictions': total_predictions,
            'evaluated_predictions': total_evaluated,
            'correct_predictions': total_correct,
            'incorrect_predictions': total_incorrect,
            'overall_accuracy': round(overall_accuracy, 1),
            'pending_predictions': len(prediction_system.get_pending_predictions())
        },
        'system_info': {
            'version': '1.0',
            'status': 'active',
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_games_in_memory': len(all_games_data)
        }
    })

@app.route('/api/prediction-history')
def prediction_history():
    """Get prediction history"""
    start_background_fetcher()
    
    limit = int(request.args.get('limit', 50))
    include_pending = request.args.get('include_pending', 'false').lower() == 'true'
    
    history = prediction_system.get_prediction_history(limit, include_pending)
    
    return jsonify({
        'success': True,
        'history': history,
        'count': len(history),
        'pending_count': len(prediction_system.get_pending_predictions())
    })

@app.route('/api/prediction-performance')
def prediction_performance():
    """Get detailed prediction performance"""
    start_background_fetcher()
    
    stats = get_prediction_stats()
    
    # Get detailed performance breakdown
    history = prediction_system.get_prediction_history(100, include_pending=False)
    
    performance = {
        'total_predictions': stats['total_predictions'],
        'correct_predictions': stats['correct_predictions'],
        'accuracy': stats['accuracy_percentage'],
        'recent_accuracy': stats['recent_accuracy'],
        'category_accuracy': stats['category_accuracy'],
        
        # Recent performance
        'last_10': {'total': 0, 'correct': 0, 'accuracy': 0},
        'last_50': {'total': 0, 'correct': 0, 'accuracy': 0},
        
        # Time-based performance
        'performance_by_hour': defaultdict(lambda: {'total': 0, 'correct': 0}),
        
        # Streak analysis
        'current_streak': 0,
        'longest_streak': 0,
        'current_wrong_streak': 0
    }
    
    # Calculate recent performance
    if history:
        last_10 = history[-10:] if len(history) >= 10 else history
        last_50 = history[-50:] if len(history) >= 50 else history
        
        performance['last_10']['total'] = len(last_10)
        performance['last_10']['correct'] = sum(1 for p in last_10 if p.get('is_correct', False))
        performance['last_10']['accuracy'] = (performance['last_10']['correct'] / performance['last_10']['total'] * 100) if performance['last_10']['total'] > 0 else 0
        
        performance['last_50']['total'] = len(last_50)
        performance['last_50']['correct'] = sum(1 for p in last_50 if p.get('is_correct', False))
        performance['last_50']['accuracy'] = (performance['last_50']['correct'] / performance['last_50']['total'] * 100) if performance['last_50']['total'] > 0 else 0
        
        # Calculate streaks
        current_streak = 0
        longest_streak = 0
        current_wrong_streak = 0
        
        for pred in reversed(history):
            if pred.get('is_correct', False):
                current_streak += 1
                current_wrong_streak = 0
                if current_streak > longest_streak:
                    longest_streak = current_streak
            else:
                current_streak = 0
                current_wrong_streak += 1
        
        performance['current_streak'] = current_streak
        performance['longest_streak'] = longest_streak
        performance['current_wrong_streak'] = current_wrong_streak
        
        # Time-based performance
        for pred in history:
            if 'timestamp' in pred:
                try:
                    if isinstance(pred['timestamp'], str):
                        pred_time = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                    else:
                        pred_time = pred['timestamp']
                    
                    hour = pred_time.hour
                    performance['performance_by_hour'][hour]['total'] += 1
                    if pred.get('is_correct', False):
                        performance['performance_by_hour'][hour]['correct'] += 1
                except:
                    continue
    
    # Convert defaultdict to dict
    performance['performance_by_hour'] = dict(sorted(performance['performance_by_hour'].items()))
    
    return jsonify({
        'success': True,
        'overall': stats,
        'performance': performance
    })

@app.route('/api/games-with-predictions')
def games_with_predictions():
    """Get games with prediction data - MAIN ENDPOINT FOR TABLE"""
    start_background_fetcher()
    
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    
    with DATA_LOCK:
        total_games = len(all_games_data)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        if start_idx >= total_games:
            page_games = []
        else:
            page_games = all_games_data[start_idx:end_idx]
    
    # Get predictions history
    predictions = prediction_system.get_prediction_history(200, include_pending=False)
    
    # Match predictions with games
    games_with_preds = []
    matched_count = 0
    
    for game in page_games:
        game_with_pred = game.copy()
        
        # Try to find matching prediction
        best_match = None
        best_time_diff = float('inf')
        
        for pred in predictions:
            pred_time = pred.get('timestamp')
            game_time = game['timestamp']
            
            if isinstance(pred_time, str):
                try:
                    pred_time = datetime.fromisoformat(pred_time.replace('Z', '+00:00'))
                except:
                    continue
            
            if isinstance(pred_time, datetime) and isinstance(game_time, datetime):
                time_diff = abs((pred_time - game_time).total_seconds())
                
                # Match if within 2 minutes (for real-time prediction)
                if time_diff < 120 and time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_match = pred
        
        if best_match:
            # Add prediction info to game
            game_with_pred['ai_prediction'] = best_match['prediction']
            game_with_pred['ai_confidence'] = best_match['confidence']
            game_with_pred['ai_reason'] = best_match['reason']
            game_with_pred['prediction_time'] = best_match.get('timestamp')
            game_with_pred['prediction_id'] = best_match.get('prediction_id')
            
            # Check if prediction was correct
            actual_category = prediction_system._get_category(game['total'])
            game_with_pred['actual_category'] = actual_category
            game_with_pred['prediction_correct'] = best_match['prediction'] == actual_category
            game_with_pred['prediction_evaluated'] = best_match.get('status') in ['correct', 'incorrect']
            
            matched_count += 1
        else:
            # No prediction found for this game
            game_with_pred['ai_prediction'] = None
            game_with_pred['actual_category'] = prediction_system._get_category(game['total'])
            game_with_pred['prediction_correct'] = None
        
        games_with_preds.append(game_with_pred)
    
    return jsonify({
        'success': True,
        'games': games_with_preds,
        'total_count': len(page_games),
        'predictions_matched': matched_count,
        'page': page,
        'per_page': per_page,
        'total_pages': (total_games + per_page - 1) // per_page,
        'total_games': total_games,
        'matched_percentage': (matched_count / len(page_games) * 100) if page_games else 0
    })

# ✅ নতুন API এন্ডপয়েন্ট: টোটাল prediction summary
@app.route('/api/total-prediction-summary')
def total_prediction_summary():
    """Get TOTAL prediction summary (all predictions, not page-based)"""
    start_background_fetcher()
    
    # Get ALL predictions from AI system
    all_predictions = prediction_system.predictions_history
    
    # Filter only evaluated predictions (correct/incorrect)
    evaluated_predictions = [p for p in all_predictions if p.get('status') in ['correct', 'incorrect']]
    
    # Calculate totals
    total_all_predictions = len(all_predictions)
    total_evaluated = len(evaluated_predictions)
    total_correct = sum(1 for p in evaluated_predictions if p.get('is_correct', False))
    total_incorrect = total_evaluated - total_correct
    
    # Calculate accuracy
    overall_accuracy = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
    
    # Get pending predictions
    pending_predictions = len(prediction_system.get_pending_predictions())
    
    # Get AI stats for additional info
    ai_stats = get_prediction_stats()
    
    return jsonify({
        'success': True,
        'summary': {
            'total_predictions': total_all_predictions,  # সব predictions
            'evaluated_predictions': total_evaluated,    # evaluated predictions
            'correct_predictions': total_correct,        # সঠিক predictions
            'incorrect_predictions': total_incorrect,    # ভুল predictions
            'overall_accuracy': round(overall_accuracy, 1),  # সামগ্রিক accuracy
            'pending_predictions': pending_predictions,  # pending predictions
            'total_games_analyzed': ai_stats.get('total_games_analyzed', 0),
            'system_status': ai_stats.get('system_status', 'active')
        },
        'timestamp': datetime.now().isoformat()
    })

# ✅ নতুন API এন্ডপয়েন্ট: টোটাল-ভিত্তিক স্ট্যাটস
@app.route('/api/total-prediction-stats')
def total_prediction_stats():
    """Get total prediction statistics (not page-based)"""
    start_background_fetcher()
    
    with DATA_LOCK:
        total_games = len(all_games_data)
    
    # Get AI stats
    ai_stats = get_prediction_stats()
    
    # Calculate total match rate
    total_predictions_made = ai_stats['total_predictions']
    match_rate = (total_predictions_made / total_games * 100) if total_games > 0 else 0
    
    # Total accuracy (from AI system)
    total_accuracy = ai_stats['accuracy_percentage']
    
    return jsonify({
        'success': True,
        'total_games': total_games,
        'total_predictions': total_predictions_made,
        'match_rate': round(match_rate, 1),
        'accuracy': total_accuracy,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/get-all-data')
def get_all_data():
    """Get complete dataset (for analysis)"""
    start_background_fetcher()
    
    limit = int(request.args.get('limit', 1000))
    
    with DATA_LOCK:
        if limit == 0:
            games_data = all_games_data
        else:
            games_data = all_games_data[:limit]
    
    return jsonify({
        'success': True,
        'total_games': len(games_data),
        'games': games_data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/quick-check')
def quick_check():
    """Quick check for new games"""
    start_background_fetcher()
    
    new_games = fetch_new_games()
    
    with DATA_LOCK:
        recent_games = all_games_data[:20]
    
    return jsonify({
        'success': True,
        'new_games_count': len(new_games),
        'new_games': new_games[:5],
        'recent_games': recent_games,
        'total_in_memory': len(all_games_data),
        'last_fetch': last_fetch_time.strftime('%H:%M:%S.%f')[:-3] if last_fetch_time else '--:--:--',
        'check_time': datetime.now().strftime('%H:%M:%S')
    })

@app.route('/api/force-refresh')
def force_refresh():
    """Force refresh"""
    start_background_fetcher()
    
    new_games = fetch_new_games()
    
    with DATA_LOCK:
        total_games = len(all_games_data)
    
    return jsonify({
        'success': True,
        'message': f'Refresh completed. {total_games} games in memory.',
        'new_games': len(new_games),
        'total_games': total_games,
        'last_fetch': datetime.now().strftime('%H:%M:%S')
    })

@app.before_request
def before_request():
    """Before each request"""
    start_background_fetcher()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 LIGHTNING DICE TRACKER - AI PREDICTION SYSTEM")
    print("="*60)
    print("📊 FEATURES: 72-hour data, Real-time AI Prediction")
    print("🧠 INTELLIGENT: Self-learning, Pattern Recognition")
    print("="*60)
    
    print("📥 Loading COMPLETE 72-hour game data from API...")
    initial_games = fetch_initial_data()
    
    with DATA_LOCK:
        all_games_data = initial_games
        print(f"✅ Loaded {len(initial_games)} games from 72-hour history")
    
    # Initialize prediction system
    initialize_prediction_system(initial_games)
    
    # Show prediction system stats
    stats = get_prediction_stats()
    print(f"🧠 AI System Ready: {stats['total_games_analyzed']} games analyzed")
    print(f"🎯 Prediction Accuracy: {stats.get('accuracy_percentage', 0)}%")
    print(f"📊 Distribution: {stats.get('class_distribution', {})}")
    
    print("\n" + "="*60)
    print("✅ AI PREDICTION SYSTEM READY!")
    print("="*60)
    print(f"🎮 Games in memory: {len(all_games_data)}")
    print(f"🧠 AI Analysis: {stats['total_games_analyzed']} games")
    print(f"🎯 Recent Accuracy: {stats.get('recent_accuracy', 0)}%")
    print(f"⚡ Auto-refresh: Every {FETCH_INTERVAL} seconds")
    print(f"🌐 New API Endpoints:")
    print(f"   /api/games-with-predictions - Main table with AI predictions")
    print(f"   /api/prediction-performance - Performance analytics")
    print(f"   /api/total-prediction-summary - Total prediction summary (NEW)")  # নতুন এন্ডপয়েন্ট
    print(f"   /api/total-prediction-stats - Total-based stats")
    print(f"🌍 Server: http://localhost:8083")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=8083, debug=False, threaded=True)