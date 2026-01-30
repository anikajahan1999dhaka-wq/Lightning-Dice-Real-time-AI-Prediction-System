"""
🎲 Lightning Dice AI Prediction System - Main Application
With enhanced AI tracking and analysis APIs - FIXED VERSION
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import threading
import time
import logging
from collections import deque, defaultdict
import os
import sys
import numpy as np

# Import modules
from data_fetcher import data_fetcher
from ai_model import ai_system
from grid_system import grid_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
USE_MOCK_DATA = False  # Set to True for testing, False for production
FETCH_INTERVAL = 5  # seconds
MAX_GAMES = 50000
PORT = int(os.environ.get('PORT', 5000))

# Global variables
all_games = deque(maxlen=MAX_GAMES)
last_fetch_time = None
is_fetching = False
fetcher_started = False
system_start_time = datetime.now()
current_view = "all"

# Data fetcher configuration
if hasattr(data_fetcher, 'use_mock'):
    data_fetcher.use_mock = USE_MOCK_DATA

# Helper functions
def get_game_summary(game):
    """Safely get game summary string"""
    try:
        total = game.get('total', '?')
        category = game.get('category', 'UNKNOWN')
        
        # Get first character of category safely
        if isinstance(category, str) and category:
            category_char = category[0]
        else:
            category_char = 'U'
            
        return f"{total}({category_char})"
    except Exception:
        return "?"

def get_category_distribution(games_list, limit=1000):
    """Get category distribution from games list"""
    categories = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
    
    for game in games_list[:limit]:
        try:
            cat = game.get('category')
            if cat in categories:
                categories[cat] += 1
            elif cat:
                # Handle unexpected categories
                categories['LOW'] += 1
        except (AttributeError, KeyError, TypeError):
            continue
    
    return categories

# Background fetcher thread
def background_fetcher():
    """Background thread for fetching data"""
    global all_games, last_fetch_time, is_fetching
    
    logger.info("🔄 Starting 24/7 background fetcher...")
    fetch_count = 0
    
    while True:
        try:
            if is_fetching:
                time.sleep(1)
                continue
            
            is_fetching = True
            fetch_count += 1
            
            # Log every 100 fetches
            if fetch_count % 100 == 0:
                logger.info(f"📊 Fetch #{fetch_count} - System running for {fetch_count*FETCH_INTERVAL/60:.1f} minutes")
            
            # Fetch latest games
            new_games = data_fetcher.fetch_latest_games(limit=20)
            
            if new_games and isinstance(new_games, list):
                # Filter out duplicates
                existing_hashes = set()
                try:
                    existing_hashes = {g.get('game_hash', '') for g in all_games if g and isinstance(g, dict)}
                except Exception as hash_error:
                    logger.debug(f"Hash collection error: {hash_error}")
                
                unique_new_games = []
                
                for game in new_games:
                    if not isinstance(game, dict):
                        continue
                    
                    game_hash = game.get('game_hash')
                    if game_hash and game_hash not in existing_hashes:
                        unique_new_games.append(game)
                
                if unique_new_games:
                    # Add to beginning (newest first)
                    for game in reversed(unique_new_games):
                        all_games.appendleft(game)
                    
                    # Keep within max limit
                    while len(all_games) > MAX_GAMES:
                        all_games.pop()
                    
                    # Update AI system with new games
                    for game in unique_new_games:
                        try:
                            ai_system.add_game(game)
                        except Exception as ai_error:
                            logger.debug(f"AI add game error: {ai_error}")
                    
                    # Update grid system
                    try:
                        games_list = list(all_games)
                        grid_system.update_games(games_list)
                    except Exception as grid_error:
                        logger.debug(f"Grid update error: {grid_error}")
                    
                    last_fetch_time = datetime.now()
                    
                    logger.info(f"🎮 Added {len(unique_new_games)} new games. Total: {len(all_games)}")
                    
                    # Show notification for new games
                    if unique_new_games:
                        summaries = []
                        for g in unique_new_games[:3]:
                            summaries.append(get_game_summary(g))
                        
                        if summaries:
                            logger.info(f"✨ New games: {', '.join(summaries)}")
            
            # Periodic AI retraining (every 500 fetches ~ 4 hours)
            if fetch_count % 500 == 0 and len(all_games) > 1000:
                logger.info("🧠 Periodic AI retraining...")
                try:
                    games_list = list(all_games)
                    ai_system.train_on_72h_data(games_list)
                    logger.info(f"✅ AI retrained on {len(games_list)} games")
                except Exception as train_error:
                    logger.error(f"❌ AI retraining error: {train_error}")
            
            # Periodic logging
            if fetch_count % 20 == 0:
                logger.info(f"📊 Fetch #{fetch_count}, Total games: {len(all_games)}")
                
                # Show category distribution
                if all_games:
                    try:
                        games_list = list(all_games)
                        categories = get_category_distribution(games_list[:100])
                        
                        total = sum(categories.values())
                        if total > 0:
                            logger.info(f"📈 Recent distribution: "
                                      f"LOW: {categories['LOW']/total*100:.1f}%, "
                                      f"MIDDLE: {categories['MIDDLE']/total*100:.1f}%, "
                                      f"HIGH: {categories['HIGH']/total*100:.1f}%")
                    except Exception as dist_error:
                        logger.debug(f"Distribution error: {dist_error}")
            
            is_fetching = False
            time.sleep(FETCH_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ Fetcher error (will continue): {e}")
            is_fetching = False
            time.sleep(10)  # Wait longer on error

def start_fetcher():
    """Start background fetcher thread"""
    global fetcher_started
    
    if not fetcher_started:
        threading.Thread(target=background_fetcher, daemon=True).start()
        fetcher_started = True
        logger.info(f"⚡ Background fetcher started (interval: {FETCH_INTERVAL}s)")

# Initialize system on startup
def initialize_system():
    """Initialize the complete system - UPDATED"""
    logger.info("="*60)
    logger.info("🚀 Initializing Lightning Dice AI System...")
    logger.info("="*60)
    
    # Step 1: Fetch 72-hour data
    logger.info("📥 Fetching 72-hour historical data...")
    
    try:
        historical_games = data_fetcher.fetch_72h_data()
        
        if historical_games and isinstance(historical_games, list):
            # Store games (newest first)
            for game in historical_games:
                if isinstance(game, dict):
                    all_games.appendleft(game)
            
            logger.info(f"✅ Loaded {len(historical_games)} historical games")
            
            # Show sample data
            if historical_games:
                logger.info("📋 Sample games:")
                for i, game in enumerate(list(all_games)[:3]):
                    try:
                        logger.info(f"   Game {i+1}: Total={game.get('total', '?')}, "
                                  f"Category={game.get('category', 'UNKNOWN')}, "
                                  f"Time={game.get('time', '--:--:--')}")
                    except Exception:
                        pass
            
            # Show statistics
            try:
                games_list = list(all_games)
                categories = get_category_distribution(games_list[:1000])
                
                total_cats = sum(categories.values())
                if total_cats > 0:
                    logger.info(f"📊 Category stats (first 1000 games):")
                    logger.info(f"   LOW: {categories['LOW']} ({categories['LOW']/total_cats*100:.1f}%)")
                    logger.info(f"   MIDDLE: {categories['MIDDLE']} ({categories['MIDDLE']/total_cats*100:.1f}%)")
                    logger.info(f"   HIGH: {categories['HIGH']} ({categories['HIGH']/total_cats*100:.1f}%)")
            except Exception as stat_error:
                logger.debug(f"Stat error: {stat_error}")
        
        else:
            logger.warning("⚠️ No historical games loaded or invalid data format")
            
    except Exception as e:
        logger.error(f"❌ Error loading historical data: {e}")
    
    # Step 2: Train AI on historical data - FORCE TRAINING
    if len(all_games) > 100:
        logger.info(f"🧠 Training AI on {len(all_games)} historical games...")
        
        try:
            # Convert deque to list for training
            games_list = list(all_games)
            
            # Check if AI is already trained
            if not ai_system.is_trained:
                ai_system.train_on_72h_data(games_list)
                
                # Get AI stats after training
                ai_stats = ai_system.get_ai_stats()
                logger.info(f"✅ AI Training Complete!")
                logger.info(f"   Patterns learned: LOW={ai_stats.get('low_patterns_count', 0)}, "
                          f"MIDDLE={ai_stats.get('middle_patterns_count', 0)}, "
                          f"HIGH={ai_stats.get('high_patterns_count', 0)}")
                logger.info(f"   Training accuracy: {ai_stats.get('training_accuracy', 0)*100:.1f}%")
                
                # Make a test prediction
                test_prediction = ai_system.predict_next()
                logger.info(f"   Test prediction: {test_prediction.get('prediction', '?')} "
                          f"({test_prediction.get('confidence', 0)*100:.1f}% confidence)")
                logger.info(f"   System used: {test_prediction.get('system_used', 'unknown')}")
            else:
                logger.info("🤖 AI is already trained")
                ai_stats = ai_system.get_ai_stats()
                logger.info(f"   Current accuracy: {ai_stats.get('training_accuracy', 0)*100:.1f}%")
                logger.info(f"   Patterns: L={ai_stats.get('low_patterns_count', 0)}, "
                          f"M={ai_stats.get('middle_patterns_count', 0)}, "
                          f"H={ai_stats.get('high_patterns_count', 0)}")
            
        except Exception as e:
            logger.error(f"❌ AI training error: {e}")
            logger.info("🔄 Attempting alternative training method...")
            
            # Alternative training method
            try:
                # Train with smaller batch
                games_list = list(all_games)[:1000]  # First 1000 games
                ai_system.train_on_72h_data(games_list)
                logger.info(f"✅ Alternative training successful on {len(games_list)} games")
            except Exception as e2:
                logger.error(f"❌ Alternative training also failed: {e2}")
    else:
        logger.warning(f"⚠️ Insufficient data for AI training ({len(all_games)} games)")
    
    # Step 3: Initialize grid system
    logger.info("📊 Initializing grid system...")
    
    try:
        games_list = list(all_games)
        grid_system.update_games(games_list)
        grid_stats = grid_system.get_grid_stats()
        logger.info(f"✅ Grid initialized: {grid_stats.get('filled_cells', 0)}/"
                  f"{grid_stats.get('total_cells', 300)} cells filled "
                  f"({grid_stats.get('fill_percentage', 0):.1f}%)")
        
    except Exception as e:
        logger.error(f"❌ Grid initialization error: {e}")
    
    # Step 4: Make initial AI prediction
    try:
        if ai_system.is_trained:
            logger.info("🔮 Making initial AI prediction...")
            prediction = ai_system.predict_next()
            if prediction and isinstance(prediction, dict):
                logger.info(f"   Prediction: {prediction.get('prediction', 'UNKNOWN')} "
                          f"with {prediction.get('confidence', 0)*100:.1f}% confidence")
                logger.info(f"   Reason: {prediction.get('reason', 'No reason provided')}")
    except Exception as e:
        logger.error(f"❌ Initial prediction error: {e}")
    
    logger.info("="*60)
    logger.info("🎉 System initialization complete!")
    logger.info("="*60)
    logger.info(f"📊 Total Games: {len(all_games)}")
    logger.info(f"🤖 AI Trained: {ai_system.is_trained}")
    logger.info(f"⚡ Auto-refresh: {FETCH_INTERVAL} seconds")
    logger.info(f"🌍 Server: http://localhost:{PORT}")
    logger.info("="*60)
    
    # Step 5: Start background fetcher
    start_fetcher()

# API Routes
@app.route('/')
def index():
    """Main page"""
    start_fetcher()
    return render_template('index.html')

@app.route('/ai-analysis')
def ai_analysis():
    """AI Analysis Dashboard"""
    start_fetcher()
    return render_template('ai_analysis.html')

@app.route('/api/status')
def api_status():
    """System status"""
    global last_fetch_time
    
    try:
        total_games = len(all_games)
        latest_game = list(all_games)[0] if all_games else None
        
        # Calculate uptime
        uptime = datetime.now() - system_start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"
        
        # Get AI stats
        ai_stats = ai_system.get_ai_stats()
        grid_stats = grid_system.get_grid_stats()
        
        # Get category distribution
        categories = get_category_distribution(list(all_games)[:1000])
        
        # Prepare latest game info
        latest_info = None
        if latest_game and isinstance(latest_game, dict):
            latest_info = {
                'total': latest_game.get('total', 0),
                'category': latest_game.get('category', 'UNKNOWN'),
                'time': latest_game.get('time', '--:--:--'),
                'date': latest_game.get('date', '----:--:--'),
                'dice_icons': latest_game.get('dice_icons', '⚀⚁⚂'),
                'dice1': latest_game.get('dice1', 1),
                'dice2': latest_game.get('dice2', 1),
                'dice3': latest_game.get('dice3', 1)
            }
        
        return jsonify({
            'success': True,
            'status': 'online',
            'system': 'Lightning Dice AI Prediction System',
            'version': '2.0',
            'total_games': total_games,
            'latest_game': latest_info,
            'category_distribution': categories,
            'last_update': last_fetch_time.strftime('%H:%M:%S') if last_fetch_time else '--:--:--',
            'uptime': uptime_str,
            'fetch_interval': FETCH_INTERVAL,
            'ai_status': 'trained' if ai_stats.get('is_trained', False) else 'training',
            'ai_accuracy': ai_stats.get('prediction_accuracy', 0) * 100,
            'ai_patterns': ai_stats.get('low_patterns_count', 0) + 
                          ai_stats.get('middle_patterns_count', 0) + 
                          ai_stats.get('high_patterns_count', 0),
            'grid_status': f"{grid_stats.get('filled_cells', 0)}/{grid_stats.get('total_cells', 300)} cells",
            'grid_percentage': grid_stats.get('fill_percentage', 0),
            'api_mode': 'MOCK' if USE_MOCK_DATA else 'REAL',
            'server_time': datetime.now().strftime('%H:%M:%S'),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Status API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/ai-analysis')
def get_ai_analysis():
    """Get detailed AI analysis with better error handling"""
    try:
        # Check if AI system is properly initialized
        if not hasattr(ai_system, 'get_detailed_analysis'):
            return jsonify({
                'success': False,
                'error': 'AI system not properly initialized',
                'analysis': {},
                'timestamp': datetime.now().isoformat()
            })
        
        detailed_analysis = ai_system.get_detailed_analysis()
        ai_stats = ai_system.get_ai_stats()
        
        return jsonify({
            'success': True,
            'analysis': detailed_analysis,
            'stats': ai_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ AI Analysis API error: {e}")
        
        # Provide fallback data
        fallback_analysis = {
            'pattern_analysis': {
                'top_low_patterns': [],
                'top_middle_patterns': [],
                'top_high_patterns': [],
                'pattern_stats': [],
                'total_patterns_learned': 0
            },
            'prediction_history': [],
            'hourly_performance': [],
            'recent_debug_log': [f"Error: {str(e)}"]
        }
        
        fallback_stats = {
            'total_games_analyzed': len(ai_system.games_history) if hasattr(ai_system, 'games_history') else 0,
            'is_trained': ai_system.is_trained if hasattr(ai_system, 'is_trained') else False,
            'training_accuracy': 0.0,
            'prediction_accuracy': 0.0,
            'low_patterns_count': 0,
            'middle_patterns_count': 0,
            'high_patterns_count': 0
        }
        
        return jsonify({
            'success': False,
            'error': str(e),
            'analysis': fallback_analysis,
            'stats': fallback_stats,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/train-ai')
def train_ai_manual():
    """Manually trigger AI training"""
    try:
        if len(all_games) < 100:
            return jsonify({
                'success': False,
                'error': f'Insufficient data: {len(all_games)} games (need at least 100)',
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info("🧠 Manual AI training triggered...")
        games_list = list(all_games)
        
        # Train AI
        ai_system.train_on_72h_data(games_list)
        
        # Get stats
        ai_stats = ai_system.get_ai_stats()
        
        # Make a test prediction
        test_prediction = ai_system.predict_next()
        
        return jsonify({
            'success': True,
            'message': f'AI trained on {len(games_list)} games',
            'training_accuracy': ai_stats.get('training_accuracy', 0) * 100,
            'patterns_learned': ai_stats.get('low_patterns_count', 0) + 
                              ai_stats.get('middle_patterns_count', 0) + 
                              ai_stats.get('high_patterns_count', 0),
            'test_prediction': test_prediction,
            'is_trained': ai_system.is_trained,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Manual AI training error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/prediction-history')
def get_prediction_history():
    """Get prediction history"""
    try:
        # Get recent predictions with detailed info
        ai_stats = ai_system.get_ai_stats()
        
        # Get prediction history from AI system
        prediction_history = []
        if hasattr(ai_system, 'prediction_history'):
            pred_list = list(ai_system.prediction_history)
            
            # Get last 100 predictions
            for pred in pred_list[-100:]:
                if isinstance(pred, dict):
                    pred_copy = pred.copy()
                    
                    # Ensure all required fields exist
                    if 'timestamp' not in pred_copy:
                        pred_copy['timestamp'] = datetime.now().isoformat()
                    if 'prediction_id' not in pred_copy:
                        pred_copy['prediction_id'] = 'N/A'
                    
                    prediction_history.append(pred_copy)
        
        # Calculate accuracy stats
        checked_predictions = [p for p in prediction_history if 'correct' in p]
        accuracy_stats = {
            'total_checked': len(checked_predictions),
            'correct': sum(1 for p in checked_predictions if p.get('correct', False)),
            'incorrect': sum(1 for p in checked_predictions if not p.get('correct', True)),
            'accuracy': 0
        }
        
        if accuracy_stats['total_checked'] > 0:
            accuracy_stats['accuracy'] = accuracy_stats['correct'] / accuracy_stats['total_checked'] * 100
        
        return jsonify({
            'success': True,
            'predictions': prediction_history,
            'accuracy_stats': accuracy_stats,
            'total_predictions': len(prediction_history),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction History API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'predictions': [],
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/ai-patterns')
def get_ai_patterns():
    """Get AI patterns"""
    try:
        # Get patterns from AI system
        top_patterns = []
        
        if hasattr(ai_system, 'low_patterns'):
            for pattern, stats in list(ai_system.low_patterns.items())[:20]:
                if isinstance(stats, dict):
                    # Use .get() to avoid KeyError
                    next_low = stats.get('next_low', 0)
                    next_other = stats.get('next_other', 0)
                    total = next_low + next_other
                    
                    if total > 0:
                        top_patterns.append({
                            'pattern': pattern,
                            'type': 'LOW',
                            'count': stats.get('count', 0),
                            'effectiveness': (next_low / total * 100) if total > 0 else 0,
                            'next_low': next_low,
                            'total_next': total,
                            'last_seen': stats.get('last_seen', 'Unknown')
                        })
        
        if hasattr(ai_system, 'middle_patterns'):
            for pattern, stats in list(ai_system.middle_patterns.items())[:20]:
                if isinstance(stats, dict):
                    next_middle = stats.get('next_middle', 0)
                    next_other = stats.get('next_other', 0)
                    total = next_middle + next_other
                    
                    if total > 0:
                        top_patterns.append({
                            'pattern': pattern,
                            'type': 'MIDDLE',
                            'count': stats.get('count', 0),
                            'effectiveness': (next_middle / total * 100) if total > 0 else 0,
                            'next_middle': next_middle,
                            'total_next': total,
                            'last_seen': stats.get('last_seen', 'Unknown')
                        })
        
        if hasattr(ai_system, 'high_patterns'):
            for pattern, stats in list(ai_system.high_patterns.items())[:20]:
                if isinstance(stats, dict):
                    next_high = stats.get('next_high', 0)
                    next_other = stats.get('next_other', 0)
                    total = next_high + next_other
                    
                    if total > 0:
                        top_patterns.append({
                            'pattern': pattern,
                            'type': 'HIGH',
                            'count': stats.get('count', 0),
                            'effectiveness': (next_high / total * 100) if total > 0 else 0,
                            'next_high': next_high,
                            'total_next': total,
                            'last_seen': stats.get('last_seen', 'Unknown')
                        })
        
        # Sort by effectiveness
        top_patterns.sort(key=lambda x: x['effectiveness'], reverse=True)
        
        return jsonify({
            'success': True,
            'patterns': top_patterns[:50],  # Top 50 patterns
            'total_patterns': len(ai_system.low_patterns) + len(ai_system.middle_patterns) + len(ai_system.high_patterns),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ AI Patterns API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'patterns': [],
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/ai-timing-stats')
def get_ai_timing_stats():
    """Get AI timing statistics"""
    try:
        ai_stats = ai_system.get_ai_stats()
        timing_stats = ai_stats.get('timing_stats', {})
        
        # Calculate more detailed timing stats
        detailed_timing = {
            'low': {
                'avg': timing_stats.get('avg_low_interval', 0),
                'last': timing_stats.get('last_low_minutes', 0),
                'status': 'due' if timing_stats.get('last_low_minutes', 0) > timing_stats.get('avg_low_interval', 0) * 1.5 else 'normal',
                'intervals_count': len(ai_system.timing_stats['low_intervals']) if hasattr(ai_system, 'timing_stats') else 0
            },
            'middle': {
                'avg': timing_stats.get('avg_middle_interval', 0),
                'last': timing_stats.get('last_middle_minutes', 0),
                'status': 'due' if timing_stats.get('last_middle_minutes', 0) > timing_stats.get('avg_middle_interval', 0) * 1.5 else 'normal',
                'intervals_count': len(ai_system.timing_stats['middle_intervals']) if hasattr(ai_system, 'timing_stats') else 0
            },
            'high': {
                'avg': timing_stats.get('avg_high_interval', 0),
                'last': timing_stats.get('last_high_minutes', 0),
                'status': 'due' if timing_stats.get('last_high_minutes', 0) > timing_stats.get('avg_high_interval', 0) * 1.5 else 'normal',
                'intervals_count': len(ai_system.timing_stats['high_intervals']) if hasattr(ai_system, 'timing_stats') else 0
            }
        }
        
        return jsonify({
            'success': True,
            'timing_stats': detailed_timing,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ AI Timing Stats API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timing_stats': {},
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/grid-data')
def get_grid():
    """Get grid data"""
    start_fetcher()
    
    try:
        grid_data = grid_system.get_grid_data()
        grid_stats = grid_system.get_grid_stats()
        
        # Get recent games for context
        recent_games = list(all_games)[:20] if all_games else []
        
        return jsonify({
            'success': True,
            'grid_data': grid_data,
            'grid_stats': grid_stats,
            'recent_games': recent_games,
            'total_games': len(all_games),
            'current_view': current_view,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Grid API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load grid data',
            'grid_data': {},
            'grid_stats': {},
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/ai-prediction')
def get_ai_prediction():
    """Get AI prediction for next game"""
    start_fetcher()
    
    try:
        prediction = ai_system.predict_next()
        ai_stats = ai_system.get_ai_stats()
        
        # Get prediction history (last 5 predictions)
        prediction_history = []
        if len(all_games) >= 5:
            recent_games = list(all_games)[:5]
            for i, game in enumerate(recent_games):
                if isinstance(game, dict):
                    prediction_history.append({
                        'position': i + 1,
                        'total': game.get('total', 0),
                        'category': game.get('category', 'UNKNOWN'),
                        'time': game.get('time', '--:--:--'),
                        'date': game.get('date', '----:--:--'),
                        'dice_icons': game.get('dice_icons', '⚀⚁⚂')
                    })
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'ai_stats': ai_stats,
            'prediction_history': prediction_history,
            'games_analyzed': len(all_games),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ AI Prediction API error: {e}")
        return jsonify({
            'success': False,
            'error': 'AI prediction failed',
            'prediction': {
                'prediction': 'LOW',
                'confidence': 0.5,
                'reason': 'System error - using fallback',
                'probabilities': {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39},
                'system_used': 'fallback_error'
            },
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/category-data')
def get_category_data():
    """Get grid data filtered by category"""
    start_fetcher()
    
    try:
        category = request.args.get('category', 'all')
        if category not in ['all', 'LOW', 'MIDDLE', 'HIGH']:
            category = 'all'
        
        global current_view
        current_view = category
        
        grid_data = grid_system.get_category_grid(category)
        grid_stats = grid_system.get_grid_stats()
        
        # Get category-specific stats
        category_stats = {}
        if category != 'all':
            total_games = len(all_games)
            category_count = sum(1 for g in all_games if isinstance(g, dict) and g.get('category') == category)
            category_stats = {
                'total_in_system': category_count,
                'percentage_in_system': (category_count / total_games * 100) if total_games > 0 else 0,
                'in_grid': len(grid_data),
                'grid_percentage': (len(grid_data) / grid_stats.get('total_cells', 300) * 100) if grid_stats.get('total_cells', 300) > 0 else 0
            }
        
        return jsonify({
            'success': True,
            'category': category,
            'grid_data': grid_data,
            'grid_stats': grid_stats,
            'category_stats': category_stats,
            'count': len(grid_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Category API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load category data',
            'category': request.args.get('category', 'all'),
            'grid_data': {},
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/recent-games')
def get_recent_games():
    """Get recent games"""
    start_fetcher()
    
    try:
        limit = int(request.args.get('limit', 20))
        limit = min(limit, 100)  # Max 100 games
        
        recent = list(all_games)[:limit]
        
        # Calculate some stats
        category_counts = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        for game in recent:
            if isinstance(game, dict):
                cat = game.get('category', 'LOW')
                if cat in category_counts:
                    category_counts[cat] += 1
        
        return jsonify({
            'success': True,
            'games': recent,
            'count': len(recent),
            'category_counts': category_counts,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Recent games API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load recent games',
            'games': [],
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/search')
def search_games():
    """Search games"""
    start_fetcher()
    
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})
        
        # Search in all games
        results = []
        query_lower = query.lower()
        
        for game in all_games:
            if not isinstance(game, dict):
                continue
            
            # Search in various fields
            if (query_lower in str(game.get('total', '')).lower() or
                query_lower in str(game.get('game_id', '')).lower() or
                query_lower in str(game.get('category', '')).lower() or
                query_lower in str(game.get('time', '')).lower()):
                
                # Find cell position for this game
                cell_id = None
                grid_data = grid_system.get_grid_data()
                game_hash = game.get('game_hash')
                
                if game_hash and isinstance(grid_data, dict):
                    for cell, cell_data in grid_data.items():
                        if isinstance(cell_data, dict) and cell_data.get('game_hash') == game_hash:
                            cell_id = cell
                            break
                
                results.append({
                    'game': game,
                    'cell_id': cell_id,
                    'match_type': 'total' if query_lower in str(game.get('total', '')).lower() else 
                                  'category' if query_lower in str(game.get('category', '')).lower() else
                                  'time' if query_lower in str(game.get('time', '')).lower() else 'game_id'
                })
        
        # Limit results
        results = results[:50]
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Search API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Search failed',
            'query': request.args.get('q', ''),
            'results': [],
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/force-refresh')
def force_refresh():
    """Force refresh data"""
    start_fetcher()
    
    try:
        # Fetch new games
        new_games = data_fetcher.fetch_latest_games(limit=50)
        new_count = 0
        
        if new_games and isinstance(new_games, list):
            existing_hashes = set()
            try:
                existing_hashes = {g.get('game_hash', '') for g in all_games if isinstance(g, dict)}
            except Exception:
                pass
            
            unique_new_games = []
            for game in new_games:
                if isinstance(game, dict):
                    game_hash = game.get('game_hash')
                    if game_hash and game_hash not in existing_hashes:
                        unique_new_games.append(game)
            
            if unique_new_games:
                # Add to beginning
                for game in reversed(unique_new_games):
                    all_games.appendleft(game)
                
                # Keep within limit
                while len(all_games) > MAX_GAMES:
                    all_games.pop()
                
                # Update systems
                for game in unique_new_games:
                    try:
                        ai_system.add_game(game)
                    except Exception:
                        pass
                
                try:
                    grid_system.update_games(list(all_games))
                except Exception:
                    pass
                
                new_count = len(unique_new_games)
        
        return jsonify({
            'success': True,
            'message': 'Force refresh completed',
            'new_games': new_count,
            'total_games': len(all_games),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Force refresh error: {e}")
        return jsonify({
            'success': False,
            'error': 'Force refresh failed',
            'message': 'Force refresh failed',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/ai-stats')
def get_ai_statistics():
    """Get detailed AI statistics"""
    start_fetcher()
    
    try:
        ai_stats = ai_system.get_ai_stats()
        
        # Calculate recent accuracy
        checked_predictions = [p for p in list(ai_system.prediction_history) if 'correct' in p] if hasattr(ai_system, 'prediction_history') else []
        recent_accuracy = 0.0
        if checked_predictions:
            recent_correct = sum(1 for p in checked_predictions[-20:] if p.get('correct', False))
            recent_total = min(20, len(checked_predictions))
            recent_accuracy = recent_correct / recent_total if recent_total > 0 else 0
        
        # Get pattern examples
        pattern_examples = {'low_patterns': [], 'middle_patterns': [], 'high_patterns': []}
        
        try:
            if hasattr(ai_system, 'low_patterns'):
                pattern_examples['low_patterns'] = list(ai_system.low_patterns.keys())[:5]
        except:
            pass
        
        try:
            if hasattr(ai_system, 'middle_patterns'):
                pattern_examples['middle_patterns'] = list(ai_system.middle_patterns.keys())[:5]
        except:
            pass
        
        try:
            if hasattr(ai_system, 'high_patterns'):
                pattern_examples['high_patterns'] = list(ai_system.high_patterns.keys())[:5]
        except:
            pass
        
        return jsonify({
            'success': True,
            'ai_stats': ai_stats,
            'performance': {
                'recent_accuracy': recent_accuracy * 100,
                'overall_accuracy': ai_stats.get('prediction_accuracy', 0) * 100,
                'games_trained_on': ai_stats.get('total_games_analyzed', 0),
                'prediction_count': ai_stats.get('prediction_count', 0),
                'checked_predictions': ai_stats.get('checked_predictions', 0),
                'correct_predictions': ai_stats.get('correct_predictions', 0),
                'learning_rate': ai_system.learning_rate if hasattr(ai_system, 'learning_rate') else 0.1
            },
            'pattern_examples': pattern_examples,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ AI Stats API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load AI stats',
            'ai_stats': {},
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/system-info')
def get_system_info():
    """Get complete system information"""
    start_fetcher()
    
    try:
        # Get all stats
        ai_stats = ai_system.get_ai_stats()
        grid_stats = grid_system.get_grid_stats()
        
        # Get category distribution
        categories = get_category_distribution(list(all_games)[:1000])
        
        # Calculate percentages
        total_cats = sum(categories.values())
        if total_cats > 0:
            category_percentages = {
                'LOW': categories['LOW'] / total_cats * 100,
                'MIDDLE': categories['MIDDLE'] / total_cats * 100,
                'HIGH': categories['HIGH'] / total_cats * 100
            }
        else:
            category_percentages = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        
        # Get system memory info
        memory_info = {
            'games_stored': len(all_games),
            'max_games': MAX_GAMES,
            'memory_usage_percent': (len(all_games) / MAX_GAMES) * 100 if MAX_GAMES > 0 else 0,
            'python_version': sys.version.split()[0],
            'platform': sys.platform
        }
        
        # Get oldest and newest game times
        newest_time = None
        oldest_time = None
        
        if all_games:
            try:
                newest_game = list(all_games)[0]
                oldest_game = list(all_games)[-1]
                
                if isinstance(newest_game, dict):
                    newest_time = newest_game.get('timestamp')
                    if newest_time and isinstance(newest_time, datetime):
                        newest_time = newest_time.isoformat()
                
                if isinstance(oldest_game, dict):
                    oldest_time = oldest_game.get('timestamp')
                    if oldest_time and isinstance(oldest_time, datetime):
                        oldest_time = oldest_time.isoformat()
            except Exception:
                pass
        
        return jsonify({
            'success': True,
            'system': {
                'name': 'Lightning Dice AI Prediction System',
                'version': '2.0',
                'status': 'online',
                'uptime': str(datetime.now() - system_start_time),
                'start_time': system_start_time.isoformat(),
                'current_time': datetime.now().isoformat()
            },
            'data': {
                'total_games': len(all_games),
                'category_distribution': categories,
                'category_percentages': category_percentages,
                'newest_game_time': newest_time,
                'oldest_game_time': oldest_time
            },
            'ai': ai_stats,
            'grid': grid_stats,
            'memory': memory_info,
            'configuration': {
                'fetch_interval': FETCH_INTERVAL,
                'use_mock_data': USE_MOCK_DATA,
                'port': PORT,
                'max_games': MAX_GAMES
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ System Info API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load system info',
            'system': {},
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/test-connection')
def test_connection():
    """Test API connection"""
    try:
        # Test the API connection
        test_games = data_fetcher.fetch_latest_games(limit=1)
        
        return jsonify({
            'success': True,
            'api_working': isinstance(test_games, list) and len(test_games) > 0,
            'games_received': len(test_games) if isinstance(test_games, list) else 0,
            'api_mode': 'MOCK' if USE_MOCK_DATA else 'REAL',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Test connection error: {e}")
        return jsonify({
            'success': False,
            'api_working': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint for Railway"""
    try:
        # Check if system is running properly
        is_healthy = (
            len(all_games) > 0 and
            hasattr(ai_system, 'games_history') and
            len(ai_system.games_history) > 0
        )
        
        # Calculate uptime
        uptime = datetime.now() - system_start_time
        uptime_hours = uptime.total_seconds() / 3600
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': round(uptime_hours, 2),
            'games_count': len(all_games),
            'ai_trained': ai_system.is_trained,
            'ai_patterns_count': len(ai_system.low_patterns) + len(ai_system.middle_patterns) + len(ai_system.high_patterns),
            'memory_usage': 'normal',
            'api_mode': 'MOCK' if USE_MOCK_DATA else 'REAL',
            'system_start_time': system_start_time.isoformat(),
            'version': '2.0'
        }), 200 if is_healthy else 503
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.before_request
def before_request():
    """Ensure fetcher is running before each request"""
    start_fetcher()

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False, 
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"❌ Server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"❌ Unhandled exception: {error}")
    return jsonify({
        'success': False,
        'error': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Initialize system
    initialize_system()
    
    # Start Flask app
    logger.info(f"🌍 Starting Flask server on port {PORT}...")
    
    # Production settings
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=debug_mode,
        threaded=True,
        use_reloader=False
    )
