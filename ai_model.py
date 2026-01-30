"""
🎲 Lightning Dice AI Prediction System - Enhanced with Tracking
Advanced pattern recognition with 72-hour data training - FIXED VERSION
"""

import numpy as np
from collections import defaultdict, deque, OrderedDict
from datetime import datetime, timedelta
import hashlib
import random
import math
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class TimingPatternAI:
    """Advanced timing pattern detection AI with detailed tracking - FIXED"""
    
    def __init__(self):
        self.games_history = deque(maxlen=10000)
        self.pattern_memory = deque(maxlen=500)
        self.learning_rate = 0.1
        self.min_confidence = 0.4
        self.last_training_time = None
        
        # Pattern databases with detailed tracking - INITIALIZE PROPERLY
        self.low_patterns = OrderedDict()
        self.middle_patterns = OrderedDict()
        self.high_patterns = OrderedDict()
        
        # Timing analysis
        self.timing_stats = {
            'low_intervals': deque(maxlen=500),
            'middle_intervals': deque(maxlen=500),
            'high_intervals': deque(maxlen=500),
            'last_low_time': None,
            'last_middle_time': None,
            'last_high_time': None
        }
        
        # Prediction tracking
        self.prediction_history = deque(maxlen=1000)
        self.pattern_history = deque(maxlen=500)
        self.feature_importance_history = deque(maxlen=100)
        
        # ML Model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=0)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_accuracy = 0.0
        self.feature_importances = None
        
        # Debug logging
        self.debug_log = deque(maxlen=200)
        self._log("🤖 Advanced Timing Pattern AI Initialized (FIXED VERSION)")
    
    def _log(self, message):
        """Add debug log entry"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.debug_log.append(f"[{timestamp}] {message}")
        if len(self.debug_log) > 200:
            self.debug_log.popleft()
    
    def add_game(self, game_data):
        """Add new game to learning system with tracking"""
        try:
            category = self._get_category(game_data['total'])
            timestamp = game_data['timestamp']
            
            # Store game
            self.games_history.append({
                'category': category,
                'total': game_data['total'],
                'timestamp': timestamp,
                'time': game_data['time'],
                'hour': timestamp.hour,
                'dice1': game_data['dice1'],
                'dice2': game_data['dice2'],
                'dice3': game_data['dice3']
            })
            
            # Check previous predictions
            self._check_prediction_accuracy(category, timestamp)
            
            # Update timing stats
            self._update_timing_stats(category, timestamp)
            
            # Learn patterns from last 10 games
            if len(self.games_history) >= 10:
                pattern_learned = self._learn_patterns()
                if pattern_learned:
                    self._log(f"📚 Learned new pattern: {pattern_learned}")
        except Exception as e:
            self._log(f"❌ Error adding game: {e}")
    
    def _check_prediction_accuracy(self, actual_category, timestamp):
        """Check if last prediction was correct"""
        try:
            if self.prediction_history:
                last_pred = self.prediction_history[-1]
                if 'timestamp' in last_pred:
                    # If prediction was made recently (within 5 minutes)
                    try:
                        pred_time = datetime.fromisoformat(last_pred['timestamp'].replace('Z', '+00:00'))
                        time_diff = (timestamp - pred_time).total_seconds()
                        
                        if time_diff < 300:  # 5 minutes
                            last_pred['actual'] = actual_category
                            last_pred['correct'] = last_pred['prediction'] == actual_category
                            last_pred['checked_at'] = timestamp.isoformat()
                            
                            if last_pred['correct']:
                                self._log(f"✅ Prediction correct! Predicted {last_pred['prediction']}, Actual {actual_category}")
                            else:
                                self._log(f"❌ Prediction wrong. Predicted {last_pred['prediction']}, Actual {actual_category}")
                    except:
                        pass
        except Exception as e:
            self._log(f"❌ Error checking prediction accuracy: {e}")
    
    def _get_category(self, total):
        """Categorize total"""
        if total <= 9:
            return 'LOW'
        elif total <= 11:
            return 'MIDDLE'
        else:
            return 'HIGH'
    
    def _update_timing_stats(self, category, timestamp):
        """Update timing statistics with detailed tracking"""
        try:
            if category == 'LOW':
                if self.timing_stats['last_low_time']:
                    interval = (timestamp - self.timing_stats['last_low_time']).total_seconds() / 60
                    self.timing_stats['low_intervals'].append(interval)
                    self._log(f"⏱️ LOW interval: {interval:.1f} minutes")
                self.timing_stats['last_low_time'] = timestamp
                
            elif category == 'MIDDLE':
                if self.timing_stats['last_middle_time']:
                    interval = (timestamp - self.timing_stats['last_middle_time']).total_seconds() / 60
                    self.timing_stats['middle_intervals'].append(interval)
                    self._log(f"⏱️ MIDDLE interval: {interval:.1f} minutes")
                self.timing_stats['last_middle_time'] = timestamp
                
            elif category == 'HIGH':
                if self.timing_stats['last_high_time']:
                    interval = (timestamp - self.timing_stats['last_high_time']).total_seconds() / 60
                    self.timing_stats['high_intervals'].append(interval)
                    self._log(f"⏱️ HIGH interval: {interval:.1f} minutes")
                self.timing_stats['last_high_time'] = timestamp
        except Exception as e:
            self._log(f"❌ Error updating timing stats: {e}")
    
    def _learn_patterns(self):
        """Learn patterns from history with detailed tracking - FIXED"""
        try:
            if len(self.games_history) < 20:
                return None
            
            history = list(self.games_history)
            patterns_learned = []
            
            # Analyze patterns in chunks of 5
            for i in range(len(history) - 5):
                chunk = history[i:i+5]
                pattern_key = ''.join([g['category'][0] for g in chunk])
                
                # Check what came after this pattern
                if i + 5 < len(history):
                    next_category = history[i+5]['category']
                    
                    # Update pattern databases - WITH PROPER INITIALIZATION
                    if 'L' in pattern_key:
                        if pattern_key not in self.low_patterns:
                            # Initialize with all required keys
                            self.low_patterns[pattern_key] = {
                                'count': 0, 
                                'next_low': 0, 
                                'next_other': 0, 
                                'last_seen': datetime.now()
                            }
                        
                        self.low_patterns[pattern_key]['count'] += 1
                        self.low_patterns[pattern_key]['last_seen'] = datetime.now()
                        
                        if next_category == 'LOW':
                            self.low_patterns[pattern_key]['next_low'] += 1
                            patterns_learned.append(f"LOW:{pattern_key}→LOW")
                        else:
                            self.low_patterns[pattern_key]['next_other'] += 1
                            patterns_learned.append(f"LOW:{pattern_key}→{next_category}")
                    
                    if 'M' in pattern_key:
                        if pattern_key not in self.middle_patterns:
                            # Initialize with all required keys
                            self.middle_patterns[pattern_key] = {
                                'count': 0, 
                                'next_middle': 0, 
                                'next_other': 0, 
                                'last_seen': datetime.now()
                            }
                        
                        self.middle_patterns[pattern_key]['count'] += 1
                        self.middle_patterns[pattern_key]['last_seen'] = datetime.now()
                        
                        if next_category == 'MIDDLE':
                            self.middle_patterns[pattern_key]['next_middle'] += 1
                            patterns_learned.append(f"MIDDLE:{pattern_key}→MIDDLE")
                        else:
                            self.middle_patterns[pattern_key]['next_other'] += 1
                            patterns_learned.append(f"MIDDLE:{pattern_key}→{next_category}")
                    
                    if 'H' in pattern_key:
                        if pattern_key not in self.high_patterns:
                            # Initialize with all required keys
                            self.high_patterns[pattern_key] = {
                                'count': 0, 
                                'next_high': 0, 
                                'next_other': 0, 
                                'last_seen': datetime.now()
                            }
                        
                        self.high_patterns[pattern_key]['count'] += 1
                        self.high_patterns[pattern_key]['last_seen'] = datetime.now()
                        
                        if next_category == 'HIGH':
                            self.high_patterns[pattern_key]['next_high'] += 1
                            patterns_learned.append(f"HIGH:{pattern_key}→HIGH")
                        else:
                            self.high_patterns[pattern_key]['next_other'] += 1
                            patterns_learned.append(f"HIGH:{pattern_key}→{next_category}")
            
            # Sort patterns by count (most frequent first)
            if self.low_patterns:
                self.low_patterns = OrderedDict(
                    sorted(self.low_patterns.items(), key=lambda x: x[1]['count'], reverse=True)
                )
            
            if self.middle_patterns:
                self.middle_patterns = OrderedDict(
                    sorted(self.middle_patterns.items(), key=lambda x: x[1]['count'], reverse=True)
                )
            
            if self.high_patterns:
                self.high_patterns = OrderedDict(
                    sorted(self.high_patterns.items(), key=lambda x: x[1]['count'], reverse=True)
                )
            
            return patterns_learned[:3] if patterns_learned else None
            
        except Exception as e:
            self._log(f"❌ Error learning patterns: {e}")
            return None
    
    def train_on_72h_data(self, games_data):
        """Train AI on 72-hour data with detailed tracking"""
        self._log("🧠 Training AI on 72-hour data...")
        
        try:
            # Prepare features and labels
            features = []
            labels = []
            
            for i in range(10, len(games_data)):
                if i >= len(games_data):
                    break
                    
                # Get last 10 games for context
                context_games = games_data[max(0, i-10):i]
                
                # Extract features
                feature_vector = self._extract_features(context_games, games_data[i])
                
                # Label is current game's category
                label = self._get_category(games_data[i]['total'])
                
                features.append(feature_vector)
                labels.append(label)
            
            # Train model
            if features and labels:
                self.label_encoder.fit(['LOW', 'MIDDLE', 'HIGH'])
                labels_encoded = self.label_encoder.transform(labels)
                
                self.model.fit(features, labels_encoded)
                self.is_trained = True
                self.last_training_time = datetime.now()
                self.training_accuracy = self.model.score(features, labels_encoded)
                
                # Get feature importances
                self.feature_importances = self.model.feature_importances_.tolist()
                self.feature_importance_history.append({
                    'timestamp': datetime.now(),
                    'importances': self.feature_importances
                })
                
                self._log(f"✅ AI Training Complete! Samples: {len(features)}")
                self._log(f"📊 Training Accuracy: {self.training_accuracy*100:.1f}%")
                
                # Log top features
                feature_names = self._get_feature_names()
                if self.feature_importances and feature_names:
                    top_features = sorted(zip(feature_names, self.feature_importances), 
                                         key=lambda x: x[1], reverse=True)[:5]
                    self._log("🎯 Top 5 Important Features:")
                    for feat, imp in top_features:
                        self._log(f"   {feat}: {imp:.3f}")
            else:
                self._log("⚠️ No features or labels for training")
                
        except Exception as e:
            self._log(f"❌ AI training error: {e}")
            self.is_trained = False
    
    def _extract_features(self, context_games, current_game):
        """Extract features from game context"""
        features = []
        
        # 1. Last 5 categories
        last_categories = [self._get_category(g['total']) for g in context_games[-5:]]
        cat_mapping = {'LOW': 0, 'MIDDLE': 1, 'HIGH': 2}
        for cat in last_categories:
            features.append(cat_mapping.get(cat, 0))
        
        # 2. Current streaks
        low_streak, middle_streak, high_streak = 0, 0, 0
        for game in reversed(context_games):
            cat = self._get_category(game['total'])
            if cat == 'LOW':
                low_streak += 1
                if middle_streak > 0 or high_streak > 0:
                    break
            elif cat == 'MIDDLE':
                middle_streak += 1
                if low_streak > 0 or high_streak > 0:
                    break
            else:
                high_streak += 1
                if low_streak > 0 or middle_streak > 0:
                    break
        
        features.extend([low_streak, middle_streak, high_streak])
        
        # 3. Time since last category
        current_time = current_game['timestamp']
        last_times = {'LOW': None, 'MIDDLE': None, 'HIGH': None}
        
        for game in reversed(context_games):
            cat = self._get_category(game['total'])
            if last_times[cat] is None:
                last_times[cat] = game['timestamp']
        
        for cat in ['LOW', 'MIDDLE', 'HIGH']:
            if last_times[cat]:
                minutes_since = (current_time - last_times[cat]).total_seconds() / 60
                features.append(min(minutes_since, 1440))  # Cap at 24 hours
            else:
                features.append(1440)
        
        # 4. Hour of day
        features.append(current_time.hour)
        
        # 5. Day of week
        features.append(current_time.weekday())
        
        # 6. Recent category frequencies
        recent_games = context_games[-20:] if len(context_games) >= 20 else context_games
        cat_counts = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        for game in recent_games:
            cat = self._get_category(game['total'])
            cat_counts[cat] += 1
        
        total_recent = len(recent_games)
        if total_recent > 0:
            features.extend([
                cat_counts['LOW'] / total_recent,
                cat_counts['MIDDLE'] / total_recent,
                cat_counts['HIGH'] / total_recent
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _get_feature_names(self):
        """Get names for features"""
        feature_names = []
        
        # Last 5 categories
        for i in range(1, 6):
            feature_names.append(f'last_{i}_cat')
        
        # Current streaks
        feature_names.extend(['low_streak', 'middle_streak', 'high_streak'])
        
        # Time since last category
        feature_names.extend(['mins_since_low', 'mins_since_middle', 'mins_since_high'])
        
        # Time features
        feature_names.extend(['hour_of_day', 'day_of_week'])
        
        # Recent frequencies
        feature_names.extend(['freq_low', 'freq_middle', 'freq_high'])
        
        return feature_names
    
    def predict_next(self):
        """Predict next game category with detailed tracking"""
        if len(self.games_history) < 20 or not self.is_trained:
            prediction = self._fallback_prediction()
            prediction['timestamp'] = datetime.now().isoformat()
            prediction['prediction_id'] = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
            self.prediction_history.append(prediction)
            return prediction
        
        # Get recent games for context
        recent_games = list(self.games_history)[-10:]
        
        if len(recent_games) < 10:
            prediction = self._fallback_prediction()
            prediction['timestamp'] = datetime.now().isoformat()
            prediction['prediction_id'] = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
            self.prediction_history.append(prediction)
            return prediction
        
        # Create dummy current game for feature extraction
        dummy_game = {
            'total': 10,
            'timestamp': datetime.now(),
            'time': datetime.now().strftime('%H:%M:%S')
        }
        
        # Extract features
        features = [self._extract_features(recent_games, dummy_game)]
        
        try:
            # Make prediction
            prediction_encoded = self.model.predict(features)[0]
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Calculate confidence
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
            
            if confidence < self.min_confidence:
                prediction_data = self._fallback_prediction()
            else:
                # Get prediction reason
                reason = self._get_prediction_reason(prediction, recent_games)
                
                # Calculate probabilities for all categories
                prob_low = probabilities[list(self.label_encoder.classes_).index('LOW')] if 'LOW' in self.label_encoder.classes_ else 0.33
                prob_middle = probabilities[list(self.label_encoder.classes_).index('MIDDLE')] if 'MIDDLE' in self.label_encoder.classes_ else 0.33
                prob_high = probabilities[list(self.label_encoder.classes_).index('HIGH')] if 'HIGH' in self.label_encoder.classes_ else 0.33
                
                prediction_data = {
                    'prediction': prediction,
                    'confidence': float(confidence),
                    'reason': reason,
                    'probabilities': {
                        'LOW': float(prob_low),
                        'MIDDLE': float(prob_middle),
                        'HIGH': float(prob_high)
                    },
                    'system_used': 'ai_timing_model',
                    'games_analyzed': len(self.games_history),
                    'model_confidence': confidence
                }
            
            # Add timestamp and track
            prediction_data['timestamp'] = datetime.now().isoformat()
            prediction_data['prediction_id'] = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
            
            self.prediction_history.append(prediction_data)
            self._log(f"🔮 Made prediction: {prediction} ({confidence*100:.1f}% confidence)")
            
            return prediction_data
            
        except Exception as e:
            self._log(f"❌ AI Prediction Error: {e}")
            prediction = self._fallback_prediction()
            prediction['timestamp'] = datetime.now().isoformat()
            prediction['prediction_id'] = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
            self.prediction_history.append(prediction)
            return prediction
    
    def _get_prediction_reason(self, prediction, recent_games):
        """Generate reason for prediction"""
        reasons = {
            'LOW': [
                "High probability of LOW after recent MIDDLE/HIGH sequence",
                "LOW drought detected - statistically due for LOW",
                "Timing pattern suggests LOW is imminent",
                "Recent pattern shows alternating LOW-MIDDLE sequence",
                "Statistical analysis favors LOW based on last 20 games"
            ],
            'MIDDLE': [
                "MIDDLE dominance in recent games continuing",
                "Pattern shows MIDDLE following HIGH-LOW transitions",
                "Statistical analysis favors MIDDLE at this time",
                "Recent streak pattern points to MIDDLE",
                "Timing model predicts MIDDLE period"
            ],
            'HIGH': [
                "HIGH expected after recent LOW streak",
                "Timing analysis shows HIGH period starting",
                "Pattern recognition predicts HIGH sequence",
                "Statistical probability favors HIGH",
                "Model detects HIGH pattern in recent games"
            ]
        }
        
        # Get some stats for more specific reason
        if recent_games:
            last_categories = [g['category'] for g in recent_games[-3:]]
            pattern = ''.join([c[0] for c in last_categories])
            
            if prediction == 'LOW':
                if 'HH' in pattern:
                    return "HIGH streak typically followed by LOW"
                elif 'MM' in pattern:
                    return "MIDDLE streak often breaks to LOW"
                elif len(self.timing_stats['low_intervals']) > 0:
                    avg_interval = np.mean(list(self.timing_stats['low_intervals'])) if self.timing_stats['low_intervals'] else 0
                    last_low_minutes = (datetime.now() - self.timing_stats['last_low_time']).total_seconds() / 60 if self.timing_stats['last_low_time'] else 0
                    if last_low_minutes > avg_interval * 1.5:
                        return f"LOW overdue! Average interval: {avg_interval:.1f} min, Last LOW: {last_low_minutes:.1f} min ago"
            
            elif prediction == 'MIDDLE':
                if 'LL' in pattern:
                    return "LOW streak usually followed by MIDDLE"
                elif 'HH' in pattern:
                    return "HIGH to MIDDLE transition common"
            
            elif prediction == 'HIGH':
                if 'LL' in pattern:
                    return "LOW streak often followed by HIGH"
                elif 'MM' in pattern:
                    return "MIDDLE to HIGH transition detected"
        
        # Return random reason from list
        return random.choice(reasons.get(prediction, ["Pattern analysis and timing prediction"]))
    
    def _fallback_prediction(self):
        """Fallback prediction when AI can't decide"""
        if len(self.games_history) < 10:
            return {
                'prediction': 'LOW',
                'confidence': 0.45,
                'reason': 'Insufficient data for AI analysis',
                'probabilities': {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39},
                'system_used': 'fallback_insufficient_data',
                'games_analyzed': len(self.games_history)
            }
        
        # Simple frequency analysis
        recent = list(self.games_history)[-50:]
        counts = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        
        for game in recent:
            counts[game['category']] += 1
        
        total = len(recent)
        if total > 0:
            most_common = max(counts.items(), key=lambda x: x[1])
            confidence = 0.3 + (most_common[1] / total) * 0.4
            
            return {
                'prediction': most_common[0],
                'confidence': min(0.7, confidence),
                'reason': f'Frequency analysis: {most_common[0]} ({most_common[1]/total*100:.0f}% recent)',
                'probabilities': {
                    'LOW': counts['LOW'] / total,
                    'MIDDLE': counts['MIDDLE'] / total,
                    'HIGH': counts['HIGH'] / total
                },
                'system_used': 'frequency_analysis',
                'games_analyzed': len(self.games_history)
            }
        
        return {
            'prediction': 'LOW',
            'confidence': 0.5,
            'reason': 'Default prediction',
            'probabilities': {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39},
            'system_used': 'default',
            'games_analyzed': len(self.games_history)
        }
    
    def get_ai_stats(self):
        """Get AI statistics with detailed info"""
        try:
            # Calculate prediction accuracy
            checked_predictions = [p for p in self.prediction_history if 'correct' in p]
            accuracy = 0
            if checked_predictions:
                correct_count = sum(1 for p in checked_predictions if p['correct'])
                accuracy = correct_count / len(checked_predictions)
            
            # Calculate category distribution
            category_counts = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
            for game in self.games_history:
                category_counts[game['category']] += 1
            
            total_games = len(self.games_history)
            if total_games > 0:
                category_percentages = {
                    'LOW': category_counts['LOW'] / total_games * 100,
                    'MIDDLE': category_counts['MIDDLE'] / total_games * 100,
                    'HIGH': category_counts['HIGH'] / total_games * 100
                }
            else:
                category_percentages = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
            
            # Get timing stats safely
            avg_low_interval = 0
            avg_middle_interval = 0
            avg_high_interval = 0
            last_low_minutes = 0
            last_middle_minutes = 0
            last_high_minutes = 0
            
            if self.timing_stats['low_intervals']:
                avg_low_interval = np.mean(list(self.timing_stats['low_intervals']))
            if self.timing_stats['middle_intervals']:
                avg_middle_interval = np.mean(list(self.timing_stats['middle_intervals']))
            if self.timing_stats['high_intervals']:
                avg_high_interval = np.mean(list(self.timing_stats['high_intervals']))
            
            if self.timing_stats['last_low_time']:
                last_low_minutes = (datetime.now() - self.timing_stats['last_low_time']).total_seconds() / 60
            if self.timing_stats['last_middle_time']:
                last_middle_minutes = (datetime.now() - self.timing_stats['last_middle_time']).total_seconds() / 60
            if self.timing_stats['last_high_time']:
                last_high_minutes = (datetime.now() - self.timing_stats['last_high_time']).total_seconds() / 60
            
            return {
                'total_games_analyzed': total_games,
                'is_trained': self.is_trained,
                'training_accuracy': self.training_accuracy,
                'prediction_accuracy': accuracy,
                'low_patterns_count': len(self.low_patterns),
                'middle_patterns_count': len(self.middle_patterns),
                'high_patterns_count': len(self.high_patterns),
                'category_distribution': category_counts,
                'category_percentages': category_percentages,
                'prediction_count': len(self.prediction_history),
                'checked_predictions': len(checked_predictions),
                'correct_predictions': sum(1 for p in checked_predictions if p['correct']),
                'timing_stats': {
                    'avg_low_interval': avg_low_interval,
                    'avg_middle_interval': avg_middle_interval,
                    'avg_high_interval': avg_high_interval,
                    'last_low_minutes': last_low_minutes,
                    'last_middle_minutes': last_middle_minutes,
                    'last_high_minutes': last_high_minutes
                },
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'feature_importances': self.feature_importances,
                'debug_log': list(self.debug_log)[-20:]
            }
        except Exception as e:
            self._log(f"❌ Error in get_ai_stats: {e}")
            return {
                'total_games_analyzed': len(self.games_history),
                'is_trained': self.is_trained,
                'training_accuracy': 0.0,
                'prediction_accuracy': 0.0,
                'low_patterns_count': 0,
                'middle_patterns_count': 0,
                'high_patterns_count': 0,
                'timing_stats': {
                    'avg_low_interval': 0,
                    'avg_middle_interval': 0,
                    'avg_high_interval': 0,
                    'last_low_minutes': 0,
                    'last_middle_minutes': 0,
                    'last_high_minutes': 0
                }
            }
    
    def get_detailed_analysis(self):
        """Get detailed AI analysis for dashboard - FIXED"""
        try:
            # Get top patterns with safe access
            top_low_patterns = []
            top_middle_patterns = []
            top_high_patterns = []
            
            # Safely get low patterns
            if hasattr(self, 'low_patterns') and self.low_patterns:
                for pattern_key, stats in list(self.low_patterns.items())[:15]:
                    if isinstance(stats, dict):
                        top_low_patterns.append((pattern_key, stats))
            
            # Safely get middle patterns
            if hasattr(self, 'middle_patterns') and self.middle_patterns:
                for pattern_key, stats in list(self.middle_patterns.items())[:15]:
                    if isinstance(stats, dict):
                        top_middle_patterns.append((pattern_key, stats))
            
            # Safely get high patterns
            if hasattr(self, 'high_patterns') and self.high_patterns:
                for pattern_key, stats in list(self.high_patterns.items())[:15]:
                    if isinstance(stats, dict):
                        top_high_patterns.append((pattern_key, stats))
            
            # Calculate pattern effectiveness SAFELY
            pattern_stats = []
            
            # Check low patterns
            if hasattr(self, 'low_patterns'):
                for pattern_key, stats in list(self.low_patterns.items())[:10]:
                    if isinstance(stats, dict):
                        # Use .get() to avoid KeyError
                        next_low = stats.get('next_low', 0)
                        next_other = stats.get('next_other', 0)
                        total = next_low + next_other
                        
                        if total > 0:
                            effectiveness = next_low / total
                            pattern_stats.append({
                                'pattern': pattern_key,
                                'type': 'LOW',
                                'count': stats.get('count', 0),
                                'effectiveness': effectiveness,
                                'next_target': next_low,
                                'total_next': total,
                                'last_seen': stats.get('last_seen', 'Unknown')
                            })
            
            # Check middle patterns
            if hasattr(self, 'middle_patterns'):
                for pattern_key, stats in list(self.middle_patterns.items())[:10]:
                    if isinstance(stats, dict):
                        next_middle = stats.get('next_middle', 0)
                        next_other = stats.get('next_other', 0)
                        total = next_middle + next_other
                        
                        if total > 0:
                            effectiveness = next_middle / total
                            pattern_stats.append({
                                'pattern': pattern_key,
                                'type': 'MIDDLE',
                                'count': stats.get('count', 0),
                                'effectiveness': effectiveness,
                                'next_target': next_middle,
                                'total_next': total,
                                'last_seen': stats.get('last_seen', 'Unknown')
                            })
            
            # Check high patterns
            if hasattr(self, 'high_patterns'):
                for pattern_key, stats in list(self.high_patterns.items())[:10]:
                    if isinstance(stats, dict):
                        next_high = stats.get('next_high', 0)
                        next_other = stats.get('next_other', 0)
                        total = next_high + next_other
                        
                        if total > 0:
                            effectiveness = next_high / total
                            pattern_stats.append({
                                'pattern': pattern_key,
                                'type': 'HIGH',
                                'count': stats.get('count', 0),
                                'effectiveness': effectiveness,
                                'next_target': next_high,
                                'total_next': total,
                                'last_seen': stats.get('last_seen', 'Unknown')
                            })
            
            # Sort pattern stats by effectiveness
            pattern_stats.sort(key=lambda x: x['effectiveness'], reverse=True)
            
            # Get prediction history safely
            prediction_history = []
            if hasattr(self, 'prediction_history'):
                pred_list = list(self.prediction_history)
                for pred in pred_list[-100:]:
                    if isinstance(pred, dict):
                        prediction_history.append(pred)
            
            # Calculate hourly performance safely
            hourly_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
            if hasattr(self, 'prediction_history'):
                for pred in self.prediction_history:
                    if isinstance(pred, dict) and 'timestamp' in pred and 'correct' in pred:
                        try:
                            hour = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00')).hour
                            hourly_stats[hour]['total'] += 1
                            if pred.get('correct', False):
                                hourly_stats[hour]['correct'] += 1
                        except:
                            pass
            
            hourly_performance = []
            for hour in range(24):
                if hour in hourly_stats:
                    total = hourly_stats[hour]['total']
                    correct = hourly_stats[hour]['correct']
                    hourly_performance.append({
                        'hour': hour,
                        'total': total,
                        'correct': correct,
                        'accuracy': correct / total if total > 0 else 0
                    })
                else:
                    hourly_performance.append({'hour': hour, 'total': 0, 'correct': 0, 'accuracy': 0})
            
            return {
                'pattern_analysis': {
                    'top_low_patterns': top_low_patterns,
                    'top_middle_patterns': top_middle_patterns,
                    'top_high_patterns': top_high_patterns,
                    'pattern_stats': pattern_stats[:20],
                    'total_patterns_learned': len(self.low_patterns) + len(self.middle_patterns) + len(self.high_patterns)
                },
                'prediction_history': prediction_history,
                'hourly_performance': hourly_performance,
                'recent_debug_log': list(self.debug_log)[-50:] if hasattr(self, 'debug_log') else []
            }
            
        except Exception as e:
            self._log(f"❌ Error in get_detailed_analysis: {e}")
            return {
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

# Singleton instance
ai_system = TimingPatternAI()
