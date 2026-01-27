"""
ডাইস প্যাটার্ন ইন্টেলিজেন্স সিস্টেম v3.0
72-ঘণ্টার ডেটা + লাইভ ডেটা এনালাইসিস করে High/Low/Middle প্রেডিকশন দেয়
এডভান্সড প্যাটার্ন-বেজড ইন্টেলিজেন্ট সিস্টেম
"""

import numpy as np
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
import hashlib
import random
import math

class IntelligentPredictionSystem:
    """এডভান্সড ডাইস প্যাটার্ন ইন্টেলিজেন্স সিস্টেম Lightning Dice প্রেডিকশনের জন্য"""
    
    def __init__(self, memory_hours=72):
        self.memory_hours = memory_hours
        self.games_data = deque(maxlen=50000)
        self.predictions_history = []
        
        # ==================== উন্নত প্যাটার্ন সিস্টেম ====================
        self.pattern_system = AdvancedPatternIntelligenceSystem()
        self.session_memory = AdvancedSessionMemory()
        
        # ==================== ব্যাকআপ সিস্টেম ====================
        self.backup_system = EnhancedBackupPredictionSystem()
        
        # ==================== এডাপ্টিভ লার্নিং সিস্টেম ====================
        self.adaptive_learning = AdaptiveLearningSystem()
        
        # ==================== উন্নত পারফরম্যান্স ট্র্যাকিং ====================
        self.performance = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': deque(maxlen=100),
            'recent_errors': deque(maxlen=20),  # ভুল প্রেডিকশন ট্র্যাক
            'system_used': {'pattern': 0, 'backup': 0, 'adaptive': 0},
            'pattern_detection_rate': 0,
            'pattern_accuracy': 0,
            'backup_accuracy': 0,
            'adaptive_accuracy': 0,
            'confidence_calibration_error': 0,  # কনফিডেন্স vs accuracy gap
            'learning_effectiveness': 0.5,
            'category_accuracy': {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0},
            'category_predictions': {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        }
        
        print("🚀 উন্নত ডাইস প্যাটার্ন ইন্টেলিজেন্স সিস্টেম v3.0 শুরু হয়েছে...")
        print("🎯 ফিচার: ২-গেম প্যাটার্ন, ডাইনামিক কনফিডেন্স, সেল্ফ-লার্নিং AI")
        print("📊 ফিক্স: NaN% প্রোবাবিলিটি, ৫০% কনফিডেন্স, ২৪.৫% accuracy")
    
    def add_game_data(self, game_data):
        """নতুন গেম ডেটা যোগ করুন"""
        self.games_data.append(game_data)
        
        # 72 ঘণ্টার পুরাতন ডেটা রাখা
        cutoff_time = datetime.now() - timedelta(hours=self.memory_hours)
        while self.games_data and self.games_data[0]['timestamp'] < cutoff_time:
            self.games_data.popleft()
        
        # নতুন প্যাটার্ন সিস্টেমে ডেটা যোগ
        category = self._get_category(game_data['total'])
        timestamp = game_data['timestamp']
        
        # সব সিস্টেমে ডেটা যোগ
        self.pattern_system.add_game(category, timestamp)
        self.session_memory.add_game(category, timestamp)
        self.backup_system.add_game(category)
        self.adaptive_learning.add_game(category, timestamp)
        
        # পেন্ডিং প্রেডিকশন আপডেট
        self._update_pending_predictions(category, game_data)
    
    def _get_category(self, total):
        """টোটাল থেকে ক্যাটাগরি নির্ধারণ"""
        if 3 <= total <= 9:
            return "LOW"
        elif 10 <= total <= 11:
            return "MIDDLE"
        else:
            return "HIGH"
    
    def _update_pending_predictions(self, actual_category, game_data):
        """পেন্ডিং প্রেডিকশন আপডেট করুন"""
        if len(self.predictions_history) == 0:
            return None
        
        # সর্বশেষ পেন্ডিং প্রেডিকশন খুঁজুন
        for pred in reversed(self.predictions_history):
            if pred.get('status') == 'pending':
                predicted_category = pred['prediction']
                is_correct = predicted_category == actual_category
                
                # প্রেডিকশন রেকর্ড আপডেট
                pred['actual_category'] = actual_category
                pred['is_correct'] = is_correct
                pred['status'] = 'correct' if is_correct else 'incorrect'
                pred['evaluated_at'] = datetime.now()
                pred['game_id'] = game_data.get('game_id', '')
                pred['game_timestamp'] = game_data.get('timestamp')
                
                # পারফরম্যান্স আপডেট
                self.performance['total_predictions'] += 1
                if is_correct:
                    self.performance['correct_predictions'] += 1
                    self.performance['recent_errors'].append(0)  # ভুল না
                    
                    # Category accuracy update
                    self.performance['category_accuracy'][predicted_category] = (
                        self.performance['category_accuracy'][predicted_category] * 0.9 + 0.1
                    )
                else:
                    self.performance['recent_errors'].append(1)  # ভুল
                    self.performance['category_accuracy'][predicted_category] = (
                        self.performance['category_accuracy'][predicted_category] * 0.9
                    )
                
                # Category predictions count
                self.performance['category_predictions'][predicted_category] += 1
                
                # সিস্টেম accuracy ট্র্যাক
                system_used = pred.get('system_used', 'pattern')
                if system_used == 'pattern':
                    self.performance['pattern_accuracy'] = (
                        (self.performance['pattern_accuracy'] * 0.9) + (1 if is_correct else 0) * 0.1
                    )
                elif system_used == 'backup':
                    self.performance['backup_accuracy'] = (
                        (self.performance['backup_accuracy'] * 0.9) + (1 if is_correct else 0) * 0.1
                    )
                elif system_used == 'adaptive':
                    self.performance['adaptive_accuracy'] = (
                        (self.performance['adaptive_accuracy'] * 0.9) + (1 if is_correct else 0) * 0.1
                    )
                
                self.performance['recent_accuracy'].append(1 if is_correct else 0)
                
                # কনফিডেন্স ক্যালিব্রেশন error ট্র্যাক
                confidence_error = abs(pred['confidence'] - (1 if is_correct else 0))
                self.performance['confidence_calibration_error'] = (
                    self.performance['confidence_calibration_error'] * 0.9 + confidence_error * 0.1
                )
                
                # সব সিস্টেমকে ফলাফল জানান
                self.pattern_system.learn_from_result(
                    predicted_category, 
                    actual_category,
                    pred.get('pattern_info', {}),
                    pred['confidence']
                )
                
                self.adaptive_learning.learn_from_result(
                    predicted_category,
                    actual_category,
                    system_used,
                    pred['confidence']
                )
                
                # লার্নিং ইফেক্টিভনেস আপডেট
                recent_errors = list(self.performance['recent_errors'])[-10:]
                if len(recent_errors) >= 5:
                    error_rate = sum(recent_errors) / len(recent_errors)
                    # কম error হলে learning effective
                    self.performance['learning_effectiveness'] = 1.0 - error_rate
                
                # লগ
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"\n🎯 {status}: Predicted {predicted_category}, Actual {actual_category}")
                print(f"   System: {system_used} | Confidence: {pred['confidence']*100:.1f}%")
                if not is_correct:
                    print(f"   Learning Rate: {self.adaptive_learning.current_learning_rate:.2f}")
                
                return is_correct
        
        return None
    
    def predict_next_game(self):
        """পরবর্তী গেমের প্রেডিকশন করুন"""
        
        # পর্যাপ্ত ডেটা না থাকলে ব্যাকআপ সিস্টেম ব্যবহার করুন
        if len(self.games_data) < 30:
            return self._get_backup_prediction()
        
        # 1. অ্যাডাপ্টিভ লার্নিং থেকে প্রেডিকশন স্ট্র্যাটেজি বাছাই
        strategy = self.adaptive_learning.get_prediction_strategy()
        
        # 2. নির্বাচিত স্ট্র্যাটেজি অনুযায়ী প্রেডিকশন নিন
        if strategy == 'pattern' or strategy == 'auto':
            pattern_prediction = self.pattern_system.predict_next()
            
            if pattern_prediction['confidence'] >= 0.45:  # Lower threshold for more predictions
                self.performance['system_used']['pattern'] += 1
                return self._create_prediction_record(pattern_prediction, 'pattern')
        
        if strategy == 'adaptive' or (strategy == 'auto' and len(self.games_data) > 100):
            adaptive_prediction = self.adaptive_learning.predict()
            if adaptive_prediction['confidence'] >= 0.45:  # Lower threshold
                self.performance['system_used']['adaptive'] += 1
                return self._create_prediction_record(adaptive_prediction, 'adaptive')
        
        # 3. অন্য সবকিছু fail করলে ব্যাকআপ
        return self._get_backup_prediction()
    
    def _create_prediction_record(self, prediction, system_used):
        """প্রেডিকশন রেকর্ড তৈরি করুন"""
        prediction_id = hashlib.md5(
            f"{datetime.now().timestamp()}_{prediction['prediction']}".encode()
        ).hexdigest()[:12]
        
        # নিশ্চিত করুন সব required fields আছে
        if 'probabilities' not in prediction:
            prediction['probabilities'] = self._get_default_probabilities()
        
        prediction_record = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now(),
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'reason': prediction['reason'],
            'game_count': len(self.games_data),
            'status': 'pending',
            'system_used': system_used,
            'pattern_info': prediction.get('pattern_info', {})
        }
        
        self.predictions_history.append(prediction_record)
        
        return {
            'prediction_id': prediction_id,
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'probabilities': prediction['probabilities'],  # নিশ্চিত করুন আছে
            'reason': prediction['reason'],
            'system_status': f'{system_used}_analysis',
            'data_points': len(self.games_data),
            'pattern_detected': prediction.get('pattern_detected', False),
            'pattern_type': prediction.get('pattern_type', 'none'),
            'pattern_duration': prediction.get('pattern_duration', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_default_probabilities(self):
        """ডিফল্ট প্রোবাবিলিটি রিটার্ন করুন"""
        return {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
    
    def _get_backup_prediction(self):
        """ব্যাকআপ সিস্টেম থেকে প্রেডিকশন নিন"""
        self.performance['system_used']['backup'] += 1
        
        backup_pred = self.backup_system.predict()
        
        prediction_id = hashlib.md5(
            f"{datetime.now().timestamp()}_{backup_pred['prediction']}".encode()
        ).hexdigest()[:12]
        
        prediction_record = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now(),
            'prediction': backup_pred['prediction'],
            'confidence': backup_pred['confidence'],
            'reason': backup_pred['reason'],
            'game_count': len(self.games_data),
            'status': 'pending',
            'system_used': 'backup'
        }
        
        self.predictions_history.append(prediction_record)
        
        return {
            'prediction_id': prediction_id,
            'prediction': backup_pred['prediction'],
            'confidence': backup_pred['confidence'],
            'probabilities': backup_pred.get('probabilities', self._get_default_probabilities()),
            'reason': backup_pred['reason'],
            'system_status': 'frequency_analysis',
            'data_points': len(self.games_data),
            'pattern_detected': False,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_system_stats(self):
        """সিস্টেম স্ট্যাটিস্টিক্স পান"""
        evaluated_predictions = [p for p in self.predictions_history 
                               if p.get('status') in ['correct', 'incorrect']]
        
        total_evaluated = len(evaluated_predictions)
        correct_evaluated = sum(1 for p in evaluated_predictions if p.get('is_correct', False))
        
        accuracy = (correct_evaluated / total_evaluated * 100) if total_evaluated > 0 else 0
        
        # সাম্প্রতিক accuracy
        recent_evaluated = evaluated_predictions[-20:] if len(evaluated_predictions) >= 20 else evaluated_predictions
        recent_correct = sum(1 for p in recent_evaluated if p.get('is_correct', False))
        recent_acc = (recent_correct / len(recent_evaluated) * 100) if recent_evaluated else 0
        
        # ক্লাস ডিস্ট্রিবিউশন
        if self.games_data:
            categories = [self._get_category(g['total']) for g in self.games_data]
            total_games = len(categories)
            class_distribution = {
                'LOW': round(categories.count('LOW') / total_games * 100, 1),
                'MIDDLE': round(categories.count('MIDDLE') / total_games * 100, 1),
                'HIGH': round(categories.count('HIGH') / total_games * 100, 1)
            }
        else:
            class_distribution = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        
        # সিস্টেম usage statistics
        pattern_predictions = [p for p in evaluated_predictions if p.get('system_used') == 'pattern']
        backup_predictions = [p for p in evaluated_predictions if p.get('system_used') == 'backup']
        adaptive_predictions = [p for p in evaluated_predictions if p.get('system_used') == 'adaptive']
        
        pattern_detection_rate = (len(pattern_predictions) / total_evaluated * 100) if total_evaluated > 0 else 0
        
        # সিস্টেম accuracy
        pattern_correct = sum(1 for p in pattern_predictions if p.get('is_correct', False))
        pattern_accuracy = (pattern_correct / len(pattern_predictions) * 100) if pattern_predictions else 0
        
        backup_correct = sum(1 for p in backup_predictions if p.get('is_correct', False))
        backup_accuracy = (backup_correct / len(backup_predictions) * 100) if backup_predictions else 0
        
        adaptive_correct = sum(1 for p in adaptive_predictions if p.get('is_correct', False))
        adaptive_accuracy = (adaptive_correct / len(adaptive_predictions) * 100) if adaptive_predictions else 0
        
        # প্যাটার্ন সিস্টেম stats
        pattern_stats = self.pattern_system.get_stats()
        
        # অ্যাডাপ্টিভ লার্নিং stats
        adaptive_stats = self.adaptive_learning.get_stats()
        
        # Calculate category accuracy percentages
        category_accuracy_percent = {}
        for cat in ['LOW', 'MIDDLE', 'HIGH']:
            predictions = self.performance['category_predictions'][cat]
            if predictions > 0:
                accuracy_val = self.performance['category_accuracy'][cat] * 100
                category_accuracy_percent[cat] = round(accuracy_val, 1)
            else:
                category_accuracy_percent[cat] = 0
        
        return {
            'total_games_analyzed': len(self.games_data),
            'total_predictions': total_evaluated,
            'correct_predictions': correct_evaluated,
            'accuracy_percentage': round(accuracy, 1),
            'recent_accuracy': round(recent_acc, 1),
            'class_distribution': class_distribution,
            'category_accuracy': category_accuracy_percent,
            'system_usage': {
                'pattern': self.performance['system_used']['pattern'],
                'backup': self.performance['system_used']['backup'],
                'adaptive': self.performance['system_used']['adaptive'],
                'pattern_percentage': round(pattern_detection_rate, 1)
            },
            'system_accuracy': {
                'pattern': round(pattern_accuracy, 1),
                'backup': round(backup_accuracy, 1),
                'adaptive': round(adaptive_accuracy, 1),
                'overall': round(accuracy, 1)
            },
            'learning_metrics': {
                'confidence_error': round(self.performance['confidence_calibration_error'], 3),
                'learning_effectiveness': round(self.performance['learning_effectiveness'], 2),
                'recent_error_rate': round(sum(self.performance['recent_errors']) / len(self.performance['recent_errors']) if self.performance['recent_errors'] else 0, 2)
            },
            'pattern_system': pattern_stats,
            'adaptive_system': adaptive_stats,
            'session_info': self.session_memory.get_stats(),
            'system_status': 'active' if len(self.games_data) > 50 else 'learning'
        }
    
    def get_pending_predictions(self):
        """পেন্ডিং প্রেডিকশন পান"""
        return [p for p in self.predictions_history if p.get('status') == 'pending']
    
    def get_prediction_history(self, limit=50, include_pending=False):
        """প্রেডিকশন হিস্ট্রি পান"""
        history = self.predictions_history[-limit:] if self.predictions_history else []
        
        if not include_pending:
            history = [p for p in history if p.get('status') != 'pending']
        
        # ডেটাটাইম অবজেক্ট স্ট্রিং-এ কনভার্ট করুন
        for pred in history:
            for key in ['timestamp', 'evaluated_at', 'game_timestamp']:
                if key in pred and isinstance(pred[key], datetime):
                    pred[key] = pred[key].isoformat()
        
        return history


class AdvancedPatternIntelligenceSystem:
    """এডভান্সড প্যাটার্ন ইন্টেলিজেন্স সিস্টেম v3.0"""
    
    def __init__(self):
        # ৯টি বেসিক প্যাটার্ন (MEDIUM → MIDDLE ফিক্স)
        self.patterns = {
            'HL': {'name': 'HIGH-LOW Alternation', 'sequence': ['HIGH', 'LOW'], 'count': 0},
            'LH': {'name': 'LOW-HIGH Alternation', 'sequence': ['LOW', 'HIGH'], 'count': 0},
            'MH': {'name': 'MIDDLE-HIGH Alternation', 'sequence': ['MIDDLE', 'HIGH'], 'count': 0},
            'HM': {'name': 'HIGH-MIDDLE Alternation', 'sequence': ['HIGH', 'MIDDLE'], 'count': 0},
            'LM': {'name': 'LOW-MIDDLE Alternation', 'sequence': ['LOW', 'MIDDLE'], 'count': 0},
            'ML': {'name': 'MIDDLE-LOW Alternation', 'sequence': ['MIDDLE', 'LOW'], 'count': 0},
            'HH': {'name': 'HIGH Consecutive', 'sequence': ['HIGH', 'HIGH'], 'count': 0},
            'MM': {'name': 'MIDDLE Consecutive', 'sequence': ['MIDDLE', 'MIDDLE'], 'count': 0},
            'LL': {'name': 'LOW Consecutive', 'sequence': ['LOW', 'LOW'], 'count': 0}
        }
        
        # ২-গেম ট্রানজিশন প্যাটার্ন (নতুন)
        self.transition_patterns = {
            'LOW_LOW': {'name': 'LOW to LOW', 'prediction': 'LOW'},
            'HIGH_HIGH': {'name': 'HIGH to HIGH', 'prediction': 'HIGH'},
            'MIDDLE_MIDDLE': {'name': 'MIDDLE to MIDDLE', 'prediction': 'MIDDLE'},
            'LOW_HIGH': {'name': 'LOW to HIGH', 'prediction': 'HIGH'},
            'HIGH_LOW': {'name': 'HIGH to LOW', 'prediction': 'LOW'},
            'LOW_MIDDLE': {'name': 'LOW to MIDDLE', 'prediction': 'MIDDLE'},
            'MIDDLE_LOW': {'name': 'MIDDLE to LOW', 'prediction': 'LOW'},
            'MIDDLE_HIGH': {'name': 'MIDDLE to HIGH', 'prediction': 'HIGH'},
            'HIGH_MIDDLE': {'name': 'HIGH to MIDDLE', 'prediction': 'MIDDLE'}
        }
        
        # উন্নত প্যাটার্ন ট্র্যাকিং
        self.pattern_history = deque(maxlen=200)
        self.active_pattern = None
        self.pattern_duration = 0
        self.pattern_start_time = None
        self.last_pattern_break_reason = None
        
        # উন্নত স্ট্যাটিস্টিক্স
        self.pattern_stats = defaultdict(lambda: {
            'total_occurrences': 0,
            'successful_occurrences': 0,
            'total_duration': 0,
            'avg_duration': 2,  # ডিফল্ট 2 বার (কম করা হয়েছে)
            'break_frequency': 0,
            'strength': 0.5,
            'recent_success': deque(maxlen=10),
            'accuracy': 0.5,
            'durations': deque(maxlen=50),
            'break_points': [],
            'last_break_duration': 0,
            'next_patterns': defaultdict(int)  # Break হলে কি pattern হয়
        })
        
        # Transition matrix
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.pattern_transitions = defaultdict(lambda: defaultdict(int))
        
        # সাম্প্রতিক ডেটা
        self.recent_outcomes = deque(maxlen=20)
        self.recent_patterns = deque(maxlen=10)
        
        # এডাপ্টিভ প্যারামিটারস
        self.pattern_confidence_decay = 0.95
        self.learning_rate = 0.1
        
        # ২-গেম প্যাটার্ন ট্র্যাকিং
        self.two_game_patterns = deque(maxlen=50)
        self.two_game_stats = defaultdict(lambda: {'count': 0, 'success': 0})
        
        print("🧠 এডভান্সড প্যাটার্ন সিস্টেম v3.0 লোড করা হয়েছে")
        print("📊 ২-গেম প্যাটার্ন ডিটেকশন | স্মার্ট কনফিডেন্স | ব্রেক প্রেডিকশন")
    
    def add_game(self, category, timestamp):
        """নতুন গেম যোগ করুন"""
        # MEDIUM → MIDDLE ফিক্স
        if category == 'MEDIUM':
            category = 'MIDDLE'
        
        self.recent_outcomes.append(category)
        self.pattern_history.append(category)
        
        # ২-গেম প্যাটার্ন ট্র্যাক
        if len(self.recent_outcomes) >= 2:
            last_two = (list(self.recent_outcomes)[-2], category)
            self.two_game_patterns.append(last_two)
        
        # প্যাটার্ন আপডেট
        pattern_broken = self._update_pattern_detection(category)
        
        # প্যাটার্ন duration আপডেট
        if self.active_pattern and not pattern_broken:
            self.pattern_duration += 1
            
            # প্যাটার্ন স্ট্যাটস আপডেট
            pattern_key = self.active_pattern
            if pattern_key in self.pattern_stats:
                stats = self.pattern_stats[pattern_key]
                stats['durations'].append(self.pattern_duration)
                if stats['durations']:
                    stats['avg_duration'] = np.mean(list(stats['durations']))
    
    def _update_pattern_detection(self, current_category):
        """প্যাটার্ন ডিটেকশন আপডেট করুন - ২/৩ গেম ভিত্তিক"""
        
        if len(self.recent_outcomes) < 2:
            return False
        
        pattern_broken = False
        
        # বর্তমান প্যাটার্ন চেক
        if self.active_pattern:
            pattern = self.patterns[self.active_pattern]
            
            # Expected outcome নির্ধারণ
            if 'Alternation' in pattern['name']:
                expected_next = pattern['sequence'][self.pattern_duration % 2]
            else:  # Consecutive
                expected_next = pattern['sequence'][0]
            
            # প্যাটার্ন continue করছে কিনা চেক
            if current_category == expected_next:
                # প্যাটার্ন continue করছে
                return False
            else:
                # প্যাটার্ন break হয়েছে
                pattern_broken = True
                self._record_pattern_end(
                    success=False,
                    break_reason=f"Expected {expected_next}, got {current_category}",
                    next_pattern=current_category
                )
                self.last_pattern_break_reason = f"Pattern broke after {self.pattern_duration} cycles"
                
                # Transition matrix আপডেট
                if len(self.recent_outcomes) >= 2:
                    prev_category = list(self.recent_outcomes)[-2]
                    self.transition_matrix[prev_category][current_category] += 1
        
        # নতুন প্যাটার্ন ডিটেক্ট (২/৩ গেম ভিত্তিক)
        if not self.active_pattern or pattern_broken:
            detected_pattern = self._detect_new_pattern_flexible(current_category)
            if detected_pattern:
                self.active_pattern = detected_pattern
                self.pattern_duration = 1  # শুরুতে ১ ধরা
                self.pattern_start_time = datetime.now()
                self.patterns[detected_pattern]['count'] += 1
                self.recent_patterns.append(detected_pattern)
                print(f"🔍 প্যাটার্ন ডিটেক্ট: {self.patterns[detected_pattern]['name']}")
        
        return pattern_broken
    
    def _detect_new_pattern_flexible(self, current_category):
        """নতুন প্যাটার্ন ডিটেক্ট করুন - ২/৩ গেম flexible"""
        if len(self.recent_outcomes) < 2:
            return None
        
        recent = list(self.recent_outcomes)
        
        # ১. ২-গেম প্যাটার্ন চেক (সবচেয়ে কমন)
        if len(recent) >= 2:
            # Consecutive patterns
            if recent[-2] == current_category:
                if current_category == 'HIGH':
                    return 'HH'
                elif current_category == 'MIDDLE':
                    return 'MM'
                elif current_category == 'LOW':
                    return 'LL'
            
            # Alternation patterns (৩ গেমে চেক)
            if len(recent) >= 3:
                # A → B → A pattern
                if recent[-3] == current_category and recent[-2] != current_category:
                    if current_category == 'HIGH' and recent[-2] == 'LOW':
                        return 'HL'
                    elif current_category == 'LOW' and recent[-2] == 'HIGH':
                        return 'LH'
                    elif current_category == 'HIGH' and recent[-2] == 'MIDDLE':
                        return 'HM'
                    elif current_category == 'MIDDLE' and recent[-2] == 'HIGH':
                        return 'MH'
                    elif current_category == 'LOW' and recent[-2] == 'MIDDLE':
                        return 'LM'
                    elif current_category == 'MIDDLE' and recent[-2] == 'LOW':
                        return 'ML'
        
        return None
    
    def _record_pattern_end(self, success=True, break_reason="", next_pattern=None):
        """প্যাটার্ন শেষ রেকর্ড করুন"""
        if not self.active_pattern:
            return
        
        pattern_key = self.active_pattern
        stats = self.pattern_stats[pattern_key]
        
        stats['total_occurrences'] += 1
        
        if success:
            stats['successful_occurrences'] += 1
            stats['recent_success'].append(1)
        else:
            stats['recent_success'].append(0)
            stats['break_points'].append(self.pattern_duration)
            if len(stats['break_points']) > 10:
                stats['break_points'] = stats['break_points'][-10:]
            stats['last_break_duration'] = self.pattern_duration
            
            # Record what pattern came after break
            if next_pattern:
                stats['next_patterns'][next_pattern] += 1
        
        stats['total_duration'] += self.pattern_duration
        
        # Break frequency আপডেট
        if stats['total_occurrences'] > 0:
            stats['break_frequency'] = len(stats['break_points']) / stats['total_occurrences']
        
        # Pattern strength আপডেট
        if success:
            stats['strength'] = min(1.0, stats['strength'] + 0.1)
        else:
            stats['strength'] = max(0.1, stats['strength'] - 0.15)
        
        # Accuracy আপডেট
        if stats['recent_success']:
            stats['accuracy'] = sum(stats['recent_success']) / len(stats['recent_success'])
        
        # Reset
        self.active_pattern = None
        self.pattern_duration = 0
    
    def predict_next(self):
        """পরবর্তী outcome প্রেডিক্ট করুন"""
        
        # ১. যদি active প্যাটার্ন থাকে
        if self.active_pattern:
            return self._predict_from_active_pattern()
        
        # ২. ২-গেম transition-based prediction (নতুন)
        transition_pred = self._predict_from_transition()
        if transition_pred:
            return transition_pred
        
        # ৩. Frequency-based prediction
        return self._predict_from_frequency()
    
    def _predict_from_active_pattern(self):
        """এক্টিভ প্যাটার্ন থেকে প্রেডিকশন"""
        pattern = self.patterns[self.active_pattern]
        stats = self.pattern_stats[self.active_pattern]
        
        # Expected outcome
        if 'Alternation' in pattern['name']:
            expected_next = pattern['sequence'][self.pattern_duration % 2]
        else:
            expected_next = pattern['sequence'][0]
        
        # কনফিডেন্স ক্যালকুলেশন (improved)
        base_confidence = 0.4  # Increased from 0.3
        
        # Pattern strength factor
        pattern_strength = stats.get('strength', 0.5)
        base_confidence += pattern_strength * 0.4  # Increased weight
        
        # Duration factor
        avg_duration = stats.get('avg_duration', 2)
        if avg_duration > 0:
            duration_ratio = self.pattern_duration / avg_duration
            if duration_ratio > 1.5:
                duration_factor = 0.8  # Less penalty
            elif duration_ratio > 1.2:
                duration_factor = 0.9
            else:
                duration_factor = 1.0
            base_confidence *= duration_factor
        
        # Break probability
        break_probability = self._calculate_break_probability(stats)
        
        # Recent success rate
        recent_success = sum(stats['recent_success']) / len(stats['recent_success']) if stats['recent_success'] else 0.5
        base_confidence *= (0.6 + recent_success * 0.4)  # Adjusted weights
        
        # Final confidence
        final_confidence = min(0.85, max(0.35, base_confidence))
        
        # Alternative scenarios if break happens
        alternatives = {}
        if break_probability > 0.2 and stats['next_patterns']:
            total_breaks = sum(stats['next_patterns'].values())
            for next_cat, count in stats['next_patterns'].items():
                prob = count / total_breaks * break_probability
                alternatives[next_cat] = prob
        
        reason = f"{pattern['name']} চলছে ({self.pattern_duration} বার)"
        if break_probability > 0.25:
            reason += f", Break সম্ভাবনা: {break_probability*100:.0f}%"
        
        # Probabilities calculation
        probabilities = {}
        main_prob = final_confidence * (1 - break_probability)
        probabilities[expected_next] = main_prob
        
        # Distribute break probability among alternatives
        if alternatives:
            remaining_prob = break_probability
            for alt, alt_prob in alternatives.items():
                probabilities[alt] = alt_prob
                remaining_prob -= alt_prob
            
            # If there's remaining probability, add to main
            if remaining_prob > 0:
                probabilities[expected_next] += remaining_prob
        else:
            # If no alternatives, break probability goes to other categories
            other_cats = [cat for cat in ['LOW', 'MIDDLE', 'HIGH'] if cat != expected_next]
            if other_cats:
                prob_per_cat = break_probability / len(other_cats)
                for cat in other_cats:
                    probabilities[cat] = prob_per_cat
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return {
            'prediction': expected_next,
            'confidence': final_confidence,
            'reason': reason,
            'pattern_type': self.active_pattern,
            'pattern_duration': self.pattern_duration,
            'pattern_detected': True,
            'break_probability': break_probability,
            'pattern_info': {
                'name': pattern['name'],
                'avg_duration': avg_duration,
                'strength': pattern_strength,
                'break_points': stats.get('break_points', [])[-3:]
            },
            'probabilities': probabilities
        }
    
    def _predict_from_transition(self):
        """২-গেম transition থেকে প্রেডিকশন"""
        if len(self.recent_outcomes) < 2:
            return None
        
        last_two = list(self.recent_outcomes)[-2:]
        transition_key = f"{last_two[0]}_{last_two[1]}"
        
        # Check transition frequency
        if self.transition_matrix[last_two[0]][last_two[1]] > 0:
            total_transitions = sum(self.transition_matrix[last_two[0]].values())
            transition_prob = self.transition_matrix[last_two[0]][last_two[1]] / total_transitions
            
            # Most common next after this transition
            next_options = self.transition_matrix[last_two[1]]
            if next_options:
                most_common_next = max(next_options.items(), key=lambda x: x[1])
                confidence = 0.3 + transition_prob * 0.4  # Dynamic confidence
                
                return {
                    'prediction': most_common_next[0],
                    'confidence': min(0.7, max(0.35, confidence)),
                    'reason': f"Transition pattern: {last_two[0]}→{last_two[1]} ({transition_prob*100:.0f}%)",
                    'pattern_type': 'transition',
                    'pattern_detected': True,
                    'probabilities': self._calculate_transition_probabilities(last_two[1])
                }
        
        return None
    
    def _predict_from_frequency(self):
        """Frequency-based prediction"""
        if not self.recent_outcomes:
            return {
                'prediction': 'LOW',
                'confidence': 0.4,
                'reason': 'ডিফল্ট প্রেডিকশন',
                'pattern_type': 'default',
                'probabilities': {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
            }
        
        # Use last 15 games for frequency
        recent_list = list(self.recent_outcomes)[-15:] if len(self.recent_outcomes) >= 15 else list(self.recent_outcomes)
        freq = {
            'LOW': recent_list.count('LOW') / len(recent_list),
            'MIDDLE': recent_list.count('MIDDLE') / len(recent_list),
            'HIGH': recent_list.count('HIGH') / len(recent_list)
        }
        
        most_common = max(freq.items(), key=lambda x: x[1])
        confidence = 0.3 + most_common[1] * 0.5  # Dynamic confidence
        
        return {
            'prediction': most_common[0],
            'confidence': min(0.75, max(0.35, confidence)),
            'reason': f"সাম্প্রতিক ফ্রিকোয়েন্সি: {most_common[0]} ({most_common[1]*100:.0f}%)",
            'pattern_type': 'frequency',
            'probabilities': freq
        }
    
    def _calculate_transition_probabilities(self, current_category):
        """Calculate probabilities based on transition matrix"""
        if current_category not in self.transition_matrix:
            return {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
        
        transitions = self.transition_matrix[current_category]
        total = sum(transitions.values())
        
        if total == 0:
            return {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
        
        probs = {}
        for cat in ['LOW', 'MIDDLE', 'HIGH']:
            probs[cat] = transitions.get(cat, 1) / (total + 3)  # Add-1 smoothing
        
        # Normalize
        total_prob = sum(probs.values())
        return {k: v/total_prob for k, v in probs.items()}
    
    def _calculate_break_probability(self, stats):
        """প্যাটার্ন ব্রেক হওয়ার সম্ভাবনা ক্যালকুলেট"""
        if not stats.get('break_points'):
            return 0.0
        
        avg_break = np.mean(stats['break_points'])
        if avg_break > 0 and self.pattern_duration > 0:
            progress = self.pattern_duration / avg_break
            if progress > 0.7:  # Lower threshold (was 0.8)
                return min(0.8, (progress - 0.7) * 2)  # Less aggressive
        
        return 0.0
    
    def learn_from_result(self, predicted, actual, pattern_info, confidence):
        """ফলাফল থেকে শিখুন"""
        pattern_key = pattern_info.get('pattern_type') if pattern_info else None
        
        if pattern_key and pattern_key != 'none':
            if pattern_key in self.pattern_stats:
                stats = self.pattern_stats[pattern_key]
                
                stats['total_occurrences'] += 1
                
                if predicted == actual:
                    stats['successful_occurrences'] += 1
                    stats['recent_success'].append(1)
                    stats['strength'] = min(1.0, stats['strength'] + self.learning_rate)
                    
                    if pattern_key == self.active_pattern:
                        stats['durations'].append(self.pattern_duration)
                        if stats['durations']:
                            stats['avg_duration'] = np.mean(list(stats['durations']))
                else:
                    stats['recent_success'].append(0)
                    penalty = self.learning_rate * 1.2  # Reduced penalty
                    if confidence > 0.7:
                        penalty *= 0.8  # Less penalty for high confidence
                    stats['strength'] = max(0.1, stats['strength'] - penalty)
                
                if stats['recent_success']:
                    stats['accuracy'] = sum(stats['recent_success']) / len(stats['recent_success'])
        
        # Transition matrix আপডেট
        if len(self.recent_outcomes) >= 2:
            last = list(self.recent_outcomes)[-2]
            self.transition_matrix[last][actual] += 1
    
    def get_stats(self):
        """প্যাটার্ন সিস্টেম স্ট্যাটিস্টিক্স পান"""
        active_info = None
        if self.active_pattern:
            pattern = self.patterns[self.active_pattern]
            stats = self.pattern_stats[self.active_pattern]
            active_info = {
                'pattern': pattern['name'],
                'duration': self.pattern_duration,
                'avg_duration': stats.get('avg_duration', 0),
                'strength': stats.get('strength', 0),
                'accuracy': stats.get('accuracy', 0),
                'break_probability': self._calculate_break_probability(stats)
            }
        
        # সব প্যাটার্নের stats
        all_patterns_stats = {}
        for key, pattern in self.patterns.items():
            if key in self.pattern_stats:
                stats = self.pattern_stats[key]
                all_patterns_stats[key] = {
                    'name': pattern['name'],
                    'count': pattern['count'],
                    'avg_duration': stats.get('avg_duration', 0),
                    'strength': stats.get('strength', 0.5),
                    'accuracy': stats.get('accuracy', 0.5),
                    'total_occurrences': stats.get('total_occurrences', 0),
                    'success_rate': stats.get('successful_occurrences', 0) / stats.get('total_occurrences', 1) if stats.get('total_occurrences', 0) > 0 else 0
                }
        
        return {
            'active_pattern': active_info,
            'pattern_stats': all_patterns_stats,
            'recent_outcomes_count': len(self.recent_outcomes),
            'transition_matrix_size': len(self.transition_matrix),
            'learning_rate': self.learning_rate,
            'two_game_patterns_count': len(self.two_game_patterns)
        }


class AdvancedSessionMemory:
    """এডভান্সড সেশন-বেজড মেমোরি"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.session_games = []
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        self.session_patterns = defaultdict(lambda: defaultdict(int))
        self.session_performance = {
            'total': 0,
            'correct': 0,
            'recent_accuracy': deque(maxlen=20),
            'hourly_accuracy': defaultdict(lambda: {'total': 0, 'correct': 0})
        }
        
    def add_game(self, category, timestamp):
        """সেশনে গেম যোগ করুন"""
        # MEDIUM → MIDDLE ফিক্স
        if category == 'MEDIUM':
            category = 'MIDDLE'
            
        game_record = {
            'category': category,
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'minute': timestamp.minute
        }
        
        self.session_games.append(game_record)
        
        # Hourly stats
        hour = timestamp.hour
        self.hourly_stats[hour][category] += 1
        
        # Session patterns ট্র্যাক
        if len(self.session_games) >= 3:  # Reduced from 4 to 3
            seq = [
                self.session_games[-3]['category'],
                self.session_games[-2]['category'],
                category
            ]
            pattern_key = ''.join(seq)
            self.session_patterns[pattern_key]['count'] += 1
            self.session_patterns[pattern_key]['last_seen'] = timestamp
        
        # মেমোরি ম্যানেজমেন্ট
        if len(self.session_games) > 2000:
            self.session_games = self.session_games[-2000:]
    
    def update_performance(self, is_correct, hour):
        """সেশন পারফরম্যান্স আপডেট"""
        self.session_performance['total'] += 1
        if is_correct:
            self.session_performance['correct'] += 1
            self.session_performance['hourly_accuracy'][hour]['correct'] += 1
        
        self.session_performance['hourly_accuracy'][hour]['total'] += 1
        self.session_performance['recent_accuracy'].append(1 if is_correct else 0)
    
    def get_hourly_effectiveness(self, hour):
        """নির্দিষ্ট ঘন্টায় effectiveness"""
        if hour in self.session_performance['hourly_accuracy']:
            stats = self.session_performance['hourly_accuracy'][hour]
            if stats['total'] > 0:
                return stats['correct'] / stats['total']
        return 0.5  # ডিফল্ট
    
    def get_stats(self):
        """সেশন স্ট্যাটস পান"""
        if not self.session_games:
            return {'total_games': 0, 'session_duration': 0}
        
        session_duration = (datetime.now() - self.session_start).total_seconds() / 3600
        
        # Common patterns (top 5)
        common_patterns = dict(sorted(
            self.session_patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5])
        
        # Hourly distribution
        hourly_dist = {}
        for hour in range(24):
            if hour in self.hourly_stats:
                total = sum(self.hourly_stats[hour].values())
                if total > 0:
                    hourly_dist[hour] = {
                        'LOW': self.hourly_stats[hour].get('LOW', 0) / total * 100,
                        'MIDDLE': self.hourly_stats[hour].get('MIDDLE', 0) / total * 100,
                        'HIGH': self.hourly_stats[hour].get('HIGH', 0) / total * 100
                    }
        
        return {
            'total_games': len(self.session_games),
            'session_duration': round(session_duration, 2),
            'session_accuracy': round(self.session_performance['correct'] / self.session_performance['total'] * 100, 1) if self.session_performance['total'] > 0 else 0,
            'recent_accuracy': round(sum(self.session_performance['recent_accuracy']) / len(self.session_performance['recent_accuracy']) * 100, 1) if self.session_performance['recent_accuracy'] else 0,
            'hourly_distribution': hourly_dist,
            'common_patterns': common_patterns
        }


class EnhancedBackupPredictionSystem:
    """এনহ্যান্সড ব্যাকআপ প্রেডিকশন সিস্টেম v3.0"""
    
    def __init__(self):
        self.category_counts = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        self.recent_games = deque(maxlen=100)
        self.hourly_trends = defaultdict(lambda: defaultdict(int))
        self.last_prediction = None
        self.consecutive_same = 0
        self.default_probabilities = {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
        
    def add_game(self, category):
        """গেম যোগ করুন"""
        # MEDIUM → MIDDLE ফিক্স
        if category == 'MEDIUM':
            category = 'MIDDLE'
            
        self.category_counts[category] += 1
        self.recent_games.append(category)
        
        # Consecutive same ট্র্যাক
        if self.last_prediction == category:
            self.consecutive_same += 1
        else:
            self.consecutive_same = 0
        self.last_prediction = category
    
    def predict(self):
        """প্রেডিকশন করুন"""
        # Consecutive same বেশি হলে avoid
        if self.consecutive_same >= 3:
            recent_list = list(self.recent_games)[-15:] if len(self.recent_games) >= 15 else list(self.recent_games)
            if recent_list:
                freq = {
                    'LOW': recent_list.count('LOW') / len(recent_list),
                    'MIDDLE': recent_list.count('MIDDLE') / len(recent_list),
                    'HIGH': recent_list.count('HIGH') / len(recent_list)
                }
                
                # বর্তমান category ব্যতীত অন্য category বাছাই
                current = self.last_prediction
                other_categories = {k: v for k, v in freq.items() if k != current}
                
                if other_categories:
                    most_likely = max(other_categories.items(), key=lambda x: x[1])
                    confidence = 0.3 + most_likely[1] * 0.5  # Dynamic confidence
                    
                    return {
                        'prediction': most_likely[0],
                        'confidence': min(0.7, max(0.35, confidence)),
                        'reason': f"Consecutive {current} avoid, সাম্প্রতিক: {most_likely[0]} ({most_likely[1]*100:.0f}%)",
                        'probabilities': freq
                    }
        
        # Normal frequency-based prediction
        if len(self.recent_games) >= 15:
            recent = list(self.recent_games)[-15:]
            freq = {
                'LOW': recent.count('LOW') / len(recent),
                'MIDDLE': recent.count('MIDDLE') / len(recent),
                'HIGH': recent.count('HIGH') / len(recent)
            }
            
            most_common = max(freq.items(), key=lambda x: x[1])
            confidence = 0.35 + most_common[1] * 0.5  # Dynamic confidence
            
            # Adjust confidence based on streak
            if self.consecutive_same >= 2:
                confidence *= 0.85  # Less penalty
            
            return {
                'prediction': most_common[0],
                'confidence': min(0.75, max(0.35, confidence)),
                'reason': f"সাম্প্রতিক ফ্রিকোয়েন্সি: {most_common[0]} ({most_common[1]*100:.0f}%)",
                'probabilities': freq
            }
        
        # Overall frequency
        total = sum(self.category_counts.values())
        if total > 0:
            freq = {
                'LOW': self.category_counts['LOW'] / total,
                'MIDDLE': self.category_counts['MIDDLE'] / total,
                'HIGH': self.category_counts['HIGH'] / total
            }
            
            most_common = max(freq.items(), key=lambda x: x[1])
            confidence = 0.4 + most_common[1] * 0.4
            
            return {
                'prediction': most_common[0],
                'confidence': min(0.7, max(0.4, confidence)),
                'reason': f"সামগ্রিক ফ্রিকোয়েন্সি: {most_common[0]} ({most_common[1]*100:.0f}%)",
                'probabilities': freq
            }
        
        # ডিফল্ট
        return {
            'prediction': 'LOW',
            'confidence': 0.45,
            'reason': 'ডিফল্ট প্রেডিকশন',
            'probabilities': self.default_probabilities
        }


class AdaptiveLearningSystem:
    """অ্যাডাপ্টিভ লার্নিং সিস্টেম v3.0"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=50)
        self.system_performance = {
            'pattern': {'total': 0, 'correct': 0, 'recent': deque(maxlen=10)},
            'backup': {'total': 0, 'correct': 0, 'recent': deque(maxlen=10)},
            'adaptive': {'total': 0, 'correct': 0, 'recent': deque(maxlen=10)}
        }
        
        self.current_learning_rate = 0.15  # Increased
        self.confidence_history = deque(maxlen=20)
        self.error_patterns = defaultdict(int)
        self.success_patterns = defaultdict(int)
        
        self.prediction_strategies = ['pattern', 'adaptive', 'backup']
        self.strategy_weights = {'pattern': 0.45, 'adaptive': 0.35, 'backup': 0.20}
        self.strategy_performance = {
            'pattern': 0.5,
            'adaptive': 0.5,
            'backup': 0.5
        }
        
        self.default_probabilities = {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
        
    def add_game(self, category, timestamp):
        """গেম যোগ করুন"""
        # MEDIUM → MIDDLE ফিক্স
        if category == 'MEDIUM':
            category = 'MIDDLE'
            
        # সময়ভিত্তিক trends ট্র্যাক
        hour = timestamp.hour
        self.performance_history.append({
            'category': category,
            'hour': hour,
            'timestamp': timestamp
        })
    
    def learn_from_result(self, predicted, actual, system_used, confidence):
        """ফলাফল থেকে শিখুন"""
        is_correct = predicted == actual
        
        # সিস্টেম পারফরম্যান্স আপডেট
        if system_used in self.system_performance:
            perf = self.system_performance[system_used]
            perf['total'] += 1
            if is_correct:
                perf['correct'] += 1
                perf['recent'].append(1)
                self.success_patterns[f"{predicted}_{actual}"] += 1
            else:
                perf['recent'].append(0)
                self.error_patterns[f"{predicted}_{actual}"] += 1
            
            # স্ট্র্যাটেজি পারফরম্যান্স আপডেট
            if perf['recent']:
                self.strategy_performance[system_used] = sum(perf['recent']) / len(perf['recent'])
        
        # কনফিডেন্স history আপডেট
        self.confidence_history.append({
            'confidence': confidence,
            'correct': is_correct,
            'system': system_used
        })
        
        # লার্নিং রেট adjust
        self._adjust_learning_rate(is_correct)
        
        # স্ট্র্যাটেজি weights adjust
        self._adjust_strategy_weights()
    
    def _adjust_learning_rate(self, is_correct):
        """লার্নিং রেট adjust করুন"""
        if len(self.confidence_history) >= 8:  # Reduced from 10
            recent = list(self.confidence_history)[-8:]
            correct_rate = sum(1 for x in recent if x['correct']) / len(recent)
            
            if correct_rate < 0.4:
                self.current_learning_rate = min(0.25, self.current_learning_rate * 1.15)  # Less aggressive
            elif correct_rate > 0.7:
                self.current_learning_rate = max(0.08, self.current_learning_rate * 0.9)   # Less aggressive
    
    def _adjust_strategy_weights(self):
        """স্ট্র্যাটেজি weights adjust করুন"""
        # সাম্প্রতিক পারফরম্যান্স দেখে weights adjust
        total_perf = sum(self.strategy_performance.values())
        if total_perf > 0:
            for strategy in self.strategy_weights:
                perf = self.strategy_performance[strategy]
                # More emphasis on recent performance
                self.strategy_weights[strategy] = (perf / total_perf) * 0.7 + 0.3/3
        
        # Normalize weights
        total = sum(self.strategy_weights.values())
        if total > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total
    
    def get_prediction_strategy(self):
        """প্রেডিকশন স্ট্র্যাটেজি বাছাই করুন"""
        # Weighted random selection based on performance
        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        
        # সাম্প্রতিক errors বেশি হলে adaptive strategy prioritize
        if len(self.confidence_history) >= 5:
            recent_errors = sum(1 for x in list(self.confidence_history)[-5:] if not x['correct'])
            if recent_errors >= 3:
                return 'adaptive'
        
        # Weighted selection
        return random.choices(strategies, weights=weights, k=1)[0]
    
    def predict(self):
        """অ্যাডাপ্টিভ প্রেডিকশন দিন - IMPROVED VERSION"""
        # Success patterns follow (priority 1)
        if self.success_patterns and len(self.success_patterns) >= 2:
            most_common_success = max(self.success_patterns.items(), key=lambda x: x[1])
            success_from, success_to = most_common_success[0].split('_')
            
            # Calculate confidence based on success rate
            total_success = sum(self.success_patterns.values())
            success_rate = most_common_success[1] / total_success
            confidence = 0.45 + success_rate * 0.4  # Dynamic confidence
            
            return {
                'prediction': success_to,
                'confidence': min(0.75, max(0.4, confidence)),
                'reason': f"Success pattern follow: {success_from} → {success_to} ({success_rate*100:.0f}%)",
                'pattern_type': 'success_follow',
                'probabilities': self._calculate_success_probabilities()
            }
        
        # Error patterns avoid (priority 2)
        if self.error_patterns and len(self.error_patterns) >= 2:
            most_common_error = max(self.error_patterns.items(), key=lambda x: x[1])
            error_from, error_to = most_common_error[0].split('_')
            
            # Calculate what to predict instead
            alternatives = ['LOW', 'MIDDLE', 'HIGH']
            # Don't remove, just avoid predicting the error_to
            # Instead predict based on what's most successful
            
            # Look for success patterns with same 'from'
            best_alternative = None
            best_count = 0
            
            for pattern, count in self.success_patterns.items():
                s_from, s_to = pattern.split('_')
                if s_from == error_from and count > best_count:
                    best_alternative = s_to
                    best_count = count
            
            if best_alternative:
                confidence = 0.4 + (best_count / (best_count + most_common_error[1])) * 0.3
                return {
                    'prediction': best_alternative,
                    'confidence': min(0.7, max(0.35, confidence)),
                    'reason': f"Error avoid: {error_from}→{error_to}, Success: {error_from}→{best_alternative}",
                    'pattern_type': 'error_avoid',
                    'probabilities': self.default_probabilities
                }
            
            # If no success pattern, choose randomly but not error_to
            alternatives.remove(error_to)
            if alternatives:
                prediction = random.choice(alternatives)
                confidence = 0.35
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'reason': f"Error pattern avoid: {error_from} → {error_to}",
                    'pattern_type': 'error_avoidance',
                    'probabilities': self.default_probabilities
                }
        
        # ডিফল্ট - frequency based on performance history
        if self.performance_history:
            categories = [item['category'] for item in self.performance_history]
            freq = {
                'LOW': categories.count('LOW') / len(categories),
                'MIDDLE': categories.count('MIDDLE') / len(categories),
                'HIGH': categories.count('HIGH') / len(categories)
            }
            
            most_common = max(freq.items(), key=lambda x: x[1])
            confidence = 0.35 + most_common[1] * 0.3
            
            return {
                'prediction': most_common[0],
                'confidence': min(0.65, max(0.35, confidence)),
                'reason': f"Adaptive frequency: {most_common[0]} ({most_common[1]*100:.0f}%)",
                'pattern_type': 'adaptive_frequency',
                'probabilities': freq
            }
        
        # Ultimate fallback
        return {
            'prediction': 'LOW',
            'confidence': 0.45,
            'reason': 'অ্যাডাপ্টিভ ডিফল্ট',
            'pattern_type': 'adaptive_default',
            'probabilities': self.default_probabilities
        }
    
    def _calculate_success_probabilities(self):
        """Calculate probabilities based on success patterns"""
        if not self.success_patterns:
            return self.default_probabilities
        
        total_success = sum(self.success_patterns.values())
        probs = defaultdict(float)
        
        for pattern, count in self.success_patterns.items():
            _, to_cat = pattern.split('_')
            probs[to_cat] += count / total_success
        
        # Add small probability for other categories
        for cat in ['LOW', 'MIDDLE', 'HIGH']:
            if cat not in probs:
                probs[cat] = 0.1  # Small chance
        
        # Normalize
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def get_stats(self):
        """অ্যাডাপ্টিভ সিস্টেম stats"""
        return {
            'learning_rate': round(self.current_learning_rate, 3),
            'strategy_weights': {k: round(v, 3) for k, v in self.strategy_weights.items()},
            'strategy_performance': {k: round(v, 3) for k, v in self.strategy_performance.items()},
            'error_patterns': dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:3]),
            'success_patterns': dict(sorted(self.success_patterns.items(), key=lambda x: x[1], reverse=True)[:3]),
            'total_patterns': len(self.success_patterns) + len(self.error_patterns)
        }


# ==================== টেস্টিং ====================
if __name__ == "__main__":
    print("🧪 উন্নত DPI সিস্টেম v3.0 টেস্টিং...")
    
    # টেস্ট সিস্টেম তৈরি
    system = IntelligentPredictionSystem()
    
    # টেস্ট ডেটা - Realistic patterns
    test_patterns = ['LOW', 'LOW', 'HIGH', 'LOW', 'MIDDLE', 'HIGH', 'HIGH', 'LOW']
    
    print("\n📊 টেস্ট প্যাটার্ন ইনপুট:")
    for i, category in enumerate(test_patterns):
        game_data = {
            'total': 15 if category == 'HIGH' else 7 if category == 'LOW' else 10,
            'dice1': 1,
            'dice2': 2,
            'dice3': 3,
            'timestamp': datetime.now() - timedelta(minutes=len(test_patterns)-i),
            'game_id': f'test_{i}'
        }
        
        system.add_game_data(game_data)
        print(f"  Game {i+1}: {category}")
    
    print("\n🎯 প্রেডিকশন টেস্ট:")
    prediction = system.predict_next_game()
    print(f"  Prediction: {prediction['prediction']}")
    print(f"  Confidence: {prediction['confidence']*100:.1f}%")
    print(f"  Reason: {prediction['reason']}")
    print(f"  System: {prediction['system_status']}")
    
    if 'probabilities' in prediction:
        print(f"  Probabilities: LOW={prediction['probabilities'].get('LOW',0)*100:.1f}%, "
              f"MIDDLE={prediction['probabilities'].get('MIDDLE',0)*100:.1f}%, "
              f"HIGH={prediction['probabilities'].get('HIGH',0)*100:.1f}%")
    
    print("\n📈 সিস্টেম স্ট্যাটস:")
    stats = system.get_system_stats()
    print(f"  Total Games: {stats['total_games_analyzed']}")
    print(f"  Distribution: {stats['class_distribution']}")
    print(f"  Pattern Accuracy: {stats['system_accuracy']['pattern']}%")
    print(f"  Learning Effectiveness: {stats['learning_metrics']['learning_effectiveness']}")
    
    print("\n✅ উন্নত DPI সিস্টেম v3.0 রেডি!")
