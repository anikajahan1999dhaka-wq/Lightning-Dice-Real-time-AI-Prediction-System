"""
ডাইস প্যাটার্ন ইন্টেলিজেন্স সিস্টেম v4.0
স্মার্ট লার্নিং AI: সাম্প্রতিক ২ গেম দেখে, সব প্যাটার্নে একই লজিক, টাইম-অ্যাওয়্যার
"""

import numpy as np
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
import hashlib
import random
import math

class IntelligentPredictionSystem:
    """স্মার্ট ডাইস প্যাটার্ন ইন্টেলিজেন্স সিস্টেম"""
    
    def __init__(self, memory_hours=72):
        self.memory_hours = memory_hours
        self.games_data = deque(maxlen=50000)
        self.predictions_history = []
        
        # ==================== স্মার্ট প্যাটার্ন সিস্টেম ====================
        self.pattern_system = SmartPatternIntelligenceSystem()
        
        # ==================== সেশন মেমোরি ====================
        self.session_memory = AdvancedSessionMemory()
        
        # ==================== ব্যাকআপ সিস্টেম ====================
        self.backup_system = EnhancedBackupPredictionSystem()
        
        # ==================== পারফরম্যান্স ট্র্যাকিং ====================
        self.performance = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'recent_accuracy': deque(maxlen=100),
            'system_used': {'pattern': 0, 'backup': 0},
            'pattern_accuracy': 0,
            'backup_accuracy': 0,
            'confidence_calibration_error': 0,
            'learning_effectiveness': 0.5
        }
        
        print("🧠 স্মার্ট ডাইস প্যাটার্ন ইন্টেলিজেন্স সিস্টেম v4.0 শুরু হয়েছে...")
        print("🎯 ফিচার: সাম্প্রতিক ২-গেম এনালাইসিস, টাইম-অ্যাওয়্যার, সঠিক শেখা")
    
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
                
                self.performance['recent_accuracy'].append(1 if is_correct else 0)
                
                # কনফিডেন্স ক্যালিব্রেশন error ট্র্যাক
                confidence_error = abs(pred['confidence'] - (1 if is_correct else 0))
                self.performance['confidence_calibration_error'] = (
                    self.performance['confidence_calibration_error'] * 0.9 + confidence_error * 0.1
                )
                
                # প্যাটার্ন সিস্টেমকে ফলাফল জানান
                self.pattern_system.learn_from_result(
                    predicted_category, 
                    actual_category,
                    pred.get('pattern_info', {}),
                    pred['confidence'],
                    pred.get('timestamp')
                )
                
                # লার্নিং ইফেক্টিভনেস আপডেট
                recent_accuracy = list(self.performance['recent_accuracy'])[-10:]
                if len(recent_accuracy) >= 5:
                    accuracy_rate = sum(recent_accuracy) / len(recent_accuracy)
                    self.performance['learning_effectiveness'] = accuracy_rate
                
                # লগ
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"\n🎯 {status}: Predicted {predicted_category}, Actual {actual_category}")
                print(f"   System: {system_used} | Confidence: {pred['confidence']*100:.1f}%")
                
                return is_correct
        
        return None
    
    def predict_next_game(self):
        """পরবর্তী গেমের প্রেডিকশন করুন"""
        
        # পর্যাপ্ত ডেটা না থাকলে ব্যাকআপ সিস্টেম ব্যবহার করুন
        if len(self.games_data) < 10:
            return self._get_backup_prediction()
        
        # প্যাটার্ন সিস্টেম থেকে প্রেডিকশন নিন
        pattern_prediction = self.pattern_system.predict_next()
        
        if pattern_prediction['confidence'] >= 0.4:  # লো থ্রেশহোল্ড
            self.performance['system_used']['pattern'] += 1
            return self._create_prediction_record(pattern_prediction, 'pattern')
        
        # ব্যাকআপ সিস্টেম
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
            'probabilities': prediction['probabilities'],
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
        
        # প্যাটার্ন সিস্টেম stats
        pattern_stats = self.pattern_system.get_stats()
        
        return {
            'total_games_analyzed': len(self.games_data),
            'total_predictions': total_evaluated,
            'correct_predictions': correct_evaluated,
            'accuracy_percentage': round(accuracy, 1),
            'recent_accuracy': round(recent_acc, 1),
            'class_distribution': class_distribution,
            'system_usage': {
                'pattern': self.performance['system_used']['pattern'],
                'backup': self.performance['system_used']['backup'],
                'pattern_percentage': round(self.performance['system_used']['pattern'] / max(1, total_evaluated) * 100, 1)
            },
            'system_accuracy': {
                'pattern': round(self.performance['pattern_accuracy'] * 100, 1),
                'backup': round(self.performance['backup_accuracy'] * 100, 1),
                'overall': round(accuracy, 1)
            },
            'learning_metrics': {
                'confidence_error': round(self.performance['confidence_calibration_error'], 3),
                'learning_effectiveness': round(self.performance['learning_effectiveness'], 2)
            },
            'pattern_system': pattern_stats,
            'session_info': self.session_memory.get_stats(),
            'system_status': 'active' if len(self.games_data) > 20 else 'learning'
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


class PatternMemory:
    """একটি প্যাটার্নের জন্য মেমোরি"""
    
    def __init__(self, pattern_key):
        self.pattern_key = pattern_key
        self.success_count = 0
        self.error_count = 0
        self.success_history = []  # সঠিক হলে পরেরটা কি হয়েছিল
        self.error_history = []    # ভুল হলে পরেরটা কি হয়েছিল
        self.time_stats = defaultdict(lambda: {'success': 0, 'error': 0})
        self.last_updated = None
        
    def add_result(self, is_correct, next_outcome, timestamp):
        """রেজাল্ট যোগ করুন"""
        hour = timestamp.hour
        
        if is_correct:
            self.success_count += 1
            self.success_history.append(next_outcome)
            self.time_stats[hour]['success'] += 1
        else:
            self.error_count += 1
            self.error_history.append(next_outcome)
            self.time_stats[hour]['error'] += 1
        
        self.last_updated = timestamp
        
        # মেমোরি ম্যানেজমেন্ট
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def get_success_rate(self):
        """সাকসেস রেট পান"""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.5
    
    def get_time_effectiveness(self, hour):
        """নির্দিষ্ট সময়ে ইফেক্টিভনেস"""
        if hour in self.time_stats:
            stats = self.time_stats[hour]
            total = stats['success'] + stats['error']
            return stats['success'] / total if total > 0 else 0.5
        return 0.5
    
    def predict_next(self, current_hour):
        """পরবর্তী আউটকাম প্রেডিক্ট করুন"""
        if self.success_count + self.error_count == 0:
            return None, 0.5
        
        # সাকসেস হিস্ট্রি থেকে প্রেডিকশন
        success_pred = None
        success_confidence = 0
        if self.success_history:
            success_counter = Counter(self.success_history[-20:])  # শেষ ২০টি
            success_pred, success_count = success_counter.most_common(1)[0]
            success_confidence = success_count / len(self.success_history[-20:])
        
        # এরর হিস্ট্রি থেকে প্রেডিকশন
        error_pred = None
        error_confidence = 0
        if self.error_history:
            error_counter = Counter(self.error_history[-20:])  # শেষ ২০টি
            error_pred, error_count = error_counter.most_common(1)[0]
            error_confidence = error_count / len(self.error_history[-20:])
        
        # টাইম ইফেক্টিভনেস
        time_factor = self.get_time_effectiveness(current_hour)
        
        # কম্বাইন প্রেডিকশন
        if success_pred and error_pred:
            # ওয়েটেড এভারেজ
            total_confidence = success_confidence + error_confidence
            if total_confidence > 0:
                success_weight = success_confidence / total_confidence * time_factor
                error_weight = error_confidence / total_confidence * (1 - time_factor)
                
                # Success-এ বেশি priority
                if success_weight >= error_weight:
                    return success_pred, success_weight
                else:
                    return error_pred, error_weight
        
        elif success_pred:
            return success_pred, success_confidence * time_factor
        
        elif error_pred:
            return error_pred, error_confidence * (1 - time_factor)
        
        return None, 0.5
    
    def get_stats(self):
        """স্ট্যাটিস্টিক্স পান"""
        return {
            'pattern': self.pattern_key,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': round(self.get_success_rate() * 100, 1),
            'recent_success': self.success_history[-5:] if self.success_history else [],
            'recent_error': self.error_history[-5:] if self.error_history else [],
            'total_occurrences': self.success_count + self.error_count
        }


class SmartPatternIntelligenceSystem:
    """স্মার্ট প্যাটার্ন ইন্টেলিজেন্স সিস্টেম"""
    
    def __init__(self):
        # ৯টি বেসিক প্যাটার্ন
        self.patterns_list = [
            'HIGH_LOW',    # HIGH→LOW
            'LOW_HIGH',    # LOW→HIGH  
            'MIDDLE_HIGH', # MIDDLE→HIGH
            'HIGH_MIDDLE', # HIGH→MIDDLE
            'LOW_MIDDLE',  # LOW→MIDDLE
            'MIDDLE_LOW',  # MIDDLE→LOW
            'HIGH_HIGH',   # HIGH→HIGH
            'MIDDLE_MIDDLE', # MIDDLE→MIDDLE
            'LOW_LOW'      # LOW→LOW
        ]
        
        # প্রতিটি প্যাটার্নের জন্য আলাদা মেমোরি
        self.pattern_memories = {}
        for pattern in self.patterns_list:
            self.pattern_memories[pattern] = PatternMemory(pattern)
        
        # সাম্প্রতিক আউটকাম (শেষ ২০টি)
        self.recent_outcomes = deque(maxlen=50000)
        
        # কারেন্ট প্যাটার্ন
        self.current_pattern = None
        self.last_two_games = deque(maxlen=2)
        
        # ট্রানজিশন ম্যাট্রিক্স
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        
        print("🧠 স্মার্ট প্যাটার্ন সিস্টেম v4.0 লোড করা হয়েছে")
        print("📊 ৯টি প্যাটার্ন, সাম্প্রতিক ২-গেম এনালাইসিস, টাইম-অ্যাওয়্যার")
    
    def add_game(self, category, timestamp):
        """নতুন গেম যোগ করুন"""
        # MEDIUM → MIDDLE ফিক্স
        if category == 'MEDIUM':
            category = 'MIDDLE'
        
        self.recent_outcomes.append((category, timestamp))
        self.last_two_games.append((category, timestamp))
        
        # ট্রানজিশন ম্যাট্রিক্স আপডেট
        if len(self.recent_outcomes) >= 2:
            prev_category, _ = self.recent_outcomes[-2]
            self.transition_matrix[prev_category][category] += 1
        
        # কারেন্ট প্যাটার্ন আপডেট
        self._update_current_pattern(category, timestamp)
    
    def _update_current_pattern(self, current_category, timestamp):
        """কারেন্ট প্যাটার্ন আপডেট করুন"""
        if len(self.last_two_games) < 2:
            return
        
        # শেষ ২টি গেম
        game1_category, game1_time = self.last_two_games[0]
        game2_category, _ = self.last_two_games[1]
        
        # প্যাটার্ন কী তৈরি
        pattern_key = f"{game1_category}_{game2_category}"
        
        if pattern_key in self.pattern_memories:
            self.current_pattern = pattern_key
        else:
            self.current_pattern = None
    
    def learn_from_result(self, predicted, actual, pattern_info, confidence, timestamp=None):
        """ফলাফল থেকে শিখুন"""
        if not self.current_pattern or timestamp is None:
            return
        
        # প্যাটার্ন মেমোরি খুঁজুন
        pattern_memory = self.pattern_memories.get(self.current_pattern)
        if not pattern_memory:
            return
        
        # পরের আউটকাম কী ছিল?
        if len(self.recent_outcomes) >= 1:
            next_outcome, _ = self.recent_outcomes[-1]
            is_correct = (predicted == actual)
            
            # মেমোরিতে যোগ করুন
            pattern_memory.add_result(is_correct, next_outcome, timestamp)
    
    def predict_next(self):
        """পরবর্তী আউটকাম প্রেডিক্ট করুন"""
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # ১. যদি কারেন্ট প্যাটার্ন থাকে
        if self.current_pattern:
            pattern_memory = self.pattern_memories.get(self.current_pattern)
            if pattern_memory:
                prediction, confidence = pattern_memory.predict_next(current_hour)
                
                if prediction and confidence > 0.4:
                    # ট্রানজিশন প্রোবাবিলিটি
                    last_category = self.current_pattern.split('_')[-1]
                    transition_probs = self._get_transition_probabilities(last_category)
                    
                    reason = f"প্যাটার্ন: {self.current_pattern}"
                    if pattern_memory.success_count > 0:
                        success_rate = pattern_memory.get_success_rate()
                        reason += f", সাকসেস: {success_rate*100:.0f}%"
                    
                    return {
                        'prediction': prediction,
                        'confidence': min(0.85, confidence * 1.2),  # Boost confidence
                        'reason': reason,
                        'pattern_type': self.current_pattern,
                        'pattern_detected': True,
                        'probabilities': transition_probs
                    }
        
        # ২. ট্রানজিশন-বেসড প্রেডিকশন
        if len(self.recent_outcomes) >= 1:
            last_category, _ = self.recent_outcomes[-1]
            transition_probs = self._get_transition_probabilities(last_category)
            
            if transition_probs:
                most_likely = max(transition_probs.items(), key=lambda x: x[1])
                confidence = 0.3 + most_likely[1] * 0.5
                
                return {
                    'prediction': most_likely[0],
                    'confidence': min(0.75, confidence),
                    'reason': f"ট্রানজিশন প্রোবাবিলিটি: {last_category}→{most_likely[0]} ({most_likely[1]*100:.0f}%)",
                    'pattern_type': 'transition',
                    'pattern_detected': True,
                    'probabilities': transition_probs
                }
        
        # ৩. ফ্রিকোয়েন্সি-বেসড প্রেডিকশন
        if self.recent_outcomes:
            recent_categories = [cat for cat, _ in list(self.recent_outcomes)[-10:]]
            freq = Counter(recent_categories)
            total = len(recent_categories)
            
            if total > 0:
                most_common = freq.most_common(1)[0]
                confidence = 0.35 + (most_common[1] / total) * 0.4
                
                probs = {}
                for cat in ['LOW', 'MIDDLE', 'HIGH']:
                    probs[cat] = freq.get(cat, 0) / total
                
                return {
                    'prediction': most_common[0],
                    'confidence': min(0.7, confidence),
                    'reason': f"সাম্প্রতিক ফ্রিকোয়েন্সি: {most_common[0]} ({most_common[1]*100/total:.0f}%)",
                    'pattern_type': 'frequency',
                    'pattern_detected': False,
                    'probabilities': probs
                }
        
        # ৪. ডিফল্ট
        return {
            'prediction': 'LOW',
            'confidence': 0.45,
            'reason': 'ডিফল্ট প্রেডিকশন',
            'pattern_type': 'default',
            'pattern_detected': False,
            'probabilities': {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
        }
    
    def _get_transition_probabilities(self, from_category):
        """ট্রানজিশন প্রোবাবিলিটি পান"""
        if from_category not in self.transition_matrix:
            return {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
        
        transitions = self.transition_matrix[from_category]
        total = sum(transitions.values())
        
        if total == 0:
            return {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
        
        probs = {}
        for cat in ['LOW', 'MIDDLE', 'HIGH']:
            probs[cat] = transitions.get(cat, 1) / (total + 3)  # Add-1 smoothing
        
        # নরমালাইজ
        total_prob = sum(probs.values())
        return {k: v/total_prob for k, v in probs.items()}
    
    def get_stats(self):
        """সিস্টেম স্ট্যাটিস্টিক্স পান"""
        pattern_stats = {}
        for pattern_key, memory in self.pattern_memories.items():
            if memory.success_count + memory.error_count > 0:
                pattern_stats[pattern_key] = memory.get_stats()
        
        # টপ ৫ প্যাটার্ন
        top_patterns = dict(sorted(
            pattern_stats.items(),
            key=lambda x: x[1]['total_occurrences'],
            reverse=True
        )[:5])
        
        return {
            'total_patterns': len([p for p in self.pattern_memories.values() 
                                  if p.success_count + p.error_count > 0]),
            'top_patterns': top_patterns,
            'current_pattern': self.current_pattern,
            'recent_outcomes_count': len(self.recent_outcomes),
            'transition_matrix_size': len(self.transition_matrix)
        }


class AdvancedSessionMemory:
    """এডভান্সড সেশন-বেজড মেমোরি"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.session_games = []
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        
    def add_game(self, category, timestamp):
        """সেশনে গেম যোগ করুন"""
        # MEDIUM → MIDDLE ফিক্স
        if category == 'MEDIUM':
            category = 'MIDDLE'
            
        game_record = {
            'category': category,
            'timestamp': timestamp,
            'hour': timestamp.hour
        }
        
        self.session_games.append(game_record)
        
        # Hourly stats
        hour = timestamp.hour
        self.hourly_stats[hour][category] += 1
        
        # মেমোরি ম্যানেজমেন্ট
        if len(self.session_games) > 2000:
            self.session_games = self.session_games[-2000:]
    
    def get_stats(self):
        """সেশন স্ট্যাটস পান"""
        if not self.session_games:
            return {'total_games': 0, 'session_duration': 0}
        
        session_duration = (datetime.now() - self.session_start).total_seconds() / 3600
        
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
            'hourly_distribution': hourly_dist
        }


class EnhancedBackupPredictionSystem:
    """এনহ্যান্সড ব্যাকআপ প্রেডিকশন সিস্টেম"""
    
    def __init__(self):
        self.category_counts = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        self.recent_games = deque(maxlen=50000)
        self.default_probabilities = {'LOW': 0.44, 'MIDDLE': 0.17, 'HIGH': 0.39}
        
    def add_game(self, category):
        """গেম যোগ করুন"""
        # MEDIUM → MIDDLE ফিক্স
        if category == 'MEDIUM':
            category = 'MIDDLE'
            
        self.category_counts[category] += 1
        self.recent_games.append(category)
    
    def predict(self):
        """প্রেডিকশন করুন"""
        # সাম্প্রতিক ফ্রিকোয়েন্সি
        if len(self.recent_games) >= 10:
            recent = list(self.recent_games)[-10:]
            freq = {
                'LOW': recent.count('LOW') / len(recent),
                'MIDDLE': recent.count('MIDDLE') / len(recent),
                'HIGH': recent.count('HIGH') / len(recent)
            }
            
            most_common = max(freq.items(), key=lambda x: x[1])
            confidence = 0.4 + most_common[1] * 0.4
            
            return {
                'prediction': most_common[0],
                'confidence': min(0.7, confidence),
                'reason': f"সাম্প্রতিক ফ্রিকোয়েন্সি: {most_common[0]} ({most_common[1]*100:.0f}%)",
                'probabilities': freq
            }
        
        # সামগ্রিক ফ্রিকোয়েন্সি
        total = sum(self.category_counts.values())
        if total > 0:
            freq = {
                'LOW': self.category_counts['LOW'] / total,
                'MIDDLE': self.category_counts['MIDDLE'] / total,
                'HIGH': self.category_counts['HIGH'] / total
            }
            
            most_common = max(freq.items(), key=lambda x: x[1])
            confidence = 0.45 + most_common[1] * 0.3
            
            return {
                'prediction': most_common[0],
                'confidence': min(0.65, confidence),
                'reason': f"সামগ্রিক ফ্রিকোয়েন্সি: {most_common[0]} ({most_common[1]*100:.0f}%)",
                'probabilities': freq
            }
        
        # ডিফল্ট
        return {
            'prediction': 'LOW',
            'confidence': 0.5,
            'reason': 'ডিফল্ট প্রেডিকশন',
            'probabilities': self.default_probabilities
        }


# ==================== টেস্টিং ====================
if __name__ == "__main__":
    print("🧪 স্মার্ট DPI সিস্টেম v4.0 টেস্টিং...")
    
    # টেস্ট সিস্টেম তৈরি
    system = IntelligentPredictionSystem()
    
    # টেস্ট ডেটা - রিয়েলিস্টিক প্যাটার্ন
    test_data = [
        ('LOW', 7), ('LOW', 7), ('HIGH', 15), ('LOW', 8), 
        ('MIDDLE', 10), ('HIGH', 14), ('HIGH', 16), ('LOW', 9)
    ]
    
    print("\n📊 টেস্ট ডেটা ইনপুট:")
    for i, (category, total) in enumerate(test_data):
        game_data = {
            'total': total,
            'dice1': 1,
            'dice2': 2,
            'dice3': 3,
            'timestamp': datetime.now() - timedelta(minutes=len(test_data)-i),
            'game_id': f'test_{i}'
        }
        
        system.add_game_data(game_data)
        print(f"  Game {i+1}: {category} (Total: {total})")
    
    print("\n🎯 প্রেডিকশন টেস্ট:")
    prediction = system.predict_next_game()
    print(f"  Prediction: {prediction['prediction']}")
    print(f"  Confidence: {prediction['confidence']*100:.1f}%")
    print(f"  Reason: {prediction['reason']}")
    print(f"  System: {prediction['system_status']}")
    
    if 'probabilities' in prediction:
        probs = prediction['probabilities']
        print(f"  Probabilities: LOW={probs.get('LOW',0)*100:.1f}%, "
              f"MIDDLE={probs.get('MIDDLE',0)*100:.1f}%, "
              f"HIGH={probs.get('HIGH',0)*100:.1f}%")
    
    print("\n📈 সিস্টেম স্ট্যাটস:")
    stats = system.get_system_stats()
    print(f"  Total Games: {stats['total_games_analyzed']}")
    print(f"  Distribution: {stats['class_distribution']}")
    
    print("\n✅ স্মার্ট DPI সিস্টেম v4.0 রেডি!")
