"""
ইন্টেলিজেন্ট প্রেডিকশন সিস্টেম
72-ঘণ্টার ডেটা + লাইভ ডেটা এনালাইসিস করে High/Low/Middle প্রেডিকশন দেয়
"""

import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib

class IntelligentPredictionSystem:
    """ইন্টেলিজেন্ট AI সিস্টেম Lightning Dice প্রেডিকশনের জন্য"""
    
    def __init__(self, memory_hours=72):
        self.memory_hours = memory_hours
        self.games_data = deque(maxlen=50000)  # deque ব্যবহার করে memory optimize
        self.predictions_history = []  # প্রেডিকশন হিস্ট্রি
        self.model_knowledge = {}  # AI এর জ্ঞানভাণ্ডার
        
        # স্ট্যাটিসটিকাল মেমোরি
        self.pattern_memory = defaultdict(list)
        self.streak_memory = defaultdict(int)
        self.time_patterns = defaultdict(lambda: defaultdict(int))
        
        # মডেল পারফরম্যান্স ট্র্যাকিং
        self.performance = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'class_accuracy': {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0},
            'recent_accuracy': deque(maxlen=100),
            'predictions_by_category': {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0},
            'correct_by_category': {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        }
        
        # সেলফ-লার্নিং প্যারামিটারস
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
        print("🤖 ইন্টেলিজেন্ট প্রেডিকশন সিস্টেম শুরু হয়েছে...")
    
    def add_game_data(self, game_data):
        """নতুন গেম ডেটা যোগ করুন"""
        self.games_data.append(game_data)
        
        # শুধু শেষ 72 ঘণ্টার ডেটা রাখুন
        cutoff_time = datetime.now() - timedelta(hours=self.memory_hours)
        
        # Remove old games
        while self.games_data and self.games_data[0]['timestamp'] < cutoff_time:
            self.games_data.popleft()
        
        # প্যাটার্ন আপডেট করুন
        self._update_patterns(game_data)
    
    def _update_patterns(self, game_data):
        """ডেটা থেকে প্যাটার্ন শিখুন"""
        total = game_data['total']
        category = self._get_category(total)
        timestamp = game_data['timestamp']
        
        # টাইম প্যাটার্ন
        hour = timestamp.hour
        self.time_patterns[hour][category] += 1
        
        # স্ট্রীক ট্র্যাকিং
        if not hasattr(self, 'last_category'):
            self.last_category = category
            self.current_streak = 1
        else:
            if category == self.last_category:
                self.current_streak += 1
            else:
                self.current_streak = 1
                self.last_category = category
        
        # গ্যাপ অ্যানালাইসিস
        self._update_gap_analysis(category, timestamp)
        
        # ডাইস প্যাটার্ন
        dice_combo = (game_data['dice1'], game_data['dice2'], game_data['dice3'])
        if dice_combo not in self.pattern_memory:
            self.pattern_memory[dice_combo] = []
        self.pattern_memory[dice_combo].append(category)
    
    def _update_gap_analysis(self, category, timestamp):
        """গ্যাপ (কতক্ষণ পরে আবার আসে) অ্যানালাইসিস"""
        if not hasattr(self, 'last_seen'):
            self.last_seen = {}
        
        if category in self.last_seen:
            gap = (timestamp - self.last_seen[category]).total_seconds() / 60  # মিনিটে
            if 'gaps' not in self.model_knowledge:
                self.model_knowledge['gaps'] = defaultdict(list)
            self.model_knowledge['gaps'][category].append(gap)
        
        self.last_seen[category] = timestamp
    
    def _get_category(self, total):
        """টোটাল থেকে ক্যাটাগরি নির্ধারণ"""
        if 3 <= total <= 9:
            return "LOW"
        elif 10 <= total <= 11:
            return "MIDDLE"
        else:  # 12-18
            return "HIGH"
    
    def _calculate_probabilities(self):
        """বিভিন্ন ফ্যাক্টর থেকে প্রোবাবিলিটি ক্যালকুলেট"""
        
        if len(self.games_data) < 10:
            return {'LOW': 0.33, 'MIDDLE': 0.34, 'HIGH': 0.33}
        
        # 1. হিস্টোরিক্যাল ফ্রিকোয়েন্সি
        categories = [self._get_category(g['total']) for g in list(self.games_data)[-100:]]
        freq = {
            'LOW': categories.count('LOW') / len(categories),
            'MIDDLE': categories.count('MIDDLE') / len(categories),
            'HIGH': categories.count('HIGH') / len(categories)
        }
        
        # 2. স্ট্রীক অ্যানালাইসিস
        streak_factor = self._analyze_streaks()
        
        # 3. টাইম-বেজড প্যাটার্ন
        time_factor = self._analyze_time_patterns()
        
        # 4. গ্যাপ অ্যানালাইসিস
        gap_factor = self._analyze_gaps()
        
        # 5. ডাইস প্যাটার্ন
        dice_factor = self._analyze_dice_patterns()
        
        # সব ফ্যাক্টর combine করুন
        final_probs = {}
        for category in ['LOW', 'MIDDLE', 'HIGH']:
            prob = freq[category]
            
            # Apply factors
            if category in streak_factor:
                prob *= streak_factor[category]
            if category in time_factor:
                prob *= time_factor[category]
            if category in gap_factor:
                prob *= gap_factor[category]
            if category in dice_factor:
                prob *= dice_factor[category]
            
            final_probs[category] = prob
        
        # Normalize
        total = sum(final_probs.values())
        if total > 0:
            final_probs = {k: v/total for k, v in final_probs.items()}
        
        return final_probs
    
    def _analyze_streaks(self):
        """স্ট্রীক প্যাটার্ন অ্যানালাইসিস"""
        factors = {'LOW': 1.0, 'MIDDLE': 1.0, 'HIGH': 1.0}
        
        if hasattr(self, 'current_streak') and self.current_streak > 2:
            current_cat = self.last_category
            
            # স্ট্রীক বেশি লম্বা হলে ব্রেক হওয়ার সম্ভাবনা বেশি
            if self.current_streak >= 3:
                for cat in ['LOW', 'MIDDLE', 'HIGH']:
                    if cat != current_cat:
                        factors[cat] = 1.5  # অন্য ক্যাটাগরির সম্ভাবনা বাড়ান
                factors[current_cat] = 0.7  # বর্তমান ক্যাটাগরির সম্ভাবনা কমান
        
        return factors
    
    def _analyze_time_patterns(self):
        """সময়ভিত্তিক প্যাটার্ন"""
        factors = {'LOW': 1.0, 'MIDDLE': 1.0, 'HIGH': 1.0}
        
        current_hour = datetime.now().hour
        
        if current_hour in self.time_patterns:
            hour_data = self.time_patterns[current_hour]
            total_games = sum(hour_data.values())
            
            if total_games > 10:
                for cat, count in hour_data.items():
                    factors[cat] = 1.0 + (count / total_games) * 0.5
        
        return factors
    
    def _analyze_gaps(self):
        """গ্যাপ অ্যানালাইসিস (কতক্ষণ দেখা যায়নি)"""
        factors = {'LOW': 1.0, 'MIDDLE': 1.0, 'HIGH': 1.0}
        
        if hasattr(self, 'last_seen') and 'gaps' in self.model_knowledge:
            current_time = datetime.now()
            
            for category in ['LOW', 'MIDDLE', 'HIGH']:
                if category in self.last_seen:
                    # কতক্ষণ দেখা যায়নি
                    minutes_since_last = (current_time - self.last_seen[category]).total_seconds() / 60
                    
                    # এই ক্যাটাগরির গড় গ্যাপ
                    if category in self.model_knowledge['gaps']:
                        avg_gap = np.mean(self.model_knowledge['gaps'][category])
                        if avg_gap > 0:
                            # গড়ের চেয়ে বেশি সময় হলে আসার সম্ভাবনা বাড়ে
                            if minutes_since_last > avg_gap * 1.5:
                                factors[category] = 1.8
                            elif minutes_since_last > avg_gap:
                                factors[category] = 1.3
    
        return factors
    
    def _analyze_dice_patterns(self):
        """ডাইস কম্বিনেশন প্যাটার্ন"""
        factors = {'LOW': 1.0, 'MIDDLE': 1.0, 'HIGH': 1.0}
        
        if len(self.games_data) > 0:
            last_game = list(self.games_data)[-1] if self.games_data else None
            if last_game:
                last_dice = (last_game['dice1'], last_game['dice2'], last_game['dice3'])
                
                # এই ডাইস কম্বিনেশনের পর সাধারণত কী আসে
                if last_dice in self.pattern_memory and len(self.pattern_memory[last_dice]) > 5:
                    outcomes = self.pattern_memory[last_dice]
                    for cat in ['LOW', 'MIDDLE', 'HIGH']:
                        freq = outcomes.count(cat) / len(outcomes)
                        factors[cat] = 1.0 + freq * 0.3
        
        return factors
    
    def predict_next_game(self):
        """পরবর্তী গেমের প্রেডিকশন করুন"""
        
        if len(self.games_data) < 20:
            return {
                'prediction_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12],
                'prediction': 'LOW',
                'confidence': 0.5,
                'probabilities': {'LOW': 0.33, 'MIDDLE': 0.34, 'HIGH': 0.33},
                'reason': 'পর্যাপ্ত ডেটা নেই, ডিফল্ট প্রেডিকশন',
                'system_status': 'training',
                'timestamp': datetime.now().isoformat(),
                'game_count': len(self.games_data)
            }
        
        # প্রোবাবিলিটি ক্যালকুলেট
        probabilities = self._calculate_probabilities()
        
        # সবচেয়ে বেশি প্রোবাবিলিটি
        prediction = max(probabilities.items(), key=lambda x: x[1])[0]
        confidence = probabilities[prediction]
        
        # কনফিডেন্স ক্যালিব্রেশন
        calibrated_confidence = self._calibrate_confidence(prediction, confidence)
        
        # কারণ/যুক্তি জেনারেট
        reasoning = self._generate_reasoning(prediction, probabilities)
        
        # Generate prediction ID
        prediction_id = hashlib.md5(f"{datetime.now().timestamp()}_{prediction}".encode()).hexdigest()[:12]
        
        # Performance ট্র্যাক করার জন্য স্টোর
        prediction_record = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': calibrated_confidence,
            'probabilities': probabilities.copy(),
            'reason': reasoning,
            'game_count': len(self.games_data),
            'status': 'pending'  # pending, correct, incorrect
        }
        
        self.predictions_history.append(prediction_record)
        self.performance['total_predictions'] += 1
        self.performance['predictions_by_category'][prediction] += 1
        
        return {
            'prediction_id': prediction_id,
            'prediction': prediction,
            'confidence': calibrated_confidence,
            'probabilities': probabilities,
            'reason': reasoning,
            'system_status': 'active',
            'data_points': len(self.games_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calibrate_confidence(self, prediction, raw_confidence):
        """কনফিডেন্স লেভেল ক্যালিব্রেট করুন"""
        
        # স্ট্রীক এর উপর ভিত্তি করে adjust
        if hasattr(self, 'current_streak') and self.current_streak >= 3:
            if prediction != self.last_category:
                # স্ট্রীক ব্রেক করার প্রেডিকশন - কম কনফিডেন্ট
                calibrated = raw_confidence * 0.8
            else:
                # স্ট্রীক continue করার প্রেডিকশন - বেশি কনফিডেন্ট
                calibrated = raw_confidence * 1.2
        else:
            calibrated = raw_confidence
        
        # MIDDLE প্রেডিকশনে কম কনফিডেন্ট (কারণ rare)
        if prediction == 'MIDDLE':
            calibrated *= 0.9
        
        # Ensure between 0.3 and 0.95
        calibrated = max(0.3, min(0.95, calibrated))
        
        return round(calibrated, 2)
    
    def _generate_reasoning(self, prediction, probabilities):
        """প্রেডিকশনের কারণ জেনারেট করুন"""
        
        reasons = []
        
        # 1. প্রোবাবিলিটি ভিত্তিতে
        reasons.append(f"সম্ভাবনা: {prediction} ({probabilities[prediction]*100:.1f}%)")
        
        # 2. স্ট্রীক ভিত্তিতে
        if hasattr(self, 'current_streak') and self.current_streak >= 2:
            reasons.append(f"বর্তমান স্ট্রীক: {self.last_category} ×{self.current_streak}")
        
        # 3. টাইম ভিত্তিতে
        current_hour = datetime.now().hour
        if current_hour in self.time_patterns:
            hour_stats = self.time_patterns[current_hour]
            total = sum(hour_stats.values())
            if total > 5 and prediction in hour_stats:
                percent = (hour_stats[prediction] / total) * 100
                reasons.append(f"এই সময়ে {prediction}: {percent:.0f}%")
        
        # 4. গ্যাপ ভিত্তিতে
        if hasattr(self, 'last_seen') and prediction in self.last_seen:
            minutes_ago = (datetime.now() - self.last_seen[prediction]).total_seconds() / 60
            reasons.append(f"শেষ {prediction}: {minutes_ago:.0f} মিনিট আগে")
        
        return " | ".join(reasons[:3])  # Max 3 reasons
    
    def update_accuracy(self, actual_category, game_id=None, game_timestamp=None):
        """প্রেডিকশন এক্যুরেসি আপডেট করুন"""
        if len(self.predictions_history) == 0:
            return None
        
        # Find the latest pending prediction
        for pred in reversed(self.predictions_history):
            if pred.get('status') == 'pending':
                predicted_category = pred['prediction']
                is_correct = predicted_category == actual_category
                
                # Update prediction record
                pred['actual_category'] = actual_category
                pred['is_correct'] = is_correct
                pred['status'] = 'correct' if is_correct else 'incorrect'
                pred['evaluated_at'] = datetime.now()
                if game_id:
                    pred['game_id'] = game_id
                if game_timestamp:
                    pred['game_timestamp'] = game_timestamp
                
                # Performance আপডেট
                if is_correct:
                    self.performance['correct_predictions'] += 1
                    self.performance['class_accuracy'][predicted_category] += 1
                    self.performance['correct_by_category'][predicted_category] += 1
                
                self.performance['recent_accuracy'].append(1 if is_correct else 0)
                
                # সেলফ-লার্নিং: ভুল হলে কারণ খুঁজুন
                if not is_correct:
                    self._learn_from_mistake(predicted_category, actual_category)
                
                # Log the result
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"\n🎯 {status}: Predicted {predicted_category}, Actual {actual_category}")
                if game_id:
                    print(f"   Game ID: {game_id}")
                print(f"   Confidence: {pred['confidence']*100:.1f}% | Reason: {pred['reason']}")
                
                return is_correct
        
        return None
    
    def _learn_from_mistake(self, predicted, actual):
        """ভুল থেকে শিখুন"""
        print(f"🤔 AI শিখছে: {predicted} ভবিষ্যদ্বাণী করেছিলাম, কিন্তু {actual} এলো")
        
        # ভুলের প্যাটার্ন ট্র্যাক
        mistake_key = f"{predicted}_to_{actual}"
        if 'mistake_patterns' not in self.model_knowledge:
            self.model_knowledge['mistake_patterns'] = defaultdict(int)
        
        self.model_knowledge['mistake_patterns'][mistake_key] += 1
        
        # Learning rate adjust
        if len(self.performance['recent_accuracy']) >= 20:
            recent_acc = sum(self.performance['recent_accuracy']) / len(self.performance['recent_accuracy'])
            if recent_acc < 0.6:
                self.learning_rate = min(0.3, self.learning_rate * 1.1)  # আরও শিখুন
            elif recent_acc > 0.8:
                self.learning_rate = max(0.05, self.learning_rate * 0.9)  # কম শিখুন
    
    def get_pending_predictions(self):
        """Get all pending predictions"""
        return [p for p in self.predictions_history if p.get('status') == 'pending']
    
    def get_system_stats(self):
        """সিস্টেম স্ট্যাটিস্টিক্স পান - শুধু evaluated প্রেডিকশন গণনা"""
        # শুধু evaluated প্রেডিকশন (correct/incorrect) গণনা করুন
        evaluated_predictions = [p for p in self.predictions_history 
                               if p.get('status') in ['correct', 'incorrect']]
        
        total_evaluated = len(evaluated_predictions)
        correct_evaluated = sum(1 for p in evaluated_predictions if p.get('is_correct', False))
        
        accuracy = (correct_evaluated / total_evaluated * 100) if total_evaluated > 0 else 0
        
        # সাম্প্রতিক এক্যুরেসি (শেষ ১০টি evaluated প্রেডিকশন)
        recent_evaluated = evaluated_predictions[-10:] if len(evaluated_predictions) >= 10 else evaluated_predictions
        recent_correct = sum(1 for p in recent_evaluated if p.get('is_correct', False))
        recent_acc = (recent_correct / len(recent_evaluated) * 100) if recent_evaluated else 0
        
        # ক্যাটাগরি ডিস্ট্রিবিউশন
        categories = [self._get_category(g['total']) for g in self.games_data] if self.games_data else []
        total_games = len(categories)
        
        distribution = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        if total_games > 0:
            distribution = {
                'LOW': round(categories.count('LOW') / total_games * 100, 1),
                'MIDDLE': round(categories.count('MIDDLE') / total_games * 100, 1),
                'HIGH': round(categories.count('HIGH') / total_games * 100, 1)
            }
        
        # Category accuracy (শুধু evaluated প্রেডিকশনের জন্য)
        category_accuracy = {}
        predictions_by_category = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        correct_by_category = {'LOW': 0, 'MIDDLE': 0, 'HIGH': 0}
        
        # evaluated প্রেডিকশন থেকে category accuracy গণনা
        for pred in evaluated_predictions:
            pred_category = pred.get('prediction')
            if pred_category in predictions_by_category:
                predictions_by_category[pred_category] += 1
                if pred.get('is_correct', False):
                    correct_by_category[pred_category] += 1
        
        for cat in ['LOW', 'MIDDLE', 'HIGH']:
            total_preds = predictions_by_category[cat]
            correct_preds = correct_by_category[cat]
            cat_acc = (correct_preds / total_preds * 100) if total_preds > 0 else 0
            category_accuracy[cat] = round(cat_acc, 1)
        
        # মোট প্রেডিকশন (সব ধরণের) - শুধু তথ্যের জন্য
        all_predictions_count = len(self.predictions_history)
        pending_predictions_count = len(self.get_pending_predictions())
        
        return {
            'total_games_analyzed': len(self.games_data),
            'total_predictions': total_evaluated,  # ✅ শুধু evaluated প্রেডিকশন
            'all_predictions': all_predictions_count,  # সব প্রেডিকশন (তথ্যের জন্য)
            'pending_predictions': pending_predictions_count,
            'correct_predictions': correct_evaluated,
            'accuracy_percentage': round(accuracy, 1),
            'recent_accuracy': round(recent_acc, 1),
            'class_distribution': distribution,
            'system_status': 'active' if len(self.games_data) > 50 else 'learning',
            'learning_rate': round(self.learning_rate, 2),
            'knowledge_points': len(self.model_knowledge.get('mistake_patterns', {})),
            'category_accuracy': category_accuracy,
            'predictions_by_category': predictions_by_category,  # evaluated প্রেডিকশন
            'correct_by_category': correct_by_category  # evaluated প্রেডিকশন
        }
    
    def get_prediction_history(self, limit=50, include_pending=False):
        """Get prediction history"""
        history = self.predictions_history[-limit:] if self.predictions_history else []
        
        if not include_pending:
            history = [p for p in history if p.get('status') != 'pending']
        
        # Convert datetime objects to strings for JSON serialization
        for pred in history:
            if isinstance(pred.get('timestamp'), datetime):
                pred['timestamp'] = pred['timestamp'].isoformat()
            if isinstance(pred.get('evaluated_at'), datetime):
                pred['evaluated_at'] = pred['evaluated_at'].isoformat()
            if isinstance(pred.get('game_timestamp'), datetime):
                pred['game_timestamp'] = pred['game_timestamp'].isoformat()
        
        return history