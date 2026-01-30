/**
 * 🤖 AI Analysis Dashboard JavaScript
 * Handles AI analysis data visualization
 */

// Global variables
let aiStats = null;
let predictionHistory = [];
let patterns = [];
let timingStats = null;
let featureImportance = [];

// Load all data
async function loadAllData() {
    try {
        await loadAIStats();
        await loadPredictionHistory();
        await loadPatterns();
        await loadTimingStats();
        await loadFeatureImportance();
        
        updateStatsCards();
        updateAccuracySummary();
        updatePredictionTable();
        updatePatternsTable();
        updateTimingInfo();
        updateFeatureImportance();
        updateDebugLogs();
        
        createCharts();
        
        showNotification('✅ AI Analysis data updated', 'success');
    } catch (error) {
        console.error('Error loading AI analysis data:', error);
        showNotification('❌ Error loading AI analysis data', 'error');
    }
}

// Load AI statistics
async function loadAIStats() {
    const response = await fetch('/api/ai-stats');
    const data = await response.json();
    
    if (data.success) {
        aiStats = data.ai_stats;
    }
}

// Load prediction history
async function loadPredictionHistory() {
    const response = await fetch('/api/prediction-history');
    const data = await response.json();
    
    if (data.success) {
        predictionHistory = data.predictions;
    }
}

// Load patterns
async function loadPatterns() {
    const response = await fetch('/api/ai-patterns');
    const data = await response.json();
    
    if (data.success) {
        patterns = data.patterns;
    }
}

// Load timing stats
async function loadTimingStats() {
    const response = await fetch('/api/ai-timing-stats');
    const data = await response.json();
    
    if (data.success) {
        timingStats = data.timing_stats;
    }
}

// Load feature importance
async function loadFeatureImportance() {
    const response = await fetch('/api/ai-analysis');
    const data = await response.json();
    
    if (data.success && data.stats && data.stats.feature_importances) {
        featureImportance = data.stats.feature_importances;
    }
}

// Update stats cards
function updateStatsCards() {
    if (!aiStats) return;
    
    const statsContainer = document.getElementById('stats-cards');
    
    const accuracy = (aiStats.prediction_accuracy || 0) * 100;
    const trainingAccuracy = (aiStats.training_accuracy || 0) * 100;
    
    statsContainer.innerHTML = `
        <div class="stat-card accuracy">
            <div class="stat-label">Prediction Accuracy</div>
            <div class="stat-value">${accuracy.toFixed(1)}%</div>
            <div class="stat-label">${aiStats.checked_predictions || 0} checked predictions</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${accuracy}%; background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);"></div>
            </div>
        </div>
        
        <div class="stat-card low">
            <div class="stat-label">LOW Patterns</div>
            <div class="stat-value">${aiStats.low_patterns_count || 0}</div>
            <div class="stat-label">${(aiStats.category_percentages?.LOW || 0).toFixed(1)}% of games</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${aiStats.category_percentages?.LOW || 0}%; background: linear-gradient(135deg, #28a745 0%, #20c997 100%);"></div>
            </div>
        </div>
        
        <div class="stat-card middle">
            <div class="stat-label">MIDDLE Patterns</div>
            <div class="stat-value">${aiStats.middle_patterns_count || 0}</div>
            <div class="stat-label">${(aiStats.category_percentages?.MIDDLE || 0).toFixed(1)}% of games</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${aiStats.category_percentages?.MIDDLE || 0}%; background: linear-gradient(135deg, #ffc107 0%, #ffd54f 100%);"></div>
            </div>
        </div>
        
        <div class="stat-card high">
            <div class="stat-label">HIGH Patterns</div>
            <div class="stat-value">${aiStats.high_patterns_count || 0}</div>
            <div class="stat-label">${(aiStats.category_percentages?.HIGH || 0).toFixed(1)}% of games</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${aiStats.category_percentages?.HIGH || 0}%; background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);"></div>
            </div>
        </div>
    `;
}

// Update accuracy summary
function updateAccuracySummary() {
    if (!aiStats) return;
    
    const summaryContainer = document.getElementById('accuracy-summary');
    const accuracy = (aiStats.prediction_accuracy || 0) * 100;
    const checked = aiStats.checked_predictions || 0;
    const correct = aiStats.correct_predictions || 0;
    const incorrect = checked - correct;
    
    summaryContainer.innerHTML = `
        <div style="display: flex; gap: 30px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; font-weight: bold; color: #00b4db;">${accuracy.toFixed(1)}%</div>
                <div style="color: #aaa; font-size: 0.9rem;">Overall Accuracy</div>
            </div>
            
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; font-weight: bold; color: #28a745;">${correct}</div>
                <div style="color: #aaa; font-size: 0.9rem;">Correct Predictions</div>
            </div>
            
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; font-weight: bold; color: #dc3545;">${incorrect}</div>
                <div style="color: #aaa; font-size: 0.9rem;">Incorrect Predictions</div>
            </div>
            
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; font-weight: bold; color: #ffc107;">${checked}</div>
                <div style="color: #aaa; font-size: 0.9rem;">Total Checked</div>
            </div>
        </div>
    `;
}

// Update prediction table
function updatePredictionTable() {
    if (!predictionHistory.length) return;
    
    const tableBody = document.getElementById('prediction-table-body');
    tableBody.innerHTML = '';
    
    // Show last 100 predictions
    const recentPredictions = predictionHistory.slice(-100).reverse();
    
    recentPredictions.forEach(pred => {
        const time = new Date(pred.timestamp).toLocaleTimeString();
        const prediction = pred.prediction || 'N/A';
        const confidence = pred.confidence ? (pred.confidence * 100).toFixed(1) + '%' : 'N/A';
        const actual = pred.actual || 'Not yet';
        const system = pred.system_used || 'Unknown';
        
        let status = 'Pending';
        let statusClass = 'prediction-pending';
        
        if ('correct' in pred) {
            if (pred.correct) {
                status = 'Correct ✓';
                statusClass = 'prediction-correct';
            } else {
                status = 'Incorrect ✗';
                statusClass = 'prediction-incorrect';
            }
        }
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${time}</td>
            <td><strong>${prediction}</strong></td>
            <td>${confidence}</td>
            <td>${actual}</td>
            <td class="${statusClass}">${status}</td>
            <td>${system}</td>
        `;
        tableBody.appendChild(row);
    });
    
    document.getElementById('prediction-table-content').style.display = 'block';
    document.querySelector('#predictions-tab .loading').style.display = 'none';
}

// Update patterns table
function updatePatternsTable() {
    if (!patterns.length) return;
    
    const tableBody = document.getElementById('patterns-table-body');
    tableBody.innerHTML = '';
    
    patterns.forEach(pattern => {
        const patternType = pattern.type;
        const patternText = pattern.pattern;
        const count = pattern.count || 0;
        const effectiveness = pattern.effectiveness ? pattern.effectiveness.toFixed(1) + '%' : 'N/A';
        const nextTarget = pattern.next_target || 0;
        const lastSeen = pattern.last_seen ? new Date(pattern.last_seen).toLocaleString() : 'Unknown';
        
        const patternClass = patternType === 'LOW' ? 'pattern-low' : 
                           patternType === 'MIDDLE' ? 'pattern-middle' : 'pattern-high';
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><span class="pattern-badge ${patternClass}">${patternText}</span></td>
            <td>${patternType}</td>
            <td>${count}</td>
            <td><strong>${effectiveness}</strong></td>
            <td>${nextTarget}</td>
            <td>${lastSeen}</td>
        `;
        tableBody.appendChild(row);
    });
    
    document.getElementById('patterns-table-content').style.display = 'block';
    document.querySelector('#patterns-tab .loading').style.display = 'none';
}

// Update timing info
function updateTimingInfo() {
    if (!timingStats) return;
    
    const timingContainer = document.getElementById('timing-info');
    
    timingContainer.innerHTML = `
        <div class="timing-item">
            <div class="timing-label">LOW Average Interval</div>
            <div class="timing-value">${timingStats.low?.avg?.toFixed(1) || '0.0'} min</div>
            <div class="timing-label">
                <span class="status-dot ${timingStats.low?.status === 'due' ? 'status-due' : 'status-normal'}"></span>
                Last: ${timingStats.low?.last?.toFixed(1) || '0.0'} min ago
            </div>
        </div>
        
        <div class="timing-item">
            <div class="timing-label">MIDDLE Average Interval</div>
            <div class="timing-value">${timingStats.middle?.avg?.toFixed(1) || '0.0'} min</div>
            <div class="timing-label">
                <span class="status-dot ${timingStats.middle?.status === 'due' ? 'status-due' : 'status-normal'}"></span>
                Last: ${timingStats.middle?.last?.toFixed(1) || '0.0'} min ago
            </div>
        </div>
        
        <div class="timing-item">
            <div class="timing-label">HIGH Average Interval</div>
            <div class="timing-value">${timingStats.high?.avg?.toFixed(1) || '0.0'} min</div>
            <div class="timing-label">
                <span class="status-dot ${timingStats.high?.status === 'due' ? 'status-due' : 'status-normal'}"></span>
                Last: ${timingStats.high?.last?.toFixed(1) || '0.0'} min ago
            </div>
        </div>
        
        <div class="timing-item">
            <div class="timing-label">Total Intervals Tracked</div>
            <div class="timing-value">${(timingStats.low?.intervals_count || 0) + (timingStats.middle?.intervals_count || 0) + (timingStats.high?.intervals_count || 0)}</div>
            <div class="timing-label">LOW: ${timingStats.low?.intervals_count || 0}, MIDDLE: ${timingStats.middle?.intervals_count || 0}, HIGH: ${timingStats.high?.intervals_count || 0}</div>
        </div>
    `;
}

// Update feature importance
function updateFeatureImportance() {
    if (!featureImportance.length) return;
    
    const featureNames = [
        'Last 1 Category', 'Last 2 Category', 'Last 3 Category', 'Last 4 Category', 'Last 5 Category',
        'LOW Streak', 'MIDDLE Streak', 'HIGH Streak',
        'Mins Since LOW', 'Mins Since MIDDLE', 'Mins Since HIGH',
        'Hour of Day', 'Day of Week',
        'LOW Frequency', 'MIDDLE Frequency', 'HIGH Frequency'
    ];
    
    const container = document.getElementById('feature-importance');
    container.innerHTML = '';
    
    featureImportance.forEach((importance, index) => {
        const featureName = featureNames[index] || `Feature ${index + 1}`;
        const percentage = (importance * 100).toFixed(2);
        
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        featureItem.innerHTML = `
            <div class="feature-name">${featureName}</div>
            <div class="feature-value">${percentage}%</div>
        `;
        
        container.appendChild(featureItem);
    });
}

// Update debug logs
async function updateDebugLogs() {
    try {
        const response = await fetch('/api/ai-analysis');
        const data = await response.json();
        
        if (data.success && data.analysis && data.analysis.recent_debug_log) {
            const logsContainer = document.getElementById('debug-logs');
            const logs = data.analysis.recent_debug_log;
            
            logsContainer.innerHTML = '';
            
            logs.forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                
                // Extract time and message
                const timeMatch = log.match(/\[(\d{2}:\d{2}:\d{2})\]/);
                const time = timeMatch ? timeMatch[1] : '';
                const message = log.replace(/\[\d{2}:\d{2}:\d{2}\]/, '').trim();
                
                logEntry.innerHTML = `
                    <span class="log-time">${time}</span>
                    <span class="log-message">${message}</span>
                `;
                
                logsContainer.appendChild(logEntry);
            });
            
            // Scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
    } catch (error) {
        console.error('Error loading debug logs:', error);
    }
}

// Create charts
function createCharts() {
    createAccuracyChart();
    createPatternChart();
    createTimingChart();
    createHourlyChart();
    createFeatureChart();
}

// Create accuracy chart
function createAccuracyChart() {
    if (!predictionHistory.length) return;
    
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    // Calculate accuracy over time (last 20 predictions)
    const recentPredictions = predictionHistory.slice(-20);
    const labels = [];
    const accuracyData = [];
    
    let correctCount = 0;
    let totalCount = 0;
    
    recentPredictions.forEach((pred, index) => {
        totalCount++;
        if (pred.correct) correctCount++;
        
        const accuracy = (correctCount / totalCount) * 100;
        accuracyData.push(accuracy);
        labels.push(`Pred ${index + 1}`);
    });
    
    if (accuracyChart) accuracyChart.destroy();
    
    accuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Cumulative Accuracy (%)',
                data: accuracyData,
                borderColor: '#00b4db',
                backgroundColor: 'rgba(0, 180, 219, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Prediction Accuracy Over Time'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Recent Predictions'
                    }
                }
            }
        }
    });
}

// Create pattern chart
function createPatternChart() {
    if (!patterns.length) return;
    
    const ctx = document.getElementById('patternChart').getContext('2d');
    
    // Group patterns by type and calculate average effectiveness
    const patternTypes = ['LOW', 'MIDDLE', 'HIGH'];
    const patternCounts = [0, 0, 0];
    const effectiveness = [0, 0, 0];
    
    patterns.forEach(pattern => {
        const index = patternTypes.indexOf(pattern.type);
        if (index !== -1) {
            patternCounts[index]++;
            effectiveness[index] += pattern.effectiveness || 0;
        }
    });
    
    // Calculate average effectiveness
    for (let i = 0; i < effectiveness.length; i++) {
        if (patternCounts[i] > 0) {
            effectiveness[i] = effectiveness[i] / patternCounts[i];
        }
    }
    
    if (patternChart) patternChart.destroy();
    
    patternChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: patternTypes,
            datasets: [
                {
                    label: 'Number of Patterns',
                    data: patternCounts,
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.7)',
                        'rgba(255, 193, 7, 0.7)',
                        'rgba(220, 53, 69, 0.7)'
                    ],
                    borderColor: [
                        '#28a745',
                        '#ffc107',
                        '#dc3545'
                    ],
                    borderWidth: 1
                },
                {
                    label: 'Avg Effectiveness (%)',
                    data: effectiveness,
                    type: 'line',
                    borderColor: '#00b4db',
                    backgroundColor: 'rgba(0, 180, 219, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Pattern Analysis by Category'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Patterns'
                    }
                },
                y1: {
                    beginAtZero: true,
                    max: 100,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Effectiveness (%)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Create timing chart
function createTimingChart() {
    if (!timingStats) return;
    
    const ctx = document.getElementById('timingChart').getContext('2d');
    
    if (timingChart) timingChart.destroy();
    
    timingChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['LOW', 'MIDDLE', 'HIGH'],
            datasets: [{
                label: 'Average Interval (minutes)',
                data: [
                    timingStats.low?.avg || 0,
                    timingStats.middle?.avg || 0,
                    timingStats.high?.avg || 0
                ],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',
                    'rgba(255, 193, 7, 0.7)',
                    'rgba(220, 53, 69, 0.7)'
                ],
                borderColor: [
                    '#28a745',
                    '#ffc107',
                    '#dc3545'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Average Time Between Categories'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Minutes'
                    }
                }
            }
        }
    });
}

// Create hourly chart
function createHourlyChart() {
    // This would require additional data from the API
    // For now, create a placeholder chart
    const ctx = document.getElementById('hourlyChart').getContext('2d');
    
    // Sample data - in a real implementation, this would come from the API
    const hours = Array.from({length: 24}, (_, i) => i);
    const sampleData = hours.map(hour => 50 + Math.sin(hour * 0.3) * 30 + Math.random() * 20);
    
    if (hourlyChart) hourlyChart.destroy();
    
    hourlyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: hours.map(h => `${h}:00`),
            datasets: [{
                label: 'Average Prediction Accuracy by Hour',
                data: sampleData,
                borderColor: '#00b4db',
                backgroundColor: 'rgba(0, 180, 219, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Hourly Performance (Sample Data)'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Hour of Day'
                    }
                }
            }
        }
    });
}

// Create feature importance chart
function createFeatureChart() {
    if (!featureImportance.length) return;
    
    const ctx = document.getElementById('featureChart').getContext('2d');
    
    const featureNames = [
        'Last 1 Cat', 'Last 2 Cat', 'Last 3 Cat', 'Last 4 Cat', 'Last 5 Cat',
        'LOW Streak', 'MID Streak', 'HIGH Streak',
        'Since LOW', 'Since MID', 'Since HIGH',
        'Hour', 'Weekday',
        'LOW Freq', 'MID Freq', 'HIGH Freq'
    ];
    
    // Take only the first 16 features to match featureNames
    const importanceData = featureImportance.slice(0, 16);
    
    if (featureChart) featureChart.destroy();
    
    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureNames,
            datasets: [{
                label: 'Feature Importance',
                data: importanceData,
                backgroundColor: 'rgba(0, 180, 219, 0.7)',
                borderColor: '#00b4db',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Feature Importance in AI Model'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance'
                    }
                }
            }
        }
    });
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification container if it doesn't exist
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 2000;
        `;
        document.body.appendChild(container);
    }
    
    const notification = document.createElement('div');
    notification.style.cssText = `
        padding: 15px 25px;
        border-radius: 10px;
        margin-bottom: 10px;
        animation: slideIn 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
        max-width: 400px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
    `;
    
    if (type === 'success') {
        notification.style.background = 'linear-gradient(135deg, rgba(40, 167, 69, 0.9) 0%, rgba(32, 201, 151, 0.9) 100%)';
    } else if (type === 'error') {
        notification.style.background = 'linear-gradient(135deg, rgba(220, 53, 69, 0.9) 0%, rgba(232, 62, 140, 0.9) 100%)';
    } else {
        notification.style.background = 'linear-gradient(135deg, rgba(0, 180, 219, 0.9) 0%, rgba(0, 131, 176, 0.9) 100%)';
    }
    
    notification.innerHTML = message;
    container.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => {
            if (notification.parentNode === container) {
                container.removeChild(notification);
            }
        }, 300);
    }, 3000);
    
    // Add CSS animation
    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
}
