/**
 * 🎲 Lightning Dice AI Grid Tracker - Main JavaScript
 * Handles grid display, AI predictions, and real-time updates
 */

// Global variables
let currentFilter = 'all';
let autoRefresh = true;
let autoRefreshInterval;
let currentGridData = {};
let aiPrediction = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log("🎲 Lightning Dice AI System Initializing...");
    
    // Load initial data
    loadSystemStatus();
    loadGridData();
    loadRecentGames();
    getAIPrediction();
    
    // Start auto refresh
    startAutoRefresh();
    
    // Setup event listeners
    setupEventListeners();
    
    // Setup search
    setupSearch();
});

// Setup event listeners
function setupEventListeners() {
    // Auto refresh toggle
    const toggle = document.getElementById('auto-refresh-toggle');
    if (toggle) {
        toggle.addEventListener('change', function() {
            autoRefresh = this.checked;
            if (autoRefresh) {
                startAutoRefresh();
                showNotification('🔄 Auto-refresh enabled', 'info');
            } else {
                stopAutoRefresh();
                showNotification('⏸️ Auto-refresh disabled', 'info');
            }
        });
    }
    
    // Search input
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchGames(this.value);
            }
        });
    }
}

// Setup search functionality
function setupSearch() {
    const searchInput = document.getElementById('search-input');
    let searchTimeout;
    
    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        const query = this.value.trim();
        
        if (query.length >= 2) {
            searchTimeout = setTimeout(() => {
                searchGames(query);
            }, 500);
        } else if (query.length === 0) {
            // Reset to current filter
            filterGrid(currentFilter);
        }
    });
}

// Load system status
async function loadSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('status-text').textContent = 'Connected';
            document.getElementById('total-games').textContent = data.total_games.toLocaleString();
            document.getElementById('ai-status').textContent = data.ai_status === 'trained' ? 'Ready' : 'Training';
            document.getElementById('last-update').textContent = data.last_update;
            document.getElementById('grid-status').textContent = data.grid_status;
            
            // Update status dot
            const statusDot = document.querySelector('.status-dot');
            statusDot.classList.add('online');
        }
    } catch (error) {
        console.error('Error loading status:', error);
        document.getElementById('status-text').textContent = 'Disconnected';
        document.querySelector('.status-dot').classList.remove('online');
    }
}

// Load grid data
async function loadGridData() {
    try {
        const response = await fetch('/api/grid-data');
        const data = await response.json();
        
        if (data.success) {
            currentGridData = data.grid_data;
            displayGrid();
            updateStatistics(data.grid_stats);
            
            // Update grid status
            document.getElementById('grid-status').textContent = 
                `${data.grid_stats.filled_cells}/${data.grid_stats.total_cells}`;
            
            showNotification('✅ Grid updated successfully', 'success');
        }
    } catch (error) {
        console.error('Error loading grid:', error);
        showNotification('❌ Failed to load grid data', 'error');
    }
}

// Display grid
function displayGrid() {
    const gridContainer = document.getElementById('grid-container');
    
    if (!currentGridData || Object.keys(currentGridData).length === 0) {
        gridContainer.innerHTML = '<div class="loading">No grid data available</div>';
        return;
    }
    
    let gridHTML = '';
    
    // Grid header
    gridHTML += '<div class="grid-header">';
    gridHTML += '<div class="grid-corner">Row/Col</div>';
    
    // Column headers (A to T)
    for (let col = 0; col < 20; col++) {
        const columnLabel = String.fromCharCode(65 + col);
        gridHTML += `<div class="grid-column-header">${columnLabel}</div>`;
    }
    gridHTML += '</div>';
    
    // Grid body
    gridHTML += '<div class="grid-body">';
    gridHTML += '<div class="grid-row-headers">';
    
    // Row headers (1 to 15)
    for (let row = 1; row <= 15; row++) {
        gridHTML += `<div class="grid-row-header">${row}</div>`;
    }
    gridHTML += '</div>';
    
    gridHTML += '<div class="grid-cells">';
    
    // Grid cells
    for (let row = 1; row <= 15; row++) {
        gridHTML += '<div class="grid-row">';
        
        for (let col = 0; col < 20; col++) {
            const columnLabel = String.fromCharCode(65 + col);
            const cellId = `${columnLabel}${row}`;
            const cellData = currentGridData[cellId];
            
            if (cellData) {
                const category = cellData.category;
                const categoryClass = category.toLowerCase();
                
                gridHTML += `
                    <div class="grid-cell ${categoryClass}" onclick="showGameDetails('${cellId}')">
                        <div class="cell-category ${categoryClass}">${category.charAt(0)}</div>
                        <div class="cell-total">${cellData.total}</div>
                        <div class="cell-dice">${cellData.dice_icons}</div>
                        <div class="cell-time">${cellData.time}</div>
                        
                        <div class="cell-details">
                            <div class="details-title">${cellId} - ${category}</div>
                            <div class="details-dice">${cellData.dice_icons}</div>
                            <div class="details-numbers">Dice: ${cellData.dice1} + ${cellData.dice2} + ${cellData.dice3}</div>
                            <div class="details-numbers">Total: ${cellData.total}</div>
                            <div class="details-time">${cellData.date} ${cellData.time}</div>
                            <div class="details-time">ID: ${cellData.game_id.substring(0, 15)}...</div>
                        </div>
                    </div>
                `;
            } else {
                gridHTML += `
                    <div class="grid-cell empty">
                        ${cellId}
                        <div style="font-size: 0.7rem; margin-top: 5px;">Empty</div>
                    </div>
                `;
            }
        }
        
        gridHTML += '</div>';
    }
    
    gridHTML += '</div></div>';
    
    gridContainer.innerHTML = gridHTML;
}

// Update statistics
function updateStatistics(stats) {
    document.getElementById('stat-total').textContent = stats.filled_cells.toLocaleString();
    
    const categories = stats.category_distribution || {};
    document.getElementById('stat-low').textContent = (categories.LOW || 0).toLocaleString();
    document.getElementById('stat-middle').textContent = (categories.MIDDLE || 0).toLocaleString();
    document.getElementById('stat-high').textContent = (categories.HIGH || 0).toLocaleString();
    document.getElementById('stat-filled').textContent = `${stats.fill_percentage.toFixed(1)}%`;
}

// Filter grid by category
async function filterGrid(category) {
    currentFilter = category;
    
    // Update active button
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Set active button
    let activeButton;
    if (category === 'all') {
        activeButton = document.querySelector('.filter-btn:not(.low-btn):not(.middle-btn):not(.high-btn)');
    } else if (category === 'LOW') {
        activeButton = document.querySelector('.low-btn');
    } else if (category === 'MIDDLE') {
        activeButton = document.querySelector('.middle-btn');
    } else if (category === 'HIGH') {
        activeButton = document.querySelector('.high-btn');
    }
    
    if (activeButton) {
        activeButton.classList.add('active');
    }
    
    // Update view text
    document.getElementById('current-view').textContent = 
        `Showing: ${category === 'all' ? 'All Games' : category + ' Games'}`;
    
    if (category === 'all') {
        // Use existing grid data
        displayGrid();
    } else {
        // Fetch filtered data
        try {
            const response = await fetch(`/api/category-data?category=${category}`);
            const data = await response.json();
            
            if (data.success) {
                currentGridData = data.grid_data;
                displayGrid();
                
                showNotification(`✅ Showing ${category} games (${data.count} total)`, 'success');
            }
        } catch (error) {
            console.error('Error filtering grid:', error);
            showNotification('❌ Error loading filtered data', 'error');
        }
    }
}

// Search games
async function searchGames(query) {
    if (!query.trim()) {
        filterGrid(currentFilter);
        return;
    }
    
    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (data.success) {
            if (data.results.length > 0) {
                // Create a new grid with only search results
                const searchGrid = {};
                data.results.forEach(result => {
                    searchGrid[result.cell_id] = result.cell_data;
                });
                
                currentGridData = searchGrid;
                displayGrid();
                
                showNotification(`🔍 Found ${data.results.length} matching games`, 'info');
            } else {
                showNotification('🔍 No matching games found', 'info');
            }
        }
    } catch (error) {
        console.error('Error searching:', error);
    }
}

// Load recent games
async function loadRecentGames() {
    try {
        const response = await fetch('/api/recent-games?limit=10');
        const data = await response.json();
        
        if (data.success) {
            const recentContainer = document.getElementById('recent-games');
            recentContainer.innerHTML = '';
            
            data.games.forEach(game => {
                const category = game.category || 'UNKNOWN';
                const categoryClass = category.toLowerCase();
                
                recentContainer.innerHTML += `
                    <div class="recent-game">
                        <div class="recent-game-header">
                            <div class="recent-game-total">${game.total}</div>
                            <div class="recent-game-category ${categoryClass}">${category}</div>
                        </div>
                        <div class="recent-game-dice">${game.dice_icons}</div>
                        <div class="recent-game-time">
                            <span>${game.time}</span>
                            <span>${game.date}</span>
                        </div>
                    </div>
                `;
            });
        }
    } catch (error) {
        console.error('Error loading recent games:', error);
    }
}

// Get AI prediction
async function getAIPrediction() {
    try {
        const response = await fetch('/api/ai-prediction');
        const data = await response.json();
        
        if (data.success) {
            aiPrediction = data.prediction;
            displayPrediction(aiPrediction);
            
            // Update AI status
            const aiStats = data.ai_stats;
            document.getElementById('ai-status').textContent = 
                aiStats.is_trained ? 'Ready' : 'Training';
        }
    } catch (error) {
        console.error('Error getting AI prediction:', error);
        showNotification('❌ Error getting AI prediction', 'error');
    }
}

// Display prediction
function displayPrediction(prediction) {
    const predictionCard = document.getElementById('prediction-card');
    
    if (!prediction) {
        predictionCard.innerHTML = '<div class="loading">No prediction available</div>';
        return;
    }
    
    const category = prediction.prediction;
    const confidence = (prediction.confidence * 100).toFixed(1);
    const reason = prediction.reason || 'Pattern analysis and timing prediction';
    
    // Color based on category
    let colorClass, icon;
    if (category === 'LOW') {
        colorClass = 'low-color';
        icon = '👇';
    } else if (category === 'MIDDLE') {
        colorClass = 'middle-color';
        icon = '🎯';
    } else {
        colorClass = 'high-color';
        icon = '👆';
    }
    
    predictionCard.innerHTML = `
        <div class="prediction-main">
            <div class="prediction-category" style="color: var(--${category.toLowerCase()})">
                ${icon} ${category}
            </div>
            <div class="prediction-confidence">
                ${confidence}% Confidence
            </div>
        </div>
        <div class="prediction-reason">
            ${reason}
        </div>
        <div style="margin-top: 10px; font-size: 0.8rem; color: #aaa;">
            System: ${prediction.system_used} | Games analyzed: ${prediction.games_analyzed || 0}
        </div>
    `;
}

// Force refresh
async function forceRefresh() {
    try {
        const response = await fetch('/api/force-refresh');
        const data = await response.json();
        
        if (data.success) {
            // Reload all data
            loadSystemStatus();
            loadGridData();
            loadRecentGames();
            getAIPrediction();
            
            showNotification(`✅ Refreshed! Added ${data.new_games} new games`, 'success');
        }
    } catch (error) {
        console.error('Error forcing refresh:', error);
        showNotification('❌ Error refreshing data', 'error');
    }
}

// Show game details modal
function showGameDetails(cellId) {
    const cellData = currentGridData[cellId];
    if (!cellData) return;
    
    const modal = document.getElementById('game-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    
    modalTitle.textContent = `Game Details - ${cellId}`;
    
    modalBody.innerHTML = `
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 3rem; margin-bottom: 10px;">${cellData.dice_icons}</div>
            <div style="font-size: 2.5rem; font-weight: bold; color: var(--${cellData.category.toLowerCase()})">
                Total: ${cellData.total}
            </div>
            <div style="padding: 5px 15px; background: rgba(var(--${cellData.category.toLowerCase()}-rgb), 0.2); 
                 border-radius: 20px; display: inline-block; margin: 10px 0;">
                <strong>${cellData.category}</strong>
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Dice 1:</span>
                <span><strong>${cellData.dice1}</strong> (${getDiceIcon(cellData.dice1)})</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Dice 2:</span>
                <span><strong>${cellData.dice2}</strong> (${getDiceIcon(cellData.dice2)})</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Dice 3:</span>
                <span><strong>${cellData.dice3}</strong> (${getDiceIcon(cellData.dice3)})</span>
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Date:</span>
                <span><strong>${cellData.date}</strong></span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Time:</span>
                <span><strong>${cellData.time}</strong></span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Cell ID:</span>
                <span><strong>${cellData.cell_id}</strong></span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Game ID:</span>
                <span style="font-family: monospace; font-size: 0.9rem;">${cellData.game_id}</span>
            </div>
        </div>
    `;
    
    modal.style.display = 'flex';
}

// Get dice icon from number
function getDiceIcon(number) {
    const diceIcons = ['⚀', '⚁', '⚂', '⚃', '⚄', '⚅'];
    return diceIcons[number - 1] || '⚀';
}

// Close modal
function closeModal() {
    document.getElementById('game-modal').style.display = 'none';
}

// Auto refresh functions
function startAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    autoRefreshInterval = setInterval(() => {
        loadSystemStatus();
        loadGridData();
        loadRecentGames();
        
        // Update prediction every 30 seconds
        if (Math.floor(Date.now() / 1000) % 30 === 0) {
            getAIPrediction();
        }
    }, 5000); // 5 seconds
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const container = document.getElementById('notification-container');
    if (!container) return;
    
    // Remove existing notifications
    const existingNotifications = container.querySelectorAll('.notification');
    existingNotifications.forEach(notification => {
        if (notification.parentNode === container) {
            container.removeChild(notification);
        }
    });
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = message;
    
    container.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentNode === container) {
            notification.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => {
                if (notification.parentNode === container) {
                    container.removeChild(notification);
                }
            }, 300);
        }
    }, 3000);
}

// Handle page visibility
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden, stop auto refresh
        stopAutoRefresh();
    } else if (autoRefresh) {
        // Page is visible, restart auto refresh
        startAutoRefresh();
    }
});

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('game-modal');
    if (event.target === modal) {
        closeModal();
    }
};
