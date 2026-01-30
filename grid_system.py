"""
📊 Google Sheets Style Grid System
Organizes games in A1 to T15 grid format
"""

from collections import defaultdict
import math

class GridSystem:
    """Manages Google Sheets style grid display"""
    
    def __init__(self, rows=15, cols=20):
        self.rows = rows
        self.cols = cols
        self.total_cells = rows * cols
        
        # Column labels A to T (20 columns)
        self.column_labels = [chr(65 + i) for i in range(cols)]
        
        # Grid data storage
        self.grid_data = {}
        self.all_games = []
        
        print(f"📊 Grid System Initialized: {rows}×{cols} ({self.total_cells} cells)")
    
    def update_games(self, games_data):
        """Update games and organize into grid"""
        self.all_games = games_data.copy()
        self._organize_grid()
    
    def _organize_grid(self):
        """Organize games into grid format"""
        self.grid_data = {}
        cell_counter = 0
        
        # Start from newest games (beginning of list)
        for game in self.all_games:
            if cell_counter >= self.total_cells:
                break
            
            col_idx = cell_counter // self.rows
            row_idx = cell_counter % self.rows
            
            if col_idx >= len(self.column_labels):
                break
            
            cell_id = f"{self.column_labels[col_idx]}{row_idx + 1}"
            
            self.grid_data[cell_id] = {
                'cell_id': cell_id,
                'total': game['total'],
                'dice_icons': game['dice_icons'],
                'dice1': game['dice1'],
                'dice2': game['dice2'],
                'dice3': game['dice3'],
                'time': game['time'],
                'date': game['date'],
                'category': game.get('category', 'LOW'),
                'game_id': game['game_id'],
                'game_hash': game['game_hash'],
                'col_label': self.column_labels[col_idx],
                'row_number': row_idx + 1
            }
            
            cell_counter += 1
    
    def get_grid_data(self):
        """Get organized grid data"""
        return self.grid_data
    
    def get_category_grid(self, category):
        """Get grid filtered by category"""
        if category == 'all':
            return self.grid_data
        
        filtered_grid = {}
        for cell_id, cell_data in self.grid_data.items():
            if cell_data['category'] == category:
                filtered_grid[cell_id] = cell_data
        
        return filtered_grid
    
    def get_grid_stats(self):
        """Get grid statistics"""
        categories = defaultdict(int)
        for cell_data in self.grid_data.values():
            categories[cell_data['category']] += 1
        
        total_filled = len(self.grid_data)
        
        return {
            'total_cells': self.total_cells,
            'filled_cells': total_filled,
            'empty_cells': self.total_cells - total_filled,
            'fill_percentage': (total_filled / self.total_cells * 100) if self.total_cells > 0 else 0,
            'category_distribution': dict(categories),
            'rows': self.rows,
            'columns': self.cols,
            'column_labels': self.column_labels
        }
    
    def get_cell_info(self, cell_id):
        """Get detailed info for a specific cell"""
        return self.grid_data.get(cell_id)
    
    def search_in_grid(self, query):
        """Search for games in grid"""
        results = []
        query = str(query).lower()
        
        for cell_id, cell_data in self.grid_data.items():
            # Search in various fields
            if (query in str(cell_data['total']).lower() or
                query in cell_data['game_id'].lower() or
                query in cell_data['category'].lower() or
                query in cell_id.lower()):
                results.append({
                    'cell_id': cell_id,
                    'cell_data': cell_data,
                    'match_type': self._get_match_type(cell_data, query)
                })
        
        return results
    
    def _get_match_type(self, cell_data, query):
        """Determine what matched in search"""
        query = str(query)
        if query in str(cell_data['total']):
            return 'total'
        elif query in cell_data['category'].lower():
            return 'category'
        elif query in cell_data['game_id'].lower():
            return 'game_id'
        else:
            return 'cell_id'
    
    def get_row_data(self, row_number):
        """Get all data for a specific row"""
        row_data = {}
        for col_label in self.column_labels:
            cell_id = f"{col_label}{row_number}"
            if cell_id in self.grid_data:
                row_data[col_label] = self.grid_data[cell_id]
        
        return row_data
    
    def get_column_data(self, column_label):
        """Get all data for a specific column"""
        column_data = {}
        for row in range(1, self.rows + 1):
            cell_id = f"{column_label}{row}"
            if cell_id in self.grid_data:
                column_data[row] = self.grid_data[cell_id]
        
        return column_data

# Singleton instance
grid_system = GridSystem(rows=15, cols=20)
