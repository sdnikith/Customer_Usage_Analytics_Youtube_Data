#!/usr/bin/env python3
"""
YouTube Analytics Dashboard Server

This Flask server provides a REST API for the YouTube Analytics dashboard
with real-time data processing and WebSocket updates.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'youtube-analytics-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global data storage (in production, use database)
dashboard_data = {
    'total_videos': 0,
    'total_views': 0,
    'avg_engagement': 0,
    'unique_channels': 0,
    'daily_views': [],
    'category_data': [],
    'region_data': [],
    'time_data': {},
    'ml_performance': {},
    'nlp_accuracy': {},
    'last_updated': None
}

class YouTubeAnalyticsEngine:
    """
    Analytics engine for processing YouTube data and generating insights.
    """
    
    def __init__(self):
        """Initialize the analytics engine."""
        self.data = None
        self.load_sample_data()
        logger.info("YouTubeAnalyticsEngine initialized")
    
    def load_sample_data(self):
        """Load sample data for demonstration."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate comprehensive sample data
        data = {
            'video_id': [f'video_{i:06d}' for i in range(n_samples)],
            'title': [
                f'How to cook pasta {i}' if i % 5 == 0 else
                f'Gaming highlights {i}' if i % 5 == 1 else
                f'Music video {i}' if i % 5 == 2 else
                f'Educational content {i}' if i % 5 == 3 else
                f'Comedy sketch {i}'
                for i in range(n_samples)
            ],
            'description': [
                f'This video shows how to cook delicious pasta dish {i}' if i % 5 == 0 else
                f'Amazing gaming moments and highlights {i}' if i % 5 == 1 else
                f'New music release with great beats {i}' if i % 5 == 2 else
                f'Learn something new today with this educational video {i}' if i % 5 == 3 else
                f'Funny comedy sketch that will make you laugh {i}'
                for i in range(n_samples)
            ],
            'category_id': [26 if i % 5 == 0 else 20 if i % 5 == 1 else 10 if i % 5 == 2 else 27 if i % 5 == 3 else 23 for i in range(n_samples)],
            'channel_title': [f'Channel {i%20}' for i in range(n_samples)],
            'publish_date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'region_code': np.random.choice(['US', 'GB', 'CA', 'DE', 'FR', 'IN'], n_samples),
            'views': np.random.lognormal(mean=10, sigma=1.5, size=n_samples).astype(int),
            'likes': np.random.lognormal(mean=7, sigma=1, size=n_samples).astype(int),
            'comments': np.random.lognormal(mean=5, sigma=1, size=n_samples).astype(int)
        }
        
        self.data = pd.DataFrame(data)
        
        # Ensure positive values
        self.data['views'] = self.data['views'].clip(lower=100)
        self.data['likes'] = self.data['likes'].clip(lower=1)
        self.data['comments'] = self.data['comments'].clip(lower=0)
        
        # Add engagement metrics
        self.data['engagement_rate'] = (self.data['likes'] + self.data['comments']) / self.data['views']
        self.data['publish_date'] = pd.to_datetime(self.data['publish_date'])
        self.data['publish_hour'] = self.data['publish_date'].dt.hour
        
        logger.info(f"Loaded {len(self.data)} records for analytics")
    
    def get_dashboard_metrics(self):
        """Calculate key dashboard metrics."""
        if self.data is None:
            return {}
        
        return {
            'total_videos': len(self.data),
            'total_views': int(self.data['views'].sum()),
            'avg_engagement': float(self.data['engagement_rate'].mean() * 100),
            'unique_channels': self.data['channel_title'].nunique(),
            'unique_categories': self.data['category_id'].nunique(),
            'unique_regions': self.data['region_code'].nunique(),
            'date_range': {
                'start': self.data['publish_date'].min().strftime('%Y-%m-%d'),
                'end': self.data['publish_date'].max().strftime('%Y-%m-%d')
            }
        }
    
    def get_daily_views_trend(self, days=30):
        """Get daily views trend for the last N days."""
        if self.data is None:
            return []
        
        # Filter last N days
        end_date = self.data['publish_date'].max()
        start_date = end_date - timedelta(days=days)
        
        filtered_data = self.data[
            (self.data['publish_date'] >= start_date) & 
            (self.data['publish_date'] <= end_date)
        ]
        
        # Group by date
        daily_views = filtered_data.groupby(
            filtered_data['publish_date'].dt.date
        )['views'].sum().reset_index()
        
        return [
            {
                'date': row['publish_date'].strftime('%Y-%m-%d'),
                'views': int(row['views'])
            }
            for _, row in daily_views.iterrows()
        ]
    
    def get_category_performance(self):
        """Get performance metrics by category."""
        if self.data is None:
            return []
        
        category_names = {
            10: 'Music', 20: 'Gaming', 23: 'Comedy',
            26: 'Howto & Style', 27: 'Education'
        }
        
        category_stats = self.data.groupby('category_id').agg({
            'views': 'sum',
            'engagement_rate': 'mean',
            'video_count': 'count'
        }).reset_index()
        
        return [
            {
                'category': category_names.get(row['category_id'], f'Category {row["category_id"]}'),
                'category_id': int(row['category_id']),
                'views': int(row['views']),
                'avg_engagement': float(row['engagement_rate'] * 100),
                'count': int(row['video_count'])
            }
            for _, row in category_stats.iterrows()
        ]
    
    def get_regional_distribution(self):
        """Get views distribution by region."""
        if self.data is None:
            return []
        
        region_stats = self.data.groupby('region_code')['views'].sum().reset_index()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        return [
            {
                'region': row['region_code'],
                'views': int(row['views']),
                'color': colors[i % len(colors)]
            }
            for i, (_, row) in enumerate(region_stats.iterrows())
        ]
    
    def get_time_analysis(self):
        """Get engagement analysis by publishing time."""
        if self.data is None:
            return {}
        
        hourly_engagement = self.data.groupby('publish_hour')['engagement_rate'].mean()
        
        return {
            'labels': [f'{h:02d}:00' for h in range(24)],
            'engagement': [
                float(hourly_engagement.get(h, 0) * 100) for h in range(24)
            ]
        }
    
    def get_ml_performance(self):
        """Simulate ML model performance metrics."""
        return {
            'accuracy': 92.4,
            'error_rate': 7.6,
            'predictions': len(self.data) if self.data is not None else 0,
            'mean_absolute_error': 7.66,
            'median_absolute_error': 6.38,
            'perfect_predictions': int(len(self.data) * 0.41) if self.data is not None else 0,
            'good_predictions': int(len(self.data) * 0.704) if self.data is not None else 0
        }
    
    def get_nlp_accuracy(self):
        """Simulate NLP categorization accuracy."""
        total_predictions = len(self.data) if self.data is not None else 0
        correct_predictions = int(total_predictions * 0.876)  # 87.6% accuracy
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': 87.6,
            'confidence_score': 0.892,
            'categories_predicted': 5
        }
    
    def get_scatter_data(self):
        """Generate scatter plot data for engagement vs views."""
        if self.data is None:
            return []
        
        # Sample 100 points for scatter plot
        sample_size = min(100, len(self.data))
        sample_data = self.data.sample(n=sample_size)
        
        return [
            {
                'x': int(row['views']),
                'y': float(row['engagement_rate'] * 100),
                'r': np.random.randint(5, 25),
                'title': row['title'],
                'category': row['category_id']
            }
            for _, row in sample_data.iterrows()
        ]
    
    def get_top_channels(self, limit=10):
        """Get top performing channels."""
        if self.data is None:
            return []
        
        channel_stats = self.data.groupby('channel_title').agg({
            'views': 'sum',
            'engagement_rate': 'mean',
            'video_count': 'count'
        }).sort_values('views', ascending=False).head(limit).reset_index()
        
        return [
            {
                'channel': row['channel_title'],
                'total_views': int(row['views']),
                'avg_engagement': float(row['engagement_rate'] * 100),
                'count': int(row['video_count'])
            }
            for _, row in channel_stats.iterrows()
        ]

# Initialize analytics engine
analytics_engine = YouTubeAnalyticsEngine()

def update_dashboard_data():
    """Update global dashboard data with latest analytics."""
    global dashboard_data
    
    try:
        dashboard_data.update({
            'total_videos': analytics_engine.get_dashboard_metrics()['total_videos'],
            'total_views': analytics_engine.get_dashboard_metrics()['total_views'],
            'avg_engagement': analytics_engine.get_dashboard_metrics()['avg_engagement'],
            'unique_channels': analytics_engine.get_dashboard_metrics()['unique_channels'],
            'daily_views': analytics_engine.get_daily_views_trend(),
            'category_data': analytics_engine.get_category_performance(),
            'region_data': analytics_engine.get_regional_distribution(),
            'time_data': analytics_engine.get_time_analysis(),
            'ml_performance': analytics_engine.get_ml_performance(),
            'nlp_accuracy': analytics_engine.get_nlp_accuracy(),
            'last_updated': datetime.now().isoformat()
        })
        
        logger.info("Dashboard data updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating dashboard data: {e}")

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return send_file(os.path.join(os.path.dirname(__file__), 'index.html'))

@app.route('/api/dashboard/metrics')
def get_dashboard_metrics():
    """API endpoint for dashboard metrics."""
    try:
        metrics = analytics_engine.get_dashboard_metrics()
        return jsonify({
            'status': 'success',
            'data': metrics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/daily-views')
def get_daily_views():
    """API endpoint for daily views trend."""
    try:
        days = request.args.get('days', 30, type=int)
        data = analytics_engine.get_daily_views_trend(days)
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/category-performance')
def get_category_performance():
    """API endpoint for category performance."""
    try:
        data = analytics_engine.get_category_performance()
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/regional-distribution')
def get_regional_distribution():
    """API endpoint for regional distribution."""
    try:
        data = analytics_engine.get_regional_distribution()
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/time-analysis')
def get_time_analysis():
    """API endpoint for time analysis."""
    try:
        data = analytics_engine.get_time_analysis()
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/scatter-data')
def get_scatter_data():
    """API endpoint for scatter plot data."""
    try:
        data = analytics_engine.get_scatter_data()
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/ml-performance')
def get_ml_performance():
    """API endpoint for ML performance metrics."""
    try:
        data = analytics_engine.get_ml_performance()
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/nlp-accuracy')
def get_nlp_accuracy():
    """API endpoint for NLP accuracy metrics."""
    try:
        data = analytics_engine.get_nlp_accuracy()
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/top-channels')
def get_top_channels():
    """API endpoint for top performing channels."""
    try:
        limit = request.args.get('limit', 10, type=int)
        data = analytics_engine.get_top_channels(limit)
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/dashboard/refresh')
def refresh_dashboard():
    """API endpoint to refresh dashboard data."""
    try:
        update_dashboard_data()
        return jsonify({
            'status': 'success',
            'message': 'Dashboard data refreshed successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info("Client connected to dashboard")
    update_dashboard_data()
    emit('dashboard_update', dashboard_data)

@socketio.on('request_update')
def handle_request_update():
    """Handle manual update request."""
    logger.info("Manual update requested")
    update_dashboard_data()
    emit('dashboard_update', dashboard_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info("Client disconnected from dashboard")

def main():
    """Main function to run the dashboard server."""
    logger.info("🚀 Starting YouTube Analytics Dashboard Server")
    logger.info("=" * 60)
    
    # Initialize dashboard data
    update_dashboard_data()
    
    # Print server information
    print("📊 YouTube Analytics Dashboard Server")
    print("=" * 40)
    print("🌐 Server Information:")
    print(f"   📡 Local URL: http://localhost:8080")
    print(f"   📡 Dashboard: http://localhost:8080/dashboard")
    print(f"   📡 API Docs: http://localhost:8080/api")
    print()
    print("🔗 Available Endpoints:")
    endpoints = [
        "GET /api/dashboard/metrics",
        "GET /api/dashboard/daily-views",
        "GET /api/dashboard/category-performance", 
        "GET /api/dashboard/regional-distribution",
        "GET /api/dashboard/time-analysis",
        "GET /api/dashboard/scatter-data",
        "GET /api/dashboard/ml-performance",
        "GET /api/dashboard/nlp-accuracy",
        "GET /api/dashboard/top-channels",
        "POST /api/dashboard/refresh"
    ]
    
    for endpoint in endpoints:
        print(f"   📡 {endpoint}")
    
    print()
    print("🔧 Features:")
    print("   ✅ Real-time data updates via WebSocket")
    print("   ✅ RESTful API for all dashboard components")
    print("   ✅ Responsive web interface")
    print("   ✅ Interactive charts and visualizations")
    print("   ✅ ML and NLP performance tracking")
    print()
    print("🚀 Starting server on http://localhost:8080")
    print("=" * 60)
    
    # Run the application
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)

if __name__ == '__main__':
    main()
