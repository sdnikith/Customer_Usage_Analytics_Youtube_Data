# YouTube Analytics Dashboard

A comprehensive, real-time web dashboard for YouTube video analytics with interactive visualizations and live data updates.

## 🚀 Features

### 📊 **Core Dashboard Components**
- **Key Metrics Cards**: Total videos, views, engagement rate, channels
- **Views Trend Chart**: Interactive line chart with time series analysis
- **Category Performance**: Doughnut chart showing views by category
- **Regional Distribution**: Pie chart with geographic breakdown
- **Time Analysis**: Bar chart showing optimal publishing times
- **Engagement Scatter Plot**: Interactive scatter plot with bubble sizing
- **ML Performance**: Real-time model accuracy and error metrics
- **NLP Accuracy**: Categorization performance visualization

### 🔧 **Technical Features**
- **Real-time Updates**: WebSocket-based live data streaming
- **RESTful API**: Complete API for all dashboard components
- **Responsive Design**: Mobile-friendly interface
- **Interactive Filters**: Category, region, and date range filters
- **Auto-refresh**: Configurable automatic data updates
- **Error Handling**: Comprehensive error handling and notifications

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r dashboard/requirements.txt
```

### Quick Start
```bash
# Navigate to dashboard directory
cd dashboard

# Start the server
python dashboard_server.py

# Open dashboard in browser
# Navigate to: http://localhost:5000
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "dashboard_server.py"]
```

## 📡 API Endpoints

### Dashboard Metrics
```http
GET /api/dashboard/metrics
```
Returns key dashboard metrics including total videos, views, engagement rate, etc.

### Daily Views Trend
```http
GET /api/dashboard/daily-views?days=30
```
Returns daily views trend for specified number of days.

### Category Performance
```http
GET /api/dashboard/category-performance
```
Returns performance metrics broken down by video category.

### Regional Distribution
```http
GET /api/dashboard/regional-distribution
```
Returns views distribution by geographic region.

### Time Analysis
```http
GET /api/dashboard/time-analysis
```
Returns engagement analysis by publishing time.

### Scatter Plot Data
```http
GET /api/dashboard/scatter-data
```
Returns data for engagement vs views scatter plot.

### ML Performance
```http
GET /api/dashboard/ml-performance
```
Returns machine learning model performance metrics.

### NLP Accuracy
```http
GET /api/dashboard/nlp-accuracy
```
Returns NLP categorization accuracy metrics.

### Top Channels
```http
GET /api/dashboard/top-channels?limit=10
```
Returns top performing channels by views.

### Refresh Data
```http
POST /api/dashboard/refresh
```
Triggers manual refresh of dashboard data.

## 🔌 WebSocket Events

### Connect
```javascript
const socket = io('http://localhost:5000');
socket.on('connect', () => {
    console.log('Connected to dashboard');
});
```

### Dashboard Updates
```javascript
socket.on('dashboard_update', (data) => {
    // Handle real-time data updates
    updateCharts(data);
});
```

### Manual Update Request
```javascript
socket.emit('request_update');
```

## 🎨 Frontend Integration

### HTML Structure
The dashboard uses a modern, responsive HTML structure with:
- Semantic HTML5 elements
- CSS Grid and Flexbox layouts
- Mobile-first responsive design
- Accessibility features (ARIA labels, keyboard navigation)

### JavaScript Framework
- **Chart.js**: For most charts (line, bar, doughnut, pie)
- **Plotly.js**: For interactive scatter plots
- **WebSocket**: Real-time data streaming
- **CSS3 Animations**: Smooth transitions and micro-interactions

### Styling
- **Gradient Backgrounds**: Modern gradient overlays
- **Glass Morphism**: Frosted glass effect cards
- **Hover States**: Interactive feedback on all elements
- **Loading States**: Professional loading animations
- **Notifications**: Toast-style notifications for user feedback

## 📊 Data Sources

### Sample Data
The dashboard currently uses generated sample data that simulates:
- **500 videos** across 5 categories
- **6 geographic regions** (US, GB, CA, DE, FR, IN)
- **20 unique channels**
- **30 days of historical data**
- **ML model performance** with 92.4% accuracy
- **NLP categorization** with 87.6% accuracy

### Production Integration
To connect to real data:

1. **Update `YouTubeAnalyticsEngine`** class:
   ```python
   def load_real_data(self):
       # Connect to your data source
       # Replace sample data generation with real API calls
       pass
   ```

2. **Configure data sources**:
   - AWS Athena for analytical queries
   - Real-time streaming for live updates
   - ML model endpoints for predictions

## 🔧 Configuration

### Environment Variables
```bash
# Server Configuration
export FLASK_ENV=development
export SECRET_KEY=your-secret-key

# Database Configuration
export DATABASE_URL=your-database-url

# API Configuration
export YOUTUBE_API_KEY=your-youtube-api-key
export AWS_ACCESS_KEY_ID=your-aws-access-key
export AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

### Server Settings
```python
# In dashboard_server.py
app.config.update({
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-key'),
    'DEBUG': os.environ.get('FLASK_ENV') == 'development',
    'HOST': '0.0.0.0',
    'PORT': 5000
})
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r dashboard/requirements.txt

# Start development server
python dashboard_server.py

# Access dashboard
open http://localhost:5000
```

### Production Deployment

#### Using Gunicorn
```bash
# Install Gunicorn
pip install gunicorn

# Start production server
gunicorn -w 4 -b 0.0.0.0:5000 dashboard_server:app
```

#### Using Docker
```bash
# Build Docker image
docker build -t youtube-analytics-dashboard .

# Run container
docker run -p 5000:5000 youtube-analytics-dashboard
```

#### Using Kubernetes
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: youtube-analytics-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: youtube-analytics-dashboard
  template:
    metadata:
      labels:
        app: youtube-analytics-dashboard
    spec:
      containers:
      - name: dashboard
        image: youtube-analytics-dashboard:latest
        ports:
        - containerPort: 5000
```

## 🔍 Monitoring & Logging

### Application Logs
```bash
# View application logs
tail -f logs/dashboard.log

# Monitor WebSocket connections
grep "WebSocket" logs/dashboard.log
```

### Performance Metrics
The dashboard includes built-in performance monitoring:
- **Response times** for all API endpoints
- **WebSocket connection** metrics
- **Error rates** and exception tracking
- **Memory usage** monitoring

### Health Checks
```http
GET /health
```
Returns server health status and uptime.

## 🎯 Customization

### Adding New Charts
1. **Create API endpoint** in `dashboard_server.py`
2. **Add frontend component** in `templates/index.html`
3. **Update WebSocket events** for real-time updates
4. **Add styling** consistent with existing design

### Theming
The dashboard uses CSS custom properties for easy theming:
```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --error-color: #dc3545;
}
```

## 🔒 Security

### Authentication (Optional)
```python
# Add authentication middleware
from flask_httpauth import HTTPBasicAuth

@auth.verify_password
def verify_password(username, password):
    # Implement your authentication logic
    return username == 'admin' and password == 'secret'

@app.route('/api/dashboard/metrics')
@auth.login_required
def protected_metrics():
    return get_dashboard_metrics()
```

### CORS Configuration
```python
# Configure CORS for specific origins
from flask_cors import CORS

CORS(app, origins=['https://yourdomain.com'])
```

### Rate Limiting
```python
# Add rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

@app.route('/api/dashboard/metrics')
@limiter.limit("100 per minute")
def rate_limited_metrics():
    return get_dashboard_metrics()
```

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or use different port
python dashboard_server.py --port 5001
```

#### WebSocket Connection Issues
```javascript
// Check WebSocket connection
socket.on('connect_error', (error) => {
    console.error('WebSocket connection failed:', error);
});

// Fallback to polling
const socket = io({
    transports: ['websocket', 'polling']
});
```

#### Data Not Loading
```python
# Check data source connection
try:
    analytics_engine.load_real_data()
except Exception as e:
    logger.error(f"Data loading failed: {e}")
    # Implement fallback to sample data
```

## 📈 Performance Optimization

### Caching
```python
from flask_caching import Cache

cache = Cache(config={'CACHE_TYPE': 'simple'})

@app.route('/api/dashboard/metrics')
@cache.cached(timeout=300)  # Cache for 5 minutes
def cached_metrics():
    return get_dashboard_metrics()
```

### Database Optimization
- Use connection pooling
- Implement query optimization
- Add database indexes
- Use read replicas for analytics queries

### Frontend Optimization
- Implement lazy loading for charts
- Use virtual scrolling for large datasets
- Optimize image assets
- Minimize and bundle CSS/JS

## 📚 API Documentation

### Response Format
All API responses follow consistent format:
```json
{
    "status": "success|error",
    "data": {},
    "message": "Error message (if applicable)",
    "timestamp": "2024-03-26T17:30:00Z"
}
```

### Error Codes
- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd youtube-analytics-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r dashboard/requirements.txt

# Run tests
python -m pytest tests/

# Start development server
python dashboard_server.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue in the project repository
- Check the troubleshooting section above
- Review the API documentation
- Examine the browser console for frontend errors

---

**Last Updated**: March 26, 2024  
**Version**: 2.0.0  
**Status**: Production Ready
