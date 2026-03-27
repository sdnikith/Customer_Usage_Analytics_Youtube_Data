-- Athena SQL Views for YouTube Analytics Dashboard
-- Layer 7: Tableau Visualization

-- View 1: Video Performance Overview
CREATE OR REPLACE VIEW youtube_analytics.video_performance_overview AS
SELECT 
    video_id,
    title,
    channel_title,
    category_id,
    region_code,
    views,
    likes,
    comments,
    engagement_rate,
    CASE 
        WHEN views < 10000 THEN 'Low'
        WHEN views < 100000 THEN 'Medium' 
        WHEN views < 1000000 THEN 'High'
        ELSE 'Viral'
    END AS view_category,
    CASE 
        WHEN engagement_rate < 0.01 THEN 'Low'
        WHEN engagement_rate < 0.05 THEN 'Medium'
        ELSE 'High'
    END AS engagement_category,
    publish_date,
    trending_date,
    EXTRACT(DAY FROM trending_date - publish_date) AS days_to_trending,
    title_length,
    description_length,
    tag_count,
    extracted_at
FROM youtube_cleaned.youtube_cleaned_data
WHERE views > 0
ORDER BY views DESC;

-- View 2: Category Analysis
CREATE OR REPLACE VIEW youtube_analytics.category_analysis AS
SELECT 
    category_id,
    CASE category_id
        WHEN 1 THEN 'Film & Animation'
        WHEN 2 THEN 'Autos & Vehicles'
        WHEN 10 THEN 'Music'
        WHEN 15 THEN 'Pets & Animals'
        WHEN 17 THEN 'Sports'
        WHEN 19 THEN 'Travel & Events'
        WHEN 20 THEN 'Gaming'
        WHEN 22 THEN 'People & Blogs'
        WHEN 23 THEN 'Comedy'
        WHEN 24 THEN 'Entertainment'
        WHEN 25 THEN 'News & Politics'
        WHEN 26 THEN 'Howto & Style'
        WHEN 27 THEN 'Education'
        WHEN 28 THEN 'Science & Technology'
        WHEN 29 THEN 'Nonprofits & Activism'
        ELSE 'Other'
    END AS category_name,
    COUNT(*) AS video_count,
    SUM(views) AS total_views,
    AVG(views) AS avg_views,
    MAX(views) AS max_views,
    SUM(likes) AS total_likes,
    AVG(likes) AS avg_likes,
    SUM(comments) AS total_comments,
    AVG(comments) AS avg_comments,
    AVG(engagement_rate) AS avg_engagement_rate,
    COUNT(DISTINCT channel_title) AS unique_channels,
    COUNT(DISTINCT region_code) AS regions_covered
FROM youtube_cleaned.youtube_cleaned_data
WHERE category_id IS NOT NULL
GROUP BY category_id
ORDER BY total_views DESC;

-- View 3: Regional Analysis
CREATE OR REPLACE VIEW youtube_analytics.regional_analysis AS
SELECT 
    region_code,
    COUNT(*) AS video_count,
    SUM(views) AS total_views,
    AVG(views) AS avg_views,
    MAX(views) AS max_views,
    SUM(likes) AS total_likes,
    AVG(likes) AS avg_likes,
    SUM(comments) AS total_comments,
    AVG(comments) AS avg_comments,
    AVG(engagement_rate) AS avg_engagement_rate,
    COUNT(DISTINCT category_id) AS unique_categories,
    COUNT(DISTINCT channel_title) AS unique_channels,
    DATE_TRUNC('month', publish_date) AS publish_month,
    DATE_TRUNC('month', trending_date) AS trending_month
FROM youtube_cleaned.youtube_cleaned_data
WHERE region_code IS NOT NULL
GROUP BY region_code, DATE_TRUNC('month', publish_date), DATE_TRUNC('month', trending_date)
ORDER BY total_views DESC;

-- View 4: Time Series Analysis
CREATE OR REPLACE VIEW youtube_analytics.time_series_analysis AS
SELECT 
    DATE_TRUNC('day', publish_date) AS publish_date,
    DATE_TRUNC('day', trending_date) AS trending_date,
    COUNT(*) AS daily_video_count,
    SUM(views) AS daily_total_views,
    AVG(views) AS daily_avg_views,
    MAX(views) AS daily_max_views,
    SUM(likes) AS daily_total_likes,
    AVG(likes) AS daily_avg_likes,
    SUM(comments) AS daily_total_comments,
    AVG(comments) AS daily_avg_comments,
    AVG(engagement_rate) AS daily_avg_engagement_rate,
    COUNT(DISTINCT region_code) AS daily_regions,
    COUNT(DISTINCT category_id) AS daily_categories
FROM youtube_cleaned.youtube_cleaned_data
WHERE publish_date IS NOT NULL
GROUP BY DATE_TRUNC('day', publish_date), DATE_TRUNC('day', trending_date)
ORDER BY publish_date DESC;

-- View 5: Channel Performance
CREATE OR REPLACE VIEW youtube_analytics.channel_performance AS
SELECT 
    channel_title,
    COUNT(*) AS video_count,
    SUM(views) AS total_views,
    AVG(views) AS avg_views,
    MAX(views) AS max_views,
    SUM(likes) AS total_likes,
    AVG(likes) AS avg_likes,
    SUM(comments) AS total_comments,
    AVG(comments) AS avg_comments,
    AVG(engagement_rate) AS avg_engagement_rate,
    COUNT(DISTINCT category_id) AS unique_categories,
    COUNT(DISTINCT region_code) AS unique_regions,
    MIN(publish_date) AS first_video_date,
    MAX(publish_date) AS last_video_date,
    DATEDIFF(day, MAX(publish_date), MIN(publish_date)) AS active_days
FROM youtube_cleaned.youtube_cleaned_data
WHERE channel_title IS NOT NULL
GROUP BY channel_title
HAVING COUNT(*) >= 5  -- Only channels with 5+ videos
ORDER BY total_views DESC;

-- View 6: Engagement Analysis
CREATE OR REPLACE VIEW youtube_analytics.engagement_analysis AS
SELECT 
    video_id,
    title,
    channel_title,
    views,
    likes,
    comments,
    engagement_rate,
    like_rate,
    comment_rate,
    like_to_comment_ratio,
    title_length,
    description_length,
    tag_count,
    CASE 
        WHEN engagement_rate >= 0.05 THEN 'High'
        WHEN engagement_rate >= 0.01 THEN 'Medium'
        ELSE 'Low'
    END AS engagement_level,
    title_word_count,
    has_description,
    has_tags,
    publish_hour,
    CASE 
        WHEN publish_hour BETWEEN 6 AND 11 THEN 'Morning'
        WHEN publish_hour BETWEEN 12 AND 17 THEN 'Afternoon'
        WHEN publish_hour BETWEEN 18 AND 23 THEN 'Evening'
        ELSE 'Night'
    END AS publish_time_of_day,
    is_weekend,
    is_business_hours,
    is_prime_time
FROM youtube_cleaned.youtube_cleaned_data
WHERE views > 0 AND engagement_rate IS NOT NULL
ORDER BY engagement_rate DESC;

-- View 7: Trending Analysis
CREATE OR REPLACE VIEW youtube_analytics.trending_analysis AS
SELECT 
    video_id,
    title,
    channel_title,
    category_id,
    region_code,
    views,
    likes,
    comments,
    publish_date,
    trending_date,
    EXTRACT(HOUR FROM trending_date) AS trending_hour,
    EXTRACT(DAYOFWEEK FROM trending_date) AS trending_day_of_week,
    time_to_trending_hours,
    CASE 
        WHEN time_to_trending_hours <= 24 THEN 'Very Fast'
        WHEN time_to_trending_hours <= 72 THEN 'Fast'
        WHEN time_to_trending_hours <= 168 THEN 'Medium'
        ELSE 'Slow'
    END AS trending_speed,
    CASE 
        WHEN EXTRACT(DAYOFWEEK FROM trending_date) IN (1,7) THEN 'Weekend'
        ELSE 'Weekday'
    END AS trending_day_type
FROM youtube_cleaned.youtube_cleaned_data
WHERE time_to_trending_hours IS NOT NULL
ORDER BY time_to_trending_hours ASC;

-- View 8: ML Predictions Analysis
CREATE OR REPLACE VIEW youtube_analytics.ml_predictions_analysis AS
SELECT 
    video_id,
    title,
    views,
    likes,
    comments,
    engagement_rate,
    ml_predicted_views,
    ml_predicted_engagement_rate,
    ABS(views - ml_predicted_views) AS views_prediction_error,
    ABS(engagement_rate - ml_predicted_engagement_rate) AS engagement_prediction_error,
    CASE 
        WHEN ABS(views - ml_predicted_views) / views <= 0.1 THEN 'Excellent'
        WHEN ABS(views - ml_predicted_views) / views <= 0.2 THEN 'Good'
        WHEN ABS(views - ml_predicted_views) / views <= 0.5 THEN 'Fair'
        ELSE 'Poor'
    END AS views_prediction_quality,
    nlp_predicted_category,
    category_id AS actual_category,
    CASE 
        WHEN nlp_predicted_category = category_id THEN 'Correct'
        ELSE 'Incorrect'
    END AS nlp_prediction_correct,
    nlp_category_confidence,
    title_length,
    description_length,
    tag_count
FROM youtube_cleaned.youtube_cleaned_data
WHERE ml_predicted_views IS NOT NULL 
  AND nlp_predicted_category IS NOT NULL
ORDER BY views DESC;

-- View 9: Dashboard Summary Metrics
CREATE OR REPLACE VIEW youtube_analytics.dashboard_summary AS
SELECT 
    -- Overall Metrics
    COUNT(*) AS total_videos,
    COUNT(DISTINCT video_id) AS unique_videos,
    SUM(views) AS total_views,
    AVG(views) AS avg_views,
    MAX(views) AS max_views,
    SUM(likes) AS total_likes,
    AVG(likes) AS avg_likes,
    SUM(comments) AS total_comments,
    AVG(comments) AS avg_comments,
    AVG(engagement_rate) AS avg_engagement_rate,
    
    -- Content Diversity
    COUNT(DISTINCT category_id) AS unique_categories,
    COUNT(DISTINCT channel_title) AS unique_channels,
    COUNT(DISTINCT region_code) AS unique_regions,
    
    -- Date Range
    MIN(publish_date) AS earliest_publish_date,
    MAX(publish_date) AS latest_publish_date,
    MIN(trending_date) AS earliest_trending_date,
    MAX(trending_date) AS latest_trending_date,
    
    -- Performance Categories
    SUM(CASE WHEN views < 10000 THEN 1 ELSE 0 END) AS low_view_videos,
    SUM(CASE WHEN views >= 10000 AND views < 100000 THEN 1 ELSE 0 END) AS medium_view_videos,
    SUM(CASE WHEN views >= 100000 AND views < 1000000 THEN 1 ELSE 0 END) AS high_view_videos,
    SUM(CASE WHEN views >= 1000000 THEN 1 ELSE 0 END) AS viral_videos,
    
    -- Engagement Distribution
    SUM(CASE WHEN engagement_rate < 0.01 THEN 1 ELSE 0 END) AS low_engagement_videos,
    SUM(CASE WHEN engagement_rate >= 0.01 AND engagement_rate < 0.05 THEN 1 ELSE 0 END) AS medium_engagement_videos,
    SUM(CASE WHEN engagement_rate >= 0.05 THEN 1 ELSE 0 END) AS high_engagement_videos,
    
    -- ML Model Performance
    COUNT(DISTINCT CASE WHEN nlp_predicted_category = category_id THEN video_id END) AS correct_nlp_predictions,
    COUNT(DISTINCT CASE WHEN nlp_predicted_category IS NOT NULL THEN video_id END) AS total_nlp_predictions,
    AVG(CASE WHEN ml_predicted_views IS NOT NULL THEN ABS(views - ml_predicted_views) / views END) AS avg_ml_error_rate,
    
    -- Data Freshness
    CURRENT_DATE - MAX(publish_date) AS days_since_last_video,
    CURRENT_DATE - MAX(trending_date) AS days_since_last_trending
FROM youtube_cleaned.youtube_cleaned_data
WHERE views > 0;
