#!/usr/bin/env python3
"""
Age-Based Video Categorization Demo

Simplified version demonstrating age-based video categorization
with comprehensive analysis and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def create_age_based_data():
    """Create sample data with age-based categorization."""
    print("🎯 Creating Age-Based Video Data")
    print("-" * 50)
    
    np.random.seed(42)
    n_samples = 1000
    
    # Age groups
    age_groups = ['Kids (0-12)', 'Teens (13-17)', 'Young Adults (18-24)', 
                 'Adults (25-34)', 'Middle-Aged Adults (35-49)', 
                 'Older Adults (50-64)', 'Seniors (65+)']
    
    # Content types by age group
    content_mapping = {
        'Kids (0-12)': {
            'categories': [1, 15],  # Film & Animation, Pets & Animals
            'keywords': ['learn', 'abc', 'colors', 'shapes', 'cartoon', 'play', 'fun'],
            'publish_hours': [8, 9, 10, 14, 15, 16],
            'duration_range': (60, 300)
        },
        'Teens (13-17)': {
            'categories': [10, 20, 23, 24],  # Music, Gaming, Comedy, Entertainment
            'keywords': ['gaming', 'music', 'challenge', 'trend', 'viral', 'funny'],
            'publish_hours': [15, 16, 17, 18, 19, 20],
            'duration_range': (180, 600)
        },
        'Young Adults (18-24)': {
            'categories': [10, 20, 22, 23, 24],  # Music, Gaming, People & Blogs, Comedy, Entertainment
            'keywords': ['music', 'gaming', 'lifestyle', 'vlog', 'reaction', 'trend'],
            'publish_hours': [18, 19, 20, 21, 22],
            'duration_range': (300, 900)
        },
        'Adults (25-34)': {
            'categories': [22, 25, 26, 27, 28],  # People & Blogs, News, Howto, Education, Science
            'keywords': ['tutorial', 'education', 'career', 'business', 'tech', 'productivity'],
            'publish_hours': [19, 20, 21, 22],
            'duration_range': (300, 1200)
        },
        'Middle-Aged Adults (35-49)': {
            'categories': [25, 26, 27, 28],  # News, Howto, Education, Science
            'keywords': ['news', 'politics', 'business', 'finance', 'health', 'family'],
            'publish_hours': [19, 20, 21],
            'duration_range': (600, 1800)
        },
        'Older Adults (50-64)': {
            'categories': [25, 26, 27],  # News, Howto, Education
            'keywords': ['health', 'retirement', 'finance', 'lifestyle', 'travel'],
            'publish_hours': [9, 10, 11, 19, 20],
            'duration_range': (300, 900)
        },
        'Seniors (65+)': {
            'categories': [26, 27],  # Howto, Education
            'keywords': ['health', 'retirement', 'hobbies', 'family', 'simple'],
            'publish_hours': [9, 10, 11, 14, 15],
            'duration_range': (180, 600)
        }
    }
    
    # Generate data
    data = []
    for i in range(n_samples):
        # Randomly assign age group
        age_group = np.random.choice(age_groups)
        content_info = content_mapping[age_group]
        
        # Generate video data based on age group
        category_id = np.random.choice(content_info['categories'])
        keyword = np.random.choice(content_info['keywords'])
        publish_hour = np.random.choice(content_info['publish_hours'])
        duration = np.random.randint(*content_info['duration_range'])
        
        # Create title and description
        if age_group == 'Kids (0-12)':
            title = f'Learn {keyword} for kids - Episode {i}'
            description = f'Fun educational content about {keyword} for young children'
        elif age_group == 'Teens (13-17)':
            title = f'{keyword.title()} Challenge #{i} - Viral Trend'
            description = f'Exciting {keyword} content that teens will love'
        elif age_group == 'Young Adults (18-24)':
            title = f'Living My Best Life: {keyword.title()} Journey'
            description = f'My personal experience with {keyword} and life lessons'
        elif age_group == 'Adults (25-34)':
            title = f'Complete Guide to {keyword.title()} - Tips & Tricks'
            description = f'Professional {keyword} tutorial for career growth'
        elif age_group == 'Middle-Aged Adults (35-49)':
            title = f'{keyword.title()} for Busy Professionals'
            description = f'Time-saving {keyword} strategies for working adults'
        elif age_group == 'Older Adults (50-64)':
            title = f'{keyword.title()} After 50: What I Learned'
            description = f'Reflections on {keyword} and life experience'
        else:  # Seniors
            title = f'Simple {keyword.title()} for Seniors'
            description = f'Easy-to-follow {keyword} guide for older adults'
        
        # Generate engagement based on age group
        if age_group == 'Kids (0-12)':
            engagement_rate = np.random.uniform(0.08, 0.15)
            views = np.random.randint(50000, 200000)
        elif age_group == 'Teens (13-17)':
            engagement_rate = np.random.uniform(0.05, 0.12)
            views = np.random.randint(100000, 500000)
        elif age_group == 'Young Adults (18-24)':
            engagement_rate = np.random.uniform(0.03, 0.08)
            views = np.random.randint(50000, 300000)
        elif age_group == 'Adults (25-34)':
            engagement_rate = np.random.uniform(0.02, 0.06)
            views = np.random.randint(25000, 150000)
        elif age_group == 'Middle-Aged Adults (35-49)':
            engagement_rate = np.random.uniform(0.02, 0.05)
            views = np.random.randint(20000, 100000)
        elif age_group == 'Older Adults (50-64)':
            engagement_rate = np.random.uniform(0.03, 0.07)
            views = np.random.randint(30000, 120000)
        else:  # Seniors
            engagement_rate = np.random.uniform(0.05, 0.12)
            views = np.random.randint(40000, 180000)
        
        data.append({
            'video_id': f'video_{i:06d}',
            'title': title,
            'description': description,
            'category_id': category_id,
            'age_group': age_group,
            'target_age_min': get_age_min(age_group),
            'target_age_max': get_age_max(age_group),
            'channel_title': f'Channel_{i%15}',
            'publish_hour': publish_hour,
            'publish_day_of_week': np.random.randint(0, 6),
            'duration': duration,
            'views': views,
            'likes': int(views * np.random.uniform(0.02, 0.08)),
            'comments': int(views * np.random.uniform(0.005, 0.02)),
            'engagement_rate': engagement_rate,
            'content_keyword': keyword,
            'content_complexity': get_content_complexity(age_group)
        })
    
    df = pd.DataFrame(data)
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']
    
    print(f"✅ Created {len(df)} videos across {len(age_groups)} age groups")
    return df

def get_age_min(age_group):
    """Get minimum age for age group."""
    age_mapping = {
        'Kids (0-12)': 0,
        'Teens (13-17)': 13,
        'Young Adults (18-24)': 18,
        'Adults (25-34)': 25,
        'Middle-Aged Adults (35-49)': 35,
        'Older Adults (50-64)': 50,
        'Seniors (65+)': 65
    }
    return age_mapping.get(age_group, 0)

def get_age_max(age_group):
    """Get maximum age for age group."""
    age_mapping = {
        'Kids (0-12)': 12,
        'Teens (13-17)': 17,
        'Young Adults (18-24)': 24,
        'Adults (25-34)': 34,
        'Middle-Aged Adults (35-49)': 49,
        'Older Adults (50-64)': 64,
        'Seniors (65+)': 80
    }
    return age_mapping.get(age_group, 80)

def get_content_complexity(age_group):
    """Get content complexity level for age group."""
    complexity_mapping = {
        'Kids (0-12)': 'Simple',
        'Teens (13-17)': 'Moderate',
        'Young Adults (18-24)': 'Moderate',
        'Adults (25-34)': 'Advanced',
        'Middle-Aged Adults (35-49)': 'Advanced',
        'Older Adults (50-64)': 'Moderate',
        'Seniors (65+)': 'Simple'
    }
    return complexity_mapping.get(age_group, 'Moderate')

def analyze_age_preferences(df):
    """Analyze content preferences by age groups."""
    print("\n📊 Analyzing Age-Based Content Preferences")
    print("-" * 50)
    
    analysis = {}
    
    # Age group distribution
    age_dist = df['age_group'].value_counts().to_dict()
    analysis['age_distribution'] = age_dist
    
    # Category preferences by age group
    category_prefs = df.groupby('age_group')['category_id'].apply(lambda x: x.value_counts().index[0]).to_dict()
    analysis['category_preferences'] = category_prefs
    
    # Engagement by age group
    engagement_by_age = df.groupby('age_group')['engagement_rate'].mean().to_dict()
    analysis['engagement_by_age'] = engagement_by_age
    
    # Views by age group
    views_by_age = df.groupby('age_group')['views'].mean().to_dict()
    analysis['avg_views_by_age'] = views_by_age
    
    # Publishing time preferences
    time_by_age = df.groupby('age_group')['publish_hour'].mean().to_dict()
    analysis['avg_publish_time_by_age'] = time_by_age
    
    # Content complexity by age group
    complexity_by_age = df.groupby('age_group')['content_complexity'].value_counts().to_dict()
    analysis['content_complexity_by_age'] = complexity_by_age
    
    # Duration preferences
    duration_by_age = df.groupby('age_group')['duration'].mean().to_dict()
    analysis['avg_duration_by_age'] = duration_by_age
    
    return analysis

def create_age_visualizations(df, analysis, output_dir='age_analysis_output'):
    """Create visualizations for age-based analysis."""
    print("\n📈 Creating Age-Based Visualizations")
    print("-" * 50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Age Group Distribution
    plt.figure(figsize=(12, 8))
    age_counts = df['age_group'].value_counts()
    colors = plt.cm.Set3(np.arange(len(age_counts)))
    bars = plt.bar(range(len(age_counts)), age_counts.values, color=colors)
    plt.title('Video Distribution by Target Age Group', fontsize=16, fontweight='bold')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Number of Videos', fontsize=12)
    plt.xticks(range(len(age_counts)), age_counts.index, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, age_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(age_counts.values)*0.01,
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Engagement by Age Group
    plt.figure(figsize=(12, 8))
    engagement_data = analysis['engagement_by_age']
    age_groups = list(engagement_data.keys())
    engagement_rates = list(engagement_data.values())
    
    bars = plt.bar(age_groups, engagement_rates, color='skyblue', alpha=0.7)
    plt.title('Average Engagement Rate by Age Group', fontsize=16, fontweight='bold')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Engagement Rate', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, engagement_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/engagement_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Views by Age Group
    plt.figure(figsize=(12, 8))
    views_data = analysis['avg_views_by_age']
    age_groups = list(views_data.keys())
    avg_views = list(views_data.values())
    
    bars = plt.bar(age_groups, avg_views, color='lightcoral', alpha=0.7)
    plt.title('Average Views by Age Group', fontsize=16, fontweight='bold')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Average Views', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, avg_views):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_views) * 0.01,
                f'{value:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/views_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Publishing Time Heatmap
    plt.figure(figsize=(14, 8))
    
    # Create pivot table for heatmap
    pivot_data = []
    
    for age_group in df['age_group'].unique():
        age_df = df[df['age_group'] == age_group]
        hour_counts = []
        for hour in range(24):
            count = len(age_df[age_df['publish_hour'] == hour])
            hour_counts.append(count)
        pivot_data.append(hour_counts)
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(
        pivot_data,
        index=df['age_group'].unique(),
        columns=[f'{h:02d}:00' for h in range(24)]
    )
    
    sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlOrRd', 
               cbar_kws={'label': 'Number of Videos'})
    plt.title('Publishing Time Heatmap by Age Group', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Age Group', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/publishing_time_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Content Complexity by Age Group
    plt.figure(figsize=(12, 8))
    complexity_data = analysis['content_complexity_by_age']
    
    # Create stacked bar chart
    complexity_levels = ['Simple', 'Moderate', 'Advanced']
    age_groups = list(complexity_data.keys())
    
    bottom = np.zeros(len(age_groups))
    colors = ['#lightgreen', 'orange', 'red']
    
    for i, level in enumerate(complexity_levels):
        values = []
        for age_group in age_groups:
            values.append(complexity_data[age_group].get(level, 0))
        
        plt.bar(age_groups, values, bottom=bottom, color=colors[i], label=level)
        bottom += values
    
    plt.title('Content Complexity by Age Group', fontsize=16, fontweight='bold')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Number of Videos', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Complexity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/content_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}/")
    return output_dir

def generate_age_insights(analysis):
    """Generate actionable insights from age-based analysis."""
    print("\n💡 Generating Age-Based Insights")
    print("-" * 50)
    
    insights = {}
    
    # Most engaged age group
    most_engaged = max(analysis['engagement_by_age'].items(), key=lambda x: x[1])
    insights['most_engaged_group'] = most_engaged[0]
    insights['highest_engagement'] = most_engaged[1]
    
    # Most views
    most_views = max(analysis['avg_views_by_age'].items(), key=lambda x: x[1])
    insights['most_viewed_group'] = most_views[0]
    insights['highest_avg_views'] = most_views[1]
    
    # Peak publishing times
    peak_times = {}
    for age_group, avg_hour in analysis['avg_publish_time_by_age'].items():
        if not np.isnan(avg_hour):
            hour = int(avg_hour)
            if 6 <= hour <= 11:
                time_period = 'Morning'
            elif 12 <= hour <= 17:
                time_period = 'Afternoon'
            elif 18 <= hour <= 22:
                time_period = 'Evening'
            else:
                time_period = 'Night'
            peak_times[age_group] = time_period
    
    insights['peak_publishing_times'] = peak_times
    
    # Content preferences
    insights['content_preferences'] = analysis['category_preferences']
    
    # Complexity trends
    insights['complexity_trends'] = analysis['content_complexity_by_age']
    
    return insights

def main():
    """Main function for age-based video categorization demo."""
    try:
        print("🎯 Age-Based Video Categorization System")
        print("=" * 60)
        
        # Create age-based data
        df = create_age_based_data()
        
        # Analyze preferences
        analysis = analyze_age_preferences(df)
        
        # Generate insights
        insights = generate_age_insights(analysis)
        
        # Create visualizations
        viz_dir = create_age_visualizations(df, analysis)
        
        # Display results
        print("\n📊 AGE-BASED ANALYSIS RESULTS:")
        print("=" * 50)
        
        print(f"Total videos analyzed: {len(df)}")
        print(f"Age groups: {df['age_group'].nunique()}")
        
        print("\n📈 Age Group Distribution:")
        for age_group, count in analysis['age_distribution'].items():
            percentage = (count / len(df)) * 100
            print(f"  {age_group}: {count} videos ({percentage:.1f}%)")
        
        print("\n🎯 Key Insights:")
        print(f"  Most engaged group: {insights['most_engaged_group']}")
        print(f"  Highest engagement: {insights['highest_engagement']:.3f}")
        print(f"  Most viewed group: {insights['most_viewed_group']}")
        print(f"  Highest avg views: {insights['highest_avg_views']:,.0f}")
        
        print("\n⏰ Peak Publishing Times:")
        for age_group, time_period in insights['peak_publishing_times'].items():
            print(f"  {age_group}: {time_period}")
        
        print("\n📂 Files Created:")
        print(f"  Output directory: {viz_dir}")
        print("  - age_distribution.png")
        print("  - engagement_by_age.png") 
        print("  - views_by_age.png")
        print("  - publishing_time_heatmap.png")
        print("  - content_complexity.png")
        
        print("\n🎉 AGE-BASED CATEGORIZATION COMPLETE!")
        print("=" * 60)
        print("✅ Comprehensive age group analysis completed")
        print("✅ Actionable insights generated")
        print("✅ Professional visualizations created")
        print("✅ Ready for content strategy optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in age-based analysis: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
