#!/usr/bin/env python3
"""
Simple Age-Based Video Analysis

Demonstrates age-based video categorization and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def create_age_sample_data():
    """Create sample data with age-based categorization."""
    print("🎯 Creating Age-Based Video Data")
    print("-" * 50)
    
    np.random.seed(42)
    n_samples = 500
    
    # Age groups and their characteristics
    age_groups = ['Kids (0-12)', 'Teens (13-17)', 'Young Adults (18-24)', 
                 'Adults (25-34)', 'Middle-Aged Adults (35-49)']
    
    # Generate data
    data = []
    for i in range(n_samples):
        age_group = np.random.choice(age_groups)
        
        if age_group == 'Kids (0-12)':
            title = f'Learn ABC with {i}'
            category_id = 1  # Film & Animation
            engagement_rate = np.random.uniform(0.08, 0.15)
            views = np.random.randint(50000, 200000)
            publish_hour = np.random.choice([8, 9, 10, 14, 15, 16])
        elif age_group == 'Teens (13-17)':
            title = f'Gaming Challenge #{i}'
            category_id = 20  # Gaming
            engagement_rate = np.random.uniform(0.05, 0.12)
            views = np.random.randint(100000, 500000)
            publish_hour = np.random.choice([15, 16, 17, 18, 19, 20])
        elif age_group == 'Young Adults (18-24)':
            title = f'Life Vlog: Day {i}'
            category_id = 22  # People & Blogs
            engagement_rate = np.random.uniform(0.03, 0.08)
            views = np.random.randint(50000, 300000)
            publish_hour = np.random.choice([18, 19, 20, 21, 22])
        elif age_group == 'Adults (25-34)':
            title = f'Career Tips #{i}'
            category_id = 26  # Howto & Style
            engagement_rate = np.random.uniform(0.02, 0.06)
            views = np.random.randint(25000, 150000)
            publish_hour = np.random.choice([19, 20, 21, 22])
        else:  # Middle-Aged Adults
            title = f'Business Strategy {i}'
            category_id = 25  # News & Politics
            engagement_rate = np.random.uniform(0.02, 0.05)
            views = np.random.randint(20000, 100000)
            publish_hour = np.random.choice([9, 10, 11, 19, 20])
        
        data.append({
            'video_id': f'video_{i:06d}',
            'title': title,
            'age_group': age_group,
            'category_id': category_id,
            'publish_hour': publish_hour,
            'views': views,
            'engagement_rate': engagement_rate,
            'likes': int(views * np.random.uniform(0.02, 0.08)),
            'comments': int(views * np.random.uniform(0.005, 0.02))
        })
    
    df = pd.DataFrame(data)
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']
    
    print(f"✅ Created {len(df)} videos across {len(age_groups)} age groups")
    return df

def analyze_by_age(df):
    """Analyze video metrics by age groups."""
    print("\n📊 Analyzing by Age Groups")
    print("-" * 50)
    
    # Age group distribution
    age_dist = df['age_group'].value_counts()
    print("Age Group Distribution:")
    for age_group, count in age_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {age_group}: {count} videos ({percentage:.1f}%)")
    
    # Engagement by age group
    engagement_by_age = df.groupby('age_group')['engagement_rate'].mean()
    print("\nAverage Engagement Rate by Age Group:")
    for age_group, engagement in engagement_by_age.items():
        print(f"  {age_group}: {engagement:.3f}")
    
    # Views by age group
    views_by_age = df.groupby('age_group')['views'].mean()
    print("\nAverage Views by Age Group:")
    for age_group, avg_views in views_by_age.items():
        print(f"  {age_group}: {avg_views:,.0f}")
    
    # Publishing time by age group
    time_by_age = df.groupby('age_group')['publish_hour'].mean()
    print("\nAverage Publishing Hour by Age Group:")
    for age_group, avg_hour in time_by_age.items():
        print(f"  {age_group}: {avg_hour:.1f}:00")
    
    return {
        'age_distribution': age_dist.to_dict(),
        'engagement_by_age': engagement_by_age.to_dict(),
        'views_by_age': views_by_age.to_dict(),
        'time_by_age': time_by_age.to_dict()
    }

def create_age_visualizations(df, analysis):
    """Create visualizations for age-based analysis."""
    print("\n📈 Creating Visualizations")
    print("-" * 50)
    
    os.makedirs('age_analysis_output', exist_ok=True)
    
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
    plt.savefig('age_analysis_output/age_distribution.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('age_analysis_output/engagement_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Views by Age Group
    plt.figure(figsize=(12, 8))
    views_data = analysis['views_by_age']
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
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_views)*0.01,
                f'{value:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('age_analysis_output/views_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Publishing Time Analysis
    plt.figure(figsize=(12, 8))
    time_data = analysis['time_by_age']
    age_groups = list(time_data.keys())
    avg_hours = list(time_data.values())
    
    bars = plt.bar(age_groups, avg_hours, color='lightgreen', alpha=0.7)
    plt.title('Average Publishing Hour by Age Group', fontsize=16, fontweight='bold')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Hour of Day (24h)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, avg_hours):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('age_analysis_output/publishing_time_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved to age_analysis_output/")
    print("  - age_distribution.png")
    print("  - engagement_by_age.png")
    print("  - views_by_age.png")
    print("  - publishing_time_by_age.png")

def main():
    """Main function for age-based analysis."""
    try:
        print("🎯 Age-Based Video Analysis System")
        print("=" * 60)
        
        # Create sample data
        df = create_age_sample_data()
        
        # Analyze by age groups
        analysis = analyze_by_age(df)
        
        # Create visualizations
        create_age_visualizations(df, analysis)
        
        # Generate insights
        print("\n💡 Key Insights:")
        print("-" * 50)
        
        # Most engaged age group
        most_engaged = max(analysis['engagement_by_age'].items(), key=lambda x: x[1])
        print(f"Most engaged group: {most_engaged[0]}")
        print(f"Highest engagement: {most_engaged[1]:.3f}")
        
        # Most views
        most_views = max(analysis['views_by_age'].items(), key=lambda x: x[1])
        print(f"Most viewed group: {most_views[0]}")
        print(f"Highest avg views: {most_views[1]:,.0f}")
        
        # Peak publishing times
        print("\nPeak Publishing Times:")
        for age_group, avg_hour in analysis['time_by_age'].items():
            hour = int(avg_hour)
            if 6 <= hour <= 11:
                time_period = 'Morning'
            elif 12 <= hour <= 17:
                time_period = 'Afternoon'
            elif 18 <= hour <= 22:
                time_period = 'Evening'
            else:
                time_period = 'Night'
            print(f"  {age_group}: {time_period} ({hour:.1f}:00)")
        
        print("\n🎉 AGE-BASED ANALYSIS COMPLETE!")
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
