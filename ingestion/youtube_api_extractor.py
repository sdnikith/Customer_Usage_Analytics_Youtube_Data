"""
YouTube Data API v3 Extractor

This module extracts structured and semi-structured video data from YouTube Data API v3
including titles, descriptions, tags, views, likes, comments, category, publish_date, 
and channel information.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeAPIExtractor:
    """
    Extracts video data from YouTube Data API v3 with retry logic and error handling.
    
    Attributes:
        api_key (str): YouTube Data API v3 key
        api_version (str): API version
        youtube (Resource): YouTube API resource object
        max_retries (int): Maximum retry attempts for API calls
        retry_delay (int): Delay between retries in seconds
    """
    
    def __init__(self, api_key: Optional[str] = None, api_version: str = "v3"):
        """
        Initialize YouTube API extractor.
        
        Args:
            api_key (Optional[str]): YouTube Data API key. If None, loads from environment.
            api_version (str): API version to use. Defaults to "v3".
            
        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key must be provided or set in YOUTUBE_API_KEY environment variable")
        
        self.api_version = api_version
        self.max_retries = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY_SECONDS', '5'))
        
        try:
            self.youtube = build('youtube', self.api_version, developerKey=self.api_key)
            logger.info("YouTube API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API client: {e}")
            raise
    
    def _make_api_call_with_retry(self, api_call: callable, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make API call with retry logic.
        
        Args:
            api_call (callable): The API function to call
            *args: Arguments to pass to the API call
            **kwargs: Keyword arguments to pass to the API call
            
        Returns:
            Optional[Dict[str, Any]]: API response or None if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = api_call(*args, **kwargs)
                return response
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit exceeded
                    wait_time = (2 ** attempt) * self.retry_delay
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {attempt + 1}")
                    time.sleep(wait_time)
                elif e.resp.status == 403:  # Quota exceeded
                    logger.error(f"API quota exceeded: {e}")
                    return None
                else:
                    logger.error(f"API error on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        return None
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.retry_delay)
        
        return None
    
    def get_video_categories(self, region_code: str = "US") -> List[Dict[str, Any]]:
        """
        Get video categories for a specific region.
        
        Args:
            region_code (str): Two-letter ISO country code. Defaults to "US".
            
        Returns:
            List[Dict[str, Any]]: List of category dictionaries
        """
        logger.info(f"Fetching video categories for region: {region_code}")
        
        response = self._make_api_call_with_retry(
            self.youtube.videoCategories().list,
            part="snippet",
            regionCode=region_code
        )
        
        if not response:
            logger.error(f"Failed to fetch categories for region {region_code}")
            return []
        
        categories = []
        for item in response.get('items', []):
            category = {
                'category_id': int(item['id']),
                'title': item['snippet']['title'],
                'assignable': item['snippet']['assignable'],
                'channel_id': item['snippet']['channelId']
            }
            categories.append(category)
        
        logger.info(f"Retrieved {len(categories)} categories for region {region_code}")
        return categories
    
    def get_trending_videos(self, region_code: str = "US", max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Get trending videos for a specific region.
        
        Args:
            region_code (str): Two-letter ISO country code. Defaults to "US".
            max_results (int): Maximum number of results to return (1-50). Defaults to 50.
            
        Returns:
            List[Dict[str, Any]]: List of video dictionaries with detailed information
        """
        logger.info(f"Fetching trending videos for region: {region_code}, max_results: {max_results}")
        
        response = self._make_api_call_with_retry(
            self.youtube.videos().list,
            part="snippet,statistics,contentDetails,status",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=min(max_results, 50)
        )
        
        if not response:
            logger.error(f"Failed to fetch trending videos for region {region_code}")
            return []
        
        videos = []
        for item in response.get('items', []):
            try:
                video_data = self._extract_video_data(item, region_code)
                if video_data:
                    videos.append(video_data)
            except Exception as e:
                logger.error(f"Error processing video {item.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Retrieved {len(videos)} trending videos for region {region_code}")
        return videos
    
    def get_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed information for specific video IDs.
        
        Args:
            video_ids (List[str]): List of YouTube video IDs
            
        Returns:
            List[Dict[str, Any]]: List of video dictionaries with detailed information
        """
        logger.info(f"Fetching details for {len(video_ids)} videos")
        
        videos = []
        # Process in batches of 50 (API limit)
        batch_size = 50
        
        for i in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[i:i + batch_size]
            
            response = self._make_api_call_with_retry(
                self.youtube.videos().list,
                part="snippet,statistics,contentDetails,status",
                id=",".join(batch_ids)
            )
            
            if not response:
                logger.error(f"Failed to fetch details for video batch {i//batch_size + 1}")
                continue
            
            for item in response.get('items', []):
                try:
                    video_data = self._extract_video_data(item)
                    if video_data:
                        videos.append(video_data)
                except Exception as e:
                    logger.error(f"Error processing video {item.get('id', 'unknown')}: {e}")
                    continue
        
        logger.info(f"Retrieved details for {len(videos)} videos")
        return videos
    
    def _extract_video_data(self, item: Dict[str, Any], region_code: str = "US") -> Optional[Dict[str, Any]]:
        """
        Extract and standardize video data from API response item.
        
        Args:
            item (Dict[str, Any]): API response item
            region_code (str): Region code for the video
            
        Returns:
            Optional[Dict[str, Any]]: Standardized video data dictionary
        """
        try:
            snippet = item.get('snippet', {})
            statistics = item.get('statistics', {})
            content_details = item.get('contentDetails', {})
            status = item.get('status', {})
            
            # Parse duration
            duration = content_details.get('duration', '')
            duration_seconds = self._parse_duration(duration)
            
            # Parse publish date
            publish_date_str = snippet.get('publishedAt', '')
            publish_date = datetime.fromisoformat(publish_date_str.replace('Z', '+00:00')) if publish_date_str else None
            
            # Extract tags
            tags = snippet.get('tags', [])
            
            video_data = {
                'video_id': item.get('id'),
                'title': snippet.get('title', ''),
                'description': snippet.get('description', ''),
                'tags': tags,
                'tag_count': len(tags),
                'category_id': int(snippet.get('categoryId', 0)),
                'channel_id': snippet.get('channelId', ''),
                'channel_title': snippet.get('channelTitle', ''),
                'publish_date': publish_date.isoformat() if publish_date else None,
                'trending_date': datetime.now(timezone.utc).isoformat(),
                'region_code': region_code,
                'views': int(statistics.get('viewCount', 0)),
                'likes': int(statistics.get('likeCount', 0)),
                'comments': int(statistics.get('commentCount', 0)),
                'duration': duration,
                'duration_seconds': duration_seconds,
                'definition': content_details.get('definition', ''),
                'caption': content_details.get('caption', ''),
                'status': status.get('uploadStatus', ''),
                'privacy_status': status.get('privacyStatus', ''),
                'license': content_details.get('licensedContent', False),
                'projection': content_details.get('projection', ''),
                'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                'extracted_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Calculate engagement rate
            if video_data['views'] > 0:
                video_data['engagement_rate'] = (video_data['likes'] + video_data['comments']) / video_data['views']
            else:
                video_data['engagement_rate'] = 0.0
            
            # Calculate text metrics
            video_data['title_length'] = len(video_data['title'])
            video_data['description_length'] = len(video_data['description'])
            video_data['title_word_count'] = len(video_data['title'].split())
            
            return video_data
            
        except Exception as e:
            logger.error(f"Error extracting video data: {e}")
            return None
    
    def _parse_duration(self, duration: str) -> int:
        """
        Parse ISO 8601 duration string to seconds.
        
        Args:
            duration (str): ISO 8601 duration string (e.g., "PT4M13S")
            
        Returns:
            int: Duration in seconds
        """
        if not duration or not duration.startswith('PT'):
            return 0
        
        import re
        
        # Extract hours, minutes, seconds
        hours = 0
        minutes = 0
        seconds = 0
        
        hour_match = re.search(r'(\d+)H', duration)
        if hour_match:
            hours = int(hour_match.group(1))
        
        minute_match = re.search(r'(\d+)M', duration)
        if minute_match:
            minutes = int(minute_match.group(1))
        
        second_match = re.search(r'(\d+)S', duration)
        if second_match:
            seconds = int(second_match.group(1))
        
        return hours * 3600 + minutes * 60 + seconds
    
    def search_videos(self, query: str, max_results: int = 50, region_code: str = "US") -> List[str]:
        """
        Search for videos by query and return video IDs.
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            region_code (str): Two-letter ISO country code
            
        Returns:
            List[str]: List of video IDs
        """
        logger.info(f"Searching videos for query: '{query}' in region {region_code}")
        
        response = self._make_api_call_with_retry(
            self.youtube.search().list,
            part="id",
            q=query,
            type="video",
            maxResults=min(max_results, 50),
            regionCode=region_code
        )
        
        if not response:
            logger.error(f"Failed to search videos for query: {query}")
            return []
        
        video_ids = []
        for item in response.get('items', []):
            if item['id']['kind'] == 'youtube#video':
                video_ids.append(item['id']['videoId'])
        
        logger.info(f"Found {len(video_ids)} videos for query: '{query}'")
        return video_ids


def main():
    """
    Main function to test the YouTube API extractor.
    """
    try:
        extractor = YouTubeAPIExtractor()
        
        # Test getting categories
        categories = extractor.get_video_categories("US")
        print(f"Retrieved {len(categories)} categories")
        
        # Test getting trending videos
        trending_videos = extractor.get_trending_videos("US", 10)
        print(f"Retrieved {len(trending_videos)} trending videos")
        
        # Save sample data
        if trending_videos:
            with open('data/sample/sample_videos.json', 'w') as f:
                json.dump(trending_videos, f, indent=2)
            print("Sample data saved to data/sample/sample_videos.json")
        
        # Save categories
        if categories:
            with open('data/sample/sample_categories.json', 'w') as f:
                json.dump(categories, f, indent=2)
            print("Categories saved to data/sample/sample_categories.json")
                
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
