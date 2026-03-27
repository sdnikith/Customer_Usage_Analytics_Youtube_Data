"""
Schema Definitions for YouTube Analytics

This module defines comprehensive schemas for YouTube data including
raw, cleaned, and analytics schemas with validation rules.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class FieldDefinition:
    """Definition of a data field with validation rules."""
    name: str
    data_type: str
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    description: str = ""


class YouTubeSchema:
    """
    Comprehensive schema definitions for YouTube data.
    
    Attributes:
        raw_schema: Schema for raw ingested data
        cleaned_schema: Schema for cleaned data
        analytics_schema: Schema for analytics data
    """
    
    def __init__(self):
        """Initialize schema definitions."""
        self.raw_schema = self._define_raw_schema()
        self.cleaned_schema = self._define_cleaned_schema()
        self.analytics_schema = self._define_analytics_schema()
    
    def _define_raw_schema(self) -> Dict[str, Any]:
        """
        Define schema for raw YouTube data.
        
        Returns:
            Dict[str, Any]: Raw schema definition
        """
        fields = [
            FieldDefinition(
                name="video_id",
                data_type="string",
                nullable=False,
                min_length=1,
                max_length=50,
                description="Unique YouTube video identifier"
            ),
            FieldDefinition(
                name="title",
                data_type="string",
                nullable=False,
                min_length=1,
                max_length=200,
                description="Video title"
            ),
            FieldDefinition(
                name="description",
                data_type="string",
                nullable=True,
                max_length=5000,
                description="Video description"
            ),
            FieldDefinition(
                name="tags",
                data_type="string",
                nullable=True,
                max_length=1000,
                description="Video tags separated by | or ,"
            ),
            FieldDefinition(
                name="category_id",
                data_type="integer",
                nullable=False,
                min_value=1,
                max_value=44,
                description="YouTube category ID (1-44)"
            ),
            FieldDefinition(
                name="channel_id",
                data_type="string",
                nullable=True,
                max_length=50,
                description="Channel unique identifier"
            ),
            FieldDefinition(
                name="channel_title",
                data_type="string",
                nullable=False,
                max_length=100,
                description="Channel name"
            ),
            FieldDefinition(
                name="publish_date",
                data_type="datetime",
                nullable=True,
                description="Video publication date"
            ),
            FieldDefinition(
                name="trending_date",
                data_type="datetime",
                nullable=False,
                description="Date when video was trending"
            ),
            FieldDefinition(
                name="region_code",
                data_type="string",
                nullable=False,
                min_length=2,
                max_length=2,
                pattern=r"^[A-Z]{2}$",
                description="Two-letter ISO country code"
            ),
            FieldDefinition(
                name="views",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_value=10**12,
                description="Number of views"
            ),
            FieldDefinition(
                name="likes",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_value=10**11,
                description="Number of likes"
            ),
            FieldDefinition(
                name="comments",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_value=10**10,
                description="Number of comments"
            ),
            FieldDefinition(
                name="thumbnail_url",
                data_type="string",
                nullable=True,
                max_length=500,
                description="URL to video thumbnail"
            ),
            FieldDefinition(
                name="extracted_at",
                data_type="datetime",
                nullable=True,
                description="Timestamp when data was extracted"
            )
        ]
        
        return {
            "schema_name": "youtube_raw",
            "version": "1.0",
            "description": "Schema for raw YouTube data ingestion",
            "fields": [self._field_to_dict(field) for field in fields],
            "primary_keys": ["video_id"],
            "partition_keys": ["region_code"]
        }
    
    def _define_cleaned_schema(self) -> Dict[str, Any]:
        """
        Define schema for cleaned YouTube data.
        
        Returns:
            Dict[str, Any]: Cleaned schema definition
        """
        fields = [
            FieldDefinition(
                name="video_id",
                data_type="string",
                nullable=False,
                min_length=1,
                max_length=50,
                description="Unique YouTube video identifier"
            ),
            FieldDefinition(
                name="title",
                data_type="string",
                nullable=False,
                min_length=1,
                max_length=200,
                description="Cleaned video title"
            ),
            FieldDefinition(
                name="description",
                data_type="string",
                nullable=False,
                min_length=1,
                max_length=5000,
                description="Cleaned video description"
            ),
            FieldDefinition(
                name="tags",
                data_type="string",
                nullable=False,
                max_length=1000,
                description="Cleaned video tags"
            ),
            FieldDefinition(
                name="tag_count",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_value=100,
                description="Number of tags"
            ),
            FieldDefinition(
                name="category_id",
                data_type="integer",
                nullable=False,
                min_value=1,
                max_value=44,
                description="Validated YouTube category ID"
            ),
            FieldDefinition(
                name="category_name",
                data_type="string",
                nullable=True,
                max_length=100,
                description="YouTube category name"
            ),
            FieldDefinition(
                name="channel_id",
                data_type="string",
                nullable=True,
                max_length=50,
                description="Channel unique identifier"
            ),
            FieldDefinition(
                name="channel_title",
                data_type="string",
                nullable=False,
                min_length=1,
                max_length=100,
                description="Channel name"
            ),
            FieldDefinition(
                name="publish_date",
                data_type="datetime",
                nullable=False,
                description="Validated publication date"
            ),
            FieldDefinition(
                name="trending_date",
                data_type="datetime",
                nullable=False,
                description="Trending date"
            ),
            FieldDefinition(
                name="region_code",
                data_type="string",
                nullable=False,
                min_length=2,
                max_length=2,
                pattern=r"^[A-Z]{2}$",
                description="Two-letter ISO country code"
            ),
            FieldDefinition(
                name="views",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_value=10**12,
                description="Validated view count"
            ),
            FieldDefinition(
                name="likes",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_value=10**11,
                description="Validated like count"
            ),
            FieldDefinition(
                name="comments",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_value=10**10,
                description="Validated comment count"
            ),
            FieldDefinition(
                name="engagement_rate",
                data_type="float",
                nullable=False,
                min_value=0,
                max_value=1,
                description="Engagement rate (likes + comments) / views"
            ),
            FieldDefinition(
                name="title_length",
                data_type="integer",
                nullable=False,
                min_value=1,
                max_length=200,
                description="Length of title in characters"
            ),
            FieldDefinition(
                name="description_length",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_length=5000,
                description="Length of description in characters"
            ),
            FieldDefinition(
                name="title_word_count",
                data_type="integer",
                nullable=False,
                min_value=1,
                max_value=100,
                description="Number of words in title"
            ),
            FieldDefinition(
                name="thumbnail_url",
                data_type="string",
                nullable=True,
                max_length=500,
                description="URL to video thumbnail"
            ),
            FieldDefinition(
                name="extracted_at",
                data_type="datetime",
                nullable=False,
                description="Data extraction timestamp"
            ),
            FieldDefinition(
                name="has_description",
                data_type="boolean",
                nullable=False,
                description="Whether video has a meaningful description"
            ),
            FieldDefinition(
                name="has_tags",
                data_type="boolean",
                nullable=False,
                description="Whether video has tags"
            ),
            FieldDefinition(
                name="engagement_category",
                data_type="string",
                nullable=False,
                allowed_values=["Low", "Medium", "High"],
                description="Engagement rate category"
            ),
            FieldDefinition(
                name="view_category",
                data_type="string",
                nullable=False,
                allowed_values=["Low", "Medium", "High", "Viral"],
                description="View count category"
            )
        ]
        
        return {
            "schema_name": "youtube_cleaned",
            "version": "1.0",
            "description": "Schema for cleaned and validated YouTube data",
            "fields": [self._field_to_dict(field) for field in fields],
            "primary_keys": ["video_id"],
            "partition_keys": ["region_code", "category_id"]
        }
    
    def _define_analytics_schema(self) -> Dict[str, Any]:
        """
        Define schema for analytics-ready YouTube data.
        
        Returns:
            Dict[str, Any]: Analytics schema definition
        """
        fields = [
            FieldDefinition(
                name="video_id",
                data_type="string",
                nullable=False,
                description="Unique YouTube video identifier"
            ),
            FieldDefinition(
                name="title",
                data_type="string",
                nullable=False,
                description="Video title"
            ),
            FieldDefinition(
                name="description",
                data_type="string",
                nullable=False,
                description="Video description"
            ),
            FieldDefinition(
                name="tags",
                data_type="string",
                nullable=False,
                description="Video tags"
            ),
            FieldDefinition(
                name="category_id",
                data_type="integer",
                nullable=False,
                description="YouTube category ID"
            ),
            FieldDefinition(
                name="category_name",
                data_type="string",
                nullable=True,
                description="YouTube category name"
            ),
            FieldDefinition(
                name="channel_title",
                data_type="string",
                nullable=False,
                description="Channel name"
            ),
            FieldDefinition(
                name="publish_date",
                data_type="datetime",
                nullable=False,
                description="Publication date"
            ),
            FieldDefinition(
                name="trending_date",
                data_type="datetime",
                nullable=False,
                description="Trending date"
            ),
            FieldDefinition(
                name="region_code",
                data_type="string",
                nullable=False,
                description="Two-letter ISO country code"
            ),
            FieldDefinition(
                name="views",
                data_type="integer",
                nullable=False,
                description="View count"
            ),
            FieldDefinition(
                name="likes",
                data_type="integer",
                nullable=False,
                description="Like count"
            ),
            FieldDefinition(
                name="comments",
                data_type="integer",
                nullable=False,
                description="Comment count"
            ),
            FieldDefinition(
                name="engagement_rate",
                data_type="float",
                nullable=False,
                description="Engagement rate"
            ),
            FieldDefinition(
                name="time_to_trending_hours",
                data_type="float",
                nullable=True,
                description="Hours from publish to trending"
            ),
            FieldDefinition(
                name="publish_hour",
                data_type="integer",
                nullable=False,
                min_value=0,
                max_value=23,
                description="Hour of publication (0-23)"
            ),
            FieldDefinition(
                name="publish_day_of_week",
                data_type="integer",
                nullable=False,
                min_value=1,
                max_value=7,
                description="Day of week (1=Monday, 7=Sunday)"
            ),
            FieldDefinition(
                name="publish_week",
                data_type="integer",
                nullable=False,
                min_value=1,
                max_value=53,
                description="Week of year"
            ),
            FieldDefinition(
                name="publish_month",
                data_type="integer",
                nullable=False,
                min_value=1,
                max_value=12,
                description="Month of year"
            ),
            FieldDefinition(
                name="publish_year",
                data_type="integer",
                nullable=False,
                min_value=2005,
                max_value=2030,
                description="Year of publication"
            ),
            FieldDefinition(
                name="title_length",
                data_type="integer",
                nullable=False,
                description="Title length in characters"
            ),
            FieldDefinition(
                name="description_length",
                data_type="integer",
                nullable=False,
                description="Description length in characters"
            ),
            FieldDefinition(
                name="tag_count",
                data_type="integer",
                nullable=False,
                description="Number of tags"
            ),
            FieldDefinition(
                name="title_word_count",
                data_type="integer",
                nullable=False,
                description="Number of words in title"
            ),
            FieldDefinition(
                name="has_description",
                data_type="boolean",
                nullable=False,
                description="Has meaningful description"
            ),
            FieldDefinition(
                name="has_tags",
                data_type="boolean",
                nullable=False,
                description="Has tags"
            ),
            FieldDefinition(
                name="engagement_category",
                data_type="string",
                nullable=False,
                description="Engagement category"
            ),
            FieldDefinition(
                name="view_category",
                data_type="string",
                nullable=False,
                description="View category"
            ),
            FieldDefinition(
                name="nlp_predicted_category",
                data_type="integer",
                nullable=True,
                description="NLP predicted category ID"
            ),
            FieldDefinition(
                name="nlp_category_confidence",
                data_type="float",
                nullable=True,
                min_value=0,
                max_value=1,
                description="NLP prediction confidence"
            ),
            FieldDefinition(
                name="ml_predicted_views",
                data_type="float",
                nullable=True,
                description="ML predicted views"
            ),
            FieldDefinition(
                name="ml_predicted_engagement_rate",
                data_type="float",
                nullable=True,
                min_value=0,
                max_value=1,
                description="ML predicted engagement rate"
            ),
            FieldDefinition(
                name="data_quality_score",
                data_type="float",
                nullable=True,
                min_value=0,
                max_value=100,
                description="Data quality score"
            ),
            FieldDefinition(
                name="processed_at",
                data_type="datetime",
                nullable=False,
                description="Processing timestamp"
            )
        ]
        
        return {
            "schema_name": "youtube_analytics",
            "version": "1.0",
            "description": "Schema for analytics-ready YouTube data with ML predictions",
            "fields": [self._field_to_dict(field) for field in fields],
            "primary_keys": ["video_id"],
            "partition_keys": ["region_code", "category_id", "publish_year"]
        }
    
    def _field_to_dict(self, field: FieldDefinition) -> Dict[str, Any]:
        """
        Convert FieldDefinition to dictionary.
        
        Args:
            field (FieldDefinition): Field definition
            
        Returns:
            Dict[str, Any]: Field definition as dictionary
        """
        return {
            "name": field.name,
            "data_type": field.data_type,
            "nullable": field.nullable,
            "min_value": field.min_value,
            "max_value": field.max_value,
            "min_length": field.min_length,
            "max_length": field.max_length,
            "pattern": field.pattern,
            "allowed_values": field.allowed_values,
            "description": field.description
        }
    
    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema by name.
        
        Args:
            schema_name (str): Name of schema
            
        Returns:
            Optional[Dict[str, Any]]: Schema definition or None if not found
        """
        schemas = {
            "raw": self.raw_schema,
            "cleaned": self.cleaned_schema,
            "analytics": self.analytics_schema
        }
        
        return schemas.get(schema_name)
    
    def get_field_names(self, schema_name: str) -> List[str]:
        """
        Get field names for a schema.
        
        Args:
            schema_name (str): Name of schema
            
        Returns:
            List[str]: List of field names
        """
        schema = self.get_schema(schema_name)
        if schema:
            return [field["name"] for field in schema["fields"]]
        return []
    
    def get_partition_keys(self, schema_name: str) -> List[str]:
        """
        Get partition keys for a schema.
        
        Args:
            schema_name (str): Name of schema
            
        Returns:
            List[str]: List of partition keys
        """
        schema = self.get_schema(schema_name)
        if schema:
            return schema.get("partition_keys", [])
        return []
    
    def get_primary_keys(self, schema_name: str) -> List[str]:
        """
        Get primary keys for a schema.
        
        Args:
            schema_name (str): Name of schema
            
        Returns:
            List[str]: List of primary keys
        """
        schema = self.get_schema(schema_name)
        if schema:
            return schema.get("primary_keys", [])
        return []
    
    def validate_field(self, field_name: str, value: Any, schema_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a field value against schema rules.
        
        Args:
            field_name (str): Name of the field
            value (Any): Value to validate
            schema_name (str): Name of schema
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        schema = self.get_schema(schema_name)
        if not schema:
            return False, f"Schema {schema_name} not found"
        
        # Find field definition
        field_def = None
        for field in schema["fields"]:
            if field["name"] == field_name:
                field_def = field
                break
        
        if not field_def:
            return False, f"Field {field_name} not found in schema"
        
        # Check nullability
        if value is None:
            if not field_def["nullable"]:
                return False, f"Field {field_name} cannot be null"
            return True, None
        
        # Check data type
        expected_type = field_def["data_type"]
        if not self._check_data_type(value, expected_type):
            return False, f"Field {field_name} must be of type {expected_type}"
        
        # Check numeric constraints
        if expected_type in ["integer", "float"] and isinstance(value, (int, float)):
            if field_def.get("min_value") is not None and value < field_def["min_value"]:
                return False, f"Field {field_name} must be >= {field_def['min_value']}"
            if field_def.get("max_value") is not None and value > field_def["max_value"]:
                return False, f"Field {field_name} must be <= {field_def['max_value']}"
        
        # Check string constraints
        if expected_type == "string" and isinstance(value, str):
            if field_def.get("min_length") is not None and len(value) < field_def["min_length"]:
                return False, f"Field {field_name} must have length >= {field_def['min_length']}"
            if field_def.get("max_length") is not None and len(value) > field_def["max_length"]:
                return False, f"Field {field_name} must have length <= {field_def['max_length']}"
            if field_def.get("pattern") and not re.match(field_def["pattern"], value):
                return False, f"Field {field_name} does not match pattern {field_def['pattern']}"
            if field_def.get("allowed_values") and value not in field_def["allowed_values"]:
                return False, f"Field {field_name} must be one of {field_def['allowed_values']}"
        
        return True, None
    
    def _check_data_type(self, value: Any, expected_type: str) -> bool:
        """
        Check if value matches expected data type.
        
        Args:
            value (Any): Value to check
            expected_type (str): Expected data type
            
        Returns:
            bool: True if type matches
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "datetime": (str, datetime)  # Datetime can be string representation
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid
        
        return isinstance(value, expected_python_type)
    
    def save_schema_to_file(self, schema_name: str, file_path: str) -> bool:
        """
        Save schema definition to JSON file.
        
        Args:
            schema_name (str): Name of schema
            file_path (str): Output file path
            
        Returns:
            bool: True if saved successfully
        """
        try:
            schema = self.get_schema(schema_name)
            if schema is None:
                return False
            
            with open(file_path, 'w') as f:
                json.dump(schema, f, indent=2, default=str)
            
            return True
        except Exception:
            return False
    
    def load_schema_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load schema definition from JSON file.
        
        Args:
            file_path (str): Input file path
            
        Returns:
            Optional[Dict[str, Any]]: Schema definition or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None


def main():
    """
    Main function to test schema definitions.
    """
    try:
        # Initialize schema
        schema = YouTubeSchema()
        
        # Test getting schemas
        raw_schema = schema.get_schema("raw")
        cleaned_schema = schema.get_schema("cleaned")
        analytics_schema = schema.get_schema("analytics")
        
        print("Available schemas:", ["raw", "cleaned", "analytics"])
        print("\nRaw schema fields:", schema.get_field_names("raw"))
        print("Cleaned schema fields:", len(schema.get_field_names("cleaned")))
        print("Analytics schema fields:", len(schema.get_field_names("analytics")))
        
        # Test field validation
        is_valid, error = schema.validate_field("views", 1000, "raw")
        print(f"\nValidation test - views=1000: {is_valid}, error: {error}")
        
        is_valid, error = schema.validate_field("views", -100, "raw")
        print(f"Validation test - views=-100: {is_valid}, error: {error}")
        
        # Save schemas to files
        schema.save_schema_to_file("raw", "data/sample/raw_schema.json")
        schema.save_schema_to_file("cleaned", "data/sample/cleaned_schema.json")
        schema.save_schema_to_file("analytics", "data/sample/analytics_schema.json")
        
        print("\nSchemas saved to data/sample/")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    import re
    from typing import Tuple
    main()
