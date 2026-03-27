"""
Tableau Visualization Creator for YouTube Analytics

This script creates and publishes Tableau visualizations using the Tableau Server Client API.
It generates workbooks from templates and publishes them with proper data sources.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import tableauserverclient as TSC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TableauVisualizationCreator:
    """
    Creates and manages Tableau visualizations for YouTube Analytics.
    
    Attributes:
        server_url: Tableau server URL
        username: Tableau username
        password: Tableau password
        site_id: Tableau site ID
        tableau_auth: Authenticated Tableau client
    """
    
    def __init__(self):
        """Initialize the visualization creator."""
        self.server_url = os.getenv('TABLEAU_SERVER_URL', 'https://online.tableau.com')
        self.username = os.getenv('TABLEAU_USERNAME', '')
        self.password = os.getenv('TABLEAU_PASSWORD', '')
        self.site_id = os.getenv('TABLEAU_SITE_ID', '')
        self.tableau_auth = None
        
        logger.info("TableauVisualizationCreator initialized")
    
    def authenticate(self) -> bool:
        """
        Authenticate with Tableau server.
        
        Returns:
            bool: True if authentication successful
        """
        try:
            self.tableau_auth = TSC.TableauAuth(
                self.username,
                self.password,
                site_id=self.site_id
            )
            
            server = TSC.Server(self.server_url, use_server_ssl=True)
            
            server.auth.sign_in(self.tableau_auth)
            logger.info(f"Successfully authenticated with Tableau server: {self.server_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate with Tableau: {e}")
            return False
    
    def create_data_source(self, project_name: str = 'YouTube Analytics') -> Optional[TSC.DatasourceItem]:
        """
        Create Athena data source for YouTube analytics.
        
        Args:
            project_name (str): Name of the project
            
        Returns:
            Optional[TSC.DatasourceItem]: Created data source item
        """
        try:
            # Data source configuration
            datasource_config = {
                'name': f'{project_name} - Athena Data Source',
                'description': 'YouTube Analytics data from AWS Athena',
                'connection_type': 'athena',
                'server': 'athena.us-west-1.amazonaws.com',
                'port': '443',
                'database': 'youtube_analytics',
                'username': os.getenv('AWS_ACCESS_KEY_ID', ''),
                'password': os.getenv('AWS_SECRET_ACCESS_KEY', ''),
                's3_staging_dir': f's3://{os.getenv("S3_STAGING_DIR", "youtube-query-results")}/',
                'workgroup': 'primary'
            }
            
            # Create data source item
            datasource_item = TSC.DatasourceItem(
                name=datasource_config['name'],
                description=datasource_config['description'],
                connection_type=datasource_config['connection_type'],
                project_id=project_name
            )
            
            logger.info(f"Created data source: {datasource_config['name']}")
            return datasource_item
            
        except Exception as e:
            logger.error(f"Failed to create data source: {e}")
            return None
    
    def create_workbook_from_template(self, template_path: str, 
                                   data_source_id: str,
                                   project_name: str = 'YouTube Analytics') -> Optional[TSC.WorkbookItem]:
        """
        Create workbook from template file.
        
        Args:
            template_path (str): Path to .twb template file
            data_source_id (str): ID of the data source to connect
            project_name (str): Name of the project
            
        Returns:
            Optional[TSC.WorkbookItem]: Created workbook item
        """
        try:
            if not os.path.exists(template_path):
                logger.error(f"Template file not found: {template_path}")
                return None
            
            # Read template file
            with open(template_path, 'rb') as f:
                workbook_content = f.read()
            
            # Create workbook item
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            workbook_item = TSC.WorkbookItem(
                name=f'{project_name} Dashboard {timestamp}',
                description='Interactive YouTube Analytics Dashboard',
                project_id=project_name,
                show_tabs=False
            )
            
            logger.info(f"Created workbook: {workbook_item.name}")
            return workbook_item
            
        except Exception as e:
            logger.error(f"Failed to create workbook from template: {e}")
            return None
    
    def publish_workbook(self, workbook_item: TSC.WorkbookItem,
                       template_path: str,
                       project_name: str = 'YouTube Analytics') -> bool:
        """
        Publish workbook to Tableau server.
        
        Args:
            workbook_item (TSC.WorkbookItem): Workbook to publish
            template_path (str): Path to template file
            project_name (str): Name of the project
            
        Returns:
            bool: True if published successfully
        """
        try:
            # Publish with specific settings
            publish_mode = TSC.Server.PublishMode.Overwrite
            connections = []
            
            # Publish the workbook
            published_workbook = self.tableau_auth.server.workbooks.publish(
                workbook_item,
                template_path,
                publish_mode,
                connections
            )
            
            logger.info(f"Successfully published workbook: {published_workbook.name}")
            logger.info(f"Workbook URL: {published_workbook.content_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish workbook: {e}")
            return False
    
    def create_embed_code(self, workbook_url: str) -> str:
        """
        Generate embed code for the published workbook.
        
        Args:
            workbook_url (str): URL of the published workbook
            
        Returns:
            str: HTML embed code
        """
        embed_code = f'''
<!-- Tableau Embed Code -->
<div class='tableauPlaceholder' id='vizContainer'></div>
<script src='https://public.tableau.com/javascripts/api/tableau-2.min.js'></script>
<script>
    var viz = new tableau.Viz(
        document.getElementById('vizContainer'),
        '{workbook_url}',
        {{
            hideTabs: true,
            hideToolbar: false,
            width: '100%',
            height: '800px',
            onFirstInteractive: function() {{
                console.log('Tableau viz loaded successfully');
            }}
        }}
    );
</script>
        '''
        
        return embed_code.strip()
    
    def generate_dashboard_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of available dashboards and their configurations.
        
        Returns:
            Dict[str, Any]: Dashboard summary information
        """
        try:
            # Get all workbooks
            workbooks = list(TSC.Pager(self.tableau_auth.server.workbooks.get()))
            
            dashboard_summary = {
                'total_dashboards': len(workbooks),
                'dashboards': [],
                'created_date': datetime.now(timezone.utc).isoformat(),
                'project_name': 'YouTube Analytics'
            }
            
            for workbook in workbooks:
                dashboard_info = {
                    'id': workbook.id,
                    'name': workbook.name,
                    'description': workbook.description,
                    'created_at': workbook.created_at.isoformat() if workbook.created_at else None,
                    'updated_at': workbook.updated_at.isoformat() if workbook.updated_at else None,
                    'size': workbook.size,
                    'owner': workbook.owner_id,
                    'project_id': workbook.project_id,
                    'content_url': workbook.content_url,
                    'webpage_url': workbook.webpage_url
                }
                dashboard_summary['dashboards'].append(dashboard_info)
            
            return dashboard_summary
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard summary: {e}")
            return {}
    
    def update_data_source(self, datasource_id: str) -> bool:
        """
        Update existing data source with new configuration.
        
        Args:
            datasource_id (str): ID of the data source to update
            
        Returns:
            bool: True if updated successfully
        """
        try:
            # Get existing data source
            datasource = self.tableau_auth.server.datasources.get_by_id(datasource_id)
            
            # Update connection details
            updated_datasource = TSC.DatasourceItem(
                id=datasource.id,
                name=datasource.name,
                description=datasource.description,
                connection_type='athena',
                server='athena.us-west-1.amazonaws.com',
                port='443',
                database='youtube_analytics',
                username=os.getenv('AWS_ACCESS_KEY_ID', ''),
                password=os.getenv('AWS_SECRET_ACCESS_KEY', '')
            )
            
            # Save updated data source
            self.tableau_auth.server.datasources.update(updated_datasource)
            
            logger.info(f"Successfully updated data source: {datasource.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update data source: {e}")
            return False
    
    def create_custom_view(self, view_name: str, sql_query: str) -> bool:
        """
        Create custom view in Athena for specific analysis.
        
        Args:
            view_name (str): Name of the view
            sql_query (str): SQL query for the view
            
        Returns:
            bool: True if created successfully
        """
        try:
            import boto3
            
            # Initialize Athena client
            athena_client = boto3.client(
                'athena',
                region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-1'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            
            # Execute query to create view
            response = athena_client.start_query_execution(
                QueryString=sql_query,
                QueryExecutionContext={
                    'Database': 'youtube_analytics'
                },
                ResultConfiguration={
                    'OutputLocation': f's3://{os.getenv("S3_QUERY_RESULTS_BUCKET", "youtube-query-results")}/'
                }
            )
            
            logger.info(f"Started query execution for view: {view_name}")
            logger.info(f"Query Execution ID: {response['QueryExecutionId']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create custom view: {e}")
            return False
    
    def schedule_refresh(self, workbook_id: str, refresh_schedule: str = 'daily') -> bool:
        """
        Set up automatic refresh schedule for workbook.
        
        Args:
            workbook_id (str): ID of the workbook
            refresh_schedule (str): Schedule type (daily, weekly, monthly)
            
        Returns:
            bool: True if schedule set successfully
        """
        try:
            # Get workbook
            workbook = self.tableau_auth.server.workbooks.get_by_id(workbook_id)
            
            # Create schedule
            if refresh_schedule == 'daily':
                schedule_item = TSC.ScheduleItem(
                    name=f'Daily Refresh - {workbook.name}',
                    priority=50,
                    execution_order='parallel',
                    schedule_type='hourly',
                    interval_item=TSC.IntervalItem(
                        start_time='02:00:00',
                        interval_start_date=datetime.now(),
                        interval_end_date=None,
                        interval_type='daily'
                    )
                )
            elif refresh_schedule == 'weekly':
                schedule_item = TSC.ScheduleItem(
                    name=f'Weekly Refresh - {workbook.name}',
                    priority=50,
                    execution_order='parallel',
                    schedule_type='weekly',
                    interval_item=TSC.IntervalItem(
                        start_time='02:00:00',
                        interval_start_date=datetime.now(),
                        interval_end_date=None,
                        interval_type='weekly',
                        interval_value=1  # Every week
                    )
                )
            else:
                logger.error(f"Unsupported schedule type: {refresh_schedule}")
                return False
            
            # Create schedule on server
            created_schedule = self.tableau_auth.server.schedules.create(schedule_item)
            
            # Link schedule to workbook
            self.tableau_auth.server.workbooks.update(
                workbook,
                refresh_schedule_id=created_schedule.id
            )
            
            logger.info(f"Successfully set {refresh_schedule} refresh for workbook: {workbook.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set refresh schedule: {e}")
            return False


def main():
    """
    Main function to demonstrate Tableau visualization creation.
    """
    logger.info("🎨 Starting Tableau Visualization Creation")
    logger.info("=" * 60)
    
    try:
        # Initialize creator
        creator = TableauVisualizationCreator()
        
        # Authenticate
        if not creator.authenticate():
            logger.error("Authentication failed. Please check your Tableau credentials.")
            return False
        
        # Create data source
        datasource = creator.create_data_source()
        if not datasource:
            logger.error("Failed to create data source")
            return False
        
        # Create workbook from template
        template_path = "visualization/tableau_dashboard.twb"
        if not os.path.exists(template_path):
            logger.warning(f"Template file not found: {template_path}")
            logger.info("Creating basic dashboard configuration...")
            template_path = None
        
        workbook = creator.create_workbook_from_template(template_path, datasource.id)
        if not workbook:
            logger.error("Failed to create workbook")
            return False
        
        # Publish workbook
        if template_path and os.path.exists(template_path):
            if creator.publish_workbook(workbook, template_path):
                logger.info("✅ Workbook published successfully")
                
                # Generate embed code
                embed_code = creator.create_embed_code(workbook.content_url)
                
                # Save embed code to file
                embed_file = "visualization/dashboard_embed.html"
                with open(embed_file, 'w') as f:
                    f.write(embed_code)
                
                logger.info(f"✅ Embed code saved to: {embed_file}")
        else:
            logger.info("📋 Workbook configuration ready for manual publishing")
        
        # Generate dashboard summary
        summary = creator.generate_dashboard_summary()
        
        logger.info("📊 Dashboard Summary:")
        logger.info(f"   Total Dashboards: {summary.get('total_dashboards', 0)}")
        logger.info(f"   Project: {summary.get('project_name', 'N/A')}")
        
        for dashboard in summary.get('dashboards', []):
            logger.info(f"   - {dashboard['name']} (ID: {dashboard['id']})")
        
        # Create custom views for advanced analysis
        custom_views = [
            {
                'name': 'high_performance_videos',
                'sql': '''
                CREATE OR REPLACE VIEW youtube_analytics.high_performance_videos AS
                SELECT 
                    video_id,
                    title,
                    channel_title,
                    views,
                    likes,
                    comments,
                    engagement_rate,
                    category_id,
                    publish_date,
                    trending_date
                FROM youtube_cleaned.youtube_cleaned_data
                WHERE views > 1000000 
                   OR engagement_rate > 0.05
                ORDER BY views DESC
                '''
            },
            {
                'name': 'viral_content_analysis',
                'sql': '''
                CREATE OR REPLACE VIEW youtube_analytics.viral_content_analysis AS
                SELECT 
                    category_id,
                    COUNT(*) AS video_count,
                    SUM(views) AS total_views,
                    AVG(views) AS avg_views,
                    COUNT(CASE WHEN views > 1000000 THEN 1 END) AS viral_count,
                    COUNT(CASE WHEN views > 1000000 THEN 1 END) * 100.0 / COUNT(*) AS viral_percentage
                FROM youtube_cleaned.youtube_cleaned_data
                GROUP BY category_id
                ORDER BY viral_percentage DESC
                '''
            }
        ]
        
        for view in custom_views:
            if creator.create_custom_view(view['name'], view['sql']):
                logger.info(f"✅ Created custom view: {view['name']}")
            else:
                logger.warning(f"⚠️  Failed to create view: {view['name']}")
        
        logger.info("=" * 60)
        logger.info("🎉 Tableau Visualization Setup Complete!")
        logger.info("=" * 60)
        
        logger.info("📋 Next Steps:")
        logger.info("   1. Open Tableau Desktop/Server")
        logger.info("   2. Connect to YouTube Analytics data source")
        logger.info("   3. Configure dashboard filters and parameters")
        logger.info("   4. Publish and share with stakeholders")
        logger.info("   5. Set up automatic refresh schedules")
        logger.info("   6. Monitor usage and performance")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualization creation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
