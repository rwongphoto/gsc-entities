import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import json
import re
from collections import defaultdict, Counter

# Google Cloud NLP
from google.cloud import language_v1
from google.oauth2 import service_account

# Sentence Transformers for similarity analysis
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class GSCEntityAnalyzer:
    def __init__(self, gcp_credentials_path=None, gcp_credentials_info=None):
        """Initialize the GSC Entity Analyzer with Google Cloud NLP."""
        self.gcp_credentials_path = gcp_credentials_path
        self.gcp_credentials_info = gcp_credentials_info
        self.nlp_client = None
        
        # Initialize Google Cloud NLP client
        self._initialize_gcp_client()
        
        # Load embedding model for query similarity analysis
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _initialize_gcp_client(self):
        """Initialize Google Cloud Natural Language API client."""
        try:
            if self.gcp_credentials_info:
                credentials = service_account.Credentials.from_service_account_info(self.gcp_credentials_info)
                self.nlp_client = language_v1.LanguageServiceClient(credentials=credentials)
                print("✅ Google Cloud NLP client initialized from credentials info")
            elif self.gcp_credentials_path:
                credentials = service_account.Credentials.from_service_account_file(self.gcp_credentials_path)
                self.nlp_client = language_v1.LanguageServiceClient(credentials=credentials)
                print("✅ Google Cloud NLP client initialized from file")
            else:
                # Try to use default credentials
                self.nlp_client = language_v1.LanguageServiceClient()
                print("✅ Google Cloud NLP client initialized with default credentials")
        except Exception as e:
            print(f"❌ Failed to initialize Google Cloud NLP client: {e}")
            self.nlp_client = None
    
    def is_number_entity(self, entity_name):
        """Check if an entity is primarily numeric and should be filtered out."""
        if not entity_name:
            return True

        # Remove common separators and whitespace
        cleaned = re.sub(r'[,\s\-\.]', '', entity_name)

        # Check if it's purely numeric
        if cleaned.isdigit():
            return True

        # Check if it's a percentage
        if entity_name.strip().endswith('%') and re.sub(r'[%,\s\-\.]', '', entity_name).isdigit():
            return True

        # Check if it's a year (4 digits)
        if re.match(r'^\d{4}$', cleaned):
            return True

        # Check if it's mostly numeric (>70% digits)
        digit_count = sum(1 for char in entity_name if char.isdigit())
        total_chars = len(re.sub(r'\s', '', entity_name))

        if total_chars > 0 and (digit_count / total_chars) > 0.7:
            return True

        # Filter out very short numeric-heavy entities
        if len(entity_name.strip()) <= 4 and any(char.isdigit() for char in entity_name):
            return True

        return False
    
    def extract_entities_with_google_nlp(self, text):
        """Extract entities from text using Google Cloud Natural Language API."""
        if not self.nlp_client or not text:
            return {}

        try:
            # Handle text size limits (Google NLP API limit)
            max_bytes = 900000
            text_bytes = text.encode('utf-8')
            if len(text_bytes) > max_bytes:
                text = text_bytes[:max_bytes].decode('utf-8', 'ignore')
                print(f"⚠️ Text truncated to fit Google NLP API size limit")

            document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
            response = self.nlp_client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)

            entities_dict = {}
            for entity in response.entities:
                entity_name = entity.name.strip()

                # Filter out number entities
                if self.is_number_entity(entity_name):
                    continue

                key = entity_name.lower()
                if key not in entities_dict or entity.salience > entities_dict[key]['salience']:
                    entities_dict[key] = {
                        'name': entity_name,
                        'type': language_v1.Entity.Type(entity.type_).name,
                        'salience': entity.salience,
                        'mentions': len(entity.mentions)
                    }
            return entities_dict

        except Exception as e:
            print(f"❌ Google Cloud NLP API Error: {e}")
            return {}
    
    def load_gsc_data(self, file_path, year_label):
        """Load and clean GSC data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Clean and standardize column names
            df.columns = df.columns.str.strip()
            
            # Convert CTR from percentage string to float
            if 'CTR' in df.columns:
                df['CTR'] = df['CTR'].str.replace('%', '').astype(float) / 100
            
            # Add year label for comparison
            df['Year'] = year_label
            
            # Clean query strings
            df['Top queries'] = df['Top queries'].str.lower().str.strip()
            
            print(f"✅ Loaded {len(df)} queries for {year_label}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def extract_entities_from_queries(self, df, batch_size=50):
        """Extract entities from search queries using Google Cloud NLP."""
        if not self.nlp_client:
            print("❌ Google Cloud NLP client not available")
            return pd.DataFrame()
        
        entity_data = []
        total_queries = len(df)
        
        print(f"🔍 Extracting entities from {total_queries} queries...")
        
        # Process queries in batches to optimize API calls
        for i in range(0, total_queries, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_text = ". ".join(batch_df['Top queries'].tolist())
            
            # Extract entities from the batch
            entities = self.extract_entities_with_google_nlp(batch_text)
            
            # Map entities back to individual queries
            for idx, row in batch_df.iterrows():
                query = row['Top queries']
                query_entities = []
                
                # Find which entities appear in this specific query
                for entity_key, entity_info in entities.items():
                    entity_name = entity_info['name'].lower()
                    if entity_name in query or any(word in query.split() for word in entity_name.split()):
                        query_entities.append({
                            'entity_key': entity_key,
                            'entity_name': entity_info['name'],
                            'entity_type': entity_info['type'],
                            'salience': entity_info['salience'],
                            'mentions': entity_info['mentions']
                        })
                
                # If no specific entities found, create a fallback categorization
                if not query_entities:
                    query_entities = [self._categorize_query_fallback(query)]
                
                # Create records for each entity found in the query
                for entity_info in query_entities:
                    # Calculate query relevance if embedding model is available
                    query_relevance = 0.0
                    if self.embedding_model:
                        try:
                            query_embedding = self.embedding_model.encode([query])
                            entity_embedding = self.embedding_model.encode([entity_info['entity_name']])
                            query_relevance = cosine_similarity(query_embedding, entity_embedding)[0][0]
                        except:
                            pass
                    
                    entity_data.append({
                        'Query': query,
                        'Entity': entity_info['entity_name'],
                        'Entity_Type': entity_info['entity_type'],
                        'Entity_Salience': entity_info.get('salience', 0.0),
                        'Entity_Mentions': entity_info.get('mentions', 1),
                        'Query_Relevance': query_relevance,
                        'Clicks': row['Clicks'],
                        'Impressions': row['Impressions'],
                        'CTR': row['CTR'],
                        'Position': row['Position'],
                        'Year': row['Year']
                    })
            
            if (i + batch_size) % 200 == 0:
                print(f"   Processed {min(i + batch_size, total_queries)}/{total_queries} queries")
        
        print(f"✅ Entity extraction complete! Found {len(entity_data)} entity-query mappings")
        return pd.DataFrame(entity_data)
    
    def _categorize_query_fallback(self, query):
        """Fallback categorization when no entities are detected."""
        query_lower = query.lower()
        
        # Photography-specific categorizations
        if any(term in query_lower for term in ['photographer', 'photography', 'photo']):
            return {'entity_name': 'Photography', 'entity_type': 'OTHER', 'salience': 0.3, 'mentions': 1}
        elif any(term in query_lower for term in ['gallery', 'museum', 'exhibition']):
            return {'entity_name': 'Art Gallery', 'entity_type': 'LOCATION', 'salience': 0.3, 'mentions': 1}
        elif any(term in query_lower for term in ['nature', 'landscape', 'wildlife']):
            return {'entity_name': 'Nature Photography', 'entity_type': 'OTHER', 'salience': 0.3, 'mentions': 1}
        elif 'how' in query_lower or 'tutorial' in query_lower:
            return {'entity_name': 'Tutorial Content', 'entity_type': 'OTHER', 'salience': 0.2, 'mentions': 1}
        elif 'best' in query_lower or 'top' in query_lower:
            return {'entity_name': 'Comparison Query', 'entity_type': 'OTHER', 'salience': 0.2, 'mentions': 1}
        else:
            return {'entity_name': 'General Photography', 'entity_type': 'OTHER', 'salience': 0.1, 'mentions': 1}
    
    def aggregate_entity_performance(self, entity_df):
        """Aggregate performance metrics by entity and year."""
        return entity_df.groupby(['Entity', 'Entity_Type', 'Year']).agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR': 'mean',
            'Position': 'mean',
            'Entity_Salience': 'mean',
            'Query_Relevance': 'mean',
            'Query': 'count'
        }).rename(columns={'Query': 'Query_Count'}).reset_index()
    
    def calculate_yoy_changes(self, agg_df):
        """Calculate year-over-year changes for each entity."""
        pivot_data = {}
        metrics = ['Clicks', 'Impressions', 'CTR', 'Position', 'Query_Count']
        
        for metric in metrics:
            pivot_data[metric] = agg_df.pivot(index='Entity', columns='Year', values=metric).fillna(0)
        
        years = pivot_data['Clicks'].columns.tolist()
        if len(years) < 2:
            print("⚠️ Need at least 2 years of data for YOY comparison")
            return None
        
        current_year, previous_year = sorted(years)[-1], sorted(years)[-2]
        
        yoy_data = []
        for entity in pivot_data['Clicks'].index:
            # Get entity type
            entity_type = agg_df[agg_df['Entity'] == entity]['Entity_Type'].iloc[0] if len(agg_df[agg_df['Entity'] == entity]) > 0 else 'UNKNOWN'
            
            # Calculate changes for each metric
            changes = {}
            for metric in metrics:
                current_val = pivot_data[metric].loc[entity, current_year]
                previous_val = pivot_data[metric].loc[entity, previous_year]
                
                if metric == 'Position':
                    # For position, negative change means improvement
                    changes[f'{metric}_Change'] = previous_val - current_val
                else:
                    # For other metrics, calculate percentage change
                    if previous_val > 0:
                        changes[f'{metric}_Change_%'] = ((current_val - previous_val) / previous_val) * 100
                    else:
                        changes[f'{metric}_Change_%'] = 100.0 if current_val > 0 else 0.0
                
                changes[f'Current_{metric}'] = current_val
                changes[f'Previous_{metric}'] = previous_val
            
            # Calculate combined performance score
            clicks_weight = 0.4
            impressions_weight = 0.3
            ctr_weight = 0.2
            position_weight = 0.1
            
            performance_score = (
                changes.get('Clicks_Change_%', 0) * clicks_weight +
                changes.get('Impressions_Change_%', 0) * impressions_weight +
                changes.get('CTR_Change_%', 0) * ctr_weight +
                changes.get('Position_Change', 0) * position_weight * 10  # Scale position change
            )
            
            yoy_record = {
                'Entity': entity,
                'Entity_Type': entity_type,
                'Performance_Score': performance_score,
                **changes
            }
            
            yoy_data.append(yoy_record)
        
        return pd.DataFrame(yoy_data)

def create_entity_performance_dashboard():
    """Create a Streamlit dashboard for GSC entity performance analysis."""
    
    st.set_page_config(page_title="GSC Entity Dashboard", layout="wide")
    
    st.title("🎯 GSC Entity Performance Dashboard")
    st.markdown("**Advanced Entity Analysis using Google Cloud NLP | by Richard Wong, The SEO Consultant.ai**")
    
    # Sidebar configuration
    st.sidebar.header("🔑 Configuration")
    
    # Google Cloud credentials
    st.sidebar.subheader("Google Cloud NLP Setup")
    credentials_option = st.sidebar.radio(
        "Authentication Method:",
        ["Upload JSON Key", "Use Default Credentials"]
    )
    
    gcp_credentials_info = None
    if credentials_option == "Upload JSON Key":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Google Cloud Service Account JSON",
            type="json",
            help="Upload your GCP service account key with Natural Language API access"
        )
        if uploaded_file:
            gcp_credentials_info = json.load(uploaded_file)
            st.sidebar.success("✅ GCP credentials loaded!")
    
    # File uploads
    st.sidebar.subheader("📁 GSC Data Files")
    current_file = st.sidebar.file_uploader(
        "Current Year GSC Data (CSV)",
        type="csv",
        help="Upload your current year's GSC query data"
    )
    
    previous_file = st.sidebar.file_uploader(
        "Previous Year GSC Data (CSV)",
        type="csv",
        help="Upload your previous year's GSC query data"
    )
    
    # Analysis parameters
    st.sidebar.subheader("⚙️ Analysis Settings")
    min_clicks = st.sidebar.slider("Minimum Clicks Filter", 0, 100, 10)
    min_impressions = st.sidebar.slider("Minimum Impressions Filter", 0, 1000, 50)
    batch_size = st.sidebar.slider("NLP Batch Size", 10, 100, 50)
    
    if st.sidebar.button("🚀 Start Analysis", type="primary"):
        if not current_file or not previous_file:
            st.error("Please upload both current and previous year CSV files")
            return
        
        # Initialize analyzer
        analyzer = GSCEntityAnalyzer(gcp_credentials_info=gcp_credentials_info)
        
        if not analyzer.nlp_client:
            st.error("❌ Google Cloud NLP client not initialized. Please check your credentials.")
            return
        
        # Load data
        with st.spinner("Loading GSC data..."):
            current_df = analyzer.load_gsc_data(current_file, "Current Year")
            previous_df = analyzer.load_gsc_data(previous_file, "Previous Year")
            
            if current_df is None or previous_df is None:
                st.error("Failed to load data files")
                return
            
            # Apply filters
            current_df = current_df[
                (current_df['Clicks'] >= min_clicks) & 
                (current_df['Impressions'] >= min_impressions)
            ]
            previous_df = previous_df[
                (previous_df['Clicks'] >= min_clicks) & 
                (previous_df['Impressions'] >= min_impressions)
            ]
            
            combined_df = pd.concat([current_df, previous_df], ignore_index=True)
        
        # Extract entities
        with st.spinner("Extracting entities using Google Cloud NLP..."):
            entity_df = analyzer.extract_entities_from_queries(combined_df, batch_size=batch_size)
            
            if entity_df.empty:
                st.error("No entities were extracted. Please check your data and credentials.")
                return
        
        # Aggregate and calculate YOY changes
        with st.spinner("Calculating YOY performance changes..."):
            agg_df = analyzer.aggregate_entity_performance(entity_df)
            yoy_df = analyzer.calculate_yoy_changes(agg_df)
            
            if yoy_df is None:
                st.error("Unable to calculate YOY changes - need data from both years")
                return
        
        # Display Results
        st.success(f"✅ Analysis complete! Found {len(yoy_df)} entities across {len(entity_df)} query-entity mappings")
        
        # Key Metrics Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_entities = len(yoy_df)
            st.metric("Total Entities", total_entities)
        
        with col2:
            winners = len(yoy_df[yoy_df['Performance_Score'] > 0])
            st.metric("Improving Entities", winners, delta=f"{winners/total_entities*100:.1f}%")
        
        with col3:
            avg_clicks_change = yoy_df['Clicks_Change_%'].mean()
            st.metric("Avg Clicks Change", f"{avg_clicks_change:.1f}%")
        
        with col4:
            avg_impressions_change = yoy_df['Impressions_Change_%'].mean()
            st.metric("Avg Impressions Change", f"{avg_impressions_change:.1f}%")
        
        # Top Performers Table
        st.subheader("🏆 Top Performing Entities YOY")
        top_performers = yoy_df.nlargest(10, 'Performance_Score')[
            ['Entity', 'Entity_Type', 'Performance_Score', 'Clicks_Change_%', 
             'Impressions_Change_%', 'CTR_Change_%', 'Current_Clicks']
        ].round(2)
        
        st.dataframe(
            top_performers,
            use_container_width=True,
            column_config={
                "Performance_Score": st.column_config.ProgressColumn(
                    "Performance Score",
                    format="%.1f",
                    min_value=-100,
                    max_value=100,
                ),
                "Clicks_Change_%": st.column_config.ProgressColumn(
                    "Clicks Change %",
                    format="%.1f%%",
                    min_value=-100,
                    max_value=200,
                ),
                "Impressions_Change_%": st.column_config.ProgressColumn(
                    "Impressions Change %",
                    format="%.1f%%", 
                    min_value=-100,
                    max_value=200,
                ),
            }
        )
        
        # Declining Entities
        st.subheader("⚠️ Declining Entities - Optimization Opportunities")
        declining = yoy_df.nsmallest(10, 'Performance_Score')[
            ['Entity', 'Entity_Type', 'Performance_Score', 'Clicks_Change_%', 
             'Previous_Clicks', 'Current_Clicks']
        ].round(2)
        
        st.dataframe(declining, use_container_width=True)
        
        # Visualizations
        st.subheader("📊 Entity Performance Visualizations")
        
        # Performance Score Distribution
        fig_dist = px.histogram(
            yoy_df, 
            x='Performance_Score',
            title="Entity Performance Score Distribution",
            nbins=30,
            color_discrete_sequence=['#1f77b4']
        )
        fig_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Entity Type Performance
        entity_type_perf = yoy_df.groupby('Entity_Type').agg({
            'Performance_Score': 'mean',
            'Clicks_Change_%': 'mean',
            'Entity': 'count'
        }).reset_index().rename(columns={'Entity': 'Count'})
        
        fig_type = px.scatter(
            entity_type_perf,
            x='Clicks_Change_%',
            y='Performance_Score',
            size='Count',
            hover_name='Entity_Type',
            title="Performance by Entity Type",
            labels={'Clicks_Change_%': 'Average Clicks Change %', 'Performance_Score': 'Average Performance Score'}
        )
        st.plotly_chart(fig_type, use_container_width=True)
        
        # Detailed Entity Analysis
        st.subheader("🔍 Detailed Entity Analysis")
        
        selected_entity = st.selectbox(
            "Select an entity for detailed analysis:",
            options=yoy_df['Entity'].tolist()
        )
        
        if selected_entity:
            entity_details = yoy_df[yoy_df['Entity'] == selected_entity].iloc[0]
            entity_queries = entity_df[entity_df['Entity'] == selected_entity]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Entity:** {selected_entity}")
                st.markdown(f"**Type:** {entity_details['Entity_Type']}")
                st.markdown(f"**Performance Score:** {entity_details['Performance_Score']:.2f}")
                
                st.markdown("**YOY Changes:**")
                st.markdown(f"- Clicks: {entity_details['Clicks_Change_%']:.1f}%")
                st.markdown(f"- Impressions: {entity_details['Impressions_Change_%']:.1f}%")
                st.markdown(f"- CTR: {entity_details['CTR_Change_%']:.1f}%")
                st.markdown(f"- Position: {entity_details['Position_Change']:.2f}")
            
            with col2:
                st.markdown("**Related Queries:**")
                top_queries = entity_queries.groupby(['Query', 'Year']).agg({
                    'Clicks': 'sum',
                    'Impressions': 'sum'
                }).reset_index().nlargest(5, 'Clicks')
                
                for _, query_row in top_queries.iterrows():
                    st.markdown(f"- {query_row['Query']} ({query_row['Year']}): {query_row['Clicks']} clicks")
        
        # Export functionality
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        with col1:
            csv_data = yoy_df.to_csv(index=False)
            st.download_button(
                label="📥 Download YOY Analysis CSV",
                data=csv_data,
                file_name=f"entity_yoy_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            entity_csv = entity_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Entity Mappings CSV", 
                data=entity_csv,
                file_name=f"entity_query_mappings_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    create_entity_performance_dashboard()
