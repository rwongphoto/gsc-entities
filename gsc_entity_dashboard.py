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
                self.nlp_client = language_v1.LanguageServiceClient()
                print("✅ Google Cloud NLP client initialized with default credentials")
        except Exception as e:
            print(f"❌ Failed to initialize Google Cloud NLP client: {e}")
            self.nlp_client = None
    
    def is_number_entity(self, entity_name):
        """Check if an entity is primarily numeric and should be filtered out."""
        if not entity_name:
            return True
        cleaned = re.sub(r'[,\s\-\.]', '', entity_name)
        if cleaned.isdigit():
            return True
        if entity_name.strip().endswith('%') and re.sub(r'[%,\s\-\.]', '', entity_name).isdigit():
            return True
        if re.match(r'^\d{4}$', cleaned):
            return True
        digit_count = sum(1 for char in entity_name if char.isdigit())
        total_chars = len(re.sub(r'\s', '', entity_name))
        if total_chars > 0 and (digit_count / total_chars) > 0.7:
            return True
        if len(entity_name.strip()) <= 4 and any(char.isdigit() for char in entity_name):
            return True
        return False
    
    def extract_entities_with_google_nlp(self, text):
        """Extract entities from text using Google Cloud Natural Language API."""
        if not self.nlp_client or not text:
            return {}
        try:
            max_bytes = 800000
            text_bytes = text.encode('utf-8')
            if len(text_bytes) > max_bytes:
                text = text_bytes[:max_bytes].decode('utf-8', 'ignore')
                print(f"⚠️ Text truncated to {len(text)} characters for API limits")
            text = re.sub(r'\|MINIBATCH_SEPARATOR\|', '. ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
            response = self.nlp_client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)
            entities_dict = {}
            for entity in response.entities:
                entity_name = entity.name.strip()
                if self.is_number_entity(entity_name) or len(entity_name) < 2:
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
            df.columns = df.columns.str.strip()
            if 'CTR' in df.columns and df['CTR'].dtype == 'object':
                df['CTR'] = df['CTR'].str.replace('%', '', regex=False).astype(float) / 100
            df['Year'] = year_label
            df['Top queries'] = df['Top queries'].str.lower().str.strip()
            print(f"✅ Loaded {len(df)} queries for {year_label}")
            return df
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def extract_entities_from_queries(self, df, batch_size=5):
        """Extract entities from search queries using optimized mini-batch processing for accurate entity mapping."""
        if not self.nlp_client:
            print("❌ Google Cloud NLP client not available")
            return pd.DataFrame()
        
        unique_queries = df['Top queries'].unique()
        total_unique = len(unique_queries)
        print(f"🔍 Processing {total_unique} unique queries...")
        
        query_entities_cache = {}
        processed_count = 0
        api_calls_made = 0
        progress_placeholder = st.empty()
        
        for i in range(0, total_unique, batch_size):
            batch_queries = unique_queries[i:i+batch_size]
            separator = " |MINIBATCH_SEPARATOR| "
            batch_text = separator.join(batch_queries)
            batch_entities = self.extract_entities_with_google_nlp(batch_text)
            api_calls_made += 1
            
            for query in batch_queries:
                query_entities = []
                query_lower = query.lower().strip()
                
                for entity_key, entity_info in batch_entities.items():
                    entity_name = entity_info['name']
                    entity_name_lower = entity_name.lower().strip()
                    
                    # <<< FIX #2: More Flexible Entity Mapping Logic
                    is_match = False
                    try:
                        if re.search(r'\b' + re.escape(entity_name_lower) + r'\b', query_lower):
                            is_match = True
                    except re.error:
                        if entity_name_lower in query_lower:
                            is_match = True
                    if not is_match and entity_name_lower in query_lower:
                        is_match = True
                    if not is_match and len(entity_name_lower.split()) > 1:
                        entity_words = {w for w in entity_name_lower.split() if len(w) > 2}
                        query_words = set(query_lower.split())
                        if entity_words and len(entity_words.intersection(query_words)) / len(entity_words) >= 0.7:
                            is_match = True
                    
                    if is_match:
                        query_entities.append({
                            'entity_name': entity_info['name'],
                            'entity_type': entity_info['type'],
                            'salience': entity_info['salience'],
                            'mentions': entity_info['mentions']
                        })
                
                if not query_entities:
                    query_entities = [self._categorize_query_fallback(query)]
                query_entities_cache[query] = query_entities
            
            processed_count += len(batch_queries)
            progress_percentage = (processed_count / total_unique) * 100
            progress_placeholder.progress(
                processed_count / total_unique,
                text=f"🎯 Processed {processed_count}/{total_unique} queries ({progress_percentage:.1f}%) | API calls: {api_calls_made}"
            )
        
        progress_placeholder.empty()
        entity_data = []
        for idx, row in df.iterrows():
            query = row['Top queries']
            for entity_info in query_entities_cache.get(query, []):
                entity_data.append({
                    'Query': query,
                    'Entity': entity_info['entity_name'],
                    'Entity_Type': entity_info['entity_type'],
                    'Clicks': row['Clicks'],
                    'Impressions': row['Impressions'],
                    'CTR': row['CTR'],
                    'Position': row['Position'],
                    'Year': row['Year']
                })
        print(f"✅ Entity analysis complete! Found {len(entity_data)} mappings.")
        return pd.DataFrame(entity_data)

    def _categorize_query_fallback(self, query):
        """Fallback categorization when no entities are detected."""
        query_lower = query.lower()
        if any(term in query_lower for term in ['photographer', 'photography', 'photo']):
            return {'entity_name': 'Photography', 'entity_type': 'OTHER'}
        elif any(term in query_lower for term in ['gallery', 'museum', 'exhibition']):
            return {'entity_name': 'Art Gallery', 'entity_type': 'LOCATION'}
        elif any(term in query_lower for term in ['nature', 'landscape', 'wildlife']):
            return {'entity_name': 'Nature Photography', 'entity_type': 'OTHER'}
        elif 'how' in query_lower or 'tutorial' in query_lower:
            return {'entity_name': 'Tutorial Content', 'entity_type': 'OTHER'}
        elif 'best' in query_lower or 'top' in query_lower:
            return {'entity_name': 'Comparison Query', 'entity_type': 'OTHER'}
        else:
            return {'entity_name': 'General Photography', 'entity_type': 'OTHER'}

    def aggregate_entity_performance(self, entity_df):
        """Aggregate performance metrics by entity and year."""
        agg_df = entity_df.groupby(['Entity', 'Entity_Type', 'Year']).agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR': 'mean',
            'Position': 'mean',
            'Query': 'count'
        }).rename(columns={'Query': 'Query_Count'}).reset_index()
        return agg_df

    def calculate_yoy_changes(self, agg_df):
        """Calculate year-over-year changes for each entity."""
        # <<< FIX #1: Correct YOY Calculation Logic
        duplicate_check = agg_df.groupby(['Entity', 'Year']).size()
        if (duplicate_check > 1).any():
            print("🔧 Resolving duplicate entities by keeping highest-performing entries...")
            agg_df = agg_df.loc[agg_df.groupby(['Entity', 'Year'])['Clicks'].idxmax()].reset_index(drop=True)
        
        pivot_data = {}
        metrics = ['Clicks', 'Impressions', 'CTR', 'Position', 'Query_Count']
        
        try:
            for metric in metrics:
                pivot_data[metric] = agg_df.pivot(index='Entity', columns='Year', values=metric).fillna(0)
        except ValueError as e:
            print(f"❌ Pivot error: {e}")
            return None

        if "Current Year" not in pivot_data['Clicks'].columns or "Previous Year" not in pivot_data['Clicks'].columns:
            print("⚠️ Both 'Current Year' and 'Previous Year' must be present for YOY comparison.")
            return None
            
        current_year_label = "Current Year"
        previous_year_label = "Previous Year"
        print(f"📊 Comparing {previous_year_label} vs {current_year_label}")
        
        yoy_data = []
        for entity in pivot_data['Clicks'].index:
            entity_type = agg_df[agg_df['Entity'] == entity]['Entity_Type'].iloc[0]
            changes = {}
            for metric in metrics:
                current_val = pivot_data[metric].loc[entity, current_year_label]
                previous_val = pivot_data[metric].loc[entity, previous_year_label]
                
                if metric == 'Position':
                    if previous_val > 0 and current_val > 0:
                        changes[f'{metric}_Change'] = previous_val - current_val
                    else:
                        changes[f'{metric}_Change'] = 0
                else:
                    if previous_val > 0:
                        changes[f'{metric}_Change_%'] = ((current_val - previous_val) / previous_val) * 100
                    elif current_val > 0:
                        changes[f'{metric}_Change_%'] = 100.0
                    else:
                        changes[f'{metric}_Change_%'] = 0.0
                
                changes[f'Current_{metric}'] = current_val
                changes[f'Previous_{metric}'] = previous_val
            
            performance_score = (
                changes.get('Clicks_Change_%', 0) * 0.4 +
                changes.get('Impressions_Change_%', 0) * 0.3 +
                changes.get('CTR_Change_%', 0) * 0.2 +
                changes.get('Position_Change', 0) * 0.1 * 5
            )
            yoy_data.append({'Entity': entity, 'Entity_Type': entity_type, 'Performance_Score': performance_score, **changes})
        
        result_df = pd.DataFrame(yoy_data)
        print(f"✅ YOY analysis complete for {len(result_df)} entities")
        return result_df

def create_entity_performance_dashboard():
    """Create a Streamlit dashboard for GSC entity performance analysis."""
    st.set_page_config(page_title="GSC Entity Dashboard", layout="wide")
    st.title("🎯 GSC Entity Performance Dashboard")
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Sidebar
    st.sidebar.header("🔑 Configuration")
    credentials_option = st.sidebar.radio("Authentication Method:", ["Upload JSON Key", "Use Default Credentials"])
    gcp_credentials_info = None
    if credentials_option == "Upload JSON Key":
        uploaded_file = st.sidebar.file_uploader("Upload Google Cloud Service Account JSON", type="json")
        if uploaded_file:
            gcp_credentials_info = json.load(uploaded_file)
            st.sidebar.success("✅ GCP credentials loaded!")
    
    current_file = st.sidebar.file_uploader("Current Year GSC Data (CSV)", type="csv")
    previous_file = st.sidebar.file_uploader("Previous Year GSC Data (CSV)", type="csv")
    batch_size = st.sidebar.slider("Mini-Batch Size", 3, 10, 5)

    if st.sidebar.button("🚀 Start Analysis", type="primary"):
        if not current_file or not previous_file:
            st.error("Please upload both CSV files.")
        else:
            analyzer = GSCEntityAnalyzer(gcp_credentials_info=gcp_credentials_info)
            if not analyzer.nlp_client:
                st.error("❌ GCP client not initialized.")
                return

            with st.spinner("Processing..."):
                current_df = analyzer.load_gsc_data(current_file, "Current Year")
                previous_df = analyzer.load_gsc_data(previous_file, "Previous Year")
                combined_df = pd.concat([current_df, previous_df], ignore_index=True)
                
                entity_df = analyzer.extract_entities_from_queries(combined_df, batch_size=batch_size)
                agg_df = analyzer.aggregate_entity_performance(entity_df)
                yoy_df = analyzer.calculate_yoy_changes(agg_df)

                st.session_state.yoy_df = yoy_df
                st.session_state.entity_df = entity_df
                st.session_state.analysis_complete = True
                st.success("✅ Analysis complete!")

    if st.session_state.analysis_complete:
        yoy_df = st.session_state.yoy_df
        entity_df = st.session_state.entity_df

        st.subheader("🏆 Top Performing Entities YOY")
        st.dataframe(yoy_df.nlargest(10, 'Performance_Score').round(2))

        st.subheader("⚠️ Declining Entities - Optimization Opportunities")
        st.dataframe(yoy_df.nsmallest(10, 'Performance_Score').round(2))
        
        st.subheader("🔍 Detailed Entity Analysis")
        selected_entity = st.selectbox("Select an entity:", options=yoy_df['Entity'].tolist())
        if selected_entity:
            entity_queries = entity_df[entity_df['Entity'] == selected_entity]
            st.dataframe(entity_queries)

if __name__ == "__main__":
    create_entity_performance_dashboard()
