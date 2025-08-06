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
            max_bytes = 800000  # Leave some buffer
            text_bytes = text.encode('utf-8')
            if len(text_bytes) > max_bytes:
                text = text_bytes[:max_bytes].decode('utf-8', 'ignore')
                print(f"⚠️ Text truncated to {len(text)} characters for API limits")

            # Clean text for better NLP processing
            text = re.sub(r'\|MINIBATCH_SEPARATOR\|', '. ', text)  # Replace mini-batch separators with periods
            text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

            document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
            response = self.nlp_client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)

            entities_dict = {}
            for entity in response.entities:
                entity_name = entity.name.strip()

                # Filter out number entities and very short entities
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
            
            # Clean and standardize column names
            df.columns = df.columns.str.strip()
            
            # Convert CTR from percentage string to float
            if 'CTR' in df.columns:
                df['CTR'] = df['CTR'].str.replace('%', '').astype(float) / 100
            
            # Add year label for comparison
            df['Year'] = year_label
            
            # Clean query strings
            df['Top queries'] = df['Top queries'].str.lower().str.strip()
            
            # Debug: Show data summary
            total_clicks = df['Clicks'].sum()
            total_impressions = df['Impressions'].sum()
            avg_position = df['Position'].mean()
            
            print(f"✅ Loaded {len(df)} queries for {year_label}")
            print(f"   📊 Total clicks: {total_clicks:,}")
            print(f"   👁️ Total impressions: {total_impressions:,}")
            print(f"   📍 Average position: {avg_position:.2f}")
            
            # DEBUG: Show Ansel Adams data in this file
            ansel_queries = df[df['Top queries'].str.contains('ansel adams', na=False, case=False)]
            if not ansel_queries.empty:
                ansel_total_clicks = ansel_queries['Clicks'].sum()
                print(f"   🎭 Ansel Adams in this file ({year_label}): {ansel_total_clicks} total clicks")
                for _, row in ansel_queries.head(3).iterrows():
                    print(f"      - '{row['Top queries']}': {row['Clicks']} clicks")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def extract_entities_from_queries(self, df, batch_size=5):
        """Extract entities from search queries using optimized mini-batch processing for accurate entity mapping."""
        if not self.nlp_client:
            print("❌ Google Cloud NLP client not available")
            return pd.DataFrame()
        
        # Step 1: Get unique queries and prepare for mini-batch processing
        unique_queries = df['Top queries'].unique()
        total_unique = len(unique_queries)
        total_queries = len(df)
        
        print(f"🔍 Processing {total_unique} unique queries from {total_queries} total queries...")
        print(f"🎯 Using optimized mini-batch approach (batch size: {batch_size}) for accurate entity mapping...")
        
        # Step 2: Process in mini-batches for accurate entity-to-query mapping
        query_entities_cache = {}
        processed_count = 0
        api_calls_made = 0
        
        # Create progress bar placeholder
        progress_placeholder = st.empty()
        
        for i in range(0, total_unique, batch_size):
            batch_queries = unique_queries[i:i+batch_size]
            
            # Create mini-batch text with clear separators
            separator = " |MINIBATCH_SEPARATOR| "
            batch_text = separator.join(batch_queries)
            
            # Extract entities from this mini-batch
            batch_entities = self.extract_entities_with_google_nlp(batch_text)
            api_calls_made += 1
            
            # Now do PRECISE mapping - only associate entities with queries where they actually appear
            for query in batch_queries:
                query_entities = []
                query_lower = query.lower().strip()
                
                # For each entity found in the mini-batch, check if it belongs to this query
                for entity_key, entity_info in batch_entities.items():
                    entity_name = entity_info['name']
                    entity_name_lower = entity_name.lower().strip()
                    
                    # MUCH MORE RELAXED MATCHING
                    is_match = False
                    
                    # Method 1: Direct contains check for most cases
                    if entity_name_lower in query_lower or query_lower in entity_name_lower:
                        is_match = True
                    
                    # Method 2: Word-based matching for names and topics
                    else:
                        entity_words = [w for w in entity_name_lower.split() if len(w) > 2]
                        query_words = query_lower.split()
                        
                        # If any significant entity word appears in query, it's a match
                        for entity_word in entity_words:
                            if entity_word in query_words:
                                is_match = True
                                break
                    
                    if is_match:
                        query_entities.append({
                            'entity_name': entity_info['name'],
                            'entity_type': entity_info['type'],
                            'salience': entity_info['salience'],
                            'mentions': entity_info['mentions']
                        })
                
                # Fallback: If no precise entities found, create a broad category
                if not query_entities:
                    query_entities = [self._categorize_query_fallback(query)]
                
                # Cache the results
                query_entities_cache[query] = query_entities
            
            processed_count += len(batch_queries)
            progress_percentage = (processed_count / total_unique) * 100
            
            # Update progress with API call info
            progress_placeholder.progress(
                processed_count / total_unique,
                text=f"🎯 Processed {processed_count}/{total_unique} queries ({progress_percentage:.1f}%) | API calls: {api_calls_made}"
            )
        
        progress_placeholder.empty()
        
        # Step 3: Build final dataset using precise entity mappings
        print(f"🔄 Building precise entity-query mappings... (made {api_calls_made} API calls)")
        entity_data = []
        
        # Pre-compute embeddings for unique entities if available
        entity_embeddings_cache = {}
        if self.embedding_model:
            all_entities = set()
            for entities_list in query_entities_cache.values():
                for entity in entities_list:
                    all_entities.add(entity['entity_name'])
            
            if all_entities:
                try:
                    entity_names_list = list(all_entities)
                    embeddings = self.embedding_model.encode(entity_names_list)
                    entity_embeddings_cache = dict(zip(entity_names_list, embeddings))
                    print(f"✅ Pre-computed embeddings for {len(all_entities)} unique entities")
                except Exception as e:
                    print(f"⚠️ Failed to pre-compute embeddings: {e}")
        
        # Process each row in the original dataframe
        precise_mappings = 0
        fallback_mappings = 0
        
        for idx, row in df.iterrows():
            query = row['Top queries']
            cached_entities = query_entities_cache.get(query, [])
            
            for entity_info in cached_entities:
                # Calculate query relevance efficiently using cached embeddings
                query_relevance = 0.0
                if self.embedding_model and entity_info['entity_name'] in entity_embeddings_cache:
                    try:
                        query_embedding = self.embedding_model.encode([query])
                        entity_embedding = entity_embeddings_cache[entity_info['entity_name']].reshape(1, -1)
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
                
                # Track mapping quality
                if entity_info.get('salience', 0) > 0.1:
                    precise_mappings += 1
                else:
                    fallback_mappings += 1
        
        print(f"✅ Precise entity analysis complete!")
        print(f"   📊 Made {api_calls_made} API calls (vs {total_unique} for individual processing)")
        print(f"   🎯 Found {len(entity_data)} entity-query mappings from {total_unique} unique queries")
        print(f"   ✅ Precise mappings: {precise_mappings}")
        print(f"   📂 Fallback categorizations: {fallback_mappings}")
        print(f"   🚀 Mapping accuracy: {(precise_mappings / (precise_mappings + fallback_mappings) * 100):.1f}%")
        
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
        # First, handle potential duplicates by creating a unique entity identifier
        entity_df['Entity_Unique'] = entity_df['Entity'] + '_' + entity_df['Entity_Type']
        
        # Aggregate by the unique entity identifier
        agg_df = entity_df.groupby(['Entity_Unique', 'Entity', 'Entity_Type', 'Year']).agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR': 'mean',
            'Position': 'mean',
            'Entity_Salience': 'mean',
            'Query_Relevance': 'mean',
            'Query': 'count'
        }).rename(columns={'Query': 'Query_Count'}).reset_index()
        
        # If there are still Entity-Year duplicates, resolve them by taking the one with higher clicks
        duplicates = agg_df.groupby(['Entity', 'Year']).size()
        if (duplicates > 1).any():
            print(f"⚠️ Found {(duplicates > 1).sum()} entities with multiple types. Resolving by highest performance...")
            # Keep the version with highest clicks for each Entity-Year combination
            agg_df = agg_df.loc[agg_df.groupby(['Entity', 'Year'])['Clicks'].idxmax()].reset_index(drop=True)
        
        return agg_df
    
    def calculate_yoy_changes(self, agg_df):
        """Calculate year-over-year changes for each entity."""
        # Check for duplicates before pivoting
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
            print("Available data structure:")
            print(f"Unique entities: {agg_df['Entity'].nunique()}")
            print(f"Unique years: {agg_df['Year'].nunique()}")
            print(f"Total rows: {len(agg_df)}")
            print("Sample data:")
            print(agg_df[['Entity', 'Year', 'Clicks']].head(10))
            return None
        
        years = pivot_data['Clicks'].columns.tolist()
        if len(years) < 2:
            print("⚠️ Need at least 2 years of data for YOY comparison")
            return None
        
        # FIX: Explicitly assign current and previous years based on labels
        print(f"📊 Available years: {years}")
        
        # Define which year is current and which is previous based on the labels we use
        if "Current Year" in years and "Previous Year" in years:
            current_year = "Current Year"
            previous_year = "Previous Year"
            print(f"✅ Using explicit labels: {previous_year} vs {current_year}")
        else:
            # Fallback to sorting if actual year numbers are used
            current_year, previous_year = sorted(years)[-1], sorted(years)[-2]
            print(f"⚠️ Using sorted years: {previous_year} vs {current_year}")
        
        yoy_data = []
        
        for entity in pivot_data['Clicks'].index:
            # Get entity type (safely)
            entity_type_series = agg_df[agg_df['Entity'] == entity]['Entity_Type']
            entity_type = entity_type_series.iloc[0] if len(entity_type_series) > 0 else 'UNKNOWN'
            
            # Calculate changes for each metric
            changes = {}
            for metric in metrics:
                # Get the values from pivot table with CORRECT year assignment
                current_val = pivot_data[metric].loc[entity, current_year]
                previous_val = pivot_data[metric].loc[entity, previous_year]
                
                if metric == 'Position':
                    # For position, negative change means improvement (lower position number is better)
                    changes[f'{metric}_Change'] = previous_val - current_val
                else:
                    # For other metrics, calculate percentage change: (current - previous) / previous * 100
                    if previous_val > 0:
                        changes[f'{metric}_Change_%'] = ((current_val - previous_val) / previous_val) * 100
                    elif current_val > 0:
                        # If previous was 0 but current has value, that's 100% growth
                        changes[f'{metric}_Change_%'] = 100.0
                    else:
                        # Both are 0
                        changes[f'{metric}_Change_%'] = 0.0
                
                # Store the values correctly labeled
                changes[f'Current_{metric}'] = current_val  
                changes[f'Previous_{metric}'] = previous_val
            
            # Calculate CTR values from clicks and impressions
            current_ctr = (changes['Current_Clicks'] / changes['Current_Impressions'] * 100) if changes['Current_Impressions'] > 0 else 0
            previous_ctr = (changes['Previous_Clicks'] / changes['Previous_Impressions'] * 100) if changes['Previous_Impressions'] > 0 else 0
            ctr_change = current_ctr - previous_ctr
            
            changes['Current_CTR'] = current_ctr
            changes['Previous_CTR'] = previous_ctr
            changes['CTR_Change_%'] = ((current_ctr - previous_ctr) / previous_ctr * 100) if previous_ctr > 0 else (100.0 if current_ctr > 0 else 0.0)
            
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
            
            # DEBUG: Show what's being stored for Ansel Adams
            if 'ansel adams' in entity.lower():
                print(f"\n🔍 CORRECTED RECORD FOR {entity}:")
                print(f"   Previous Year ({previous_year}): {yoy_record['Previous_Clicks']} clicks")
                print(f"   Current Year ({current_year}): {yoy_record['Current_Clicks']} clicks")
                print(f"   Clicks_Change_%: {yoy_record['Clicks_Change_%']:.1f}%")
                print(f"   Performance_Score: {yoy_record['Performance_Score']:.1f}")
                
                # Additional verification
                if yoy_record['Current_Clicks'] > yoy_record['Previous_Clicks']:
                    print(f"   ✅ CORRECT: Current > Previous (positive growth)")
                else:
                    print(f"   ❌ INCORRECT: Current <= Previous")
            
            yoy_data.append(yoy_record)
        
        result_df = pd.DataFrame(yoy_data)
        print(f"✅ FIXED YOY analysis complete for {len(result_df)} entities")
        return result_df

def create_entity_performance_dashboard():
    """Create a Streamlit dashboard for GSC entity performance analysis."""
    
    st.set_page_config(page_title="GSC Entity Dashboard", layout="wide")
    
    st.title("🎯 GSC Entity Performance Dashboard")
    st.markdown("**Advanced Entity Analysis using Google Cloud NLP | by Richard Wong, The SEO Consultant.ai**")
    st.markdown("**🔄 Code Version: 12.0 - TOTAL IMPACT FOCUS + ENHANCED DATA**")
    
    st.markdown("""
    **🚀 MAJOR IMPROVEMENTS:**
    - ✅ **NEW**: Analysis Overview includes Overall CTR metric  
    - ✅ **ENHANCED**: Tables now include Previous/Current Impressions & CTR columns
    - ✅ **CHANGED**: Sorting by absolute click changes (total impact) instead of percentages
    - ✅ **ENHANCED**: Visualizations focus on total changes with comprehensive hover data
    - ✅ **IMPROVED**: Winners vs Losers shows total click gains/losses in labels
    - ✅ **ENHANCED**: Scatter plot sized by current clicks, colored by absolute change
    - ✅ **IMPROVED**: Treemap and bar charts show absolute changes with full context in tooltips
    - ✅ **FOCUS**: All metrics prioritize business impact over percentage changes
    """)

    
    # Initialize session state for persistent data
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'entity_df' not in st.session_state:
        st.session_state.entity_df = None
    if 'agg_df' not in st.session_state:
        st.session_state.agg_df = None
    if 'yoy_df' not in st.session_state:
        st.session_state.yoy_df = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
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
    min_clicks = st.sidebar.slider("Minimum Clicks Filter", 0, 100, 1)
    min_impressions = st.sidebar.slider("Minimum Impressions Filter", 0, 1000, 10)
    batch_size = st.sidebar.slider("Mini-Batch Size", 3, 10, 5, help="Number of queries per API call (smaller = more precise)")
    
    # Add caching option
    use_cache = st.sidebar.checkbox("Enable Caching", value=True, help="Cache results to speed up repeated analysis")
    
    # Analysis button
    analysis_button = st.sidebar.button("🚀 Start Analysis", type="primary")
    
    # Clear results button
    if st.session_state.analysis_complete:
        if st.sidebar.button("🗑️ Clear Results & Use New Code", help="Clear current analysis to see code updates"):
            st.session_state.analysis_complete = False
            st.session_state.entity_df = None
            st.session_state.agg_df = None
            st.session_state.yoy_df = None
            st.session_state.analyzer = None
            if 'entity_cache' in st.session_state:
                st.session_state.entity_cache = {}
            st.success("✅ Cache cleared! Run analysis again to see new code changes.")
            st.rerun()
    
    # Only run analysis if button is clicked and files are uploaded
    if analysis_button:
        if not current_file or not previous_file:
            st.error("Please upload both current and previous year CSV files")
        else:
            # Clear previous results
            st.session_state.analysis_complete = False
            st.session_state.entity_df = None
            st.session_state.agg_df = None
            st.session_state.yoy_df = None
            
            # Initialize analyzer
            analyzer = GSCEntityAnalyzer(gcp_credentials_info=gcp_credentials_info)
            st.session_state.analyzer = analyzer
            
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
                
                # Debug: Show file content summary before filtering
                st.info("📋 **File Content Summary (before filtering):**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Current Year File:**")
                    st.write(f"- Rows: {len(current_df)}")
                    st.write(f"- Total Clicks: {current_df['Clicks'].sum():,}")
                    st.write(f"- Top query: '{current_df.loc[current_df['Clicks'].idxmax(), 'Top queries']}'")
                
                with col2:
                    st.write("**Previous Year File:**") 
                    st.write(f"- Rows: {len(previous_df)}")
                    st.write(f"- Total Clicks: {previous_df['Clicks'].sum():,}")
                    st.write(f"- Top query: '{previous_df.loc[previous_df['Clicks'].idxmax(), 'Top queries']}'")
                
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
                
                # Debug: Show overlap analysis
                current_queries = set(current_df['Top queries'])
                previous_queries = set(previous_df['Top queries'])
                overlap = current_queries & previous_queries
                
                st.info(f"📊 **Query Overlap Analysis:**")
                st.write(f"- Current year unique queries: {len(current_queries)}")
                st.write(f"- Previous year unique queries: {len(previous_queries)}")
                st.write(f"- Overlapping queries: {len(overlap)} ({len(overlap)/max(len(current_queries), len(previous_queries))*100:.1f}%)")
                
                if len(overlap) < 10:
                    st.warning("⚠️ Very low query overlap detected! This may indicate different time periods or data sources.")
                    st.write("**Sample overlapping queries:**", list(overlap)[:5] if overlap else "None")
                    st.write("**Current-only queries:**", list(current_queries - previous_queries)[:5])
                    st.write("**Previous-only queries:**", list(previous_queries - current_queries)[:5])
            
            # Extract entities (with caching)
            cache_key = f"{hash(str(combined_df['Top queries'].tolist()))}_batch{batch_size}"
            
            if use_cache and 'entity_cache' in st.session_state and cache_key in st.session_state.entity_cache:
                st.info("📦 Using cached entity extraction results...")
                entity_df = st.session_state.entity_cache[cache_key]
            else:
                with st.spinner("Extracting entities using Google Cloud NLP..."):
                    entity_df = analyzer.extract_entities_from_queries(combined_df, batch_size=batch_size)
                    
                    if entity_df.empty:
                        st.error("No entities were extracted. Please check your data and credentials.")
                        return
                    
                    # Cache results
                    if use_cache:
                        if 'entity_cache' not in st.session_state:
                            st.session_state.entity_cache = {}
                        st.session_state.entity_cache[cache_key] = entity_df
            
            # Store in session state
            st.session_state.entity_df = entity_df
            
            # Aggregate and calculate YOY changes
            with st.spinner("Calculating YOY performance changes..."):
                agg_df = analyzer.aggregate_entity_performance(entity_df)
                yoy_df = analyzer.calculate_yoy_changes(agg_df)
                
                if yoy_df is None:
                    st.error("Unable to calculate YOY changes - need data from both years")
                    return
            
            # Store results in session state
            st.session_state.agg_df = agg_df
            st.session_state.yoy_df = yoy_df
            st.session_state.analysis_complete = True
            
            st.success(f"✅ Analysis complete! Found {len(yoy_df)} entities across {len(entity_df)} query-entity mappings")
            
            # DEBUG: Check Ansel Adams in final YOY results
            ansel_yoy = yoy_df[yoy_df['Entity'].str.contains('ansel adams', case=False, na=False)]
            if not ansel_yoy.empty:
                row = ansel_yoy.iloc[0]
                st.success("🔍 **VERIFICATION: Year calculation fix working correctly!**")
                st.write(f"- Previous_Clicks: {row['Previous_Clicks']}")
                st.write(f"- Current_Clicks: {row['Current_Clicks']}")
                st.write(f"- Clicks_Change_%: {row['Clicks_Change_%']:.1f}%")
                
                if row['Clicks_Change_%'] > 0 and row['Current_Clicks'] > row['Previous_Clicks']:
                    st.success("✅ **CORRECT: Years are properly assigned - showing positive growth!**")
                else:
                    st.error("❌ **Still incorrect - please check data files**")
    
    # Display Results (only if analysis is complete)
    if st.session_state.analysis_complete and st.session_state.yoy_df is not None:
        yoy_df = st.session_state.yoy_df
        entity_df = st.session_state.entity_df
        
        # Key Metrics Overview
        st.subheader("📊 Analysis Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_entities = len(yoy_df)
            st.metric("Total Entities", total_entities)
        
        with col2:
            winners = len(yoy_df[yoy_df['Clicks_Change_%'] > 0])
            st.metric("Improving Entities", winners, delta=f"{winners/total_entities*100:.1f}%")
        
        with col3:
            # Calculate actual total percentage change with totals
            total_current_clicks = yoy_df['Current_Clicks'].sum()
            total_previous_clicks = yoy_df['Previous_Clicks'].sum()
            total_clicks_change = ((total_current_clicks - total_previous_clicks) / total_previous_clicks * 100) if total_previous_clicks > 0 else 0
            clicks_delta = total_current_clicks - total_previous_clicks
            st.metric(
                "Total Clicks", 
                f"{total_current_clicks:,}",
                delta=f"{clicks_delta:+,} ({total_clicks_change:+.1f}%)"
            )
        
        with col4:
            # Calculate actual total percentage change with totals
            total_current_impressions = yoy_df['Current_Impressions'].sum()
            total_previous_impressions = yoy_df['Previous_Impressions'].sum()
            total_impressions_change = ((total_current_impressions - total_previous_impressions) / total_previous_impressions * 100) if total_previous_impressions > 0 else 0
            impressions_delta = total_current_impressions - total_previous_impressions
            st.metric(
                "Total Impressions", 
                f"{total_current_impressions:,}",
                delta=f"{impressions_delta:+,} ({total_impressions_change:+.1f}%)"
            )
        
        with col5:
            # Calculate weighted average CTR change
            total_current_ctr = (total_current_clicks / total_current_impressions * 100) if total_current_impressions > 0 else 0
            total_previous_ctr = (total_previous_clicks / total_previous_impressions * 100) if total_previous_impressions > 0 else 0
            ctr_change = total_current_ctr - total_previous_ctr
            st.metric(
                "Overall CTR", 
                f"{total_current_ctr:.2f}%",
                delta=f"{ctr_change:+.2f}pp"
            )
        
        # Filters for results (these won't trigger rerun now)
        st.subheader("🔍 Filter Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            entity_types = ['All'] + sorted(yoy_df['Entity_Type'].unique().tolist())
            selected_type = st.selectbox("Filter by Entity Type:", entity_types, key="entity_type_filter")
        
        with col2:
            min_performance = st.slider(
                "Minimum Performance Score:", 
                float(yoy_df['Performance_Score'].min()), 
                float(yoy_df['Performance_Score'].max()), 
                float(yoy_df['Performance_Score'].min()),
                key="performance_filter"
            )
        
        with col3:
            min_current_clicks = st.slider(
                "Minimum Current Clicks:", 
                0, 
                int(yoy_df['Current_Clicks'].max()), 
                0,
                key="clicks_filter"
            )
        
        # Apply filters
        filtered_df = yoy_df.copy()
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['Entity_Type'] == selected_type]
        filtered_df = filtered_df[
            (filtered_df['Performance_Score'] >= min_performance) & 
            (filtered_df['Current_Clicks'] >= min_current_clicks)
        ]
        
        # Top Performers Table
        st.subheader("🏆 Top Performing Entities YOY")
        if len(filtered_df) > 0:
            # Calculate absolute click change for sorting by total impact
            filtered_df['Clicks_Absolute_Change'] = filtered_df['Current_Clicks'] - filtered_df['Previous_Clicks']
            
            # Sort by highest absolute click increase (total impact)
            top_performers = filtered_df.nlargest(15, 'Clicks_Absolute_Change')[
                ['Entity', 'Entity_Type', 'Clicks_Absolute_Change', 'Previous_Clicks', 'Current_Clicks',
                 'Previous_Impressions', 'Current_Impressions', 'Previous_CTR', 'Current_CTR', 'Clicks_Change_%']
            ].round(2)
            
            st.dataframe(
                top_performers,
                use_container_width=True,
                column_config={
                    "Clicks_Absolute_Change": st.column_config.NumberColumn(
                        "Total Click Gain",
                        format="%d",
                    ),
                    "Clicks_Change_%": st.column_config.ProgressColumn(
                        "Clicks Change %",
                        format="%.1f%%",
                        min_value=-100,
                        max_value=500,
                    ),
                    "Previous_CTR": st.column_config.NumberColumn(
                        "Previous CTR",
                        format="%.2f%%",
                    ),
                    "Current_CTR": st.column_config.NumberColumn(
                        "Current CTR", 
                        format="%.2f%%",
                    ),
                }
            )
        else:
            st.info("No entities match the current filters.")
        
        # Declining Entities
        st.subheader("⚠️ Declining Entities - Optimization Opportunities")
        if len(filtered_df) > 0:
            # Filter to only declining entities and sort by most negative absolute change
            declining_df = filtered_df[filtered_df['Clicks_Absolute_Change'] < 0]
            if len(declining_df) > 0:
                declining = declining_df.nsmallest(15, 'Clicks_Absolute_Change')[
                    ['Entity', 'Entity_Type', 'Clicks_Absolute_Change', 'Previous_Clicks', 'Current_Clicks',
                     'Previous_Impressions', 'Current_Impressions', 'Previous_CTR', 'Current_CTR', 'Clicks_Change_%']
                ].round(2)
                
                st.dataframe(
                    declining, 
                    use_container_width=True,
                    column_config={
                        "Clicks_Absolute_Change": st.column_config.NumberColumn(
                            "Total Click Loss",
                            format="%d",
                        ),
                        "Clicks_Change_%": st.column_config.ProgressColumn(
                            "Clicks Change %",
                            format="%.1f%%",
                            min_value=-100,
                            max_value=100,
                        ),
                        "Previous_CTR": st.column_config.NumberColumn(
                            "Previous CTR",
                            format="%.2f%%",
                        ),
                        "Current_CTR": st.column_config.NumberColumn(
                            "Current CTR",
                            format="%.2f%%",
                        ),
                    }
                )
            else:
                st.success("🎉 No declining entities found! All entities are performing well.")
        else:
            st.info("No entities match the current filters.")
        
        # Visualizations
        st.subheader("📊 Entity Performance Visualizations")
        
        if len(filtered_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Winners vs Losers Analysis based on absolute changes
                winners_count = len(filtered_df[filtered_df['Clicks_Absolute_Change'] > 0])
                losers_count = len(filtered_df[filtered_df['Clicks_Absolute_Change'] < 0])
                stable_count = len(filtered_df[filtered_df['Clicks_Absolute_Change'] == 0])
                
                # Calculate total impacts
                total_gains = filtered_df[filtered_df['Clicks_Absolute_Change'] > 0]['Clicks_Absolute_Change'].sum()
                total_losses = abs(filtered_df[filtered_df['Clicks_Absolute_Change'] < 0]['Clicks_Absolute_Change'].sum())
                
                performance_summary = pd.DataFrame({
                    'Performance': [
                        f'📈 Winners (+{total_gains:,} clicks)', 
                        f'📉 Losers (-{total_losses:,} clicks)', 
                        '➡️ Stable'
                    ],
                    'Count': [winners_count, losers_count, stable_count],
                    'Percentage': [
                        (winners_count / len(filtered_df)) * 100,
                        (losers_count / len(filtered_df)) * 100, 
                        (stable_count / len(filtered_df)) * 100
                    ]
                })
                
                fig_summary = px.pie(
                    performance_summary, 
                    values='Count', 
                    names='Performance',
                    title="Entity Performance Distribution<br><sub>Showing total click impact</sub>",
                    color_discrete_map={
                        performance_summary.iloc[0]['Performance']: '#2E8B57',
                        performance_summary.iloc[1]['Performance']: '#DC143C', 
                        '➡️ Stable': '#808080'
                    }
                )
                fig_summary.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_summary, use_container_width=True)
            
            with col2:
                # Current vs Previous Performance Scatter with absolute changes
                # Add absolute change data for hover
                scatter_data = filtered_df.head(50).copy()
                scatter_data['Impressions_Absolute_Change'] = scatter_data['Current_Impressions'] - scatter_data['Previous_Impressions']
                
                fig_scatter = px.scatter(
                    scatter_data,
                    x='Previous_Clicks',
                    y='Current_Clicks',
                    size='Current_Clicks',
                    color='Clicks_Absolute_Change',
                    hover_name='Entity',
                    hover_data={
                        'Entity_Type': True,
                        'Clicks_Absolute_Change': ':+,',
                        'Impressions_Absolute_Change': ':+,',
                        'Previous_Clicks': ':,',
                        'Current_Clicks': ':,',
                        'Previous_Impressions': ':,',
                        'Current_Impressions': ':,'
                    },
                    title="Current vs Previous Performance (Top 50)<br><sub>Size = Current Clicks, Color = Total Change</sub>",
                    color_continuous_scale=['red', 'yellow', 'green'],
                    labels={
                        'Previous_Clicks': 'Previous Year Clicks',
                        'Current_Clicks': 'Current Year Clicks',
                        'Clicks_Absolute_Change': 'Total Click Change'
                    }
                )
                
                # Add diagonal line for reference (y = x means no change)
                max_val = max(filtered_df['Current_Clicks'].max(), filtered_df['Previous_Clicks'].max())
                fig_scatter.add_shape(
                    type="line",
                    x0=0, y0=0, x1=max_val, y1=max_val,
                    line=dict(color="gray", width=2, dash="dash"),
                )
                fig_scatter.add_annotation(
                    x=max_val*0.7, y=max_val*0.8,
                    text="No Change Line",
                    showarrow=False,
                    font=dict(color="gray")
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Enhanced Entity Type Performance with treemap
            if len(filtered_df['Entity_Type'].unique()) > 1:
                entity_type_detailed = filtered_df.groupby('Entity_Type').agg({
                    'Current_Clicks': 'sum',
                    'Previous_Clicks': 'sum',
                    'Current_Impressions': 'sum',
                    'Previous_Impressions': 'sum',
                    'Clicks_Absolute_Change': 'sum',
                    'Entity': 'count'
                }).reset_index()
                entity_type_detailed.columns = ['Entity_Type', 'Total_Current_Clicks', 'Total_Previous_Clicks', 'Total_Current_Impressions', 'Total_Previous_Impressions', 'Total_Click_Change', 'Entity_Count']
                
                # Calculate impressions change
                entity_type_detailed['Total_Impressions_Change'] = entity_type_detailed['Total_Current_Impressions'] - entity_type_detailed['Total_Previous_Impressions']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Treemap showing entity types by current volume and colored by absolute change
                    fig_treemap = px.treemap(
                        entity_type_detailed,
                        path=['Entity_Type'],
                        values='Total_Current_Clicks',
                        color='Total_Click_Change',
                        title="Entity Types by Current Traffic Volume<br><sub>Size = Current Clicks, Color = Total Change</sub>",
                        color_continuous_scale=['red', 'yellow', 'green'],
                        hover_data={
                            'Entity_Count': True,
                            'Total_Click_Change': ':+,',
                            'Total_Impressions_Change': ':+,',
                            'Total_Current_Clicks': ':,',
                            'Total_Previous_Clicks': ':,'
                        }
                    )
                    fig_treemap.update_traces(
                        textinfo="label+value",
                        textposition="middle center",
                        hovertemplate='<b>%{label}</b><br>' +
                                    'Current Clicks: %{value:,}<br>' +
                                    'Click Change: %{color:+,}<br>' +
                                    'Entities: %{customdata[0]}<br>' +
                                    'Impressions Change: %{customdata[1]:+,}<br>' +
                                    '<extra></extra>'
                    )
                    st.plotly_chart(fig_treemap, use_container_width=True)
                
                with col2:
                    # Bar chart showing absolute change by entity type
                    fig_bar = px.bar(
                        entity_type_detailed.sort_values('Total_Click_Change', ascending=True),
                        x='Total_Click_Change',
                        y='Entity_Type',
                        orientation='h',
                        title="Total Click Change by Entity Type<br><sub>Absolute numbers show real impact</sub>",
                        color='Total_Click_Change',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        hover_data={
                            'Entity_Count': True,
                            'Total_Current_Clicks': ':,',
                            'Total_Previous_Clicks': ':,',
                            'Total_Impressions_Change': ':+,'
                        }
                    )
                    fig_bar.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    fig_bar.update_traces(
                        hovertemplate='<b>%{y}</b><br>' +
                                    'Click Change: %{x:+,}<br>' +
                                    'Current Clicks: %{customdata[1]:,}<br>' +
                                    'Previous Clicks: %{customdata[2]:,}<br>' +
                                    'Entities: %{customdata[0]}<br>' +
                                    'Impressions Change: %{customdata[3]:+,}<br>' +
                                    '<extra></extra>'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed Entity Analysis
        st.subheader("🔍 Detailed Entity Analysis")
        
        if len(filtered_df) > 0:
            selected_entity = st.selectbox(
                "Select an entity for detailed analysis:",
                options=filtered_df['Entity'].tolist(),
                key="entity_detail_selector"
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
                    # Show queries from both years for better debugging
                    entity_queries_detailed = entity_df[entity_df['Entity'] == selected_entity]
                    
                    # Group by query and year, then show top queries from each year
                    queries_by_year = entity_queries_detailed.groupby(['Query', 'Year']).agg({
                        'Clicks': 'sum',
                        'Impressions': 'sum'
                    }).reset_index()
                    
                    # Sort by clicks and show top 5 overall
                    top_queries = queries_by_year.nlargest(10, 'Clicks')
                    
                    st.markdown("**All Related Queries (both years):**")
                    for _, query_row in top_queries.iterrows():
                        year_icon = "📅" if query_row['Year'] == "Current Year" else "📆"
                        st.markdown(f"- {year_icon} {query_row['Query']} ({query_row['Year']}): {query_row['Clicks']} clicks, {query_row['Impressions']} impressions")
                    
                    # Also show year-specific totals
                    year_totals = entity_queries_detailed.groupby('Year').agg({
                        'Clicks': 'sum',
                        'Impressions': 'sum'
                    }).reset_index()
                    
                    st.markdown("**Entity Totals by Year:**")
                    for _, total_row in year_totals.iterrows():
                        year_icon = "📅" if total_row['Year'] == "Current Year" else "📆"
                        st.markdown(f"- {year_icon} {total_row['Year']}: {total_row['Clicks']} total clicks, {total_row['Impressions']} total impressions")
        
        # Export functionality (moved to bottom to prevent rerun issues)
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered YOY Analysis CSV",
                data=csv_data,
                file_name=f"entity_yoy_analysis_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_yoy"
            )
        
        with col2:
            entity_csv = entity_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Entity Mappings CSV", 
                data=entity_csv,
                file_name=f"entity_query_mappings_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_entity"
            )
    
    elif not st.session_state.analysis_complete:
        st.info("👆 Please upload your GSC data files and click 'Start Analysis' to begin.")

if __name__ == "__main__":
    create_entity_performance_dashboard()

