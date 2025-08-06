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
        self._initialize_gcp_client()
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
                creds = service_account.Credentials.from_service_account_info(self.gcp_credentials_info)
                self.nlp_client = language_v1.LanguageServiceClient(credentials=creds)
                print("✅ Google Cloud NLP client initialized from credentials info")
            elif self.gcp_credentials_path:
                creds = service_account.Credentials.from_service_account_file(self.gcp_credentials_path)
                self.nlp_client = language_v1.LanguageServiceClient(credentials=creds)
                print("✅ Google Cloud NLP client initialized from file")
            else:
                self.nlp_client = language_v1.LanguageServiceClient()
                print("✅ Google Cloud NLP client initialized with default credentials")
        except Exception as e:
            print(f"❌ Failed to initialize Google Cloud NLP client: {e}")
            self.nlp_client = None

    def is_number_entity(self, entity_name):
        """Filter out primarily numeric entities."""
        if not entity_name:
            return True
        cleaned = re.sub(r'[\s,\-\.]', '', entity_name)
        if cleaned.isdigit():
            return True
        if entity_name.strip().endswith('%') and re.sub(r'[%,\s\-\.]', '', entity_name).isdigit():
            return True
        if re.match(r'^\d{4}$', cleaned):
            return True
        digits = sum(c.isdigit() for c in entity_name)
        total = len(re.sub(r'\s', '', entity_name))
        if total > 0 and (digits / total) > 0.7:
            return True
        if len(entity_name.strip()) <= 4 and any(c.isdigit() for c in entity_name):
            return True
        return False

    def extract_entities_with_google_nlp(self, text):
        """Extract entities from text using Google Cloud NLP."""
        if not self.nlp_client or not text:
            return {}
        try:
            max_bytes = 800000
            tb = text.encode('utf-8')
            if len(tb) > max_bytes:
                text = tb[:max_bytes].decode('utf-8', 'ignore')
            text = re.sub(r'\|MINIBATCH_SEPARATOR\|', '. ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            doc = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
            resp = self.nlp_client.analyze_entities(document=doc, encoding_type=language_v1.EncodingType.UTF8)
            entities = {}
            for ent in resp.entities:
                name = ent.name.strip()
                if self.is_number_entity(name) or len(name) < 2:
                    continue
                key = name.lower()
                if key not in entities or ent.salience > entities[key]['salience']:
                    entities[key] = {
                        'name': name,
                        'type': language_v1.Entity.Type(ent.type_).name,
                        'salience': ent.salience,
                        'mentions': len(ent.mentions)
                    }
            return entities
        except Exception as e:
            print(f"❌ NLP API Error: {e}")
            return {}

    def load_gsc_data(self, file_path, year_label):
        """Load and clean GSC query CSV data."""
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            if 'CTR' in df.columns:
                df['CTR'] = df['CTR'].str.replace('%', '').astype(float) / 100
            df['Year'] = year_label
            df['Top queries'] = df['Top queries'].str.lower().str.strip()
            print(f"✅ Loaded {len(df)} rows for {year_label}")
            return df
        except Exception as e:
            print(f"❌ Load error: {e}")
            return None

    def extract_entities_from_queries(self, df, batch_size=5):
        """Mini-batch entity extraction and precise query-entity mapping."""
        unique_queries = df['Top queries'].unique()
        cache = {}
        for i in range(0, len(unique_queries), batch_size):
            batch = unique_queries[i:i+batch_size]
            text = ' |MINIBATCH_SEPARATOR| '.join(batch)
            batch_entities = self.extract_entities_with_google_nlp(text)
            for q in batch:
                matches = []
                q_lower = q.lower()
                for key, info in batch_entities.items():
                    entity_lower = info['name'].lower()
                    if entity_lower in q_lower or q_lower in entity_lower or any(w in q_lower.split() for w in entity_lower.split() if len(w) > 2):
                        matches.append({
                            'entity_name': info['name'],
                            'entity_type': info['type'],
                            'salience': info['salience'],
                            'mentions': info['mentions']
                        })
                if not matches:
                    matches = [self._categorize_query_fallback(q)]
                cache[q] = matches

        rows = []
        embeddings_cache = {}
        if self.embedding_model:
            all_entities = list({e['entity_name'] for lst in cache.values() for e in lst})
            try:
                vecs = self.embedding_model.encode(all_entities)
                embeddings_cache = dict(zip(all_entities, vecs))
            except Exception:
                pass

        for _, row in df.iterrows():
            q = row['Top queries']
            for ent in cache.get(q, []):
                relevance = 0.0
                if ent['entity_name'] in embeddings_cache:
                    try:
                        q_vec = self.embedding_model.encode([q])
                        e_vec = embeddings_cache[ent['entity_name']].reshape(1, -1)
                        relevance = cosine_similarity(q_vec, e_vec)[0][0]
                    except Exception:
                        pass
                rows.append({
                    'Query': q,
                    'Entity': ent['entity_name'],
                    'Entity_Type': ent['entity_type'],
                    'Entity_Salience': ent['salience'],
                    'Entity_Mentions': ent['mentions'],
                    'Query_Relevance': relevance,
                    'Clicks': row['Clicks'],
                    'Impressions': row['Impressions'],
                    'CTR': row['CTR'],
                    'Position': row['Position'],
                    'Year': row['Year']
                })
        return pd.DataFrame(rows)

    def _categorize_query_fallback(self, query):
        """Fallback entity categorization for uncaptured queries."""
        q = query.lower()
        if any(t in q for t in ['photographer', 'photography', 'photo']):
            return {'entity_name': 'Photography', 'entity_type': 'OTHER', 'salience': 0.3, 'mentions': 1}
        if any(t in q for t in ['gallery', 'museum', 'exhibition']):
            return {'entity_name': 'Art Gallery', 'entity_type': 'LOCATION', 'salience': 0.3, 'mentions': 1}
        if any(t in q for t in ['nature', 'landscape', 'wildlife']):
            return {'entity_name': 'Nature Photography', 'entity_type': 'OTHER', 'salience': 0.3, 'mentions': 1}
        if 'how' in q or 'tutorial' in q:
            return {'entity_name': 'Tutorial Content', 'entity_type': 'OTHER', 'salience': 0.2, 'mentions': 1}
        if any(t in q for t in ['best', 'top']):
            return {'entity_name': 'Comparison Query', 'entity_type': 'OTHER', 'salience': 0.2, 'mentions': 1}
        return {'entity_name': 'General Photography', 'entity_type': 'OTHER', 'salience': 0.1, 'mentions': 1}

    def aggregate_entity_performance(self, entity_df):
        """Aggregate clicks, impressions, and other metrics by entity/year."""
        df = entity_df.copy()
        df['Entity_Unique'] = df['Entity'] + '_' + df['Entity_Type']
        agg = df.groupby(['Entity_Unique', 'Entity', 'Entity_Type', 'Year']).agg({
            'Clicks': 'sum',
            'Impressions': 'sum',
            'CTR': 'mean',
            'Position': 'mean',
            'Entity_Salience': 'mean',
            'Query_Relevance': 'mean',
            'Query': 'count'
        }).rename(columns={'Query': 'Query_Count'}).reset_index()
        dup = agg.groupby(['Entity', 'Year']).size()
        if (dup > 1).any():
            agg = agg.loc[agg.groupby(['Entity', 'Year'])['Clicks'].idxmax()].reset_index(drop=True)
        return agg

    def calculate_yoy_changes(self, agg_df):
        """Calculate YOY changes and performance score for each entity."""
        dup = agg_df.groupby(['Entity', 'Year']).size()
        if (dup > 1).any():
            agg_df = agg_df.loc[agg_df.groupby(['Entity', 'Year'])['Clicks'].idxmax()].reset_index(drop=True)

        metrics = ['Clicks', 'Impressions', 'CTR', 'Position', 'Query_Count']
        pivot = {m: agg_df.pivot(index='Entity', columns='Year', values=m).fillna(0) for m in metrics}
        years = list(pivot['Clicks'].columns)

        # Explicitly assign current/previous labels to avoid swapping
        if set(years) == {"Current Year", "Previous Year"}:
            current_year, previous_year = "Current Year", "Previous Year"
        else:
            sorted_years = sorted(years)
            previous_year, current_year = sorted_years[-2], sorted_years[-1]

        print(f"📊 Comparing {previous_year} vs {current_year}")

        results = []
        for entity in pivot['Clicks'].index:
            entity_type = agg_df[agg_df['Entity'] == entity]['Entity_Type'].iloc[0]
            rec = {}
            for m in metrics:
                curr = pivot[m].loc[entity, current_year]
                prev = pivot[m].loc[entity, previous_year]
                if m == 'Position':
                    rec[f'{m}_Change'] = prev - curr
                else:
                    if prev > 0:
                        rec[f'{m}_Change_%'] = ((curr - prev) / prev) * 100
                    elif curr > 0:
                        rec[f'{m}_Change_%'] = 100.0
                    else:
                        rec[f'{m}_Change_%'] = 0.0
                rec[f'Current_{m}'] = curr
                rec[f'Previous_{m}'] = prev

            score = (
                rec['Clicks_Change_%'] * 0.4 +
                rec['Impressions_Change_%'] * 0.3 +
                rec['CTR_Change_%'] * 0.2 +
                rec.get('Position_Change', 0) * 0.1 * 10
            )

            results.append({
                'Entity': entity,
                'Entity_Type': entity_type,
                'Performance_Score': score,
                **rec
            })

        return pd.DataFrame(results)


def create_entity_performance_dashboard():
    st.set_page_config(page_title="GSC Entity Dashboard", layout="wide")
    st.title("🎯 GSC Entity Performance Dashboard")
    st.markdown("**Advanced Entity Analysis using Google Cloud NLP | by Richard Wong, The SEO Consultant.ai**")
    st.markdown("**🔄 Code Version: 8.0 - FINAL FIX: Explicit Column Access**")
    st.markdown("""
    **Performance Optimizations:**
    - ✅ Query deduplication (processes unique queries only once)
    - ✅ Batch processing with larger batch sizes
    - ✅ Pre-computed entity embeddings for faster similarity calculations
    - ✅ Smart caching to avoid reprocessing identical datasets
    - ✅ Real-time progress tracking
    """)

    # Session state init
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'entity_cache' not in st.session_state:
        st.session_state.entity_cache = {}

    # Sidebar config
    st.sidebar.header("🔑 Configuration")
    auth_method = st.sidebar.radio("Authentication Method:", ["Upload JSON Key", "Use Default Credentials"]) 
    creds_info = None
    if auth_method == "Upload JSON Key":
        key_file = st.sidebar.file_uploader("Upload GCP JSON Key", type="json")
        if key_file:
            creds_info = json.load(key_file)
            st.sidebar.success("✅ GCP credentials loaded!")

    st.sidebar.subheader("📁 GSC Data Files")
    current_file = st.sidebar.file_uploader("Current Year GSC Data (CSV)", type="csv")
    previous_file = st.sidebar.file_uploader("Previous Year GSC Data (CSV)", type="csv")

    st.sidebar.subheader("⚙️ Analysis Settings")
    min_clicks = st.sidebar.slider("Minimum Clicks Filter", 0, 100, 1)
    min_impr = st.sidebar.slider("Minimum Impressions Filter", 0, 1000, 10)
    batch_size = st.sidebar.slider("Mini-Batch Size", 3, 10, 5)
    use_cache = st.sidebar.checkbox("Enable Caching", True)

    if st.sidebar.button("🚀 Start Analysis"):
        if not current_file or not previous_file:
            st.error("Please upload both CSV files to proceed.")
            return

        analyzer = GSCEntityAnalyzer(gcp_credentials_info=creds_info)
        df_cur = analyzer.load_gsc_data(current_file, "Current Year")
        df_prev = analyzer.load_gsc_data(previous_file, "Previous Year")
        df_cur = df_cur[(df_cur['Clicks'] >= min_clicks) & (df_cur['Impressions'] >= min_impr)]
        df_prev = df_prev[(df_prev['Clicks'] >= min_clicks) & (df_prev['Impressions'] >= min_impr)]
        combined = pd.concat([df_cur, df_prev], ignore_index=True)

        cache_key = hash(tuple(combined['Top queries']))
        if use_cache and cache_key in st.session_state.entity_cache:
            entity_df = st.session_state.entity_cache[cache_key]
        else:
            entity_df = analyzer.extract_entities_from_queries(combined, batch_size=batch_size)
            if use_cache:
                st.session_state.entity_cache[cache_key] = entity_df

        agg_df = analyzer.aggregate_entity_performance(entity_df)
        yoy_df = analyzer.calculate_yoy_changes(agg_df)

        st.session_state.entity_df = entity_df
        st.session_state.agg_df = agg_df
        st.session_state.yoy_df = yoy_df
        st.session_state.analysis_complete = True

    if st.session_state.analysis_complete:
        yoy = st.session_state.yoy_df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Entities", len(yoy))
        improving = len(yoy[yoy['Performance_Score'] > 0])
        c2.metric("Improving Entities", improving, delta=f"{improving/len(yoy)*100:.1f}%")
        c3.metric("Avg Clicks Change", f"{yoy['Clicks_Change_%'].mean():.1f}%")
        c4.metric("Avg Impr. Change", f"{yoy['Impressions_Change_%'].mean():.1f}%")

        st.subheader("🔍 Filter Results")
        types = ['All'] + sorted(yoy['Entity_Type'].unique())
        selected = st.selectbox("Filter by Entity Type:", types)
        min_ps = st.slider("Min Performance Score:", float(yoy['Performance_Score'].min()), float(yoy['Performance_Score'].max()), float(yoy['Performance_Score'].min()))
        min_cc = st.slider("Min Current Clicks:", 0, int(yoy['Current_Clicks'].max()), 0)

        filtered = yoy[(yoy['Performance_Score'] >= min_ps) & (yoy['Current_Clicks'] >= min_cc)]
        if selected != 'All':
            filtered = filtered[filtered['Entity_Type'] == selected]

        st.subheader("🏆 Top Performing Entities YOY")
        top = filtered.nlargest(10, 'Performance_Score')[['Entity', 'Entity_Type', 'Performance_Score', 'Clicks_Change_%', 'Impressions_Change_%', 'CTR_Change_%', 'Current_Clicks']]
        st.dataframe(top, use_container_width=True)

        st.subheader("⚠️ Declining Entities - Optimization Opportunities")
        decline = filtered.nsmallest(10, 'Performance_Score')[['Entity', 'Entity_Type', 'Performance_Score', 'Clicks_Change_%', 'Previous_Clicks', 'Current_Clicks']]
        st.dataframe(decline, use_container_width=True)

        # Visualizations
        st.subheader("📊 Entity Performance Visualizations")
        fig = px.histogram(filtered, x='Performance_Score', nbins=30, title="Performance Score Distribution")
        fig.add_vline(x=0, line_dash="dash", annotation_text="Break-even")
        st.plotly_chart(fig, use_container_width=True)

        if selected == 'All' or len(filtered['Entity_Type'].unique()) > 1:
            type_perf = filtered.groupby('Entity_Type').agg({'Performance_Score':'mean','Clicks_Change_%':'mean','Entity':'count'}).rename(columns={'Entity':'Count'}).reset_index()
            fig2 = px.scatter(type_perf, x='Clicks_Change_%', y='Performance_Score', size='Count', hover_name='Entity_Type', title="Performance by Entity Type")
            st.plotly_chart(fig2, use_container_width=True)

        # Detailed Entity Analysis
        st.subheader("🔍 Detailed Entity Analysis")
        sel_ent = st.selectbox("Select an entity:", filtered['Entity'].tolist())
        if sel_ent:
            details = yoy[yoy['Entity'] == sel_ent].iloc[0]
            queries = st.session_state.entity_df[st.session_state.entity_df['Entity'] == sel_ent]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Entity:** {sel_ent}")
                st.markdown(f"**Type:** {details['Entity_Type']}")
                st.markdown(f"**Perf Score:** {details['Performance_Score']:.2f}")
                st.markdown("**YOY Changes:**")
                st.markdown(f"- Clicks: {details['Clicks_Change_%']:.1f}%")
                st.markdown(f"- Impr: {details['Impressions_Change_%']:.1f}%")
                st.markdown(f"- CTR: {details['CTR_Change_%']:.1f}%")
                st.markdown(f"- Pos: {details['Position_Change']:.2f}")
            with col2:
                st.markdown("**Related Queries:**")
                qr = queries.groupby(['Query','Year']).agg({'Clicks':'sum','Impressions':'sum'}).reset_index()
                topq = qr.nlargest(10,'Clicks')
                for _, r in topq.iterrows():
                    icon = "📅" if r['Year']=='Current Year' else "📆"
                    st.markdown(f"- {icon} {r['Query']} ({r['Year']}): {r['Clicks']} clicks, {r['Impressions']} impressions")

        # Export buttons
        st.subheader("💾 Export Results")
        csv = filtered.to_csv(index=False)
        st.download_button("Download YOY CSV", csv, file_name=f"yoy_{datetime.now().strftime('%Y%m%d')}.csv")
        entity_csv = st.session_state.entity_df.to_csv(index=False)
        st.download_button("Download Entity Mappings CSV", entity_csv, file_name=f"entities_{datetime.now().strftime('%Y%m%d')}.csv")

    else:
        st.info("👆 Upload CSVs and click 'Start Analysis' to begin.")

if __name__ == "__main__":
    create_entity_performance_dashboard()

