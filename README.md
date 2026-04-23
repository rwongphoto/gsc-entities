# GSC Entities

Streamlit dashboard that extracts and clusters entities out of your Google Search Console query data. Turns raw GSC exports into topic/entity-level performance views.

## What it does

The `GSCEntityAnalyzer` class:

1. Loads a GSC query CSV (Query, Clicks, Impressions, CTR, Position).
2. Sends each query through **Google Cloud Natural Language API** for entity recognition (type + salience).
3. Embeds queries with `all-MiniLM-L6-v2` Sentence Transformers for semantic similarity.
4. Aggregates GSC performance metrics **by entity** and by entity cluster.
5. Renders interactive Plotly dashboards — entity performance tables, similarity heatmaps, entity-type breakdowns.

Helpful for seeing which entities drive the majority of impressions, which rank poorly relative to their volume, and where query clusters overlap semantically.

## Stack

- Streamlit UI
- Google Cloud Natural Language API (entity extraction + salience)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- scikit-learn (cosine similarity)
- Plotly for visualizations

## Setup

```bash
pip install -r requirements.txt
streamlit run gsc_entity_dashboard.py
```

Provide GCP credentials either by uploading the service-account JSON in-app, or by pointing `GOOGLE_APPLICATION_CREDENTIALS` at a key file. The service account needs the **Cloud Natural Language API User** role.

## Input format

GSC Performance report CSV export with `Query`, `Clicks`, `Impressions`, `CTR`, `Position` columns.
