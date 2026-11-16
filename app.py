"""Optional Streamlit interface for the no-neural OLS/Ridge research pipeline.

Requires requirements-dashboard.txt. Never treats missing earnings revisions as
an observed sixth input; model metadata determines whether 5F or 6F is allowed.
"""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from finance.model import ModelSettings, run_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
st.set_page_config(page_title='ML Finance | Stock Ranking',page_icon='📈',layout='wide')
st.title('ML Finance — Six-Signal Equity Ranking')
st.caption('Research dashboard · OLS/Ridge · Walk-forward · No neural networks')

meta_path=ROOT/'data'/'features_metadata.json'
if meta_path.exists():
    metadata=json.loads(meta_path.read_text(encoding='utf-8'))
    verified=metadata.get('earnings_revisions_verified') is True
    st.info(f"Dataset as-of: {metadata.get('data_asof','unknown')} · "
            f"{'Six features: EPS revision history verified' if verified else 'Five-feature baseline only: historical EPS revisions missing'}")
else:
    st.warning('No private dataset found. Add data/features.csv and data/features_metadata.json locally to run the model.')

with st.sidebar:
    st.header('Experiment settings')
    model=st.selectbox('Linear estimator',['ols','ridge'])
    top_n=st.slider('Top-N stocks',1,15,5)
    cost_bps=st.number_input('Trading cost, bps per side',min_value=0.0,max_value=100.0,value=5.0,step=1.0)
    min_train=st.slider('Minimum training months',12,60,24)
    ridge_alpha=st.number_input('Ridge alpha',min_value=0.01,max_value=1000.0,value=10.0)
    pressed=st.button('Run walk-forward evaluation',type='primary',use_container_width=True)

if pressed:
    if not meta_path.exists() or not (ROOT/'data'/'features.csv').exists():
        st.error('No validated private input panel. Run the collection scripts first.')
    else:
        with st.spinner('Running linear model on local data...'):
            try:
                run_model(ROOT,ModelSettings(model=model,top_n=top_n,cost_bps=cost_bps,
                                             min_train_months=min_train,ridge_alpha=ridge_alpha))
                st.success('Run finished. Inspect its status below; success does not imply positive trading performance.')
            except Exception as exc:
                st.error(f'Run failed: {exc}')

status_path=OUT/'run_status.json'
if status_path.exists():
    status=json.loads(status_path.read_text(encoding='utf-8'))
    st.subheader('Validation status')
    st.json({key:status.get(key) for key in ['status','reason','model','feature_count','data_asof','n_nonoverlapping_periods']})
    if status.get('feature_count')==5:
        st.warning('Five-factor fallback. Not a completed six-feature study.')

for title,filename in [('Out-of-sample evaluation','prediction_metrics.csv'),
                       ('Historical portfolio backtest','portfolio_metrics.csv'),
                       ('Latest stock rankings','latest_ranking.csv'),
                       ('Factor coefficients','model_coefficients.csv')]:
    path=OUT/filename
    if path.exists():
        st.subheader(title)
        frame=pd.read_csv(path)
        if filename=='prediction_metrics.csv' and 'scope' in frame.columns:
            frame=frame.head(1)
        if filename=='model_coefficients.csv' and 'decision_date' in frame.columns:
            frame=frame[frame.decision_date.eq(frame.decision_date.max())]
        st.dataframe(frame,use_container_width=True,hide_index=True)
periods=OUT/'backtest_periods.csv'
if periods.exists():
    df=pd.read_csv(periods,parse_dates=['exit_date'])
    available=[c for c in ['strategy_wealth','equal_weight_wealth','sector_equal_wealth'] if c in df]
    if available:
        st.subheader('Illustrative compounded historical wealth')
        st.line_chart(df.set_index('exit_date')[available])
st.caption('Research only. Point-in-time limits, selected-survivor bias and vendor data may distort backtests. No trading recommendations.')
