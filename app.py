import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="NIFTY 500 Stock Screener",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 NIFTY 500 Technical Stock Screener")
st.markdown("""
This screener identifies stocks from NIFTY 500 based on technical indicators:
- **ADX > 25** (Strong trend)
- **20 DMA > 50 DMA** (Bullish trend)
- **Close < 20 DMA** (Potential pullback entry)
- **OBV > OBV MA20** (Volume confirmation)
- **+DI > -DI** (Bullish momentum)
- **Risk Percentage < 10%** (Based on 50 DMA - ATR stop loss)
""")

# Sidebar for parameters
st.sidebar.header("📊 Screening Parameters")
adx_threshold = st.sidebar.slider("ADX Threshold", 15, 40, 25)
max_risk_percentage = st.sidebar.slider("Maximum Risk Percentage", 3.0, 15.0, 10.0)
atr_multiplier = st.sidebar.slider("ATR Multiplier (Stop Loss)", 1.0, 3.0, 1.5)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_nifty500_symbols():
    """Load NIFTY 500 symbols from a comprehensive list"""
    # Complete NIFTY 500 symbol list
    symbols = [
        'NETWEB.NS', 'FIRSTCRY.NS', 'GMDCLTD.NS', 'IDEA.NS', 'SKFINDIA.NS', 'GODFRYPHLP.NS', 'RTNINDIA.NS', 'JPPOWER.NS', 'BSE.NS', 'SCHNEIDER.NS', 'NIVABUPA.NS', 'RBLBANK.NS', 'INDUSTOWER.NS', 'SWIGGY.NS', 'ABFRL.NS', 'BEML.NS', 'INDGN.NS', 'APTUS.NS', 'GLENMARK.NS', 'CUB.NS', 'COHANCE.NS', 'KFINTECH.NS', 'CLEAN.NS', 'NATIONALUM.NS', 'ASHOKLEY.NS', 'GODREJAGRO.NS', 'MSUMI.NS', 'ASTRAL.NS', 'GRAVITA.NS', 'CENTURYPLY.NS', 'LEMONTREE.NS', 'RENUKA.NS', 'M&MFIN.NS', 'EICHERMOT.NS', 'HYUNDAI.NS', 'M&M.NS', 'VEDL.NS', 'MMTC.NS', 'VOLTAS.NS', 'CHOLAFIN.NS', 'PNCINFRA.NS', 'GODREJIND.NS', 'CAMPUS.NS', 'CAMS.NS', 'EXIDEIND.NS', 'KARURVYSYA.NS', 'VIJAYA.NS', 'NSLNISP.NS', 'BIOCON.NS', 'ARE&M.NS', 'IGIL.NS', 'ANGELONE.NS', 'SWANCORP.NS', 'IRFC.NS', 'SHRIRAMFIN.NS', 'MARUTI.NS', 'ESCORTS.NS', 'IRCON.NS', 'IOB.NS', 'SARDAEN.NS', 'NMDC.NS', 'TBOTEK.NS', 'PPLPHARMA.NS', 'VGUARD.NS', 'JSL.NS', 'RITES.NS', 'JYOTICNC.NS', 'NYKAA.NS', 'MRPL.NS', 'TTML.NS', 'GVT&D.NS', 'HBLENGINE.NS', 'RHIM.NS', 'SAIL.NS', 'SAILIFE.NS', 'MAHABANK.NS', 'PHOENIXLTD.NS', 'DRREDDY.NS', 'UNOMINDA.NS', 'CHENNPETRO.NS', 'POWERGRID.NS', 'IREDA.NS', 'TVSMOTOR.NS', 'AMBER.NS', 'BDL.NS', 'TANLA.NS', 'AFFLE.NS', 'JMFINANCIL.NS', 'WELSPUNLIV.NS', 'PNBHOUSING.NS', 'ABCAPITAL.NS', 'PVRINOX.NS', 'JSWHL.NS', 'KAYNES.NS', 'JBMA.NS', 'PRAJIND.NS', 'RELIANCE.NS', 'ZEEL.NS', 'INOXWIND.NS', 'AEGISLOG.NS', 'LTF.NS', 'PFC.NS', 'AUROPHARMA.NS', 'NUVAMA.NS', 'AAVAS.NS', 'CANBK.NS', 'ENGINERSIN.NS', 'PAYTM.NS', 'J&KBANK.NS', 'ROUTE.NS', 'FIVESTAR.NS', 'DATAPATTNS.NS', 'CDSL.NS', 'ETERNAL.NS', 'NBCC.NS', 'NAUKRI.NS', 'LAURUSLABS.NS', 'AADHARHFC.NS', 'KPITTECH.NS', 'CHALET.NS', 'RVNL.NS', 'LTFOODS.NS', 'SUZLON.NS', 'TATAMOTORS.NS', 'CHOLAHLDNG.NS', 'IIFL.NS', 'BHEL.NS', 'JKTYRE.NS', 'GLAND.NS', 'SBILIFE.NS', 'TATAPOWER.NS', 'TATAELXSI.NS', 'HINDALCO.NS', 'HINDCOPPER.NS', 'HOMEFIRST.NS', 'SAMMAANCAP.NS', 'HINDZINC.NS', 'AJANTPHARM.NS', 'UCOBANK.NS', 'HDFCLIFE.NS', 'IKS.NS', 'BHARATFORG.NS', 'IFCI.NS', 'SONACOMS.NS', 'YESBANK.NS', 'OIL.NS', 'ANANTRAJ.NS', 'RECLTD.NS', 'BHARTIARTL.NS', 'BANKINDIA.NS', 'ALKEM.NS', 'SIGNATURE.NS', 'RPOWER.NS', 'SJVN.NS', 'GRSE.NS', 'MFSL.NS', 'KPIL.NS', 'SUNPHARMA.NS', 'BAJAJFINSV.NS', 'NTPCGREEN.NS', 'MANAPPURAM.NS', 'CAPLIPOINT.NS', 'SHYAMMETL.NS', '360ONE.NS', 'TATASTEEL.NS', 'WAAREEENER.NS', 'TITAGARH.NS', 'IDFCFIRSTB.NS', 'BANDHANBNK.NS', 'RAYMOND.NS', 'AXISBANK.NS', 'BERGEPAINT.NS', 'ASIANPAINT.NS', 'BSOFT.NS', 'HUDCO.NS', 'MAXHEALTH.NS', 'JUBLPHARMA.NS', 'FINCABLES.NS', 'JUBLFOOD.NS', 'USHAMART.NS', 'UNIONBANK.NS', 'MINDACORP.NS', 'INDUSINDBK.NS', 'HDFCAMC.NS', 'CONCOR.NS', 'COALINDIA.NS', 'PEL.NS', 'VMM.NS', 'ENDURANCE.NS', 'IRCTC.NS', 'WOCKPHARMA.NS', 'TRIVENI.NS', 'DEVYANI.NS', 'BOSCHLTD.NS', 'RAILTEL.NS', 'BANKBARODA.NS', 'TARIL.NS', 'BLUEDART.NS', 'HEROMOTOCO.NS', 'TATACONSUM.NS', 'MOTILALOFS.NS', 'FEDERALBNK.NS', 'POLYMED.NS', 'JSWENERGY.NS', 'NETWORK18.NS', 'SUNDARMFIN.NS', 'ABSLAMC.NS', 'ACMESOLAR.NS', 'ZYDUSLIFE.NS', 'ZENTEC.NS', 'IOC.NS', 'CEATLTD.NS', 'KANSAINER.NS', 'ADANIPOWER.NS', 'BAJAJHFL.NS', 'KIMS.NS', 'BAJFINANCE.NS', 'LUPIN.NS', 'EMCURE.NS', 'ASTERDM.NS', 'JSWSTEEL.NS', 'GPPL.NS', 'WELCORP.NS', 'PNB.NS', 'STARHEALTH.NS', 'JIOFIN.NS', 'MAZDOCK.NS', 'CESC.NS', 'INDHOTEL.NS', 'ADANIENT.NS', 'LICHSGFIN.NS', 'HDFCBANK.NS', 'INDIANB.NS', 'TATAINVEST.NS', 'MUTHOOTFIN.NS', 'KALYANKJIL.NS', 'SBICARD.NS', 'ALOKINDS.NS', 'SAREGAMA.NS', 'BPCL.NS', 'LTIM.NS', 'ADANIGREEN.NS', 'ASTRAZEN.NS', 'MOTHERSON.NS', 'PGEL.NS', 'INDIAMART.NS', 'CRAFTSMAN.NS', 'DALBHARAT.NS', 'JINDALSTEL.NS', 'AUBANK.NS', 'IDBI.NS', 'BAJAJ-AUTO.NS', 'WHIRLPOOL.NS', 'DEEPAKNTR.NS', 'SUNDRMFAST.NS', 'CONCORDBIO.NS', 'SBFC.NS', 'HFCL.NS', 'MAPMYINDIA.NS', 'RKFORGE.NS', 'GICRE.NS', 'CGCL.NS', 'JWL.NS', 'POONAWALLA.NS', 'KPRMILL.NS', 'CRISIL.NS', 'SWSOLAR.NS', 'NCC.NS', 'BRITANNIA.NS', 'AFCONS.NS', 'INOXINDIA.NS', 'HSCL.NS', 'DIXON.NS', 'BIKAJI.NS', 'BLUESTARCO.NS', 'GRASIM.NS', 'EIHOTEL.NS', 'TRIDENT.NS', 'GRANULES.NS', 'LICI.NS', 'HONASA.NS', 'ADANIENSOL.NS', 'AMBUJACEM.NS', 'SHREECEM.NS', 'LATENTVIEW.NS', 'CARBORUNIV.NS', 'DELHIVERY.NS', 'TIMKEN.NS', 'DCMSHRIRAM.NS', 'DLF.NS', 'CROMPTON.NS', 'HINDPETRO.NS', 'TATATECH.NS', 'NAVA.NS', 'MGL.NS', 'POLYCAB.NS', '3MINDIA.NS', 'BEL.NS', 'NHPC.NS', 'PETRONET.NS', 'ELECON.NS', 'ICICIBANK.NS', 'TATACOMM.NS', 'CENTRALBK.NS', 'TRENT.NS', 'BATAINDIA.NS', 'SONATSOFTW.NS', 'WIPRO.NS', 'OFSS.NS', 'JUSTDIAL.NS', 'GESHIP.NS', 'POWERINDIA.NS', 'APLAPOLLO.NS', 'SBIN.NS', 'CGPOWER.NS', 'TIINDIA.NS', 'TECHNOE.NS', 'MCX.NS', 'NATCOPHARM.NS', 'IPCALAB.NS', 'BAYERCROP.NS', 'MANYAVAR.NS', 'CUMMINSIND.NS', 'FORTIS.NS', 'BAJAJHLDNG.NS', 'SCHAEFFLER.NS', 'APOLLOTYRE.NS', 'GAIL.NS', 'BLS.NS', 'BASF.NS', 'INDIGO.NS', 'NTPC.NS', 'SYRMA.NS', 'HAVELLS.NS', 'NEULANDLAB.NS', 'TORNTPHARM.NS', 'SUNTV.NS', 'IEX.NS', 'ANANDRATHI.NS', 'ALIVUS.NS', 'TRITURBINE.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'GMRAIRPORT.NS', 'CERA.NS', 'ONGC.NS', 'SUPREMEIND.NS', 'LLOYDSME.NS', 'PCBL.NS', 'HAL.NS', 'VTL.NS', 'KOTAKBANK.NS', 'TATACHEM.NS', 'ADANIPORTS.NS', 'HAPPSTMNDS.NS', 'ASAHIINDIA.NS', 'ICICIPRULI.NS', 'GPIL.NS', 'WESTLIFE.NS', 'AIIL.NS', 'ZENSARTECH.NS', 'CASTROLIND.NS', 'UBL.NS', 'ACC.NS', 'PFIZER.NS', 'DMART.NS', 'JBCHEPHARM.NS', 'IRB.NS', 'ABB.NS', 'KIRLOSBROS.NS', 'AIAENG.NS', 'UNITDSPR.NS', 'OLECTRA.NS', 'RAYMONDLSL.NS', 'HEG.NS', 'LINDEINDIA.NS', 'ATGL.NS', 'JKCEMENT.NS', 'PIDILITIND.NS', 'PTCIL.NS', 'CHAMBLFERT.NS', 'SOLARINDS.NS', 'THERMAX.NS', 'AARTIIND.NS', 'CANFINHOME.NS', 'GRAPHITE.NS', 'TITAN.NS', 'JSWINFRA.NS', 'ABREL.NS', 'CYIENT.NS', 'ACE.NS', 'BALRAMCHIN.NS', 'PAGEIND.NS', 'RAINBOW.NS', 'APOLLOHOSP.NS', 'FLUOROCHEM.NS', 'MARICO.NS', 'MANKIND.NS', 'GILLETTE.NS', 'KNRCON.NS', 'KEC.NS', 'UPL.NS', 'FACT.NS', 'NLCINDIA.NS', 'ITI.NS', 'DABUR.NS', 'AKUMS.NS', 'LT.NS', 'MRF.NS', 'GODIGIT.NS', 'AWL.NS', 'CREDITACC.NS', 'EIDPARRY.NS', 'PREMIERENE.NS', 'INFY.NS', 'SCI.NS', 'BBTC.NS', 'NEWGEN.NS', 'CCL.NS', 'MEDANTA.NS', 'COCHINSHIP.NS', 'LODHA.NS', 'SRF.NS', 'LTTS.NS', 'KIRLOSENG.NS', 'BALKRISIND.NS', 'GODREJCP.NS', 'PATANJALI.NS', 'OBEROIRLTY.NS', 'JYOTHYLAB.NS', 'METROPOLIS.NS', 'COROMANDEL.NS', 'GSPL.NS', 'APLLTD.NS', 'INDIACEM.NS', 'SAPPHIRE.NS', 'SAGILITY.NS', 'NAM-INDIA.NS', 'APARINDS.NS', 'HINDUNILVR.NS', 'RADICO.NS', 'ELGIEQUIP.NS', 'SYNGENE.NS', 'BHARTIHEXA.NS', 'TECHM.NS', 'RRKABEL.NS', 'TCS.NS', 'IGL.NS', 'KAJARIACER.NS', 'DOMS.NS', 'ZFCVINDIA.NS', 'ICICIGI.NS', 'HCLTECH.NS', 'COLPAL.NS', 'MASTEK.NS', 'TEJASNET.NS', 'ERIS.NS', 'CIPLA.NS', 'HONAUT.NS', 'SIEMENS.NS', 'BRIGADE.NS', 'ALKYLAMINE.NS', 'GNFC.NS', 'DBREALTY.NS', 'DIVISLAB.NS', 'COFORGE.NS', 'REDINGTON.NS', 'GODREJPROP.NS', 'ITC.NS', 'POLICYBZR.NS', 'MAHSEAMLES.NS', 'GUJGASLTD.NS', 'EMAMILTD.NS', 'INTELLECT.NS', 'FINPIPE.NS', 'ATUL.NS', 'JUBLINGREA.NS', 'RCF.NS', 'NAVINFLUOR.NS', 'UTIAMC.NS', 'LALPATHLAB.NS', 'MPHASIS.NS', 'KEI.NS', 'JINDALSAW.NS', 'RAMCOCEM.NS', 'PIIND.NS', 'TORNTPOWER.NS', 'SOBHA.NS', 'SUMICHEM.NS', 'PRESTIGE.NS', 'NIACL.NS', 'DEEPAKFERT.NS', 'NH.NS', 'GLAXO.NS', 'ABBOTINDIA.NS', 'PERSISTENT.NS', 'FSL.NS', 'ECLERX.NS', 'VBL.NS', 'OLAELEC.NS'
    ]
    return symbols

def calculate_obv(df):
    """Calculate On-Balance Volume"""
    if df.empty or len(df) < 2:
        return pd.Series([], dtype=float)
    
    obv = [0.0] * len(df)
    
    # Ensure Close and Volume are pandas Series with proper values
    close_values = df['Close'].values if hasattr(df['Close'], 'values') else df['Close']
    volume_values = df['Volume'].values if hasattr(df['Volume'], 'values') else df['Volume']
    
    for i in range(1, len(df)):
        if close_values[i] > close_values[i-1]:
            obv[i] = obv[i-1] + volume_values[i]
        elif close_values[i] < close_values[i-1]:
            obv[i] = obv[i-1] - volume_values[i]
        else:
            obv[i] = obv[i-1]
    
    return pd.Series(obv, index=df.index, name='OBV')

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_and_screen_stocks(symbols, adx_thresh, max_risk_pct, atr_mult):
    """Download data and screen stocks based on criteria"""
    screened_data = []
    debug_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_symbols = len(symbols)
    processed_count = 0
    successful_downloads = 0
    criteria_failures = {}
    
    for idx, symbol in enumerate(symbols):
        try:
            status_text.text(f'Processing {symbol} ({idx+1}/{total_symbols})')
            processed_count += 1
            
            # Download data
            df = yf.download(symbol, period="150d", interval="1d", progress=False)
            
            if df.empty or len(df) < 50:
                debug_data.append({'Symbol': symbol, 'Issue': 'No data or insufficient data'})
                continue
            
            # Handle multi-index columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                debug_data.append({'Symbol': symbol, 'Issue': f'Missing required columns. Available: {list(df.columns)}'})
                continue
            
            # Convert to numeric and handle any data type issues
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values in basic data
            df = df.dropna(subset=required_cols)
            
            if len(df) < 50:
                debug_data.append({'Symbol': symbol, 'Issue': 'Insufficient clean data after processing'})
                continue
                
            successful_downloads += 1
            
            # Calculate indicators
            adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['ADX14'] = adx_indicator.adx()
            df['PLUS_DI'] = adx_indicator.adx_pos()
            df['MINUS_DI'] = adx_indicator.adx_neg()
            
            atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
            df['ATR'] = atr_indicator.average_true_range()
            
            df['DMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['DMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
            
            df['OBV'] = calculate_obv(df)
            df['OBV_MA20'] = SMAIndicator(close=df['OBV'], window=20).sma_indicator()
            
            # Drop NaN values
            df = df.dropna()
            
            if df.empty:
                debug_data.append({'Symbol': symbol, 'Issue': 'No data after indicator calculation'})
                continue
                
            latest = df.iloc[-1]
            
            # Debug: Track individual criteria failures
            criteria_check = {}
            criteria_check['ADX > threshold'] = latest['ADX14'] > adx_thresh
            criteria_check['20DMA > 50DMA'] = latest['DMA20'] > latest['DMA50']
            criteria_check['Close < 20DMA'] = latest['Close'] < latest['DMA20']
            criteria_check['OBV > OBV_MA20'] = latest['OBV'] > latest['OBV_MA20']
            criteria_check['+DI > -DI'] = latest['PLUS_DI'] > latest['MINUS_DI']
            
            # Track criteria failures for debugging
            for criteria, passed in criteria_check.items():
                if not passed:
                    if criteria not in criteria_failures:
                        criteria_failures[criteria] = 0
                    criteria_failures[criteria] += 1
            
            # Screening criteria
            if all(criteria_check.values()):
                # Risk calculation based on 50 DMA and ATR
                price = latest['Close']
                estimated_stop_loss = latest['DMA50'] - (latest['ATR'] * atr_mult)
                estimated_risk_per_share = price - estimated_stop_loss
                
                # Ensure estimated_risk_per_share is positive for meaningful Risk %
                if estimated_risk_per_share > 0:
                    risk_percentage = (estimated_risk_per_share / price) * 100
                    
                    # Check risk percentage condition
                    if risk_percentage < max_risk_pct:
                        percentage_diff = ((latest['Close'] - latest['DMA20']) / latest['DMA20']) * 100
                        
                        screened_data.append({
                            'Symbol': symbol.replace('.NS', ''),
                            'LTP': round(latest['Close'], 2),
                            'ADX14': round(latest['ADX14'], 2),
                            'DMA20': round(latest['DMA20'], 2),
                            'DMA50': round(latest['DMA50'], 2),
                            'ATR': round(latest['ATR'], 2),
                            'Stop Loss': round(estimated_stop_loss, 2),
                            'Risk %': round(risk_percentage, 2),
                            'Risk Amount': round(estimated_risk_per_share, 2),
                            'LTP vs 20DMA %': round(percentage_diff, 2),
                            '+DI': round(latest['PLUS_DI'], 2),
                            '-DI': round(latest['MINUS_DI'], 2)
                        })
                    else:
                        debug_data.append({'Symbol': symbol, 'Issue': f'Risk % too high: {risk_percentage:.2f}%'})
                else:
                    debug_data.append({'Symbol': symbol, 'Issue': 'Negative risk (stop loss above current price)'})
            else:
                failed_criteria = [k for k, v in criteria_check.items() if not v]
                debug_data.append({'Symbol': symbol, 'Issue': f'Failed criteria: {", ".join(failed_criteria)}'})
        
        except Exception as e:
            debug_data.append({'Symbol': symbol, 'Issue': f'Error: {str(e)}'})
            continue
        
        # Update progress
        progress_bar.progress((idx + 1) / total_symbols)
    
    progress_bar.empty()
    status_text.empty()
    
    # Display debug information
    st.subheader("🔍 Screening Debug Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Processed", processed_count)
    with col2:
        st.metric("Successful Downloads", successful_downloads)
    with col3:
        st.metric("Stocks Found", len(screened_data))
    
    if criteria_failures:
        st.subheader("Common Criteria Failures")
        failure_df = pd.DataFrame(list(criteria_failures.items()), columns=['Criteria', 'Failed Count'])
        st.dataframe(failure_df)
    
    if debug_data:
        st.subheader("Sample Issues (First 20)")
        debug_df = pd.DataFrame(debug_data[:20])
        st.dataframe(debug_df)
    
    return pd.DataFrame(screened_data)

def create_stock_chart(symbol):
    """Create interactive chart for a stock"""
    try:
        symbol_yf = symbol + ".NS"
        df = yf.download(symbol_yf, period="150d", interval="1d", progress=False)
        
        if df.empty:
            st.error(f"No data available for {symbol}")
            return
        
        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns for {symbol}. Available: {list(df.columns)}")
            return
        
        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop NaN rows
        df = df.dropna(subset=required_cols)
        
        if df.empty:
            st.error(f"No valid data after cleaning for {symbol}")
            return
        
        # Calculate indicators
        df['DMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['DMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['OBV'] = calculate_obv(df)
        
        # Calculate OBV SMA 9 safely
        if not df['OBV'].empty and len(df['OBV']) >= 9:
            df['OBV_SMA9'] = SMAIndicator(close=df['OBV'], window=9).sma_indicator()
        else:
            df['OBV_SMA9'] = df['OBV']  # Use OBV directly if not enough data for SMA
        
        # Drop NaN rows from indicators
        df = df.dropna()
        
        if df.empty:
            st.error(f"No data remaining after indicator calculation for {symbol}")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            subplot_titles=(f'{symbol} Price with Moving Averages', 'OBV SMA 9'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                                name='Close', line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['DMA20'], 
                                name='20 DMA', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['DMA50'], 
                                name='50 DMA', line=dict(color='red', width=1)), row=1, col=1)
        
        # OBV chart
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV_SMA9'], 
                                name='OBV SMA 9', line=dict(color='purple', width=2)), row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating chart for {symbol}: {str(e)}")

# Main app logic
def main():
    # Load symbols
    symbols = load_nifty500_symbols()
    
    # Screening section
    if st.button("🔍 Run Screening", type="primary"):
        with st.spinner("Screening stocks... This may take a few minutes."):
            results_df = download_and_screen_stocks(
                symbols, adx_threshold, max_risk_percentage, atr_multiplier
            )
        
        if not results_df.empty:
            st.session_state.screening_results = results_df
        else:
            st.warning("No stocks met the screening criteria.")
    
    # Display results
    if hasattr(st.session_state, 'screening_results') and not st.session_state.screening_results.empty:
        results_df = st.session_state.screening_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stocks Found", len(results_df))
        with col2:
            st.metric("Avg Risk %", f"{results_df['Risk %'].mean():.2f}%")
        with col3:
            st.metric("Lowest Risk %", f"{results_df['Risk %'].min():.2f}%")
        with col4:
            st.metric("Avg ADX", f"{results_df['ADX14'].mean():.1f}")
        
        # Results table
        st.subheader("📋 Screening Results")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by:", 
            ['Risk %', 'ADX14', 'LTP vs 20DMA %', 'Symbol']
        )
        ascending = st.checkbox("Ascending order")
        
        sorted_df = results_df.sort_values(by=sort_by, ascending=ascending)
        st.dataframe(sorted_df, use_container_width=True)
        
        # Download CSV
        csv = sorted_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Chart section
        st.subheader("📊 Stock Charts")
        selected_symbol = st.selectbox(
            "Select a stock to view chart:", 
            options=[''] + sorted_df['Symbol'].tolist()
        )
        
        if selected_symbol:
            create_stock_chart(selected_symbol)
    
    # Instructions
    with st.expander("ℹ️ How to use this screener"):
        st.markdown("""
        1. **Adjust Parameters**: Use the sidebar to modify screening criteria
        2. **Run Screening**: Click the "Run Screening" button to analyze stocks
        3. **Review Results**: Examine the filtered stocks in the results table
        4. **Download Data**: Export results as CSV for further analysis
        5. **View Charts**: Select individual stocks to see their technical charts
        
        **Screening Criteria Explained:**
        - **ADX > 25**: Indicates strong trending market
        - **20 DMA > 50 DMA**: Confirms uptrend
        - **Close < 20 DMA**: Shows pullback for potential entry
        - **OBV > OBV MA**: Volume supports the move
        - **+DI > -DI**: Bullish directional movement
        - **Risk % < 10%**: Stop loss at 50 DMA - (ATR × multiplier) keeps risk manageable
        """)

if __name__ == "__main__":
    main()
