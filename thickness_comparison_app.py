import dash
from dash import dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xml.etree.ElementTree as ET
from scipy import stats
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "BTM vs DMT Thickness Comparison with Statistical Analysis"

def load_btm_data(file_path):
    """Load BTM CSV data"""
    try:
        print(f"Loading BTM data from: {file_path}")
        # Read CSV file, handling potential encoding issues
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Rename columns for consistency
        column_mapping = {
            'WaferID': 'WaferID',
            'Point No ': 'Point_No',
            'Point No': 'Point_No',
            'Film Thickness ': 'Thickness',
            'Film Thickness': 'Thickness',
            'Fit Rate ': 'Fit_Rate',
            'Fit Rate': 'Fit_Rate',
            'X[mm] ': 'X_mm',
            'X[mm]': 'X_mm',
            'Y[mm] ': 'Y_mm',
            'Y[mm]': 'Y_mm'
        }
        
        # Handle variations in column names
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Add measurement tool identifier
        df['Tool'] = 'BTM'
        
        # Convert coordinates to float
        df['X_mm'] = pd.to_numeric(df['X_mm'], errors='coerce')
        df['Y_mm'] = pd.to_numeric(df['Y_mm'], errors='coerce')
        df['Thickness'] = pd.to_numeric(df['Thickness'], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna(subset=['X_mm', 'Y_mm', 'Thickness', 'WaferID'])
        
        print(f"BTM data loaded: {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error loading BTM data: {str(e)}")
        return pd.DataFrame()

def load_dmt_data(file_path):
    """Load DMT XML data"""
    try:
        print(f"Loading DMT data from: {file_path}")
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        records = []
        
        for record in root.findall('.//DataRecord'):
            label_elem = record.find('Label')
            if label_elem is not None and label_elem.text == 'Layer 1 Thickness':
                wafer_id = record.find('WaferID').text if record.find('WaferID') is not None else None
                datum = record.find('Datum').text if record.find('Datum') is not None else None
                x_loc = record.find('XWaferLoc').text if record.find('XWaferLoc') is not None else None
                y_loc = record.find('YWaferLoc').text if record.find('YWaferLoc') is not None else None
                
                if all([wafer_id, datum, x_loc is not None, y_loc is not None]):
                    records.append({
                        'WaferID': wafer_id,
                        'Thickness': float(datum),
                        'X_mm': float(x_loc),
                        'Y_mm': float(y_loc),
                        'Tool': 'DMT'
                    })
        
        df = pd.DataFrame(records)
        print(f"DMT data loaded: {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error loading DMT data: {str(e)}")
        return pd.DataFrame()

def find_matching_measurements(btm_df, dmt_df, tolerance_mm=2.0):
    """Find matching measurements between BTM and DMT data within tolerance"""
    print("Finding matching measurements...")
    matched_pairs = []
    
    for wafer_id in btm_df['WaferID'].unique():
        if wafer_id in dmt_df['WaferID'].unique():
            btm_wafer = btm_df[btm_df['WaferID'] == wafer_id]
            dmt_wafer = dmt_df[dmt_df['WaferID'] == wafer_id]
            
            # Calculate distances between all BTM and DMT points for this wafer
            btm_coords = btm_wafer[['X_mm', 'Y_mm']].values
            dmt_coords = dmt_wafer[['X_mm', 'Y_mm']].values
            
            distances = cdist(btm_coords, dmt_coords)
            
            # Find matches within tolerance
            btm_indices, dmt_indices = np.where(distances <= tolerance_mm)
            
            for btm_idx, dmt_idx in zip(btm_indices, dmt_indices):
                btm_row = btm_wafer.iloc[btm_idx]
                dmt_row = dmt_wafer.iloc[dmt_idx]
                
                matched_pairs.append({
                    'WaferID': wafer_id,
                    'BTM_X': btm_row['X_mm'],
                    'BTM_Y': btm_row['Y_mm'],
                    'DMT_X': dmt_row['X_mm'],
                    'DMT_Y': dmt_row['Y_mm'],
                    'BTM_Thickness': btm_row['Thickness'],
                    'DMT_Thickness': dmt_row['Thickness'],
                    'Distance_mm': distances[btm_idx, dmt_idx],
                    'Thickness_Difference': btm_row['Thickness'] - dmt_row['Thickness'],
                    'Abs_Thickness_Difference': abs(btm_row['Thickness'] - dmt_row['Thickness'])
                })
    
    return pd.DataFrame(matched_pairs)

def perform_statistical_analysis(matched_df):
    """Perform statistical analysis including K-Sample tests"""
    print("Performing statistical analysis...")
    if matched_df.empty:
        return {}
    
    # Basic statistics
    btm_thickness = matched_df['BTM_Thickness']
    dmt_thickness = matched_df['DMT_Thickness']
    thickness_diff = matched_df['Thickness_Difference']
    
    # K-Sample tests (comparing BTM vs DMT distributions)
    try:
        # Kruskal-Wallis test (non-parametric)
        kruskal_stat, kruskal_pvalue = stats.kruskal(btm_thickness, dmt_thickness)
        
        # Mann-Whitney U test (alternative non-parametric test)
        mannwhitney_stat, mannwhitney_pvalue = stats.mannwhitneyu(btm_thickness, dmt_thickness, alternative='two-sided')
        
        # Welch's t-test (parametric, assumes unequal variances)
        ttest_stat, ttest_pvalue = stats.ttest_ind(btm_thickness, dmt_thickness, equal_var=False)
        
        # Kolmogorov-Smirnov test (comparing distributions)
        ks_stat, ks_pvalue = stats.ks_2samp(btm_thickness, dmt_thickness)
        
        # One-sample t-test on differences (testing if mean difference is significantly different from 0)
        onesample_stat, onesample_pvalue = stats.ttest_1samp(thickness_diff, 0)
        
        # Summary statistics
        stats_summary = {
            'total_matched_points': len(matched_df),
            'unique_wafers': matched_df['WaferID'].nunique(),
            
            # BTM statistics
            'btm_mean': btm_thickness.mean(),
            'btm_std': btm_thickness.std(),
            'btm_median': btm_thickness.median(),
            'btm_min': btm_thickness.min(),
            'btm_max': btm_thickness.max(),
            
            # DMT statistics
            'dmt_mean': dmt_thickness.mean(),
            'dmt_std': dmt_thickness.std(),
            'dmt_median': dmt_thickness.median(),
            'dmt_min': dmt_thickness.min(),
            'dmt_max': dmt_thickness.max(),
            
            # Difference statistics
            'mean_difference': thickness_diff.mean(),
            'std_difference': thickness_diff.std(),
            'median_abs_difference': matched_df['Abs_Thickness_Difference'].median(),
            'mean_abs_difference': matched_df['Abs_Thickness_Difference'].mean(),
            
            # Statistical test results
            'kruskal_wallis': {
                'statistic': kruskal_stat,
                'p_value': kruskal_pvalue,
                'significant': kruskal_pvalue < 0.05
            },
            'mann_whitney': {
                'statistic': mannwhitney_stat,
                'p_value': mannwhitney_pvalue,
                'significant': mannwhitney_pvalue < 0.05
            },
            'welch_ttest': {
                'statistic': ttest_stat,
                'p_value': ttest_pvalue,
                'significant': ttest_pvalue < 0.05
            },
            'kolmogorov_smirnov': {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'significant': ks_pvalue < 0.05
            },
            'one_sample_ttest': {
                'statistic': onesample_stat,
                'p_value': onesample_pvalue,
                'significant': onesample_pvalue < 0.05
            }
        }
        
        print("Statistical analysis completed")
        return stats_summary
        
    except Exception as e:
        print(f"Error in statistical analysis: {str(e)}")
        return {}

def create_simple_plot(matched_df):
    """Create a simple correlation plot"""
    try:
        if matched_df.empty:
            return go.Figure()
        
        fig = px.scatter(matched_df, x='BTM_Thickness', y='DMT_Thickness', 
                        color='WaferID', title='BTM vs DMT Thickness Comparison',
                        labels={'BTM_Thickness': 'BTM Thickness (Å)', 'DMT_Thickness': 'DMT Thickness (Å)'})
        
        # Add diagonal line for perfect correlation
        min_thick = min(matched_df['BTM_Thickness'].min(), matched_df['DMT_Thickness'].min())
        max_thick = max(matched_df['BTM_Thickness'].max(), matched_df['DMT_Thickness'].max())
        fig.add_shape(type="line", x0=min_thick, y0=min_thick, x1=max_thick, y1=max_thick,
                     line=dict(color="red", width=2, dash="dash"))
        
        return fig
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        return go.Figure()

def load_data():
    """Load and process all data"""
    print("Starting data loading process...")
    
    # Define file paths
    btm_file = 'BTM/BTM-matching-7152-12500A.csv'
    dmt_file = 'DMT/2026-03-26T07.51.50.4648236-LN1718SS29-8333-DMT116-TZJ501-DMTDUMMY.xml'
    
    # Load data
    btm_df = load_btm_data(btm_file)
    dmt_df = load_dmt_data(dmt_file)
    
    if btm_df.empty or dmt_df.empty:
        print("Error: One or both data files could not be loaded")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    # Find matching measurements
    matched_df = find_matching_measurements(btm_df, dmt_df, tolerance_mm=2.0)
    print(f"Found {len(matched_df)} matched measurements")
    
    # Perform statistical analysis
    stats_summary = perform_statistical_analysis(matched_df)
    
    return btm_df, dmt_df, matched_df, stats_summary

# Load data when app starts
print("Initializing app and loading data...")
try:
    btm_data, dmt_data, matched_data, statistical_results = load_data()
    print(f"Data loaded successfully: {len(matched_data)} matched measurements")
    app_data_loaded = True
except Exception as e:
    print(f"Error loading data: {str(e)}")
    btm_data = pd.DataFrame()
    dmt_data = pd.DataFrame()
    matched_data = pd.DataFrame()
    statistical_results = {}
    app_data_loaded = False

# Define app layout
print("Creating app layout...")
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("BTM vs DMT Thickness Comparison Dashboard", 
                   className="text-center mb-4"),
            html.H4("K-Sample Statistical Analysis with P-Values", 
                   className="text-center text-muted mb-4"),
            html.Hr()
        ])
    ]),
    
    # Summary statistics row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Dataset Summary", className="card-title"),
                    html.P(f"Matched Measurement Points: {len(matched_data) if not matched_data.empty else 0}"),
                    html.P(f"Unique Wafers: {matched_data['WaferID'].nunique() if not matched_data.empty else 0}"),
                    html.P(f"BTM Total Points: {len(btm_data)}"),
                    html.P(f"DMT Total Points: {len(dmt_data)}"),
                    html.P(f"Spatial Tolerance: 2.0 mm"),
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Thickness Statistics", className="card-title"),
                    html.P(f"BTM Mean: {statistical_results.get('btm_mean', 0):.2f} Å"),
                    html.P(f"DMT Mean: {statistical_results.get('dmt_mean', 0):.2f} Å"),
                    html.P(f"Mean Difference: {statistical_results.get('mean_difference', 0):.2f} Å"),
                    html.P(f"Std Difference: {statistical_results.get('std_difference', 0):.2f} Å"),
                    html.P(f"Mean Abs Difference: {statistical_results.get('mean_abs_difference', 0):.2f} Å"),
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Kruskal-Wallis Test", className="card-title"),
                    html.H6("(K-Sample Non-parametric Test)", className="card-subtitle mb-2 text-muted"),
                    html.P(f"Test Statistic: {statistical_results.get('kruskal_wallis', {}).get('statistic', 0):.4f}"),
                    html.P(f"P-value: {statistical_results.get('kruskal_wallis', {}).get('p_value', 0):.2e}"),
                    html.P(f"Significant: {'Yes' if statistical_results.get('kruskal_wallis', {}).get('significant', False) else 'No'}",
                          className="text-danger" if statistical_results.get('kruskal_wallis', {}).get('significant', False) else "text-success"),
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Mann-Whitney U Test", className="card-title"),
                    html.H6("(K-Sample Rank Test)", className="card-subtitle mb-2 text-muted"),
                    html.P(f"Test Statistic: {statistical_results.get('mann_whitney', {}).get('statistic', 0):.0f}"),
                    html.P(f"P-value: {statistical_results.get('mann_whitney', {}).get('p_value', 0):.2e}"),
                    html.P(f"Significant: {'Yes' if statistical_results.get('mann_whitney', {}).get('significant', False) else 'No'}",
                          className="text-danger" if statistical_results.get('mann_whitney', {}).get('significant', False) else "text-success"),
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Statistical tests table
    dbc.Row([
        dbc.Col([
            html.H4("Complete K-Sample Statistical Analysis Results"),
            html.P("Multiple statistical tests comparing BTM and DMT thickness distributions:"),
            dash_table.DataTable(
                id='stats-table',
                columns=[
                    {'name': 'Statistical Test', 'id': 'test'},
                    {'name': 'Test Statistic', 'id': 'statistic'},
                    {'name': 'P-value', 'id': 'p_value'},
                    {'name': 'Significant (α=0.05)', 'id': 'significant'},
                    {'name': 'Interpretation', 'id': 'interpretation'}
                ],
                data=[
                    {
                        'test': 'Kruskal-Wallis (K-Sample)',
                        'statistic': f"{statistical_results.get('kruskal_wallis', {}).get('statistic', 0):.4f}",
                        'p_value': f"{statistical_results.get('kruskal_wallis', {}).get('p_value', 0):.2e}",
                        'significant': 'Yes' if statistical_results.get('kruskal_wallis', {}).get('significant', False) else 'No',
                        'interpretation': 'Distributions differ significantly' if statistical_results.get('kruskal_wallis', {}).get('significant', False) else 'No significant difference in distributions'
                    },
                    {
                        'test': 'Mann-Whitney U (K-Sample)',
                        'statistic': f"{statistical_results.get('mann_whitney', {}).get('statistic', 0):.0f}",
                        'p_value': f"{statistical_results.get('mann_whitney', {}).get('p_value', 0):.2e}",
                        'significant': 'Yes' if statistical_results.get('mann_whitney', {}).get('significant', False) else 'No',
                        'interpretation': 'Medians differ significantly' if statistical_results.get('mann_whitney', {}).get('significant', False) else 'No significant difference in medians'
                    },
                    {
                        'test': "Welch's t-test",
                        'statistic': f"{statistical_results.get('welch_ttest', {}).get('statistic', 0):.4f}",
                        'p_value': f"{statistical_results.get('welch_ttest', {}).get('p_value', 0):.2e}",
                        'significant': 'Yes' if statistical_results.get('welch_ttest', {}).get('significant', False) else 'No',
                        'interpretation': 'Means differ significantly' if statistical_results.get('welch_ttest', {}).get('significant', False) else 'No significant difference in means'
                    },
                    {
                        'test': 'Kolmogorov-Smirnov',
                        'statistic': f"{statistical_results.get('kolmogorov_smirnov', {}).get('statistic', 0):.4f}",
                        'p_value': f"{statistical_results.get('kolmogorov_smirnov', {}).get('p_value', 0):.2e}",
                        'significant': 'Yes' if statistical_results.get('kolmogorov_smirnov', {}).get('significant', False) else 'No',
                        'interpretation': 'Distribution shapes differ significantly' if statistical_results.get('kolmogorov_smirnov', {}).get('significant', False) else 'Similar distribution shapes'
                    },
                    {
                        'test': 'One-sample t-test (bias test)',
                        'statistic': f"{statistical_results.get('one_sample_ttest', {}).get('statistic', 0):.4f}",
                        'p_value': f"{statistical_results.get('one_sample_ttest', {}).get('p_value', 0):.2e}",
                        'significant': 'Yes' if statistical_results.get('one_sample_ttest', {}).get('significant', False) else 'No',
                        'interpretation': 'Systematic bias exists between tools' if statistical_results.get('one_sample_ttest', {}).get('significant', False) else 'No systematic bias detected'
                    }
                ],
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'lightblue', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{significant} = Yes'},
                        'backgroundColor': '#ffcccc',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{significant} = No'},
                        'backgroundColor': '#ccffcc',
                        'color': 'black',
                    }
                ]
            )
        ])
    ], className="mb-4"),
    
    # Correlation plot
    dbc.Row([
        dbc.Col([
            html.H4("BTM vs DMT Thickness Correlation"),
            dcc.Graph(
                id='correlation-plot',
                figure=create_simple_plot(matched_data) if not matched_data.empty else go.Figure()
            )
        ])
    ], className="mb-4"),
    
    # Summary interpretation
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H4("Statistical Analysis Summary", className="alert-heading"),
                html.P("This dashboard performs comprehensive K-Sample statistical testing to compare thickness measurements between BTM and DMT tools."),
                html.Hr(),
                html.P([
                    html.Strong("K-Sample Tests Used:"),
                    html.Br(),
                    "• Kruskal-Wallis Test: Non-parametric test comparing multiple samples",
                    html.Br(),
                    "• Mann-Whitney U Test: Rank-based test for comparing two independent samples",
                    html.Br(),
                    "• Additional tests: Welch's t-test, Kolmogorov-Smirnov, and bias detection"
                ]),
                html.Hr(),
                html.P([
                    html.Strong("Interpretation:"),
                    html.Br(),
                    f"Total matched points: {len(matched_data) if not matched_data.empty else 0}",
                    html.Br(),
                    f"Mean thickness difference (BTM - DMT): {statistical_results.get('mean_difference', 0):.2f} Å",
                    html.Br(),
                    f"Standard deviation of differences: {statistical_results.get('std_difference', 0):.2f} Å"
                ])
            ], color="info")
        ])
    ])
    
], fluid=True)

if __name__ == '__main__':
    print("Starting Thickness Comparison Dashboard...")
    print(f"Matched data points: {len(matched_data)}")
    print(f"Statistical tests completed successfully!")
    print("Starting Dash server at http://127.0.0.1:8050")
    try:
        app.run(debug=True, host='127.0.0.1', port=8050)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        # Try alternative approach
        app.run(debug=False, host='127.0.0.1', port=8050)