#!/usr/bin/env python3
"""
BTM vs DMT Thickness Comparison with K-Sample Statistical Analysis
Comprehensive command-line tool for analyzing thickness measurements
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class ThicknessAnalyzer:
    def __init__(self):
        self.btm_data = pd.DataFrame()
        self.dmt_data = pd.DataFrame()
        self.matched_data = pd.DataFrame()
        self.statistical_results = {}
    
    def load_btm_data(self, file_path):
        """Load BTM CSV data"""
        try:
            print(f"📁 Loading BTM data from: {file_path}")
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
                'X[mm] ': 'X_mm',
                'X[mm]': 'X_mm',
                'Y[mm] ': 'Y_mm',
                'Y[mm]': 'Y_mm'
            }
            
            # Apply column mapping
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)
            
            # Convert to numeric
            df['X_mm'] = pd.to_numeric(df['X_mm'], errors='coerce')
            df['Y_mm'] = pd.to_numeric(df['Y_mm'], errors='coerce')
            df['Thickness'] = pd.to_numeric(df['Thickness'], errors='coerce')
            
            # Remove NaN values
            df = df.dropna(subset=['X_mm', 'Y_mm', 'Thickness', 'WaferID'])
            df['Tool'] = 'BTM'
            
            self.btm_data = df
            print(f"✅ BTM data loaded: {len(df)} records from {df['WaferID'].nunique()} wafers")
            return df
            
        except Exception as e:
            print(f"❌ Error loading BTM data: {str(e)}")
            return pd.DataFrame()

    def load_dmt_data(self, file_path):
        """Load DMT XML data"""
        try:
            print(f"📁 Loading DMT data from: {file_path}")
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
            
            self.dmt_data = pd.DataFrame(records)
            print(f"✅ DMT data loaded: {len(records)} records from {self.dmt_data['WaferID'].nunique()} wafers")
            return self.dmt_data
            
        except Exception as e:
            print(f"❌ Error loading DMT data: {str(e)}")
            return pd.DataFrame()

    def find_matching_measurements(self, tolerance_mm=2.0):
        """Find matching measurements between BTM and DMT within tolerance"""
        print(f"🔍 Finding matching measurements within {tolerance_mm}mm tolerance...")
        
        matched_pairs = []
        
        for wafer_id in self.btm_data['WaferID'].unique():
            if wafer_id in self.dmt_data['WaferID'].unique():
                btm_wafer = self.btm_data[self.btm_data['WaferID'] == wafer_id]
                dmt_wafer = self.dmt_data[self.dmt_data['WaferID'] == wafer_id]
                
                # Calculate distances between all BTM and DMT points
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
        
        self.matched_data = pd.DataFrame(matched_pairs)
        print(f"✅ Found {len(self.matched_data)} matched measurement pairs")
        return self.matched_data

    def perform_statistical_analysis(self):
        """Perform comprehensive K-Sample statistical analysis"""
        print("\n📊 Performing K-Sample Statistical Analysis...")
        
        if self.matched_data.empty:
            print("❌ No matched data available for analysis")
            return {}
        
        btm_thickness = self.matched_data['BTM_Thickness']
        dmt_thickness = self.matched_data['DMT_Thickness']
        thickness_diff = self.matched_data['Thickness_Difference']
        
        try:
            # K-Sample tests
            kruskal_stat, kruskal_pvalue = stats.kruskal(btm_thickness, dmt_thickness)
            mannwhitney_stat, mannwhitney_pvalue = stats.mannwhitneyu(btm_thickness, dmt_thickness, alternative='two-sided')
            ttest_stat, ttest_pvalue = stats.ttest_ind(btm_thickness, dmt_thickness, equal_var=False)
            ks_stat, ks_pvalue = stats.ks_2samp(btm_thickness, dmt_thickness)
            onesample_stat, onesample_pvalue = stats.ttest_1samp(thickness_diff, 0)
            
            # Calculate correlation
            correlation_r, correlation_p = stats.pearsonr(btm_thickness, dmt_thickness)
            
            # Summary statistics
            self.statistical_results = {
                'total_matched_points': len(self.matched_data),
                'unique_wafers': self.matched_data['WaferID'].nunique(),
                
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
                'median_abs_difference': self.matched_data['Abs_Thickness_Difference'].median(),
                'mean_abs_difference': self.matched_data['Abs_Thickness_Difference'].mean(),
                'max_abs_difference': self.matched_data['Abs_Thickness_Difference'].max(),
                
                # Correlation
                'correlation_r': correlation_r,
                'correlation_p': correlation_p,
                
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
            
            print("✅ Statistical analysis completed")
            return self.statistical_results
            
        except Exception as e:
            print(f"❌ Error in statistical analysis: {str(e)}")
            return {}

    def print_detailed_report(self):
        """Print comprehensive analysis report"""
        print("\n" + "="*80)
        print("🔬 BTM vs DMT THICKNESS COMPARISON - STATISTICAL ANALYSIS REPORT")
        print("="*80)
        
        # Dataset Summary
        print(f"\n📋 DATASET SUMMARY:")
        print(f"   • Total BTM measurements: {len(self.btm_data)}")
        print(f"   • Total DMT measurements: {len(self.dmt_data)}")
        print(f"   • Matched measurement pairs: {self.statistical_results['total_matched_points']}")
        print(f"   • Unique wafers analyzed: {self.statistical_results['unique_wafers']}")
        print(f"   • Spatial matching tolerance: 2.0 mm")
        
        # Thickness Statistics
        print(f"\n📏 THICKNESS STATISTICS (Angstroms):")
        print(f"   BTM Tool:")
        print(f"     - Mean: {self.statistical_results['btm_mean']:.2f} Å")
        print(f"     - Std Dev: {self.statistical_results['btm_std']:.2f} Å")
        print(f"     - Median: {self.statistical_results['btm_median']:.2f} Å")
        print(f"     - Range: {self.statistical_results['btm_min']:.2f} - {self.statistical_results['btm_max']:.2f} Å")
        
        print(f"   DMT Tool:")
        print(f"     - Mean: {self.statistical_results['dmt_mean']:.2f} Å")
        print(f"     - Std Dev: {self.statistical_results['dmt_std']:.2f} Å")
        print(f"     - Median: {self.statistical_results['dmt_median']:.2f} Å")
        print(f"     - Range: {self.statistical_results['dmt_min']:.2f} - {self.statistical_results['dmt_max']:.2f} Å")
        
        # Difference Analysis
        print(f"\n📊 THICKNESS DIFFERENCE ANALYSIS (BTM - DMT):")
        print(f"   • Mean difference: {self.statistical_results['mean_difference']:.2f} Å")
        print(f"   • Std dev of differences: {self.statistical_results['std_difference']:.2f} Å")
        print(f"   • Mean absolute difference: {self.statistical_results['mean_abs_difference']:.2f} Å")
        print(f"   • Median absolute difference: {self.statistical_results['median_abs_difference']:.2f} Å")
        print(f"   • Maximum absolute difference: {self.statistical_results['max_abs_difference']:.2f} Å")
        
        # Correlation Analysis
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print(f"   • Pearson correlation coefficient: {self.statistical_results['correlation_r']:.4f}")
        print(f"   • Correlation p-value: {self.statistical_results['correlation_p']:.2e}")
        print(f"   • Correlation significance: {'Yes' if self.statistical_results['correlation_p'] < 0.05 else 'No'}")
        
        # K-Sample Statistical Tests
        print(f"\n🧪 K-SAMPLE STATISTICAL TEST RESULTS:")
        print(f"   (Significance level: α = 0.05)")
        print(f"\n   1. KRUSKAL-WALLIS TEST (Non-parametric K-sample test)")
        kw = self.statistical_results['kruskal_wallis']
        print(f"      • Test Statistic: {kw['statistic']:.4f}")
        print(f"      • P-value: {kw['p_value']:.2e}")
        print(f"      • Significant: {'YES' if kw['significant'] else 'NO'}")
        print(f"      • Interpretation: {'Distributions differ significantly' if kw['significant'] else 'No significant difference in distributions'}")
        
        print(f"\n   2. MANN-WHITNEY U TEST (Rank-based K-sample test)")
        mw = self.statistical_results['mann_whitney']
        print(f"      • Test Statistic: {mw['statistic']:.0f}")
        print(f"      • P-value: {mw['p_value']:.2e}")
        print(f"      • Significant: {'YES' if mw['significant'] else 'NO'}")
        print(f"      • Interpretation: {'Medians differ significantly' if mw['significant'] else 'No significant difference in medians'}")
        
        print(f"\n   3. WELCH'S T-TEST (Parametric, unequal variances)")
        wt = self.statistical_results['welch_ttest']
        print(f"      • Test Statistic: {wt['statistic']:.4f}")
        print(f"      • P-value: {wt['p_value']:.2e}")
        print(f"      • Significant: {'YES' if wt['significant'] else 'NO'}")
        print(f"      • Interpretation: {'Means differ significantly' if wt['significant'] else 'No significant difference in means'}")
        
        print(f"\n   4. KOLMOGOROV-SMIRNOV TEST (Distribution shape comparison)")
        ks = self.statistical_results['kolmogorov_smirnov']
        print(f"      • Test Statistic: {ks['statistic']:.4f}")
        print(f"      • P-value: {ks['p_value']:.2e}")
        print(f"      • Significant: {'YES' if ks['significant'] else 'NO'}")
        print(f"      • Interpretation: {'Distribution shapes differ significantly' if ks['significant'] else 'Similar distribution shapes'}")
        
        print(f"\n   5. ONE-SAMPLE T-TEST (Systematic bias detection)")
        os = self.statistical_results['one_sample_ttest']
        print(f"      • Test Statistic: {os['statistic']:.4f}")
        print(f"      • P-value: {os['p_value']:.2e}")
        print(f"      • Significant: {'YES' if os['significant'] else 'NO'}")
        print(f"      • Interpretation: {'Systematic bias exists between tools' if os['significant'] else 'No systematic bias detected'}")
        
        # Overall Conclusion
        print(f"\n🎯 OVERALL CONCLUSION:")
        significant_tests = sum([
            kw['significant'], mw['significant'], wt['significant'], 
            ks['significant'], os['significant']
        ])
        
        if significant_tests >= 3:
            print(f"   • STRONG EVIDENCE of significant differences between BTM and DMT measurements")
            print(f"   • {significant_tests}/5 statistical tests show significance")
        elif significant_tests >= 1:
            print(f"   • MODERATE EVIDENCE of differences between BTM and DMT measurements")
            print(f"   • {significant_tests}/5 statistical tests show significance")
        else:
            print(f"   • NO SIGNIFICANT EVIDENCE of differences between BTM and DMT measurements")
            print(f"   • {significant_tests}/5 statistical tests show significance")
        
        print(f"\n   • Mean difference: {self.statistical_results['mean_difference']:.2f} ± {self.statistical_results['std_difference']:.2f} Å")
        print(f"   • Correlation strength: {abs(self.statistical_results['correlation_r']):.3f}")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    print("🚀 BTM vs DMT Thickness Comparison Tool")
    print("K-Sample Statistical Analysis with P-Values")
    print("-" * 50)
    
    # Initialize analyzer
    analyzer = ThicknessAnalyzer()
    
    # Define file paths
    btm_file = 'BTM/BTM-matching-7152-12500A.csv'
    dmt_file = 'DMT/2026-03-26T07.51.50.4648236-LN1718SS29-8333-DMT116-TZJ501-DMTDUMMY.xml'
    
    # Load data
    if analyzer.load_btm_data(btm_file).empty:
        print("❌ Failed to load BTM data. Exiting.")
        return
    
    if analyzer.load_dmt_data(dmt_file).empty:
        print("❌ Failed to load DMT data. Exiting.")
        return
    
    # Find matching measurements
    if analyzer.find_matching_measurements(tolerance_mm=2.0).empty:
        print("❌ No matching measurements found. Exiting.")
        return
    
    # Perform statistical analysis
    if not analyzer.perform_statistical_analysis():
        print("❌ Statistical analysis failed. Exiting.")
        return
    
    # Print detailed report
    analyzer.print_detailed_report()
    
    # Save results to CSV for further analysis
    try:
        analyzer.matched_data.to_csv('thickness_comparison_results.csv', index=False)
        print(f"\n💾 Results saved to: thickness_comparison_results.csv")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

if __name__ == "__main__":
    main()