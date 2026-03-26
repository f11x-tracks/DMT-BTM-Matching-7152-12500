#!/usr/bin/env python3
"""
Thickness Measurement Visualization Tool
Generates comprehensive visual comparisons between DMT and BTM thickness measurements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data():
    """Load the matched thickness comparison data"""
    try:
        df = pd.read_csv('thickness_comparison_results.csv')
        print(f"📊 Loaded {len(df)} matched measurement pairs from {df['WaferID'].nunique()} wafers")
        return df
    except FileNotFoundError:
        print("❌ Error: thickness_comparison_results.csv not found!")
        return None

def create_trend_plot(df):
    """Create trend plots showing raw thickness data for each wafer"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Raw Thickness Data Trends: DMT vs BTM by Wafer', fontsize=16, fontweight='bold')
    
    wafers = df['WaferID'].unique()
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, wafer in enumerate(wafers):
        if i < len(axes):
            wafer_data = df[df['WaferID'] == wafer].sort_values('BTM_X')
            
            ax = axes[i]
            
            # Plot both measurements
            ax.plot(range(len(wafer_data)), wafer_data['BTM_Thickness'], 
                   'o-', color=colors[0], alpha=0.7, linewidth=2, markersize=4, label='BTM')
            ax.plot(range(len(wafer_data)), wafer_data['DMT_Thickness'], 
                   's-', color=colors[1], alpha=0.7, linewidth=2, markersize=4, label='DMT')
            
            ax.set_title(f'{wafer}\n(n={len(wafer_data)} points)', fontweight='bold')
            ax.set_xlabel('Measurement Point')
            ax.set_ylabel('Thickness (Å)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add correlation info
            corr = wafer_data['BTM_Thickness'].corr(wafer_data['DMT_Thickness'])
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('thickness_trend_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_boxplot(df):
    """Create box plot comparing thickness distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data for plotting
    btm_data = df['BTM_Thickness']
    dmt_data = df['DMT_Thickness']
    
    # Box plot comparing distributions
    box_data = [btm_data, dmt_data]
    box_labels = ['BTM', 'DMT']
    colors = ['#e74c3c', '#3498db']
    
    bp1 = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Thickness Distribution Comparison\nBTM vs DMT', fontweight='bold')
    ax1.set_ylabel('Thickness (Å)')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    btm_stats = f"BTM: μ={btm_data.mean():.1f}±{btm_data.std():.1f} Å"
    dmt_stats = f"DMT: μ={dmt_data.mean():.1f}±{dmt_data.std():.1f} Å"
    ax1.text(0.5, 0.02, f'{btm_stats}\n{dmt_stats}', transform=ax1.transAxes, 
             ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Box plot by wafer
    df_melted = pd.melt(df, id_vars=['WaferID'], 
                       value_vars=['BTM_Thickness', 'DMT_Thickness'],
                       var_name='Tool', value_name='Thickness')
    df_melted['Tool'] = df_melted['Tool'].str.replace('_Thickness', '')
    
    sns.boxplot(data=df_melted, x='WaferID', y='Thickness', hue='Tool', ax=ax2, palette=['#e74c3c', '#3498db'])
    ax2.set_title('Thickness Distribution by Wafer\nBTM vs DMT', fontweight='bold')
    ax2.set_ylabel('Thickness (Å)')
    ax2.set_xlabel('Wafer ID')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thickness_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_qq_plot(df):
    """Create Q-Q plots for normality comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Normal Quantile Plots: DMT vs BTM Thickness Data', fontsize=16, fontweight='bold')
    
    # Q-Q plot for BTM data
    stats.probplot(df['BTM_Thickness'], dist="norm", plot=axes[0,0])
    axes[0,0].set_title('BTM Thickness vs Normal Distribution', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Q-Q plot for DMT data
    stats.probplot(df['DMT_Thickness'], dist="norm", plot=axes[0,1])
    axes[0,1].set_title('DMT Thickness vs Normal Distribution', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Q-Q plot for differences
    thickness_diff = df['DMT_Thickness'] - df['BTM_Thickness']
    stats.probplot(thickness_diff, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Thickness Differences (DMT-BTM) vs Normal', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Scatter plot DMT vs BTM with 45-degree line
    axes[1,1].scatter(df['BTM_Thickness'], df['DMT_Thickness'], alpha=0.6, s=30)
    
    # Add perfect correlation line (45-degree)
    min_thick = min(df['BTM_Thickness'].min(), df['DMT_Thickness'].min())
    max_thick = max(df['BTM_Thickness'].max(), df['DMT_Thickness'].max())
    axes[1,1].plot([min_thick, max_thick], [min_thick, max_thick], 'r--', alpha=0.8, linewidth=2, label='Perfect Agreement')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['BTM_Thickness'], df['DMT_Thickness'])
    line = slope * df['BTM_Thickness'] + intercept
    axes[1,1].plot(df['BTM_Thickness'], line, 'b-', alpha=0.8, linewidth=2, label=f'Regression (r²={r_value**2:.3f})')
    
    axes[1,1].set_xlabel('BTM Thickness (Å)')
    axes[1,1].set_ylabel('DMT Thickness (Å)')
    axes[1,1].set_title('BTM vs DMT Thickness Correlation', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thickness_qq_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_spatial_delta_plot(df):
    """Create spatial plot showing thickness differences by X,Y position"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spatial Analysis: Thickness Differences (DMT - BTM) by Position', fontsize=16, fontweight='bold')
    
    # Calculate thickness differences
    df['Thickness_Delta'] = df['DMT_Thickness'] - df['BTM_Thickness']
    
    # Overall spatial plot
    scatter = ax1.scatter(df['BTM_X'], df['BTM_Y'], 
                         c=df['Thickness_Delta'], cmap='RdBu_r', 
                         s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Thickness Difference (DMT-BTM) by Position\nAll Wafers Combined', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Thickness Difference (Å)', rotation=270, labelpad=20)
    
    # Add statistics text
    mean_delta = df['Thickness_Delta'].mean()
    std_delta = df['Thickness_Delta'].std()
    ax1.text(0.02, 0.98, f'Mean Δ: {mean_delta:.1f}±{std_delta:.1f} Å', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Heatmap by wafer (example with first wafer)
    wafer_ids = df['WaferID'].unique()
    if len(wafer_ids) > 0:
        wafer_data = df[df['WaferID'] == wafer_ids[0]]
        scatter2 = ax2.scatter(wafer_data['BTM_X'], wafer_data['BTM_Y'], 
                              c=wafer_data['Thickness_Delta'], cmap='RdBu_r', 
                              s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('X Position (mm)')
        ax2.set_ylabel('Y Position (mm)')
        ax2.set_title(f'Thickness Difference by Position\nWafer: {wafer_ids[0]}', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Thickness Difference (Å)', rotation=270, labelpad=20)
    
    # Distance vs Delta correlation plot
    df['Spatial_Distance'] = df['Distance_mm']
    ax3.scatter(df['Spatial_Distance'], df['Thickness_Delta'], 
               alpha=0.6, s=40, color='purple')
    
    # Add trendline
    z = np.polyfit(df['Spatial_Distance'], df['Thickness_Delta'], 1)
    p = np.poly1d(z)
    ax3.plot(df['Spatial_Distance'], p(df['Spatial_Distance']), "r--", alpha=0.8, linewidth=2)
    
    # Correlation coefficient
    corr_dist_delta = df['Spatial_Distance'].corr(df['Thickness_Delta'])
    ax3.set_xlabel('Spatial Matching Distance (mm)')
    ax3.set_ylabel('Thickness Difference (Å)')
    ax3.set_title(f'Matching Distance vs Thickness Difference\nr = {corr_dist_delta:.3f}', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Difference distribution by position quadrants
    # Define quadrants based on X,Y signs
    df['Quadrant'] = ''
    df.loc[(df['BTM_X'] >= 0) & (df['BTM_Y'] >= 0), 'Quadrant'] = 'Q1 (+X,+Y)'
    df.loc[(df['BTM_X'] < 0) & (df['BTM_Y'] >= 0), 'Quadrant'] = 'Q2 (-X,+Y)'  
    df.loc[(df['BTM_X'] < 0) & (df['BTM_Y'] < 0), 'Quadrant'] = 'Q3 (-X,-Y)'
    df.loc[(df['BTM_X'] >= 0) & (df['BTM_Y'] < 0), 'Quadrant'] = 'Q4 (+X,-Y)'
    
    # Box plot by quadrants
    quadrant_data = [df[df['Quadrant'] == q]['Thickness_Delta'].values for q in 
                     ['Q1 (+X,+Y)', 'Q2 (-X,+Y)', 'Q3 (-X,-Y)', 'Q4 (+X,-Y)']]
    quadrant_labels = ['Q1\n(+X,+Y)', 'Q2\n(-X,+Y)', 'Q3\n(-X,-Y)', 'Q4\n(+X,-Y)']
    
    bp = ax4.boxplot(quadrant_data, labels=quadrant_labels, patch_artist=True)
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Thickness Difference (Å)')
    ax4.set_title('Thickness Difference Distribution\nby Wafer Quadrant', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add quadrant statistics
    quad_stats = []
    for i, q in enumerate(['Q1 (+X,+Y)', 'Q2 (-X,+Y)', 'Q3 (-X,-Y)', 'Q4 (+X,-Y)']):
        q_data = df[df['Quadrant'] == q]['Thickness_Delta']
        if len(q_data) > 0:
            quad_stats.append(f'{q}: μ={q_data.mean():.1f}±{q_data.std():.1f}Å (n={len(q_data)})')
    
    stats_text = '\n'.join(quad_stats)
    ax4.text(1.02, 0.5, stats_text, transform=ax4.transAxes, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('thickness_spatial_delta_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_radial_analysis_plots(df):
    """Create radial analysis plots showing thickness delta and overlay by radius"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Radial Analysis: Thickness Measurements vs Distance from Center', fontsize=16, fontweight='bold')
    
    # Calculate radial distance from center (0,0)
    df['Radius_mm'] = np.sqrt(df['BTM_X']**2 + df['BTM_Y']**2)
    df['Thickness_Delta'] = df['DMT_Thickness'] - df['BTM_Thickness']
    
    # 1. Delta by radius scatter plot
    ax1.scatter(df['Radius_mm'], df['Thickness_Delta'], alpha=0.7, s=50, c='red', edgecolors='black', linewidth=0.5)
    
    # Add trend line for delta vs radius
    z_delta = np.polyfit(df['Radius_mm'], df['Thickness_Delta'], 1)
    p_delta = np.poly1d(z_delta)
    radius_range = np.linspace(0, 150, 100)
    ax1.plot(radius_range, p_delta(radius_range), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z_delta[0]:.2f}x+{z_delta[1]:.1f}')
    
    # Add horizontal line at zero
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Radius from Center (mm)')
    ax1.set_ylabel('Thickness Difference (DMT - BTM, Å)')
    ax1.set_title('Thickness Difference vs Radial Position', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 150)
    ax1.legend()
    
    # Add correlation info
    corr_radius_delta = df['Radius_mm'].corr(df['Thickness_Delta'])
    ax1.text(0.02, 0.98, f'Correlation: r = {corr_radius_delta:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. BTM and DMT overlay by radius
    ax2.scatter(df['Radius_mm'], df['BTM_Thickness'], alpha=0.7, s=40, c='blue', label='BTM', marker='o')
    ax2.scatter(df['Radius_mm'], df['DMT_Thickness'], alpha=0.7, s=40, c='red', label='DMT', marker='s')
    
    # Add trend lines for both tools
    z_btm = np.polyfit(df['Radius_mm'], df['BTM_Thickness'], 1)
    z_dmt = np.polyfit(df['Radius_mm'], df['DMT_Thickness'], 1)
    p_btm = np.poly1d(z_btm)
    p_dmt = np.poly1d(z_dmt)
    
    ax2.plot(radius_range, p_btm(radius_range), "b--", alpha=0.8, linewidth=2, label=f'BTM trend: {z_btm[0]:.1f}x+{z_btm[1]:.0f}')
    ax2.plot(radius_range, p_dmt(radius_range), "r--", alpha=0.8, linewidth=2, label=f'DMT trend: {z_dmt[0]:.1f}x+{z_dmt[1]:.0f}')
    
    ax2.set_xlabel('Radius from Center (mm)')
    ax2.set_ylabel('Thickness (Å)')
    ax2.set_title('BTM vs DMT Thickness by Radial Position', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 150)
    ax2.legend()
    
    # 3. Radial binned analysis
    # Create radial bins
    radius_bins = np.arange(0, 151, 25)  # 0-25, 25-50, 50-75, 75-100, 100-125, 125-150
    bin_centers = radius_bins[:-1] + np.diff(radius_bins)/2
    
    btm_means = []
    dmt_means = []
    delta_means = []
    btm_stds = []
    dmt_stds = []
    delta_stds = []
    bin_counts = []
    
    for i in range(len(radius_bins)-1):
        mask = (df['Radius_mm'] >= radius_bins[i]) & (df['Radius_mm'] < radius_bins[i+1])
        bin_data = df[mask]
        
        if len(bin_data) > 0:
            btm_means.append(bin_data['BTM_Thickness'].mean())
            dmt_means.append(bin_data['DMT_Thickness'].mean())
            delta_means.append(bin_data['Thickness_Delta'].mean())
            btm_stds.append(bin_data['BTM_Thickness'].std())
            dmt_stds.append(bin_data['DMT_Thickness'].std())
            delta_stds.append(bin_data['Thickness_Delta'].std())
            bin_counts.append(len(bin_data))
        else:
            btm_means.append(np.nan)
            dmt_means.append(np.nan)
            delta_means.append(np.nan)
            btm_stds.append(np.nan)
            dmt_stds.append(np.nan)
            delta_stds.append(np.nan)
            bin_counts.append(0)
    
    # Plot binned means with error bars
    ax3.errorbar(bin_centers, btm_means, yerr=btm_stds, fmt='bo-', capsize=5, capthick=2, 
                label='BTM', markersize=8, linewidth=2, alpha=0.8)
    ax3.errorbar(bin_centers, dmt_means, yerr=dmt_stds, fmt='rs-', capsize=5, capthick=2,
                label='DMT', markersize=8, linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Radial Distance (mm)')
    ax3.set_ylabel('Mean Thickness (Å)')
    ax3.set_title('Mean Thickness by Radial Bins\n(with Standard Deviation)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 150)
    ax3.legend()
    
    # Add sample counts
    for i, (x, count) in enumerate(zip(bin_centers, bin_counts)):
        if count > 0:
            ax3.text(x, min(btm_means + dmt_means) * 0.9995, f'n={count}', 
                    ha='center', va='top', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # 4. Delta by radial bins with statistical testing
    ax4.errorbar(bin_centers, delta_means, yerr=delta_stds, fmt='ro-', capsize=5, capthick=2,
                markersize=8, linewidth=2, alpha=0.8, label='Mean Δ ± SD')
    
    # Add horizontal line at zero
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Highlight significant bins
    for i, (x, mean, std, count) in enumerate(zip(bin_centers, delta_means, delta_stds, bin_counts)):
        if count > 0 and not np.isnan(mean) and not np.isnan(std):
            # Simple t-test against zero (DMT = BTM hypothesis)
            t_stat = abs(mean) / (std / np.sqrt(count)) if std > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(t_stat, count-1)) if count > 1 else 1
            
            if p_val < 0.05:  # Significant difference
                ax4.plot(x, mean, 'ko', markersize=12, fillstyle='none', markeredgewidth=3, alpha=0.7)
                ax4.text(x, mean + max(delta_stds)*0.2, '*', ha='center', va='bottom', 
                        fontsize=16, fontweight='bold', color='black')
    
    ax4.set_xlabel('Radial Distance (mm)')
    ax4.set_ylabel('Mean Thickness Difference (Å)')
    ax4.set_title('Thickness Difference by Radial Bins\n(* = p < 0.05 vs zero)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 150)
    ax4.legend()
    
    # Add overall radial statistics
    radial_stats = f"""Radial Analysis Summary:
Max radius: {df['Radius_mm'].max():.1f} mm
Mean delta: {df['Thickness_Delta'].mean():.1f}±{df['Thickness_Delta'].std():.1f} Å
Radial correlation: r = {corr_radius_delta:.3f}
Slope (DMT): {z_dmt[0]:.2f} Å/mm
Slope (BTM): {z_btm[0]:.2f} Å/mm"""
    
    ax4.text(1.02, 0.5, radial_stats, transform=ax4.transAxes, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('thickness_radial_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_summary(df):
    """Create comprehensive statistical summary table"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE STATISTICAL SUMMARY")
    print("="*80)
    
    # Basic statistics
    btm_data = df['BTM_Thickness']
    dmt_data = df['DMT_Thickness']
    differences = dmt_data - btm_data
    
    print(f"\n📈 DESCRIPTIVE STATISTICS")
    print("-" * 50)
    
    summary_stats = pd.DataFrame({
        'BTM': [btm_data.count(), btm_data.mean(), btm_data.std(), btm_data.min(), 
                btm_data.quantile(0.25), btm_data.median(), btm_data.quantile(0.75), btm_data.max()],
        'DMT': [dmt_data.count(), dmt_data.mean(), dmt_data.std(), dmt_data.min(),
                dmt_data.quantile(0.25), dmt_data.median(), dmt_data.quantile(0.75), dmt_data.max()],
        'Difference (DMT-BTM)': [differences.count(), differences.mean(), differences.std(), 
                                differences.min(), differences.quantile(0.25), differences.median(),
                                differences.quantile(0.75), differences.max()]
    }, index=['Count', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max'])
    
    print(summary_stats.round(2))
    
    # Statistical tests
    print(f"\n🔬 STATISTICAL TESTS & P-VALUES")
    print("-" * 50)
    
    # Normality tests
    _, btm_shapiro_p = stats.shapiro(btm_data)
    _, dmt_shapiro_p = stats.shapiro(dmt_data)
    _, diff_shapiro_p = stats.shapiro(differences)
    
    # Comparison tests
    _, ttest_p = stats.ttest_rel(dmt_data, btm_data)  # Paired t-test
    _, wilcoxon_p = stats.wilcoxon(dmt_data, btm_data)  # Wilcoxon signed-rank
    _, mannwhitney_p = stats.mannwhitneyu(dmt_data, btm_data)  # Mann-Whitney U
    _, ks_p = stats.ks_2samp(dmt_data, btm_data)  # Kolmogorov-Smirnov
    
    # Correlation
    corr_coeff, corr_p = stats.pearsonr(btm_data, dmt_data)
    spearman_coeff, spearman_p = stats.spearmanr(btm_data, dmt_data)
    
    # F-test for equal variances
    f_stat = btm_data.var() / dmt_data.var() if dmt_data.var() != 0 else np.inf
    f_p = 2 * min(stats.f.cdf(f_stat, len(btm_data)-1, len(dmt_data)-1), 
                  1 - stats.f.cdf(f_stat, len(btm_data)-1, len(dmt_data)-1))
    
    test_results = pd.DataFrame({
        'Test': ['Shapiro-Wilk (BTM)', 'Shapiro-Wilk (DMT)', 'Shapiro-Wilk (Diff)', 
                'Paired t-test', 'Wilcoxon Signed-rank', 'Mann-Whitney U', 
                'Kolmogorov-Smirnov', 'F-test (Equal Var)', 'Pearson Correlation', 'Spearman Correlation'],
        'Statistic': ['-', '-', '-', f'{(dmt_data.mean() - btm_data.mean())/(differences.std()/np.sqrt(len(differences))):.3f}', 
                     '-', '-', '-', f'{f_stat:.3f}', f'{corr_coeff:.3f}', f'{spearman_coeff:.3f}'],
        'P-Value': [f'{btm_shapiro_p:.6f}', f'{dmt_shapiro_p:.6f}', f'{diff_shapiro_p:.6f}',
                   f'{ttest_p:.6f}', f'{wilcoxon_p:.6f}', f'{mannwhitney_p:.6f}', 
                   f'{ks_p:.6f}', f'{f_p:.6f}', f'{corr_p:.2e}', f'{spearman_p:.2e}'],
        'Interpretation': [
            'Normal' if btm_shapiro_p > 0.05 else 'Non-normal',
            'Normal' if dmt_shapiro_p > 0.05 else 'Non-normal', 
            'Normal' if diff_shapiro_p > 0.05 else 'Non-normal',
            'No significant difference' if ttest_p > 0.05 else 'Significant difference',
            'No significant difference' if wilcoxon_p > 0.05 else 'Significant difference',
            'No significant difference' if mannwhitney_p > 0.05 else 'Significant difference',
            'Same distribution' if ks_p > 0.05 else 'Different distributions',
            'Equal variances' if f_p > 0.05 else 'Unequal variances',
            'Strong correlation' if abs(corr_coeff) > 0.7 else 'Moderate correlation',
            'Strong correlation' if abs(spearman_coeff) > 0.7 else 'Moderate correlation'
        ]
    })
    
    print(test_results.to_string(index=False))
    
    # Overall conclusions
    print(f"\n🎯 CONCLUSIONS")
    print("-" * 50)
    
    significant_tests = sum([ttest_p <= 0.05, wilcoxon_p <= 0.05, mannwhitney_p <= 0.05, ks_p <= 0.05])
    
    conclusions = []
    conclusions.append(f"• Measurement Tools: DMT vs BTM thickness comparison")
    conclusions.append(f"• Sample Size: {len(df)} matched measurement pairs from {df['WaferID'].nunique()} wafers")
    conclusions.append(f"• Mean Difference: {differences.mean():.2f} ± {differences.std():.2f} Å (DMT - BTM)")
    conclusions.append(f"• Correlation: r = {corr_coeff:.3f} (p < 0.001)" if corr_p < 0.001 else f"• Correlation: r = {corr_coeff:.3f} (p = {corr_p:.3f})")
    
    if significant_tests == 0:
        conclusions.append("• Statistical Conclusion: NO SIGNIFICANT DIFFERENCES detected between DMT and BTM")
        conclusions.append("• Recommendation: Tools can be used interchangeably")
    else:
        conclusions.append(f"• Statistical Conclusion: {significant_tests}/4 tests show significant differences")
        conclusions.append("• Recommendation: Further investigation of systematic differences needed")
    
    if abs(differences.mean()) < 50:  # Less than 5nm difference
        conclusions.append("• Practical Conclusion: Mean difference is practically negligible")
    else:
        conclusions.append("• Practical Conclusion: Mean difference may be practically significant") 
        
    for conclusion in conclusions:
        print(conclusion)
    
    # Save summary to file
    with open('statistical_summary.txt', 'w') as f:
        f.write("THICKNESS MEASUREMENT COMPARISON SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write("DESCRIPTIVE STATISTICS:\n")
        f.write(summary_stats.round(2).to_string())
        f.write("\n\nSTATISTICAL TESTS:\n")
        f.write(test_results.to_string(index=False))
        f.write("\n\nCONCLUSIONS:\n")
        for conclusion in conclusions:
            f.write(conclusion + "\n")
    
    print(f"\n💾 Statistical summary saved to 'statistical_summary.txt'")

def create_worst_coordinate_analysis(df):
    """Create detailed analysis showing coordinate pairs with biggest deltas"""
    fig = plt.figure(figsize=(18, 10))
    
    # Calculate absolute thickness differences for ranking
    df['Abs_Thickness_Delta'] = abs(df['DMT_Thickness'] - df['BTM_Thickness'])
    df['Thickness_Delta'] = df['DMT_Thickness'] - df['BTM_Thickness']
    df['Distance_from_center'] = np.sqrt(df['BTM_X']**2 + df['BTM_Y']**2)
    
    # Create main title
    fig.suptitle('Coordinate Analysis: Biggest Thickness Deltas (DMT - BTM)\nIdentifying Worst-Performing Measurement Locations', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    # Define grid layout - 2 rows x 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, height_ratios=[1, 1])
    
    # 1. MAIN SPATIAL PLOT - All coordinates with delta color coding
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Create scatter plot with color coding for deltas
    scatter = ax1.scatter(df['BTM_X'], df['BTM_Y'], 
                         c=df['Thickness_Delta'], 
                         cmap='RdBu_r',  # Red=positive (DMT>BTM), Blue=negative (DMT<BTM)
                         s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Thickness Difference (DMT - BTM) [Å]', rotation=270, labelpad=20, fontsize=11)
    
    ax1.set_xlabel('X Position (mm)', fontsize=12)
    ax1.set_ylabel('Y Position (mm)', fontsize=12)
    ax1.set_title('All Coordinate Pairs: Thickness Delta Spatial Distribution\n(Red: DMT > BTM, Blue: DMT < BTM)', 
                  fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Add statistics box
    mean_delta = df['Thickness_Delta'].mean()
    std_delta = df['Thickness_Delta'].std()
    max_pos_delta = df['Thickness_Delta'].max()
    max_neg_delta = df['Thickness_Delta'].min()
    stats_text = f'Mean Δ: {mean_delta:.1f}±{std_delta:.1f}Å\nMax +Δ: {max_pos_delta:.1f}Å\nMax -Δ: {max_neg_delta:.1f}Å\nn = {len(df)} points'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=10)
    
    # 2. TOP 20 WORST COORDINATES - Ranked by absolute delta
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Get top 20 worst coordinates by absolute delta
    top_worst = df.nlargest(20, 'Abs_Thickness_Delta').copy()
    top_worst['Rank'] = range(1, len(top_worst) + 1)
    
    # Create color map for ranking (worst = red, better = yellow)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_worst)))
    
    # Horizontal bar chart of top 20 worst deltas
    bars = ax2.barh(range(len(top_worst)), top_worst['Abs_Thickness_Delta'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add coordinate labels on bars
    for i, (idx, row) in enumerate(top_worst.iterrows()):
        x_pos = row['Abs_Thickness_Delta'] / 2
        label = f"({row['BTM_X']:.0f},{row['BTM_Y']:.0f})"
        ax2.text(x_pos, i, label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Absolute Thickness Difference (Å)', fontsize=11)
    ax2.set_title('TOP 20 WORST\nCoordinate Pairs\n(Ranked by |Delta|)', fontweight='bold', fontsize=11)
    ax2.set_yticks(range(len(top_worst)))
    ax2.set_yticklabels([f'#{i+1}' for i in range(len(top_worst))], fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()  # Worst at top
    
    # 3. WORST COORDINATES SPATIAL HIGHLIGHT
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Plot all points in light gray
    ax3.scatter(df['BTM_X'], df['BTM_Y'], c='lightgray', s=30, alpha=0.4, edgecolors='none')
    
    # Highlight top 10 worst coordinates
    top_10_worst = top_worst.head(10)
    highlight_scatter = ax3.scatter(top_10_worst['BTM_X'], top_10_worst['BTM_Y'], 
                                   c=top_10_worst['Abs_Thickness_Delta'], 
                                   cmap='Reds', s=200, alpha=0.9, 
                                   edgecolors='black', linewidth=2, marker='o')
    
    # Add rank labels
    for idx, row in top_10_worst.iterrows():
        ax3.annotate(f"#{row['Rank']}", (row['BTM_X'], row['BTM_Y']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
    
    ax3.set_xlabel('X Position (mm)', fontsize=11)
    ax3.set_ylabel('Y Position (mm)', fontsize=11)
    ax3.set_title('Spatial Location of\nWORST 10 Coordinates', fontweight='bold', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # Add colorbar for worst coordinates
    cbar3 = plt.colorbar(highlight_scatter, ax=ax3, shrink=0.7)
    cbar3.set_label('|Delta| (Å)', rotation=270, labelpad=15, fontsize=10)
    
    # 4. DELTA DISTRIBUTION BY COORDINATE REGIONS
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Define coordinate regions based on distance from center
    df['Region'] = pd.cut(df['Distance_from_center'], 
                         bins=[0, 50, 100, 150], 
                         labels=['Center\n(0-50mm)', 'Inner\n(50-100mm)', 'Edge\n(100-150mm)'],
                         include_lowest=True)
    
    # Create violin plot of deltas by region
    regions = ['Center\n(0-50mm)', 'Inner\n(50-100mm)', 'Edge\n(100-150mm)']
    region_data = []
    region_colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for region in regions:
        region_deltas = df[df['Region'] == region]['Thickness_Delta'].dropna()
        if len(region_deltas) > 0:
            region_data.append(region_deltas.values)
        else:
            region_data.append([])
    
    # Create box plot with custom colors
    bp = ax4.boxplot(region_data, labels=regions, patch_artist=True)
    for patch, color in zip(bp['boxes'], region_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Perfect Match')
    ax4.set_ylabel('Thickness Difference (Å)', fontsize=11)
    ax4.set_title('Delta Distribution\nby Radial Region', fontweight='bold', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add region statistics
    region_stats = []
    for i, (region, data) in enumerate(zip(regions, region_data)):
        if len(data) > 0:
            mean_val = np.mean(data)
            std_val = np.std(data)
            region_stats.append(f'{region.split()[0]}: μ={mean_val:.1f}±{std_val:.1f}Å (n={len(data)})')
    
    stats_text = '\n'.join(region_stats)
    ax4.text(1.02, 0.5, stats_text, transform=ax4.transAxes, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8), fontsize=9)
    
    # 5. WAFER-SPECIFIC WORST COORDINATES
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Analyze worst coordinates by wafer
    wafer_worst = df.groupby('WaferID').apply(
        lambda x: x.nlargest(3, 'Abs_Thickness_Delta')[['BTM_X', 'BTM_Y', 'Abs_Thickness_Delta']]
    ).reset_index(level=0)
    
    wafer_means = df.groupby('WaferID')['Abs_Thickness_Delta'].agg(['mean', 'max', 'count']).reset_index()
    wafer_means = wafer_means.sort_values('mean', ascending=False)
    
    # Bar chart of mean absolute delta by wafer
    bars = ax5.bar(range(len(wafer_means)), wafer_means['mean'], 
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add max values as error bars
    ax5.errorbar(range(len(wafer_means)), wafer_means['mean'], 
                yerr=[np.zeros(len(wafer_means)), wafer_means['max'] - wafer_means['mean']], 
                fmt='none', color='red', capsize=5, capthick=2, alpha=0.7)
    
    # Add sample counts on bars
    for i, (bar, row) in enumerate(zip(bars, wafer_means.itertuples())):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'n={row.count}', ha='center', va='bottom', fontsize=9)
    
    ax5.set_xlabel('Wafer ID', fontsize=11)
    ax5.set_ylabel('Mean |Delta| (Å)', fontsize=11)
    ax5.set_title('Worst Performance\nby Wafer', fontweight='bold', fontsize=11)
    ax5.set_xticks(range(len(wafer_means)))
    ax5.set_xticklabels(wafer_means['WaferID'], rotation=45, ha='right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Remove the table section - it will be created separately
    
    # Save the comprehensive analysis
    plt.savefig('worst_coordinates_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary report
    print(f"\n🎯 WORST COORDINATES ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📍 Total coordinate pairs analyzed: {len(df)}")
    print(f"📊 Overall delta statistics:")
    print(f"   • Mean difference: {mean_delta:.1f} ± {std_delta:.1f} Å")
    print(f"   • Worst positive delta: {max_pos_delta:.1f} Å (DMT > BTM)")  
    print(f"   • Worst negative delta: {max_neg_delta:.1f} Å (DMT < BTM)")
    print(f"   • Worst absolute delta: {df['Abs_Thickness_Delta'].max():.1f} Å")
    
    print(f"\n🔥 TOP 5 PROBLEMATIC COORDINATES:")
    for i, (idx, row) in enumerate(top_worst.head(5).iterrows()):
        print(f"   {i+1}. ({row['BTM_X']:.0f}, {row['BTM_Y']:.0f}): {row['Thickness_Delta']:+.1f}Å [Wafer: {row['WaferID']}]")
    
    # Identify patterns in worst coordinates
    worst_coords = top_worst.head(10)
    edge_count = len(worst_coords[worst_coords['Distance_from_center'] > 100])
    print(f"\n📈 PATTERN ANALYSIS (Top 10 worst):")
    print(f"   • Edge coordinates (>100mm from center): {edge_count}/10 ({edge_count*10}%)")
    print(f"   • Mean distance from center: {worst_coords['Distance_from_center'].mean():.1f} mm")
    
    return top_worst

def create_coordinate_ranking_table(df):
    """Create a separate table showing the detailed ranking of worst coordinates"""
    # Calculate deltas
    df['Abs_Thickness_Delta'] = abs(df['DMT_Thickness'] - df['BTM_Thickness'])
    df['Thickness_Delta'] = df['DMT_Thickness'] - df['BTM_Thickness']
    
    # Get top 20 worst coordinates by absolute delta
    top_worst = df.nlargest(20, 'Abs_Thickness_Delta').copy()
    top_worst['Rank'] = range(1, len(top_worst) + 1)
    
    # Create figure for table only
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')  # Hide axes for table
    
    # Create detailed table of top 20 worst coordinates
    table_data = []
    for idx, row in top_worst.iterrows():
        table_data.append([
            f"#{row['Rank']}",
            f"({row['BTM_X']:.0f}, {row['BTM_Y']:.0f})",
            f"({row['DMT_X']:.0f}, {row['DMT_Y']:.0f})",
            f"{row['BTM_Thickness']:.0f}",
            f"{row['DMT_Thickness']:.0f}",
            f"{row['Thickness_Delta']:+.1f}",
            f"{row['Abs_Thickness_Delta']:.1f}",
            f"{row['Distance_mm']:.2f}",
            row['WaferID']
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Rank', 'BTM (X,Y)', 'DMT (X,Y)', 'BTM [Å]', 'DMT [Å]', 
                              'Delta [Å]', '|Delta| [Å]', 'Match Dist [mm]', 'Wafer'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.08, 0.12, 0.12, 0.1, 0.1, 0.1, 0.1, 0.13, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color-code the worst rows
    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            cell = table[(i+1, j)]  # +1 because row 0 is header
            if i < 5:  # Top 5 worst
                cell.set_facecolor('#ffcccc')  # Light red
            elif i < 10:  # Next 5 worst  
                cell.set_facecolor('#ffffcc')  # Light yellow
            elif i < 15:  # Next 5
                cell.set_facecolor('#ccffcc')  # Light green
            else:  # Remaining
                cell.set_facecolor('#ffffff')  # White
    
    # Color header
    for j in range(9):
        cell = table[(0, j)]
        cell.set_facecolor('#4472C4')  # Blue header
        cell.set_text_props(weight='bold', color='white')
    
    # Add title
    plt.suptitle('Detailed Ranking: TOP 20 WORST Coordinate Pairs\n' + 
                'Biggest Thickness Deltas (DMT - BTM)\n' +
                '🔴 Red: Worst 5   🟡 Yellow: Next 5   🟢 Green: Next 5   ⚪ White: Remaining', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add summary statistics at the bottom
    mean_delta = df['Thickness_Delta'].mean()
    std_delta = df['Thickness_Delta'].std()
    worst_delta = df['Abs_Thickness_Delta'].max()
    
    stats_text = f"""Key Statistics:
• Total coordinate pairs: {len(df)}
• Mean thickness difference: {mean_delta:.1f}±{std_delta:.1f} Å  
• Worst absolute delta: {worst_delta:.1f} Å
• Edge coordinates (>100mm): {len(top_worst[np.sqrt(top_worst['BTM_X']**2 + top_worst['BTM_Y']**2) > 100])}/20 ({len(top_worst[np.sqrt(top_worst['BTM_X']**2 + top_worst['BTM_Y']**2) > 100])*5}%)"""
    
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))
    
    plt.tight_layout()
    plt.savefig('coordinate_ranking_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Coordinate ranking table saved to: coordinate_ranking_table.png")
    
    return top_worst

def main():
    """Main function to generate all visualizations"""
    print("🔬 Thickness Measurement Visualization Tool")
    print("=" * 50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print("\n📊 Generating visualizations...")
    
    # Generate all plots
    print("1. Creating trend plots...")
    create_trend_plot(df)
    
    print("2. Creating box plots...")
    create_boxplot(df)
    
    print("3. Creating Q-Q plots...")
    create_qq_plot(df)
    
    print("4. Creating spatial delta plot...")
    create_spatial_delta_plot(df)
    
    print("5. Creating radial analysis plots...")
    create_radial_analysis_plots(df)
    
    print("6. Creating worst coordinates analysis...")
    create_worst_coordinate_analysis(df)
    
    print("7. Creating coordinate ranking table...")
    create_coordinate_ranking_table(df)
    
    print("8. Generating statistical summary...")
    create_statistical_summary(df)
    print("\n✅ All visualizations completed!")
    print("📁 Files generated:")
    print("   • thickness_trend_plots.png")
    print("   • thickness_boxplots.png") 
    print("   • thickness_qq_plots.png")
    print("   • thickness_spatial_delta_plot.png")
    print("   • thickness_radial_analysis.png")
    print("   • worst_coordinates_analysis.png  ⭐ FIXED: Layout corrected")
    print("   • coordinate_ranking_table.png  ⭐ NEW: Separate table image")
    print("   • statistical_summary.txt")

if __name__ == "__main__":
    main()