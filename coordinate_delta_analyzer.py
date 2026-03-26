#!/usr/bin/env python3
"""
Coordinate Delta Analysis Tool
Focused analysis showing which X,Y coordinate pairs have the biggest deltas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to work properly
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

def create_ranking_table(df, top_worst):
    """Create a separate table showing the detailed ranking of worst coordinates"""
    
    # Create figure for table only
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')  # Hide axes for table
    
    # Create detailed table of top 15 worst coordinates
    table_data = []
    for idx, row in top_worst.head(15).iterrows():
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
    table.scale(1, 1.8)
    
    # Color-code the worst rows
    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            cell = table[(i+1, j)]  # +1 because row 0 is header
            if i < 5:  # Top 5 worst
                cell.set_facecolor('#ffcccc')  # Light red
            elif i < 10:  # Next 5 worst  
                cell.set_facecolor('#ffffcc')  # Light yellow
            else:  # Remaining
                cell.set_facecolor('#ccffcc')  # Light green
    
    # Color header
    for j in range(9):
        cell = table[(0, j)]
        cell.set_facecolor('#4472C4')  # Blue header
        cell.set_text_props(weight='bold', color='white')
    
    # Add title
    plt.suptitle('Detailed Ranking: TOP 15 WORST Coordinate Pairs\n' + 
                'Biggest Thickness Deltas (DMT - BTM)\n' +
                '🔴 Red: Worst 5   🟡 Yellow: Next 5   🟢 Green: Remaining', 
                fontsize=16, fontweight='bold', y=0.92)
    
    # Add summary statistics at the bottom
    mean_delta = df['Thickness_Delta'].mean()
    std_delta = df['Thickness_Delta'].std()
    worst_delta = df['Abs_Thickness_Delta'].max()
    
    stats_text = f"""Key Statistics:  •  Total coordinate pairs: {len(df)}  •  Mean difference: {mean_delta:.1f}±{std_delta:.1f} Å  •  Worst delta: {worst_delta:.1f} Å"""
    
    ax.text(0.5, 0.05, stats_text, transform=ax.transAxes, 
            ha='center', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.8))
    
    plt.tight_layout()
    plt.savefig('coordinate_ranking_table_standalone.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📋 Detailed ranking table saved to: coordinate_ranking_table_standalone.png")

def analyze_coordinate_deltas():
    """Analyze and visualize coordinate pairs with biggest thickness deltas"""
    
    print("🎯 Coordinate Delta Analysis Tool")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_csv('thickness_comparison_results.csv')
        print(f"📊 Loaded {len(df)} coordinate pairs from {df['WaferID'].nunique()} wafers")
    except FileNotFoundError:
        print("❌ Error: thickness_comparison_results.csv not found!")
        return
    
    # Calculate deltas
    df['Thickness_Delta'] = df['DMT_Thickness'] - df['BTM_Thickness']
    df['Abs_Thickness_Delta'] = abs(df['Thickness_Delta'])
    df['Distance_from_center'] = np.sqrt(df['BTM_X']**2 + df['BTM_Y']**2)
    
    # Get worst coordinate pairs
    print("\n🔍 Identifying worst coordinate pairs...")
    top_worst = df.nlargest(25, 'Abs_Thickness_Delta').copy()
    top_worst['Rank'] = range(1, len(top_worst) + 1)
    
    # Statistics
    mean_delta = df['Thickness_Delta'].mean()
    std_delta = df['Thickness_Delta'].std()
    worst_delta = df['Abs_Thickness_Delta'].max()
    
    print(f"📈 Overall Statistics:")
    print(f"   • Mean delta: {mean_delta:.1f} ± {std_delta:.1f} Å")
    print(f"   • Worst absolute delta: {worst_delta:.1f} Å")
    print(f"   • Coordinates with |delta| > {std_delta*2:.0f}Å: {len(df[df['Abs_Thickness_Delta'] > std_delta*2])}")
    
    # Create focused visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('X,Y Coordinate Pairs with Biggest Thickness Deltas\nDMT vs BTM Mismatch Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Spatial plot with all coordinates
    scatter = ax1.scatter(df['BTM_X'], df['BTM_Y'], 
                         c=df['Thickness_Delta'], cmap='RdBu_r',
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.3)
    
    # Highlight top 10 worst
    top_10 = top_worst.head(10)
    ax1.scatter(top_10['BTM_X'], top_10['BTM_Y'], 
               s=200, facecolors='none', edgecolors='red', 
               linewidths=3, alpha=0.9, marker='o')
    
    # Add rank labels for top 5
    for idx, row in top_10.head(5).iterrows():
        ax1.annotate(f"#{row['Rank']}", 
                    (row['BTM_X'], row['BTM_Y']), 
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Spatial Distribution of Coordinate Deltas\n(Red circles = Top 10 Worst)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Thickness Delta [Å]\n(DMT - BTM)', rotation=270, labelpad=20)
    
    # 2. Ranking of worst coordinates
    y_pos = range(15)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, 15))
    
    bars = ax2.barh(y_pos, top_worst.head(15)['Abs_Thickness_Delta'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add coordinate labels
    for i, (idx, row) in enumerate(top_worst.head(15).iterrows()):
        x_pos = row['Abs_Thickness_Delta'] * 0.5
        label = f"({row['BTM_X']:.0f}, {row['BTM_Y']:.0f})"
        ax2.text(x_pos, i, label, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
    
    ax2.set_xlabel('Absolute Thickness Difference (Å)')
    ax2.set_title('Top 15 Worst Coordinate Pairs\n(Ranked by |Delta|)')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'#{i+1}' for i in range(15)])
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Delta vs Distance from center
    ax3.scatter(df['Distance_from_center'], df['Abs_Thickness_Delta'], 
               alpha=0.6, s=40, c='blue', label='All coordinates')
    
    # Highlight worst coordinates
    ax3.scatter(top_10['Distance_from_center'], top_10['Abs_Thickness_Delta'], 
               s=100, c='red', marker='s', alpha=0.9, 
               edgecolors='black', linewidth=1.5, label='Top 10 worst')
    
    # Add trendline
    z = np.polyfit(df['Distance_from_center'], df['Abs_Thickness_Delta'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, df['Distance_from_center'].max(), 100)
    ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # Correlation
    corr = df['Distance_from_center'].corr(df['Abs_Thickness_Delta'])
    ax3.set_xlabel('Distance from Center (mm)')
    ax3.set_ylabel('Absolute Thickness Delta (Å)')
    ax3.set_title(f'Delta vs Distance from Center\nCorrelation: r = {corr:.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Wafer comparison
    wafer_stats = df.groupby('WaferID').agg({
        'Abs_Thickness_Delta': ['mean', 'max', 'count'],
        'Thickness_Delta': 'mean'
    }).round(1)
    wafer_stats.columns = ['Mean_|Delta|', 'Max_|Delta|', 'Count', 'Mean_Delta']
    wafer_stats = wafer_stats.sort_values('Mean_|Delta|', ascending=False)
    
    x_pos = range(len(wafer_stats))
    bars = ax4.bar(x_pos, wafer_stats['Mean_|Delta|'], 
                   color='lightcoral', alpha=0.7, edgecolor='black')
    
    # Add max values as error bars
    ax4.errorbar(x_pos, wafer_stats['Mean_|Delta|'],
                yerr=[np.zeros(len(wafer_stats)), 
                      wafer_stats['Max_|Delta|'] - wafer_stats['Mean_|Delta|']],
                fmt='none', color='red', capsize=5, capthick=2)
    
    ax4.set_xlabel('Wafer ID')
    ax4.set_ylabel('Mean |Delta| (Å)')
    ax4.set_title('Mean Absolute Delta by Wafer\n(Error bars show max values)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(wafer_stats.index, rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('coordinate_delta_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate separate ranking table
    print("\n📋 Creating detailed ranking table...")
    create_ranking_table(df, top_worst)
    
    # Generate detailed report
    print(f"\n📋 DETAILED REPORT: TOP 10 WORST COORDINATE PAIRS")
    print("=" * 70)
    
    report_data = []
    for idx, row in top_10.iterrows():
        report_data.append({
            'Rank': row['Rank'],
            'Coordinates': f"({row['BTM_X']:.0f}, {row['BTM_Y']:.0f})",
            'BTM_Thickness': f"{row['BTM_Thickness']:.0f}",
            'DMT_Thickness': f"{row['DMT_Thickness']:.0f}",
            'Delta': f"{row['Thickness_Delta']:+.1f}",
            'Abs_Delta': f"{row['Abs_Thickness_Delta']:.1f}",
            'Distance_Center': f"{row['Distance_from_center']:.0f}",
            'Match_Distance': f"{row['Distance_mm']:.2f}",
            'Wafer': row['WaferID']
        })
    
    report_df = pd.DataFrame(report_data)
    print(report_df.to_string(index=False))
    
    # Save detailed data
    top_worst.to_csv('worst_coordinates_detailed.csv', index=False)
    print(f"\n💾 Detailed data saved to: worst_coordinates_detailed.csv")
    
    # Pattern analysis
    edge_coords = top_10[top_10['Distance_from_center'] > 100]
    print(f"\n📊 PATTERN ANALYSIS (Top 10 worst coordinates):")
    print(f"   • Edge coordinates (>100mm): {len(edge_coords)}/10 ({len(edge_coords)*10}%)")
    print(f"   • Mean distance from center: {top_10['Distance_from_center'].mean():.1f} mm")
    print(f"   • Positive deltas (DMT > BTM): {len(top_10[top_10['Thickness_Delta'] > 0])}/10")
    print(f"   • Negative deltas (DMT < BTM): {len(top_10[top_10['Thickness_Delta'] < 0])}/10")
    
    # Wafer analysis
    worst_wafer = top_10['WaferID'].mode().iloc[0] if not top_10['WaferID'].mode().empty else "N/A"
    wafer_counts = top_10['WaferID'].value_counts()
    print(f"   • Most problematic wafer: {worst_wafer} ({wafer_counts.iloc[0]} coords in top 10)")
    
    print(f"\n✅ Analysis complete! Generated:")
    print(f"   • coordinate_delta_analysis.png - Main visualization")
    print(f"   • coordinate_ranking_table_standalone.png - Detailed ranking table")
    
    return top_worst

if __name__ == "__main__":
    analyze_coordinate_deltas()