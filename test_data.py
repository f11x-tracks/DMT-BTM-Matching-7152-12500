import pandas as pd
import xml.etree.ElementTree as ET
from scipy import stats
import numpy as np

def test_data_loading():
    print("Testing BTM data loading...")
    try:
        btm_df = pd.read_csv('BTM/BTM-matching-7152-12500A.csv')
        print(f"BTM data loaded: {len(btm_df)} records")
        print("BTM columns:", btm_df.columns.tolist())
        print("BTM sample data:")
        print(btm_df.head(3))
    except Exception as e:
        print(f"Error loading BTM data: {e}")
        return False
    
    print("\nTesting DMT data loading...")
    try:
        tree = ET.parse('DMT/2026-03-26T07.51.50.4648236-LN1718SS29-8333-DMT116-TZJ501-DMTDUMMY.xml')
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
                        'Y_mm': float(y_loc)
                    })
        
        dmt_df = pd.DataFrame(records)
        print(f"DMT data loaded: {len(dmt_df)} records")
        print("DMT sample data:")
        print(dmt_df.head(3))
    except Exception as e:
        print(f"Error loading DMT data: {e}")
        return False
    
    print("\nTesting statistical analysis...")
    try:
        # Create sample data for statistical tests
        sample1 = np.random.normal(12400, 50, 100)
        sample2 = np.random.normal(12450, 60, 100)
        
        kruskal_stat, kruskal_pvalue = stats.kruskal(sample1, sample2)
        mannwhitney_stat, mannwhitney_pvalue = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
        
        print(f"Kruskal-Wallis test: statistic={kruskal_stat:.4f}, p-value={kruskal_pvalue:.2e}")
        print(f"Mann-Whitney U test: statistic={mannwhitney_stat:.0f}, p-value={mannwhitney_pvalue:.2e}")
        
    except Exception as e:
        print(f"Error in statistical tests: {e}")
        return False
    
    print("\nAll tests passed successfully!")
    return True

if __name__ == "__main__":
    test_data_loading()