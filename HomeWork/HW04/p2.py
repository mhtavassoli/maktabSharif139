import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Preparation
def load_and_prepare_data():
    """
    Load and prepare the climate dataset
    """
    # Load the dataset with proper date parsing
    df = pd.read_csv(
        r'E:\Maktab\Artificial Intelligence\Programming\HomeWork\HW04\p2\GlobalLandTemperaturesByCountry.csv'
        , parse_dates=['dt'])
    
    # Extract year from date
    df['year'] = df['dt'].dt.year
    
    # 1.1 Handle NaN values in AverageTemperature
    print(f"Original dataset shape: {df.shape}")
    print(f"Number of NaN values in temperature: {df['AverageTemperature'].isna().sum()}")
    
    # First approach: Remove rows with NaN values in temperature
    df_clean = df.dropna(subset=['AverageTemperature'])
    
    # Alternative approach: Fill NaN values with 5-year rolling average
    df_filled = df.copy()
    # Sort by date to ensure proper rolling calculation
    df_filled = df_filled.sort_values('dt')
    # Fill NaN temperatures with rolling average (5 years = 60 months)
    df_filled['AverageTemperature'] = df_filled['AverageTemperature'].fillna(
        df_filled['AverageTemperature'].rolling(window=60, min_periods=1).mean()
    )
    
    # Use the filled dataset for further analysis
    df_processed = df_filled
    
    # 1.2 Add Decade column
    df_processed['Decade'] = (df_processed['year'] // 10) * 10
    df_processed['Decade'] = df_processed['Decade'].astype(str) + 's'
    
    # 1.3 Filter countries with at least 100 years of data
    # Count unique years per country
    country_years = df_processed.groupby('Country')['year'].nunique()
    valid_countries = country_years[country_years >= 100].index
    df_filtered = df_processed[df_processed['Country'].isin(valid_countries)]
    
    print(f"Dataset shape after filtering: {df_filtered.shape}")
    print(f"Number of valid countries: {len(valid_countries)}")
    print(f"Data range: {df_filtered['year'].min()} - {df_filtered['year'].max()}")
    
    return df_filtered

# Step 2: Numerical Analysis with NumPy
def numerical_analysis(df):
    """
    Perform numerical analysis using NumPy
    """
    # 2.1 Calculate mean temperature per decade
    decade_temp = df.groupby('Decade')['AverageTemperature'].mean()
    decades = decade_temp.index
    temperatures = decade_temp.values
    
    print(f"Available decades: {list(decades)}")
    print(f"Temperatures: {temperatures}")
    
    # Create decade × temperature anomaly matrix
    decade_matrix = np.column_stack([np.arange(len(decades)), temperatures])
    
    # 2.2 Calculate linear temperature trend
    # Convert decade strings to numerical values (e.g., "1980s" -> 1980)
    decade_numeric = np.array([int(d.replace('s', '')) for d in decades])
    
    # Perform linear regression
    slope, intercept = np.polyfit(decade_numeric, temperatures, 1)
    
    # Calculate warming rate per decade
    warming_rate_per_decade = slope * 10  # Convert from per year to per decade
    
    print(f"Linear regression: slope={slope:.6f}, intercept={intercept:.4f}")
    print(f"Warming rate per decade: {warming_rate_per_decade:.4f}°C")
    
    # 2.3 Temperature prediction until 2050
    current_year = decade_numeric[-1]
    future_years = np.arange(current_year + 10, 2051, 10)
    
    # Base scenario (current rate)
    base_predictions = intercept + slope * future_years
    
    # Optimistic scenario (30% reduction in rate)
    optimistic_slope = slope * 0.7
    optimistic_predictions = intercept + optimistic_slope * future_years
    
    # Pessimistic scenario (50% increase in rate)
    pessimistic_slope = slope * 1.5
    pessimistic_predictions = intercept + pessimistic_slope * future_years
    
    print(f"Future predictions until 2050:")
    print(f"Years: {future_years}")
    print(f"Base scenario: {base_predictions}")
    
    return (decade_numeric, temperatures, slope, intercept, warming_rate_per_decade,
            future_years, base_predictions, optimistic_predictions, pessimistic_predictions)

# Step 3: Visualization
def create_visualizations(decade_numeric, temperatures, slope, intercept, 
                         future_years, base_predictions, optimistic_predictions, 
                         pessimistic_predictions, df):
    """
    Create all required visualizations
    """
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # 3.1 Temperature plot with trend line
    ax1.scatter(decade_numeric, temperatures, alpha=0.7, label='Actual Data', color='blue')
    
    # Plot trend line
    trend_line = intercept + slope * decade_numeric
    ax1.plot(decade_numeric, trend_line, 'r-', linewidth=2, 
             label=f'Trend (slope: {slope:.4f}°C/year)')
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Temperature (°C)')
    ax1.set_title('Global Temperature Trend')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3.2 Heatmap of decade changes
    # Calculate temperature changes per decade
    temp_changes = []
    decades_list = []
    for i in range(1, len(temperatures)):
        change = temperatures[i] - temperatures[i-1]
        temp_changes.append(change)
        decades_list.append(f"{decade_numeric[i-1]}-{decade_numeric[i]}")
    
    # Create heatmap data
    if temp_changes:
        heatmap_data = np.array(temp_changes).reshape(-1, 1)
        
        im = ax2.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Temp Change'])
        ax2.set_yticks(range(len(decades_list)))
        ax2.set_yticklabels(decades_list)
        ax2.set_title('Decadal Temperature Changes Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Temperature Change (°C)')
        
        # Add values to heatmap cells
        for i in range(len(temp_changes)):
            ax2.text(0, i, f'{temp_changes[i]:.2f}', 
                    ha='center', va='center', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data\nfor heatmap', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Decadal Temperature Changes Heatmap')
    
    # 3.3 Three scenarios until 2050
    # Combine historical and future data for better visualization
    all_historical_years = np.array(decade_numeric)
    all_historical_temps = np.array(temperatures)
    
    ax3.plot(all_historical_years, all_historical_temps, 'ko-', 
             label='Historical Data', linewidth=2, markersize=4)
    ax3.plot(future_years, base_predictions, 'b--', 
             label='Base Scenario', linewidth=2, marker='o')
    ax3.plot(future_years, optimistic_predictions, 'g--', 
             label='Optimistic (-30%)', linewidth=2, marker='s')
    ax3.plot(future_years, pessimistic_predictions, 'r--', 
             label='Pessimistic (+50%)', linewidth=2, marker='^')
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Average Temperature (°C)')
    ax3.set_title('Temperature Projections until 2050')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3.4 Additional analysis - Temperature distribution by decade
    recent_decades = df[df['Decade'].isin([d for d in df['Decade'].unique()][-6:])]
    boxplot_data = []
    boxplot_labels = []
    
    for decade in sorted(recent_decades['Decade'].unique()):
        decade_data = recent_decades[recent_decades['Decade'] == decade]['AverageTemperature']
        if len(decade_data) > 0:
            boxplot_data.append(decade_data)
            boxplot_labels.append(decade)
    
    if boxplot_data:
        ax4.boxplot(boxplot_data, labels=boxplot_labels)
        ax4.set_xlabel('Decade')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('Temperature Distribution by Decade')
        ax4.tick_params(axis='x', rotation=45)
    else:
        # If no boxplot data, show basic statistics
        avg_temp_by_country = df.groupby('Country')['AverageTemperature'].mean().sort_values(ascending=False)
        top_countries = avg_temp_by_country.head(10)
        
        ax4.barh(range(len(top_countries)), top_countries.values)
        ax4.set_yticks(range(len(top_countries)))
        ax4.set_yticklabels(top_countries.index)
        ax4.set_xlabel('Average Temperature (°C)')
        ax4.set_title('Top 10 Countries by Average Temperature')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('climate_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution function
def main():
    """
    Main function to execute the complete climate analysis
    """
    print("Starting Climate Data Analysis...")
    
    try:
        # Step 1: Data Preparation
        print("Step 1: Loading and preparing data...")
        df = load_and_prepare_data()
        
        if df.empty:
            print("Error: No data available after preprocessing!")
            return
        
        # Step 2: Numerical Analysis
        print("Step 2: Performing numerical analysis...")
        results = numerical_analysis(df)
        
        if results is None:
            print("Error: Numerical analysis failed!")
            return
            
        (decade_numeric, temperatures, slope, intercept, warming_rate_per_decade,
         future_years, base_predictions, optimistic_predictions, 
         pessimistic_predictions) = results
        
        # Step 3 & 4: Visualization
        print("Step 3 & 4: Creating visualizations...")
        create_visualizations(decade_numeric, temperatures, slope, intercept,
                            future_years, base_predictions, optimistic_predictions,
                            pessimistic_predictions, df)
        
        print("Analysis completed successfully!")
        print(f"Results saved as 'climate_analysis_results.png'")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

# Additional utility functions
def calculate_statistics(df):
    """
    Calculate additional statistics for the dataset
    """
    if df.empty:
        return {}
        
    stats = {
        'total_countries': df['Country'].nunique(),
        'data_range': f"{df['year'].min()} - {df['year'].max()}",
        'total_years': df['year'].nunique(),
        'mean_temperature': df['AverageTemperature'].mean(),
        'temperature_std': df['AverageTemperature'].std(),
        'total_records': len(df)
    }
    return stats

if __name__ == "__main__":
    # Execute the main analysis
    main()
    
    # Calculate and print additional statistics
    try:
        df = load_and_prepare_data()
        if not df.empty:
            stats = calculate_statistics(df)
            print("\nDataset Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")