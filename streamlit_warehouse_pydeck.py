import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import base64
import math
import random

# Set page configuration
st.set_page_config(
    page_title="Warehouse Optimizer",
    page_icon="🏭",
    layout="wide"
)

# Title and description
st.title("Warehouse Location Optimizer")
st.markdown("""
This application helps determine optimal warehouse locations based on store locations and their sales volumes.
Upload your store data, select the number of warehouses, and see the optimized locations.
""")

# Function to calculate distance between two points (haversine formula)
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Function to generate sample data
def generate_sample_data(num_stores=100):
    np.random.seed(42)
    
    # Define state boundaries (approximate, for random generation)
    regions = {
        'Northeast': {'lat_range': (40.0, 47.0), 'lon_range': (-80.0, -67.0), 'store_pct': 0.2},
        'Southeast': {'lat_range': (25.0, 39.0), 'lon_range': (-92.0, -75.0), 'store_pct': 0.25},
        'Midwest': {'lat_range': (36.0, 49.0), 'lon_range': (-104.0, -80.0), 'store_pct': 0.25},
        'West': {'lat_range': (32.0, 49.0), 'lon_range': (-124.0, -104.0), 'store_pct': 0.3}
    }
    
    data = []
    for region, bounds in regions.items():
        n_stores = int(num_stores * bounds['store_pct'])
        for _ in range(n_stores):
            lat = np.random.uniform(bounds['lat_range'][0], bounds['lat_range'][1])
            lon = np.random.uniform(bounds['lon_range'][0], bounds['lon_range'][1])
            sales = int(np.random.uniform(10000, 1000000))
            data.append({'Latitude': lat, 'Longitude': lon, 'Sales': sales, 'Region': region})
    
    # Make sure we have exactly num_stores
    if len(data) < num_stores:
        # Add more to the West if needed to reach the target
        for _ in range(num_stores - len(data)):
            region = 'West'
            bounds = regions[region]
            lat = np.random.uniform(bounds['lat_range'][0], bounds['lat_range'][1])
            lon = np.random.uniform(bounds['lon_range'][0], bounds['lon_range'][1])
            sales = int(np.random.uniform(10000, 1000000))
            data.append({'Latitude': lat, 'Longitude': lon, 'Sales': sales, 'Region': region})
    
    return pd.DataFrame(data)

# Custom K-means function to avoid sklearn dependency
def custom_kmeans(X, n_clusters, max_iter=100, sales_weights=None):
    """
    Simple implementation of K-means clustering
    
    Args:
        X: Array of shape (n_samples, n_features) - in this case, latitude and longitude
        n_clusters: Number of clusters to form
        max_iter: Maximum number of iterations
        sales_weights: Optional weights for each point (based on sales)
        
    Returns:
        centroids: Array of final centroids
        labels: Cluster labels for each point
    """
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly from the data points
    if sales_weights is not None:
        # Normalize weights
        sales_weights = sales_weights / np.sum(sales_weights)
        # Sample based on weights
        idx = np.random.choice(n_samples, size=n_clusters, replace=False, p=sales_weights)
    else:
        idx = np.random.choice(n_samples, size=n_clusters, replace=False)
        
    centroids = X[idx]
    
    # Initialize labels
    labels = np.zeros(n_samples, dtype=int)
    
    for iteration in range(max_iter):
        # Assign points to closest centroid
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            # Calculate Euclidean distance to each centroid
            # (We use Euclidean for clustering, then calculate Haversine later for real distances)
            distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
        
        # Get the closest centroid for each point
        new_labels = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.array_equal(labels, new_labels):
            break
            
        labels = new_labels
        
        # Update centroids
        for i in range(n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                if sales_weights is not None:
                    # Apply weights for centroid calculation
                    weighted_sum = np.sum(X[mask] * sales_weights[mask].reshape(-1, 1), axis=0)
                    weight_sum = np.sum(sales_weights[mask])
                    centroids[i] = weighted_sum / weight_sum if weight_sum > 0 else weighted_sum
                else:
                    centroids[i] = np.mean(X[mask], axis=0)
    
    return centroids, labels

# Function to optimize warehouse locations
def optimize_warehouse_locations(data, num_warehouses, sales_weight=0.5):
    # Extract coordinates for clustering
    X = data[['Latitude', 'Longitude']].values
    
    # Prepare sales weights if needed
    if sales_weight != 0.5:
        # Normalize sales to get weights
        sales = data['Sales'].values
        weights = sales / np.max(sales)
        
        # Adjust weights based on the sales_weight parameter
        weights = weights ** (sales_weight * 2)
    else:
        weights = None
    
    # Run clustering
    centroids, labels = custom_kmeans(X, num_warehouses, max_iter=100, sales_weights=weights)
    
    # Add cluster labels to the data
    data['Cluster'] = labels
    
    # Create warehouse dataframe
    warehouse_locations = pd.DataFrame(centroids, columns=['Latitude', 'Longitude'])
    warehouse_locations['Warehouse_ID'] = warehouse_locations.index
    
    # Calculate distance from each store to its assigned warehouse
    for i, row in data.iterrows():
        warehouse = warehouse_locations.iloc[row['Cluster']]
        distance = haversine_distance(
            row['Latitude'], row['Longitude'],
            warehouse['Latitude'], warehouse['Longitude']
        )
        data.loc[i, 'Distance_km'] = distance
    
    # Calculate metrics per warehouse
    metrics = []
    for i in range(num_warehouses):
        cluster_stores = data[data['Cluster'] == i]
        metrics.append({
            'Warehouse_ID': i,
            'Num_Stores': len(cluster_stores),
            'Total_Sales': cluster_stores['Sales'].sum(),
            'Avg_Distance_km': cluster_stores['Distance_km'].mean() if len(cluster_stores) > 0 else 0,
            'Max_Distance_km': cluster_stores['Distance_km'].max() if len(cluster_stores) > 0 else 0,
            'Min_Distance_km': cluster_stores['Distance_km'].min() if len(cluster_stores) > 0 else 0
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    return data, warehouse_locations, metrics_df

# Function to create a download link for a dataframe
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Sidebar
st.sidebar.header("Configuration")

# Option to use sample data
use_sample = st.sidebar.checkbox("Use sample data", value=True)

if not use_sample:
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with store data", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            # Check if required columns exist
            required_cols = ['Latitude', 'Longitude', 'Sales']
            if not all(col in data.columns for col in required_cols):
                st.sidebar.error(f"CSV must contain columns: {', '.join(required_cols)}")
                data = None
            else:
                st.sidebar.success(f"Successfully loaded {len(data)} stores.")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            data = None
    else:
        data = None
else:
    # Generate sample data
    data = generate_sample_data(100)
    st.sidebar.success("Using sample data with 100 stores across the US.")

# Number of warehouses slider
num_warehouses = st.sidebar.slider("Number of Warehouses", min_value=1, max_value=10, value=3)

# Sales weight slider (for optimization balance between sales and distance)
sales_weight = st.sidebar.slider(
    "Optimization Balance", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5,
    step=0.1,
    help="0 = Optimize for distance only, 1 = Heavily favor high-sales stores"
)

# Main content
if data is not None:
    # Display raw data
    with st.expander("View Raw Store Data"):
        st.dataframe(data)
    
    # Run optimization on button click
    if st.button("Optimize Warehouse Locations"):
        with st.spinner("Optimizing warehouse locations..."):
            # Run the optimization
            try:
                store_data, warehouse_locations, metrics_df = optimize_warehouse_locations(
                    data.copy(), 
                    num_warehouses,
                    sales_weight
                )
                
                # Show results in tabs
                tab1, tab2, tab3 = st.tabs(["Map", "Metrics", "Download Results"])
                
                with tab1:
                    st.subheader("Store and Warehouse Locations")
                    
                    # Prepare data for PyDeck
                    store_data_for_map = store_data.copy()
                    # Normalize sales for sizing on map
                    max_sales = store_data_for_map['Sales'].max()
                    store_data_for_map['Sales_Normalized'] = store_data_for_map['Sales'] / max_sales
                    
                    # Create tooltip function
                    store_data_for_map['tooltip'] = store_data_for_map.apply(
                        lambda row: f"Store in {row['Region']}<br>Sales: ${int(row['Sales']):,}<br>Distance to Warehouse: {row['Distance_km']:.2f} km", 
                        axis=1
                    )
                    
                    # Create colors for clusters
                    cluster_colors = [
                        [239, 83, 80],  # Red
                        [66, 165, 245], # Blue
                        [76, 175, 80],  # Green
                        [171, 71, 188], # Purple
                        [255, 167, 38], # Orange
                        [77, 208, 225], # Teal
                        [255, 87, 34],  # Deep Orange
                        [141, 110, 99], # Brown
                        [0, 137, 123],  # Dark Teal
                        [63, 81, 181]   # Indigo
                    ]
                    
                    # Create a color column for each store based on its cluster
                    store_data_for_map['color'] = store_data_for_map['Cluster'].apply(
                        lambda x: cluster_colors[x % len(cluster_colors)]
                    )
                    
                    # Create warehouse data for map
                    warehouse_data_for_map = warehouse_locations.copy()
                    warehouse_data_for_map['color'] = warehouse_data_for_map['Warehouse_ID'].apply(
                        lambda x: cluster_colors[x % len(cluster_colors)]
                    )
                    
                    # Calculate a zoom level and center point
                    # Simple approach: center on mean lat/lon and use a fixed zoom
                    center_lat = store_data_for_map['Latitude'].mean()
                    center_lon = store_data_for_map['Longitude'].mean()
                    
                    # Create layers
                    # Store layer
                    store_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=store_data_for_map,
                        get_position=["Longitude", "Latitude"],
                        get_color="color",
                        get_radius=["Sales_Normalized * 50000", "Sales_Normalized * 50000"],
                        pickable=True,
                        opacity=0.8,
                        stroked=True,
                        filled=True,
                        auto_highlight=True,
                        id="store-layer"
                    )
                    
                    # Warehouse layer
                    warehouse_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=warehouse_data_for_map,
                        get_position=["Longitude", "Latitude"],
                        get_color="color",
                        get_radius=100000,  # Larger radius for warehouses
                        pickable=True,
                        opacity=0.9,
                        stroked=True,
                        filled=True,
                        auto_highlight=True,
                        id="warehouse-layer"
                    )
                    
                    # Create connections between warehouses and stores
                    # This requires creating a dataset of line segments
                    connection_data = []
                    for _, store in store_data_for_map.iterrows():
                        warehouse = warehouse_data_for_map[warehouse_data_for_map['Warehouse_ID'] == store['Cluster']].iloc[0]
                        connection_data.append({
                            'source_lat': store['Latitude'],
                            'source_lon': store['Longitude'],
                            'target_lat': warehouse['Latitude'],
                            'target_lon': warehouse['Longitude'],
                            'color': store['color']
                        })
                    
                    connection_df = pd.DataFrame(connection_data)
                    
                    # Create line layer
                    line_layer = pdk.Layer(
                        "LineLayer",
                        data=connection_df,
                        get_source_position=["source_lon", "source_lat"],
                        get_target_position=["target_lon", "target_lat"],
                        get_color="color",
                        get_width=3,
                        opacity=0.3,
                        pickable=False,
                    )
                    
                    # Create view state
                    view_state = pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lon,
                        zoom=3.5,
                        pitch=30,
                        bearing=0
                    )
                    
                    # Create tooltips
                    tooltip = {
                        "html": "<b>Info:</b> <br/> {tooltip} <br/>",
                        "style": {
                            "backgroundColor": "white",
                            "color": "black"
                        }
                    }
                    
                    # Create deck
                    r = pdk.Deck(
                        layers=[line_layer, store_layer, warehouse_layer],
                        initial_view_state=view_state,
                        map_style="light",
                        tooltip=tooltip
                    )
                    
                    # Display the map
                    st.pydeck_chart(r)
                    
                    # Legend
                    st.markdown("""
                    **Map Legend:**
                    - Circles: Stores (size indicates sales volume)
                    - Larger Circles: Warehouse locations
                    - Lines: Connection between stores and their assigned warehouse
                    - Colors: Warehouse assignment
                    """)
                
                with tab2:
                    st.subheader("Warehouse Performance Metrics")
                    
                    # Format metrics for display
                    formatted_metrics = metrics_df.copy()
                    formatted_metrics['Total_Sales'] = formatted_metrics['Total_Sales'].apply(lambda x: f"${x:,.2f}")
                    formatted_metrics['Avg_Distance_km'] = formatted_metrics['Avg_Distance_km'].apply(lambda x: f"{x:.2f}")
                    formatted_metrics['Max_Distance_km'] = formatted_metrics['Max_Distance_km'].apply(lambda x: f"{x:.2f}")
                    formatted_metrics['Min_Distance_km'] = formatted_metrics['Min_Distance_km'].apply(lambda x: f"{x:.2f}")
                    
                    # Display metrics table
                    st.dataframe(formatted_metrics)
                    
                    # Create bar charts for metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart for number of stores
                        stores_chart_data = metrics_df.copy()
                        st.bar_chart(
                            stores_chart_data.set_index('Warehouse_ID')['Num_Stores'],
                            use_container_width=True
                        )
                        st.caption("Number of Stores per Warehouse")
                    
                    with col2:
                        # Bar chart for total sales
                        sales_chart_data = metrics_df.copy()
                        st.bar_chart(
                            sales_chart_data.set_index('Warehouse_ID')['Total_Sales'],
                            use_container_width=True
                        )
                        st.caption("Total Sales per Warehouse ($)")
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # Bar chart for average distance
                        distance_chart_data = metrics_df.copy()
                        st.bar_chart(
                            distance_chart_data.set_index('Warehouse_ID')['Avg_Distance_km'],
                            use_container_width=True
                        )
                        st.caption("Average Distance to Warehouse (km)")
                    
                    with col4:
                        # Histogram of distances (using a bar chart as proxy)
                        distance_bins = pd.cut(store_data['Distance_km'], bins=10)
                        distance_counts = distance_bins.value_counts().sort_index()
                        st.bar_chart(distance_counts, use_container_width=True)
                        st.caption("Distribution of Store-to-Warehouse Distances")
                
                with tab3:
                    st.subheader("Download Results")
                    
                    # Provide download links
                    st.markdown(get_csv_download_link(
                        warehouse_locations, 
                        'warehouse_locations.csv',
                        'Download Warehouse Locations as CSV'
                    ), unsafe_allow_html=True)
                    
                    st.markdown(get_csv_download_link(
                        store_data,
                        'store_assignments.csv',
                        'Download Store Assignments as CSV'
                    ), unsafe_allow_html=True)
                    
                    # Also display the warehouse coordinates
                    st.subheader("Warehouse Coordinates")
                    st.dataframe(warehouse_locations[['Warehouse_ID', 'Latitude', 'Longitude']])
                
            except Exception as e:
                st.error(f"An error occurred during optimization: {str(e)}")
                st.error("Error details:")
                st.exception(e)
else:
    if not use_sample:
        st.info("Please upload a CSV file with store data or use the sample data option.")
