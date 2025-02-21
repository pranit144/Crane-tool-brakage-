from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import base64
import os
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Create upload folder and logs folder
UPLOAD_FOLDER = 'uploads'
LOGS_FOLDER = 'logs'
for folder in [UPLOAD_FOLDER, LOGS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Setup logging
if not app.debug:
    file_handler = RotatingFileHandler('logs/maintenance_dashboard.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Maintenance Dashboard startup')

class DataStore:
    def __init__(self):
        self.uploaded_data = None
        self.current_index = 0
        self.processed_data = []
        self.anomalies = []

data_store = DataStore()

def detect_anomalies(df):
    """Checks each row of the dataframe for readings that exceed preset thresholds."""
    alerts = []
    thresholds = {
        'brakes': 90,
        'filters':90,
        'cables': 90
    }

    for idx, row in df.iterrows():
        row_alerts = []
        for component, threshold in thresholds.items():
            if component in df.columns:
                value = row[component]
                if pd.notna(value) and value > threshold:
                    row_alerts.append(
                        f"{component.capitalize()} reading ({value}) exceeds threshold ({threshold})"
                    )
        if row_alerts:
            alerts.append({
                'row': idx + 1,
                'messages': row_alerts
            })
    return alerts

def analyze_component_trends(df):
    """Analyze trends and patterns in component data"""
    trends = {}
    for component in ['brakes', 'filters', 'cables']:
        # Calculate rolling average to identify trends
        rolling_avg = df[component].rolling(window=3).mean()
        
        # Calculate rate of change
        rate_of_change = df[component].diff().mean()
        
        # Identify peak usage periods
        peak_threshold = df[component].quantile(0.75)
        peak_periods = df[component] > peak_threshold
        
        trends[component] = {
            'trend': 'Increasing' if rate_of_change > 1 else 'Decreasing' if rate_of_change < -1 else 'Stable',
            'rate_of_change': round(rate_of_change, 2),
            'peak_usage_frequency': round((peak_periods.sum() / len(df)) * 100, 1),
            'recent_trend': 'Up' if rolling_avg.iloc[-1] > rolling_avg.iloc[-2] else 'Down'
        }
    return trends

def generate_maintenance_insights(df, trends):
    """Generate detailed maintenance insights based on data analysis"""
    insights = {
        'critical_analysis': [],
        'maintenance_recommendations': [],
        'preventive_measures': [],
        'optimization_suggestions': []
    }
    
    for component in ['brakes', 'filters', 'cables']:
        avg = df[component].mean()
        max_val = df[component].max()
        std_dev = df[component].std()
        trend = trends[component]
        
        # Critical Analysis
        if avg > 70:
            insights['critical_analysis'].append({
                'component': component,
                'severity': 'High' if avg > 80 else 'Medium',
                'reason': f"Sustained high readings (avg: {round(avg, 1)}%)",
                'trend': trend['trend'],
                'impact': 'Immediate attention required' if avg > 80 else 'Monitor closely'
            })
        
        # Maintenance Recommendations
        if trend['trend'] == 'Increasing' and avg > 60:
            insights['maintenance_recommendations'].append({
                'component': component,
                'urgency': 'High' if avg > 75 else 'Medium',
                'action': f"Schedule maintenance within {' 24 hours' if avg > 75 else ' one week'}",
                'reason': f"Increasing trend with high average ({round(avg, 1)}%)"
            })
        
        # Preventive Measures
        if std_dev > 10 or trend['peak_usage_frequency'] > 30:
            insights['preventive_measures'].append({
                'component': component,
                'measure': f"Implement regular checks for {component}",
                'frequency': 'Daily' if std_dev > 15 else 'Weekly',
                'reason': f"High variability (±{round(std_dev, 1)}%) and frequent peak usage ({trend['peak_usage_frequency']}% of time)"
            })
        
        # Optimization Suggestions
        if trend['rate_of_change'] > 2 or max_val > 90:
            insights['optimization_suggestions'].append({
                'component': component,
                'suggestion': f"Review {component} usage patterns",
                'potential_impact': 'High',
                'expected_benefit': 'Reduced wear and extended component life'
            })
    
    return insights

def calculate_statistics(df):
    """Calculate important statistics from the data"""
    stats = {
        'component_averages': {
            'brakes': round(df['brakes'].mean(), 2),
            'filters': round(df['filters'].mean(), 2),
            'cables': round(df['cables'].mean(), 2)
        },
        'critical_components': [],
        'maintenance_suggestions': [],
        'component_health': {},  # New: Component health status
        'maintenance_priority': [],  # New: Prioritized maintenance list
        'performance_metrics': {},  # New: Detailed performance metrics
        'detailed_analysis': {}  # New: Detailed analysis
    }
    
    # Calculate component health and status
    for component in ['brakes', 'filters', 'cables']:
        avg = df[component].mean()
        max_val = df[component].max()
        min_val = df[component].min()
        std_dev = df[component].std()
        
        # Calculate health score (0-100)
        health_score = max(0, min(100, 100 - (avg / 100 * 100)))
        
        stats['component_health'][component] = {
            'health_score': round(health_score, 1),
            'average': round(avg, 2),
            'max_reading': round(max_val, 2),
            'min_reading': round(min_val, 2),
            'variability': round(std_dev, 2),
            'status': 'Good' if avg < 60 else 'Warning' if avg < 75 else 'Critical'
        }
        
        # Identify critical components
        if avg > 70:
            stats['critical_components'].append({
                'name': component,
                'avg_value': round(avg, 2),
                'max_value': round(max_val, 2),
                'health_score': round(health_score, 1),
                'status': 'Critical' if avg > 80 else 'Warning',
                'variability': round(std_dev, 2)
            })
    
    # Generate prioritized maintenance suggestions
    for component, health in stats['component_health'].items():
        if health['average'] > 80:
            stats['maintenance_priority'].append({
                'component': component,
                'priority': 'High',
                'timeline': 'Immediate',
                'reason': f"Critical readings (Avg: {health['average']}%)",
                'recommendation': f"Schedule immediate maintenance for {component}"
            })
        elif health['average'] > 70:
            stats['maintenance_priority'].append({
                'component': component,
                'priority': 'Medium',
                'timeline': 'Within 1 week',
                'reason': f"Warning levels (Avg: {health['average']}%)",
                'recommendation': f"Plan maintenance for {component} soon"
            })
        elif health['variability'] > 10:
            stats['maintenance_priority'].append({
                'component': component,
                'priority': 'Low',
                'timeline': 'Monitor',
                'reason': f"High variability (±{health['variability']}%)",
                'recommendation': f"Monitor {component} performance"
            })

    # Generate detailed maintenance suggestions
    for component, health in stats['component_health'].items():
        suggestions = []
        
        if health['average'] > 80:
            suggestions.append(f"URGENT: Immediate maintenance required - {component} showing critical wear")
        elif health['average'] > 70:
            suggestions.append(f"WARNING: Schedule maintenance soon - {component} performance degrading")
        
        if health['variability'] > 10:
            suggestions.append(f"Monitor {component} - Showing inconsistent readings (±{health['variability']}%)")
        
        if health['max_reading'] > 90:
            suggestions.append(f"Investigate {component} peak readings of {health['max_reading']}%")
        
        if suggestions:
            stats['maintenance_suggestions'].extend(suggestions)

    # Calculate performance metrics
    stats['performance_metrics'] = {
        'overall_health': round(sum(h['health_score'] for h in stats['component_health'].values()) / 3, 1),
        'critical_count': len([h for h in stats['component_health'].values() if h['status'] == 'Critical']),
        'warning_count': len([h for h in stats['component_health'].values() if h['status'] == 'Warning']),
        'healthy_count': len([h for h in stats['component_health'].values() if h['status'] == 'Good'])
    }
    
    # Add new detailed analyses
    trends = analyze_component_trends(df)
    maintenance_insights = generate_maintenance_insights(df, trends)
    
    stats['detailed_analysis'] = {
        'trends': trends,
        'insights': maintenance_insights
    }
    
    return stats

def create_graphs(df):
    """Create all visualization graphs"""
    graphs = {}
    
    # Gauge Charts for all components
    components = ['brakes', 'filters', 'cables']
    for component in components:
        latest_value = df[component].iloc[-1]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_value,
            delta={'reference': df[component].iloc[-2] if len(df) > 1 else latest_value},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Latest {component.title()} Reading"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#1f77b4"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                },
                'steps': [
                    {'range': [0, 60], 'color': "#3feb48"},
                    {'range': [60, 80], 'color': "#ebeb3f"},
                    {'range': [80, 100], 'color': "#eb3f3f"}
                ]
            }))
        gauge.update_layout(height=300)
        graphs[f'{component}_gauge'] = gauge.to_html(full_html=False)

    # Bar Chart for current readings with historical average
    current_values = [df[comp].iloc[-1] for comp in components]
    avg_values = [df[comp].mean() for comp in components]
    
    bar = go.Figure(data=[
        go.Bar(name='Current Reading', x=components, y=current_values),
        go.Bar(name='Historical Average', x=components, y=avg_values)
    ])
    bar.update_layout(
        title="Component Readings Comparison",
        barmode='group',
        height=400
    )
    graphs['bar'] = bar.to_html(full_html=False)

    # Time Series Chart with Moving Average
    fig_time = go.Figure()
    for component in components:
        # Add raw data
        fig_time.add_trace(go.Scatter(
            y=df[component],
            name=component.title(),
            mode='lines'
        ))
        # Add moving average
        ma = df[component].rolling(window=3).mean()
        fig_time.add_trace(go.Scatter(
            y=ma,
            name=f"{component.title()} MA",
            line=dict(dash='dash'),
            opacity=0.5
        ))
    
    fig_time.update_layout(
        title='Component Readings Over Time',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    graphs['timeseries'] = fig_time.to_html(full_html=False)

    # Correlation Matrix Heatmap
    corr_matrix = df[components].corr()
    heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=components,
        y=components,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    heatmap.update_layout(
        title='Component Correlation Matrix',
        height=400
    )
    graphs['heatmap'] = heatmap.to_html(full_html=False)

    # Box Plot for Distribution Analysis
    box_data = [go.Box(y=df[component], name=component.title()) for component in components]
    box_plot = go.Figure(data=box_data)
    box_plot.update_layout(
        title='Component Value Distributions',
        height=400
    )
    graphs['box_plot'] = box_plot.to_html(full_html=False)

    # Scatter Matrix
    scatter_matrix = px.scatter_matrix(
        df[components],
        dimensions=components,
        title='Component Relationships Matrix'
    )
    scatter_matrix.update_layout(height=600)
    graphs['scatter_matrix'] = scatter_matrix.to_html(full_html=False)

    return graphs

@app.route('/')
def index():
    return render_template('index.html', 
                         data=None, 
                         graphs=None, 
                         stats=None, 
                         anomalies=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    try:
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            flash('Unsupported file format. Please upload CSV or Excel file.', 'error')
            return redirect(url_for('index'))

        # Validate required columns
        required_columns = ['brakes', 'filters', 'cables']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            flash(f"Missing required columns: {', '.join(missing_columns)}", 'error')
            return redirect(url_for('index'))

        # Store the data and process it
        data_store.uploaded_data = df
        data_store.anomalies = detect_anomalies(df)
        
        # Calculate statistics
        stats = calculate_statistics(df)
        
        # Create graphs
        graphs = create_graphs(df)

        # Render template with all the data
        return render_template('index.html',
                             data=df.to_dict('records'),
                             graphs=graphs,
                             stats=stats,
                             anomalies=data_store.anomalies)

    except Exception as e:
        flash(f"Error processing file: {str(e)}", 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
