import dash
from dash import html
import dash_bootstrap_components as dbc

# Create simple test app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Simple Dash Test", className="text-center"),
    html.P("If you can see this, Dash is working!")
])

if __name__ == '__main__':
    print("Starting simple Dash test app...")
    try:
        app.run(debug=False, host='127.0.0.1', port=8051)
    except Exception as e:
        print(f"Error: {e}")
        # Try without debug mode
        app.run(debug=False, host='0.0.0.0', port=8051)