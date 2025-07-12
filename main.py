from joblib.memory import pformat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import json
import joblib
import re
from flask import Flask, request, render_template
import plotly.graph_objs as go
import sympy as sp
from sympy import symbols, lambdify, Eq, parse_expr, latex
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
coord_model = joblib.load("coord_system_model.pkl")

# Load dataset once globally
def load_dataset(file_path="dataset.json"):
    with open(file_path, "r") as file:
        return json.load(file)

dataset = load_dataset()

# Utility function to extract equations (optional right now)
def extract_equations(text: str) -> list:
    equation_pattern = r'"([^"]*)"'
    equations = re.findall(equation_pattern, text)
    return equations

# Visualize equations
def plot_explicit_surface(eq_func, x_range=(-5, 5), y_range=(-5, 5)):
    x = np.linspace(*x_range, 50)
    y = np.linspace(*y_range, 50)
    X, Y = np.meshgrid(x, y)
    Z = eq_func(X, Y)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, showscale=False)])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    return fig.to_html(full_html=False)

def plot_implicit_surface(expr_func, x_range=(-5, 5), y_range=(-5, 5), z_range=(-5, 5)):
    x = np.linspace(*x_range, 30)
    y = np.linspace(*y_range, 30)
    z = np.linspace(*z_range, 30)
    X, Y, Z = np.meshgrid(x, y, z)
    values = expr_func(X, Y, Z)

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=0,
        isomax=0,
        surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    return fig.to_html(full_html=False)

def plot_equations(equations):
    x, y, z, r, theta, phi = symbols('x y z r theta phi')
    fig = go.Figure()

    for eq in equations:
        eq_clean = eq.replace('^', '**').strip()

        eq_clean = eq_clean.replace(">=", "=").replace("<=", "=").replace(">", "=").replace("<", "=")

        # Skip inequalities
        if any(op in eq_clean for op in ['>=', '<=', '>', '<']):
            print(f"Skipping inequality: {eq_clean}")
            continue

        # Explicit z = constant (plane)
        if re.fullmatch(r"z\s*=\s*[-+]?\d+(\.\d+)?", eq_clean):
            constant = float(eq_clean.split('=')[1].strip())
            x_vals = np.linspace(-5, 5, 50)
            y_vals = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = np.full_like(X, constant)

            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, showscale=False, opacity=0.5
            ))

        # Explicit z = f(x, y)
        elif re.match(r"z\s*=", eq_clean):
            rhs = eq_clean.split('=')[1].strip()
            expr = parse_expr(rhs)
            func = lambdify((x, y), expr, 'numpy')

            x_vals = np.linspace(-5, 5, 50)
            y_vals = np.linspace(-5, 5, 50)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = func(X, Y)

            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, showscale=False, opacity=0.5
            ))

        # Implicit surfaces like x^2 + y^2 + z^2 = 16
        elif '=' in eq_clean:
            lhs, rhs = eq_clean.split('=')
            lhs_expr = parse_expr(lhs.strip())
            rhs_expr = parse_expr(rhs.strip())
            expr = lhs_expr - rhs_expr
            func = lambdify((x, y, z), expr, 'numpy')

            x_vals = np.linspace(-5, 5, 30)
            y_vals = np.linspace(-5, 5, 30)
            z_vals = np.linspace(-5, 5, 30)
            X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
            values = func(X, Y, Z)

            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                isomin=0,
                isomax=0,
                surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                opacity=0.5
            ))

        # Handle polar equations of the form r = f(theta)
        elif re.match(r"r\s*=", eq_clean):
            rhs = eq_clean.split('=')[1].strip()
            try:
                expr = parse_expr(rhs)
                func = lambdify(theta, expr, 'numpy')

                theta_vals = np.linspace(0, 2*np.pi, 200)
                R = func(theta_vals)
                X = R * np.cos(theta_vals)
                Y = R * np.sin(theta_vals)

                fig.add_trace(go.Scatter3d(
                    x=X, y=Y, z=np.zeros_like(X),
                    mode='lines',
                    line=dict(width=5),
                    name=f"{eq_clean}"
                ))
            except Exception as e:
                print(f"Error parsing polar equation {eq_clean}: {e}")
                continue  # Skip this equation if parsing fails

        else:
            print(f"Unknown or unsupported equation format: {eq}")

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    return fig.to_html(full_html=False)

    
# Predict integral setup based on predicted coordinate system

def predict_integral_setup(user_question, predicted_coord_system, dataset):
    # Filter dataset by the predicted coordinate system
    filtered_dataset = [
        item for item in dataset
        if item["coordinate_system"] == predicted_coord_system
    ]

    if not filtered_dataset:
        return None  # No matching entries found

    texts = [item["question_text"] for item in filtered_dataset]

    temp_vectorizer = TfidfVectorizer()
    X_filtered = temp_vectorizer.fit_transform(texts)

    user_vec = temp_vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vec, X_filtered)
    best_match_idx = similarities.argmax()

    return filtered_dataset[best_match_idx]["integral_setup"]

def compute_volume(bounds_dict, coordinate_system):

    # Define symbols
    rho, theta, phi, x, y, z = sp.symbols('rho theta phi x y z')

    # Define math functions
    locals_dict = {
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'sqrt': sp.sqrt,
        'pi': sp.pi,
        'exp': sp.exp,
        'ln': sp.log
    }

    # Set up coordinate-specific variables and Jacobian
    if coordinate_system == "spherical":
        variables = [rho, phi, theta]  # Correct integration order
        jacobian = rho**2 * sp.sin(phi)
    elif coordinate_system == "cylindrical":
        r, theta_cyl, z = sp.symbols('r theta z')  # define separately
        variables = [z, r, theta_cyl]  # z first, then r, then theta
        jacobian = r
    else:  # cartesian
        variables = [z, y, x]
        jacobian = 1

    # Integrand is just the Jacobian
    full_integrand = jacobian

    # Nested integration
    expr = full_integrand
    for var in variables:
        var_name = str(var)
        var_bounds = bounds_dict.get(var_name)
        if var_bounds:
            lower = sp.sympify(var_bounds["lower"], locals=locals_dict)
            upper = sp.sympify(var_bounds["upper"], locals=locals_dict)
            expr = sp.integrate(expr, (var, lower, upper))

    volume_sym = sp.simplify(expr)
    
    # Generate LaTeX and clean it properly
    volume_latex = sp.latex(volume_sym)
    
    # Systematic cleaning of LaTeX output
    volume_latex = (volume_latex
        .replace(r'\text{`', '')  # Remove text artifacts
        .replace(r'`}', '')       # Remove remaining backticks
        .replace(r'\{', '{')      # Fix escaped braces
        .replace(r'\}', '}')      # Fix escaped braces
        .replace(r'\ ', ' ')      # Fix forced spaces
        .replace('[', '')         # Remove brackets
        .replace(']', '')         # Remove brackets
    )
    
    return volume_sym, volume_latex

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    question = None
    confidence_scores = {}
    equations = []
    integral_setup = None
    plot_html = None
    volume = None
    volume_latex = None
    error_message = None  # Add this line

    if request.method == "POST":
        question = request.form["question"]
        
        try:  # Wrap all processing in try-except
            prediction = coord_model.predict([question])[0]
            probs = coord_model.predict_proba([question])[0]

            extracted = extract_equations(question)
            equations = extracted

            integral_setup = predict_integral_setup(question, prediction, dataset)

            if integral_setup:
                volume_sym, volume_latex = compute_volume(integral_setup["bounds"], prediction)
                
                # Additional safety check
                try:
                    # Verify the LaTeX compiles by parsing it back
                    parsed = parse_expr(volume_latex.replace('\\', ''), evaluate=False)
                    if not parsed.equals(volume_sym):
                        volume_latex = sp.latex(volume_sym)  # Regenerate if verification fails
                except:
                    volume_latex = sp.latex(volume_sym)  # Fallback to raw LaTeX

            confidence_scores = dict(
                sorted(zip(coord_model.classes_, probs),
                key=lambda x: x[1],
                reverse=True))
            
            plot_html = plot_equations(equations)
            
        except Exception as e:
            print(f"Error processing question: {e}")  # Log the error for debugging
            error_message = "Question cannot be solved yet."

    return render_template("index2.html",
           question=question,
           prediction=prediction,
           confidence_scores=confidence_scores,
           equations=equations,
           integral_setup=integral_setup,
           plot_html=plot_html, 
           volume=volume, 
           volume_latex=volume_latex,
           error_message=error_message)  # Add this

# For training and running manually (not needed once trained)
def main():
    dataset = load_dataset()

    valid_coords = {"rectangular", "cylindrical", "spherical"}
    filtered = [
        item for item in dataset
        if item["coordinate_system"] in valid_coords
    ]

    texts = [item["question_text"] for item in filtered]
    labels = [item["coordinate_system"] for item in filtered]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X_train, y_train)

    joblib.dump(model, "coord_system_model.pkl")

    print("Model trained and saved!")

if __name__ == "__main__":
    # main()
    app.run(host="0.0.0.0", port=81)
