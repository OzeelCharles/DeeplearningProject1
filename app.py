import math
import numpy as np
import scipy.stats as stats
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

# ==========================================
# CONSTANTES & CONFIGURATION
# ==========================================
DOMAINS = {
    "Normale": (-5.0, 5.0), "Uniforme": (0.0, 1.0),
    "Exponentielle": (0.0, 5.0), "Bêta": (0.0, 1.0), "Gamma": (0.0, 10.0),
    "Pareto": (1.0, 6.0), "Gumbel": (-2.0, 6.0), "Fréchet": (0.5, 5.0),
    "Cauchy": (-5.0, 5.0), "Chi2": (0.0, 15.0), "Fisher": (0.1, 5.0), "Student": (-5.0, 5.0),
    "arcsin": (-0.9, 0.9), "arccos": (-0.9, 0.9), "arctan": (-5.0, 5.0),
    "sin": (-math.pi, math.pi), "cos": (-math.pi, math.pi), "tan": (-1.2, 1.2),
    "log": (0.1, 4.0), "exp": (-2.0, 2.0)
}

CHART_LAYOUT = dict(
    plot_bgcolor='#1e293b',
    paper_bgcolor='#1e293b',
    font=dict(color='#f8fafc', family='Inter'),
    margin=dict(l=40, r=40, t=60, b=40)
)

# ==========================================
# FONCTIONS UTILITAIRES (Backend)
# ==========================================
def get_activation_fn(name: str) -> nn.Module:
    """Retourne la fonction d'activation PyTorch correspondante."""
    activations = {
        "Tanh": nn.Tanh(),
        "GELU": nn.GELU(),
        "RELU": nn.ReLU(),
        "sigmoïde": nn.Sigmoid()
    }
    return activations.get(name, nn.ReLU())

def build_model(in_dim: int, num_layers: int, hidden_dim: int, activation_type: str) -> nn.Module:
    """Construit un réseau de neurones séquentiel dynamique."""
    layers = []
    act_fn = get_activation_fn(activation_type)
    
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_fn)
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    
    return nn.Sequential(*layers)

def generate_math_data(func_name: str, low: float, high: float, num_samples: int):
    """Génère les données pour les fonctions mathématiques."""
    X = torch.empty(num_samples, 1).uniform_(low, high)
    x_np = X.numpy().flatten()
    X = torch.sort(X, dim=0).values
    x_np = np.sort(x_np)

    func_map = {
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "log": np.log, "exp": np.exp
    }
    
    y_np = func_map[func_name](x_np)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return X, y, x_np, y_np

# ==========================================
# INTERFACE STREAMLIT (Frontend)
# ==========================================
def setup_ui():
    """Configure la mise en page et les styles CSS de l'application."""
    st.set_page_config(page_title="Playground PyTorch - Lois & Fonctions 1D", page_icon="⚡", layout="wide")
    
    st.markdown("""
        <style>
        .main { background-color: #0f172a; color: #f8fafc; }
        [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
        h1, h2, h3 { font-family: 'Inter', sans-serif; color: #f8fafc !important; font-weight: 700; }
        .stButton > button { background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); color: white; border-radius: 8px; width: 100%; transition: all 0.3s ease; }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4); }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='padding: 1.5rem 0; text-align: center;'>
            <h1 style='font-size: 2.5rem; background: linear-gradient(90deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                ⚡ Playground Interactif : Lois & Fonctions 1D
            </h1>
            <p style='color: #94a3b8; font-size: 1.1rem; margin-top: -10px;'>
                Ajustement de densités (histogrammes) & régression de courbes propulsés par PyTorch.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

def main():
    setup_ui()
    
    # 1. BARRE LATÉRALE
    with st.sidebar:
        st.markdown("### 🎛️ Paramètres de l'Expérience")
        category = st.selectbox("Type de données :", ["Lois Statistiques Continues", "Fonctions Mathématiques"])

        if category == "Lois Statistiques Continues":
            func_name = st.selectbox("Loi de probabilité :", ["Normale", "Uniforme", "Exponentielle", "Cauchy"])
        else:
            func_name = st.selectbox("Fonction à estimer :", ["arcsin", "arccos", "arctan", "sin", "cos", "tan", "log", "exp"])

        low, high = DOMAINS.get(func_name, (-3.0, 3.0))

        st.markdown("### 📊 Entraînement")
        num_samples = st.slider("Dataset size", 1_000, 50_000, 5_000, 1_000)
        epochs = st.slider("Époques", 5, 100, 25, 5)
        batch_size = st.selectbox("Taille du batch", [16, 32, 64, 128], index=2)
        lr = st.selectbox("Learning Rate", [0.001, 0.005, 0.01, 0.05], index=2)

        st.markdown("### 🧠 Architecture")
        num_layers = st.slider("Couches cachées", 1, 10, 3)
        hidden_dim = st.slider("Neurones/couche", 16, 256, 64, 16)
        activation_type = st.selectbox("Activation", ["Tanh", "GELU", "RELU", "sigmoïde"])

        run_button = st.button("🚀 Lancer l'entraînement", type="primary")

    # Fixer les seeds pour la reproductibilité
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. GÉNÉRATION DES DONNÉES
    if category == "Fonctions Mathématiques":
        X, y, x_np, y_np = generate_math_data(func_name, low, high, num_samples)
        
        # Affichage avant entraînement
        if not run_button:
            fig = go.Figure(data=go.Scatter(x=x_np, y=y_np, mode='lines', name=func_name, line=dict(color='#38bdf8', width=3)))
            fig.update_layout(**CHART_LAYOUT, title=f"Courbe de la fonction {func_name}")
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Simplification de la démo statistique pour l'exemple (à étendre selon vos besoins initiaux)
        if func_name == "Normale": y_np = np.random.normal(0, 1, size=num_samples)
        elif func_name == "Uniforme": y_np = np.random.uniform(0, 1, size=num_samples)
        elif func_name == "Exponentielle": y_np = np.random.exponential(1.0, size=num_samples)
        elif func_name == "Cauchy": y_np = np.clip(np.random.standard_cauchy(size=num_samples), -5, 5)

        kde = stats.gaussian_kde(y_np)
        x_train_np = np.linspace(np.min(y_np), np.max(y_np), num_samples)
        y_train_np = kde(x_train_np)
        
        X = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)

        if not run_button:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=y_np, histnorm='probability density', marker_color='#6366f1', opacity=0.6, nbinsx=50))
            fig.update_layout(**CHART_LAYOUT, title=f"Distribution : {func_name}")
            st.plotly_chart(fig, use_container_width=True)

    # Message incitatif si non lancé
    if not run_button:
        st.info("👈 Ajustez les paramètres dans la barre latérale et cliquez sur 'Lancer l'entraînement' !")
        return

    # 3. ENTRAÎNEMENT DU MODÈLE
    with st.spinner("✨ Entraînement du réseau en cours..."):
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = build_model(1, num_layers, hidden_dim, activation_type)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        loss_history = []
        predictions_history = []
        
        # Grille d'évaluation
        X_eval = torch.linspace(X.min().item(), X.max().item(), 200).unsqueeze(1)
        X_eval_np = X_eval.numpy().flatten()

        # Boucle d'entraînement
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for bx, by in dataloader:
                pred = model(bx)
                loss = criterion(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            loss_history.append(epoch_loss / len(dataloader))
            
            model.eval()
            with torch.no_grad():
                predictions_history.append(model(X_eval).numpy().flatten())

        st.success("🎉 Entraînement terminé avec succès !")

    # 4. RÉSULTATS VISUELS
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Évolution de la prédiction")
        # Rendu statique du dernier état pour l'exemple (remplacez par votre logique d'animation Plotly si désiré)
        fig_res = go.Figure()
        
        if category == "Fonctions Mathématiques":
            y_eval_true = generate_math_data(func_name, X_eval.min(), X_eval.max(), 200)[3]
            fig_res.add_trace(go.Scatter(x=X_eval_np, y=y_eval_true, mode='lines', name='Vraie fonction', line=dict(color='#38bdf8', width=3)))
        else:
            fig_res.add_trace(go.Histogram(x=y_np, histnorm='probability density', marker_color='#6366f1', opacity=0.4, nbinsx=50, name="Données"))

        fig_res.add_trace(go.Scatter(x=X_eval_np, y=predictions_history[-1], mode='lines', name='Prédiction Modèle (Final)', line=dict(color='#f43f5e', width=3)))
        fig_res.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_res, use_container_width=True)

    with col2:
        st.subheader("📉 Courbe de Perte (MSE)")
        fig_loss = go.Figure(data=go.Scatter(y=loss_history, mode='lines+markers', line=dict(color='#f43f5e', width=3)))
        fig_loss.update_layout(**CHART_LAYOUT, xaxis_title="Époque", yaxis_title="MSE Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

if __name__ == "__main__":
    main()