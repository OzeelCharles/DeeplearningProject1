import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import scipy.stats as stats
import plotly.graph_objects as go

# ==========================================
# DESIGN & CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Playground PyTorch - Lois & Fonctions 1D", 
    page_icon="⚡", 
    layout="wide"
)

# Injection de style CSS moderne pour embellir l'interface et la sidebar
st.markdown("""
    <style>
    /* Style général de fond et de police */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Style de la barre latérale (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Titres et en-têtes */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #f8fafc !important;
        font-weight: 700;
    }
    
    /* Conteneurs / Cartes stylisées */
    .stCard {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #334155;
    }

    /* Boutons personnalisés */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
    }

    /* Personnalisation des widgets de la sidebar */
    .sidebar .stSelectbox, .sidebar .stSlider {
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# En-tête principal avec une mise en page moderne
st.markdown("""
    <div style='padding: 1.5rem 0; text-align: center;'>
        <h1 style='font-size: 2.5rem; background: linear-gradient(90deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ⚡ Playground Interactif : Lois & Fonctions 1D
        </h1>
        <p style='color: #94a3b8; font-size: 1.1rem; margin-top: -10px;'>
            Ajustement de densités (histogrammes) & QQ-plots pour les lois statistiques et régression de courbes pour les fonctions mathématiques.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 1. BARRE LATÉRALE : PARAMÈTRES ET CONTRÔLES
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Paramètres de l'Expérience")
    
    category = st.selectbox(
        "Type de données :",
        ["Lois Statistiques Continues", "Fonctions Mathématiques"]
    )

    if category == "Lois Statistiques Continues":
        func_name = st.selectbox(
            "Loi de probabilité :",
            ["Normale", "Uniforme", "Exponentielle", "Bêta", "Gamma", "Pareto", "Gumbel", "Fréchet", "Cauchy", "Chi2", "Fisher", "Student"]
        )
    else:
        func_name = st.selectbox(
            "Fonction à estimer :",
            ["arcsin", "arccos", "arctan", "sin", "cos", "tan", "log", "exp"]
        )

    # Domaines par défaut pour l'échantillonnage
    domains = {
        "Normale": (-5.0, 5.0), "Uniforme": (0.0, 1.0),
        "Exponentielle": (0.0, 5.0), "Bêta": (0.0, 1.0), "Gamma": (0.0, 10.0),
        "Pareto": (1.0, 6.0), "Gumbel": (-2.0, 6.0), "Fréchet": (0.5, 5.0),
        "Cauchy": (-5.0, 5.0), "Chi2": (0.0, 15.0), "Fisher": (0.1, 5.0), "Student": (-5.0, 5.0),
        "arcsin": (-0.9, 0.9), "arccos": (-0.9, 0.9), "arctan": (-5.0, 5.0),
        "sin": (-math.pi, math.pi), "cos": (-math.pi, math.pi), "tan": (-1.2, 1.2),
        "log": (0.1, 4.0), "exp": (-2.0, 2.0)
    }
    low, high = domains.get(func_name, (-3.0, 3.0))

    # Paramètres par défaut pour les lois
    mu_param, sigma_param = 0.0, 1.0
    u_low, u_high = 0.0, 1.0
    expon_scale = 1.0
    alpha_param, beta_param = 2.0, 2.0
    gamma_shape, gamma_scale = 2.0, 1.0
    pareto_a = 2.0
    gumbel_loc, gumbel_scale = 0.0, 1.0
    frechet_c = 2.0
    cauchy_loc, cauchy_scale = 0.0, 1.0
    chi2_df = 3
    fisher_dfnum, fisher_dfden = 5, 2
    student_df = 5
    noise_level = 0.0
    
    if category == "Lois Statistiques Continues":
        st.markdown(f"#### ⚙️ Paramètres : {func_name}")
        if func_name == "Normale":
            mu_param = st.slider("Moyenne (μ)", -3.0, 3.0, 0.0)
            sigma_param = st.slider("Écart-type (σ)", 0.1, 3.0, 1.0)
        elif func_name == "Uniforme":
            u_low = st.slider("Borne min (a)", -5.0, 0.0, 0.0)
            u_high = st.slider("Borne max (b)", 0.1, 5.0, 1.0)
            low, high = u_low, u_high
        elif func_name == "Exponentielle":
            expon_scale = st.slider("Échelle (scale / 1-rate)", 0.1, 5.0, 1.0)
        elif func_name == "Bêta":
            alpha_param = st.slider("Alpha (α)", 0.5, 5.0, 2.0)
            beta_param = st.slider("Beta (β)", 0.5, 5.0, 2.0)
        elif func_name == "Gamma":
            gamma_shape = st.slider("Forme (k / α)", 0.5, 5.0, 2.0)
            gamma_scale = st.slider("Échelle (θ / β)", 0.5, 5.0, 1.0)
        elif func_name == "Pareto":
            pareto_a = st.slider("Forme (a)", 0.5, 5.0, 2.0)
        elif func_name == "Gumbel":
            gumbel_loc = st.slider("Position (loc)", -3.0, 3.0, 0.0)
            gumbel_scale = st.slider("Échelle (scale)", 0.1, 3.0, 1.0)
        elif func_name == "Fréchet":
            frechet_c = st.slider("Forme (c)", 0.5, 5.0, 2.0)
        elif func_name == "Cauchy":
            cauchy_loc = st.slider("Position (loc)", -3.0, 3.0, 0.0)
            cauchy_scale = st.slider("Échelle (scale)", 0.1, 3.0, 1.0)
        elif func_name == "Chi2":
            chi2_df = st.slider("Degrés de liberté (df)", 1, 15, 3)
        elif func_name == "Fisher":
            fisher_dfnum = st.slider("Degrés de liberté num (df1)", 1, 15, 5)
            fisher_dfden = st.slider("Degrés de liberté den (df2)", 1, 15, 2)
        elif func_name == "Student":
            student_df = st.slider("Degrés de liberté (df)", 1, 20, 5)

        st.markdown("#### 🌪️ Difficulté & Bruit")
        noise_level = st.slider("Intensité du Bruit Uniforme", 0.0, 2.0, 0.0, 0.1)

    st.markdown("---")
    st.markdown("### 📊 Données & Entraînement")
    num_samples = st.slider("Nombre de données (Dataset)", min_value=1_000, max_value=50_000, value=5_000, step=1_000)
    epochs = st.slider("Nombre d'époques", min_value=5, max_value=100, value=25, step=5)
    batch_size = st.selectbox("Taille du batch", [16, 32, 64, 128], index=2)
    weight_decay = st.selectbox("Learning Rate", [0.001, 0.005, 0.01, 0.05], index=2)

    st.markdown("---")
    st.markdown("### 🧠 Architecture du Réseau")
    num_layers = st.slider("Nombre de couches cachées", min_value=1, max_value=10, value=3)
    hidden_dim = st.slider("Neurones par couche cachée", min_value=16, max_value=256, value=64, step=16)
    activation_type = st.selectbox("Fonction d'activation", ["Tanh", "GELU", "RELU", "sigmoïde"])

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("🚀 Lancer l'entraînement", type="primary")

# ==========================================
# 2. GÉNÉRATION DES DONNÉES & CALCULS THÉORIQUES
# ==========================================
torch.manual_seed(42)
np.random.seed(42)

if category == "Fonctions Mathématiques":
    X = torch.empty(num_samples, 1).uniform_(low, high)
    x_np = X.numpy().flatten()
    X = torch.sort(X, dim=0).values
    x_np = np.sort(x_np)

    if func_name == "arcsin": y_np = np.arcsin(x_np)
    elif func_name == "arccos": y_np = np.arccos(x_np)
    elif func_name == "arctan": y_np = np.arctan(x_np)
    elif func_name == "sin": y_np = np.sin(x_np)
    elif func_name == "cos": y_np = np.cos(x_np)
    elif func_name == "tan": y_np = np.tan(x_np)
    elif func_name == "log": y_np = np.log(x_np)
    elif func_name == "exp": y_np = np.exp(x_np)
    
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

else:
    # Génération de l'échantillon brut
    if func_name == "Normale": y_np = np.random.normal(mu_param, sigma_param, size=num_samples)
    elif func_name == "Uniforme": y_np = np.random.uniform(u_low, u_high, size=num_samples)
    elif func_name == "Exponentielle": y_np = np.random.exponential(scale=expon_scale, size=num_samples)
    elif func_name == "Bêta": y_np = np.random.beta(alpha_param, beta_param, size=num_samples)
    elif func_name == "Gamma": y_np = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=num_samples)
    elif func_name == "Pareto": y_np = np.random.pareto(a=pareto_a, size=num_samples) + low
    elif func_name == "Gumbel": y_np = np.random.gumbel(loc=gumbel_loc, scale=gumbel_scale, size=num_samples)
    elif func_name == "Fréchet":
        u = np.random.uniform(0, 1, size=num_samples)
        y_np = (-np.log(u)) ** (-1.0 / frechet_c)
    elif func_name == "Cauchy":
        y_np = np.random.standard_cauchy(size=num_samples) * cauchy_scale + cauchy_loc
        y_np = np.clip(y_np, low, high)
    elif func_name == "Chi2": y_np = np.random.chisquare(df=int(chi2_df), size=num_samples)
    elif func_name == "Fisher":
        y_np = np.random.f(dfnum=int(fisher_dfnum), dfden=int(fisher_dfden), size=num_samples)
        y_np = np.clip(y_np, low, high)
    elif func_name == "Student": y_np = np.random.standard_t(df=int(student_df), size=num_samples)

    if noise_level > 0.0:
        noise = np.random.uniform(-noise_level, noise_level, size=num_samples)
        y_np = y_np + noise

    # --- CORRECTION CONCEPTUELLE ---
    # Au lieu de chercher à mapper un vecteur X aléatoire vers Y,
    # on extrait la densité empirique (KDE) de l'échantillon pour en faire la cible du réseau.
    kde = stats.gaussian_kde(y_np)
    
    # On crée le domaine X d'entraînement
    x_train_np = np.linspace(np.min(y_np), np.max(y_np), num_samples)
    
    # La cible Y est la densité estimée sur ces points
    y_train_np = kde(x_train_np)
    
    X = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)


# Grille d'évaluation
eval_grid = np.linspace(np.min(y_np) if category == "Lois Statistiques Continues" else low, 
                        np.max(y_np) if category == "Lois Statistiques Continues" else high, 200)

# Calcul de la densité théorique exacte (PDF) et quantiles théoriques
pdf_exact = np.zeros_like(eval_grid)
quantiles_theo = np.linspace(0.01, 0.99, 100)
qq_theo_vals = np.zeros_like(quantiles_theo)

if category == "Lois Statistiques Continues":
    if func_name == "Normale":
        pdf_exact = stats.norm.pdf(eval_grid, mu_param, sigma_param)
        qq_theo_vals = stats.norm.ppf(quantiles_theo, mu_param, sigma_param)
    elif func_name == "Uniforme":
        pdf_exact = stats.uniform.pdf(eval_grid, u_low, u_high - u_low)
        qq_theo_vals = stats.uniform.ppf(quantiles_theo, u_low, u_high - u_low)
    elif func_name == "Exponentielle":
        pdf_exact = stats.expon.pdf(eval_grid, scale=expon_scale)
        qq_theo_vals = stats.expon.ppf(quantiles_theo, scale=expon_scale)
    elif func_name == "Bêta":
        pdf_exact = stats.beta.pdf(eval_grid, alpha_param, beta_param)
        qq_theo_vals = stats.beta.ppf(quantiles_theo, alpha_param, beta_param)
    elif func_name == "Gamma":
        pdf_exact = stats.gamma.pdf(eval_grid, a=gamma_shape, scale=gamma_scale)
        qq_theo_vals = stats.gamma.ppf(quantiles_theo, a=gamma_shape, scale=gamma_scale)
    elif func_name == "Pareto":
        pdf_exact = stats.pareto.pdf(eval_grid, b=pareto_a, loc=low-1.0 if low>1 else 0)
        qq_theo_vals = stats.pareto.ppf(quantiles_theo, b=pareto_a, loc=low-1.0 if low>1 else 0)
    elif func_name == "Gumbel":
        pdf_exact = stats.gumbel_r.pdf(eval_grid, loc=gumbel_loc, scale=gumbel_scale)
        qq_theo_vals = stats.gumbel_r.ppf(quantiles_theo, loc=gumbel_loc, scale=gumbel_scale)
    elif func_name == "Fréchet":
        pdf_exact = stats.frechet_r.pdf(eval_grid, c=frechet_c)
        qq_theo_vals = stats.frechet_r.ppf(quantiles_theo, c=frechet_c)
    elif func_name == "Cauchy":
        pdf_exact = stats.cauchy.pdf(eval_grid, loc=cauchy_loc, scale=cauchy_scale)
        qq_theo_vals = stats.cauchy.ppf(quantiles_theo, loc=cauchy_loc, scale=cauchy_scale)
    elif func_name == "Chi2":
        pdf_exact = stats.chi2.pdf(eval_grid, df=int(chi2_df))
        qq_theo_vals = stats.chi2.ppf(quantiles_theo, df=int(chi2_df))
    elif func_name == "Fisher":
        pdf_exact = stats.f.pdf(eval_grid, dfnum=int(fisher_dfnum), dfden=int(fisher_dfden))
        qq_theo_vals = stats.f.ppf(quantiles_theo, dfnum=int(fisher_dfnum), dfden=int(fisher_dfden))
    elif func_name == "Student":
        pdf_exact = stats.t.pdf(eval_grid, df=int(student_df))
        qq_theo_vals = stats.t.ppf(quantiles_theo, df=int(student_df))

chart_layout_config = dict(
    plot_bgcolor='#1e293b',
    paper_bgcolor='#1e293b',
    font=dict(color='#f8fafc', family='Inter'),
    margin=dict(l=40, r=40, t=60, b=40)
)

# ==========================================
# 3. AFFICHAGE INITIAL (Avant entrainement)
# ==========================================
if not run_button:
    col_main, col_act = st.columns([2, 1])

    with col_main:
        if category == "Lois Statistiques Continues":
            st.markdown(f"### 📊 Histogramme et Distribution : {func_name}")
            fig_init = go.Figure()
            fig_init.add_trace(go.Histogram(
                x=y_np, histnorm='probability density', name='Données Bruitées', marker_color='#6366f1', opacity=0.6, nbinsx=50
            ))
            if np.any(pdf_exact > 0):
                fig_init.add_trace(go.Scatter(
                    x=eval_grid, y=pdf_exact, mode='lines', name='Distribution Théorique', line=dict(color='#38bdf8', width=3)
                ))
            fig_init.update_layout(**chart_layout_config, title=f"Distribution de {func_name} (Bruit: {noise_level})", xaxis_title="Valeurs", yaxis_title="Densité", barmode='overlay')
            st.plotly_chart(fig_init, use_container_width=True)

            # QQ-Plot initial
            st.markdown("### 📉 QQ-Plot Comparatif (Théorique vs Données)")
            qq_data_vals = np.quantile(y_np, quantiles_theo)
            fig_qq_init = go.Figure()
            fig_qq_init.add_trace(go.Scatter(
                x=qq_theo_vals, y=qq_theo_vals, mode='lines', name='Référence Parfaite (y = x)', line=dict(color='#38bdf8', width=2, dash='dot')
            ))
            fig_qq_init.add_trace(go.Scatter(
                x=qq_theo_vals, y=qq_data_vals, mode='markers', name='QQ-Plot des Données', marker=dict(color='#6366f1', size=8)
            ))
            fig_qq_init.update_layout(**chart_layout_config, title="Diagramme Quantile-Quantile", xaxis_title="Quantiles Théoriques", yaxis_title="Quantiles des Données")
            st.plotly_chart(fig_qq_init, use_container_width=True)
        else:
            st.markdown(f"### 📈 Courbe Mathématique : {func_name}")
            X_eval_plot = torch.linspace(low, high, 200).unsqueeze(1)
            if func_name == "arcsin": y_eval_plot = torch.asin(X_eval_plot).numpy().flatten()
            elif func_name == "arccos": y_eval_plot = torch.acos(X_eval_plot).numpy().flatten()
            elif func_name == "arctan": y_eval_plot = torch.atan(X_eval_plot).numpy().flatten()
            elif func_name == "sin": y_eval_plot = torch.sin(X_eval_plot).numpy().flatten()
            elif func_name == "cos": y_eval_plot = torch.cos(X_eval_plot).numpy().flatten()
            elif func_name == "tan": y_eval_plot = torch.tan(X_eval_plot).numpy().flatten()
            elif func_name == "log": y_eval_plot = torch.log(X_eval_plot).numpy().flatten()
            elif func_name == "exp": y_eval_plot = torch.exp(X_eval_plot).numpy().flatten()

            fig_init = go.Figure(data=go.Scatter(x=X_eval_plot.numpy().flatten(), y=y_eval_plot, mode='lines', name=f'Fonction {func_name}', line=dict(color='#38bdf8', width=3)))
            fig_init.update_layout(**chart_layout_config, title=f"Courbe de la fonction {func_name}", xaxis_title="Entrée X", yaxis_title="Sortie Y")
            st.plotly_chart(fig_init, use_container_width=True)

    with col_act:
        st.markdown(f"### ⚡ Fonction d'activation ({activation_type})")
        x_act_vals = torch.linspace(-3.0, 3.0, 200)
        if activation_type == "Tanh": y_act_vals = torch.tanh(x_act_vals).numpy()
        elif activation_type == "GELU": y_act_vals = nn.GELU()(x_act_vals).numpy()
        elif activation_type == "sigmoïde": y_act_vals = torch.sigmoid(x_act_vals).numpy()
        else: y_act_vals = torch.relu(x_act_vals).numpy()

        fig_act = go.Figure(data=go.Scatter(x=x_act_vals.numpy(), y=y_act_vals, mode='lines', name=activation_type, line=dict(color='#c084fc', width=3)))
        fig_act.update_layout(**chart_layout_config, title=f"Allure de {activation_type}", xaxis_title="Entrée", yaxis_title="Sortie")
        st.plotly_chart(fig_act, use_container_width=True)

    st.markdown("""
        <div style='background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 2rem; text-align: center; margin-top: 2rem;'>
            <h3 style='color: #818cf8; margin-bottom: 0.5rem;'>Prêt à lancer l'apprentissage ?</h3>
            <p style='color: #94a3b8; font-size: 1.05rem;'>
                Cliquez sur <b style='color: #f8fafc;'>'Lancer l'entraînement'</b> dans la barre latérale pour voir le modèle s'ajuster en direct.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. LOGIQUE D'ENTRAÎNEMENT & RÉSULTATS ANIMÉS
# ==========================================
else:
    with st.spinner("✨ Entraînement du réseau de neurones en cours..."):
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if category == "Fonctions Mathématiques":
            X_eval = torch.linspace(low, high, 200).unsqueeze(1)
            if func_name == "arcsin": y_eval_true = torch.asin(X_eval).numpy().flatten()
            elif func_name == "arccos": y_eval_true = torch.acos(X_eval).numpy().flatten()
            elif func_name == "arctan": y_eval_true = torch.atan(X_eval).numpy().flatten()
            elif func_name == "sin": y_eval_true = torch.sin(X_eval).numpy().flatten()
            elif func_name == "cos": y_eval_true = torch.cos(X_eval).numpy().flatten()
            elif func_name == "tan": y_eval_true = torch.tan(X_eval).numpy().flatten()
            elif func_name == "log": y_eval_true = torch.log(X_eval).numpy().flatten()
            elif func_name == "exp": y_eval_true = torch.exp(X_eval).numpy().flatten()
        else:
            X_eval = torch.tensor(eval_grid, dtype=torch.float32).unsqueeze(1)
            y_eval_true = pdf_exact

        X_eval_np = X_eval.numpy().flatten()

        layers = []
        in_dim = 1
        if activation_type == "Tanh": act_fn = nn.Tanh()
        elif activation_type == "GELU": act_fn = nn.GELU()
        elif activation_type == "sigmoïde": act_fn = nn.Sigmoid()
        else: act_fn = nn.ReLU()

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        model = nn.Sequential(*layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=weight_decay)

        predictions_history = []
        loss_history = []

        model.eval()
        with torch.no_grad():
            predictions_history.append(model(X_eval).numpy().flatten())

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

    col1, col2 = st.columns([2, 1])

    with col1:
        if category == "Lois Statistiques Continues":
            st.subheader("📈 Ajustement du Modèle sur l'Histogramme (Animation)")
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=y_np, histnorm='probability density', name='Données Bruitées', marker_color='#6366f1', opacity=0.4, nbinsx=50
            ))
            if np.any(pdf_exact > 0):
                fig.add_trace(go.Scatter(
                    x=eval_grid, y=pdf_exact, mode='lines', name='Distribution Théorique', line=dict(color='#38bdf8', width=2, dash='dot')
                ))
            fig.add_trace(go.Scatter(
                x=X_eval_np, y=predictions_history[0], mode='lines', name='Prédiction Modèle', line=dict(color='#f43f5e', width=3)
            ))

            frames = []
            for e, preds in enumerate(predictions_history):
                frame_data = [
                    go.Histogram(x=y_np, histnorm='probability density', marker_color='#6366f1', opacity=0.4, nbinsx=50),
                ]
                if np.any(pdf_exact > 0):
                    frame_data.append(go.Scatter(x=eval_grid, y=pdf_exact, mode='lines', line=dict(color='#38bdf8', width=2, dash='dot')))
                frame_data.append(go.Scatter(x=X_eval_np, y=preds, mode='lines', line=dict(color='#f43f5e', width=3)))
                frames.append(go.Frame(data=frame_data, name=f"Epoch {e}"))

            fig.frames = frames
            fig.update_layout(
                **chart_layout_config,
                xaxis=dict(title="Valeurs", gridcolor='#334155'),
                yaxis=dict(title="Densité", gridcolor='#334155'),
                barmode='overlay',
                updatemenus=[dict(
                    type="buttons", showactive=False,
                    buttons=[
                        dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}, "transition": {"duration": 100, "easing": "cubic-in-out"}, "fromcurrent": True}]),
                        dict(label="❚❚ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                    ],
                    x=0.0, y=1.15, xanchor="left", yanchor="top"
                )]
            )
        else:
            st.subheader("📈 Évolution de l'Apprentissage par Courbe (Animation)")
            fig = go.Figure(
                data=[
                    go.Scatter(x=X_eval_np, y=y_eval_true, mode='lines', name=f'Vraie fonction ({func_name})', line=dict(color='#38bdf8', width=3)),
                    go.Scatter(x=X_eval_np, y=predictions_history[0], mode='lines', name='Prédiction Modèle', line=dict(color='#f43f5e', width=3, dash='dash'))
                ],
                layout=go.Layout(
                    **chart_layout_config,
                    xaxis=dict(title="Entrée (X)", gridcolor='#334155'),
                    yaxis=dict(title="Sortie (Y)", gridcolor='#334155'),
                    updatemenus=[dict(
                        type="buttons", showactive=False,
                        buttons=[
                            dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}, "transition": {"duration": 100, "easing": "cubic-in-out"}, "fromcurrent": True}]),
                            dict(label="❚❚ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                        ],
                        x=0.0, y=1.15, xanchor="left", yanchor="top"
                    )]
                ),
                frames=[
                    go.Frame(
                        data=[
                            go.Scatter(x=X_eval_np, y=y_eval_true, mode='lines', line=dict(color='#38bdf8', width=3)),
                            go.Scatter(x=X_eval_np, y=preds, mode='lines', line=dict(color='#f43f5e', width=3, dash='dash'))
                        ],
                        name=f"Epoch {e}"
                    )
                    for e, preds in enumerate(predictions_history)
                ]
            )

        sliders = [{
            "pad": {"t": 50, "b": 10},
            "len": 0.9,
            "x": 0.05, "y": 0,
            "currentvalue": {"prefix": "<b>Époque : </b>", "font": {"size": 14, "color": "#f8fafc"}},
            "steps": [{"args": [[f"Epoch {e}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(e), "method": "animate"} for e in range(len(predictions_history))]
        }]
        fig.update_layout(sliders=sliders)
        st.plotly_chart(fig, use_container_width=True)

        # Affichage du QQ-Plot final comparant Théorique vs Données vs Réseau de neurones
        if category == "Lois Statistiques Continues":
            st.subheader("📉 QQ-Plot Comparatif Final (Théorique vs Données vs Modèle)")
            
            # Les quantiles des données réelles
            qq_data_vals = np.quantile(y_np, quantiles_theo)
            
            # CORRECTION QQ-PLOT : 
            # Le réseau prédit une Densité (PDF). Pour trouver ses quantiles, on l'intègre en CDF
            # puis on inverse la CDF par interpolation numérique.
            pdf_pred = predictions_history[-1]
            pdf_pred = np.maximum(pdf_pred, 0)  # Éviter les densités négatives du NN
            dx = X_eval_np[1] - X_eval_np[0]
            cdf_pred = np.cumsum(pdf_pred) * dx
            if cdf_pred[-1] > 0:
                cdf_pred = cdf_pred / cdf_pred[-1] # Normalisation à 1
            
            # Inversion de la CDF pour trouver les quantiles prédits
            qq_model_vals = np.interp(quantiles_theo, cdf_pred, X_eval_np)

            fig_qq_final = go.Figure()
            fig_qq_final.add_trace(go.Scatter(
                x=qq_theo_vals, y=qq_theo_vals, mode='lines', name='Référence Parfaite (y = x)', line=dict(color='#38bdf8', width=2, dash='dot')
            ))
            fig_qq_final.add_trace(go.Scatter(
                x=qq_theo_vals, y=qq_data_vals, mode='markers', name='QQ-Plot des Données Bruitées', marker=dict(color='#6366f1', size=7, opacity=0.7)
            ))
            fig_qq_final.add_trace(go.Scatter(
                x=qq_theo_vals, y=qq_model_vals, mode='markers', name='QQ-Plot du Réseau de Neurones', marker=dict(color='#f43f5e', size=8)
            ))
            fig_qq_final.update_layout(
                **chart_layout_config, 
                title="Comparaison des Quantiles (QQ-Plot)", 
                xaxis_title="Quantiles Théoriques", 
                yaxis_title="Quantiles Observés / Prédits"
            )
            st.plotly_chart(fig_qq_final, use_container_width=True)

    with col2:
        st.subheader("📉 Courbe de Perte (Loss MSE)")
        fig_loss = go.Figure(data=go.Scatter(y=loss_history, mode='lines+markers', line=dict(color='#f43f5e', width=3)))
        fig_loss.update_layout(
            **chart_layout_config,
            xaxis_title="Époque", 
            yaxis_title="MSE Loss", 
            xaxis=dict(gridcolor='#334155'),
            yaxis=dict(gridcolor='#334155')
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        st.subheader(f"⚡ Fonction d'activation ({activation_type})")
        x_act_vals = torch.linspace(-3.0, 3.0, 200)
        if activation_type == "Tanh": y_act_vals = torch.tanh(x_act_vals).numpy()
        elif activation_type == "GELU": y_act_vals = nn.GELU()(x_act_vals).numpy()
        elif activation_type == "sigmoïde": y_act_vals = torch.sigmoid(x_act_vals).numpy()
        else: y_act_vals = torch.relu(x_act_vals).numpy()

        fig_act = go.Figure(data=go.Scatter(x=x_act_vals.numpy(), y=y_act_vals, mode='lines', name=activation_type, line=dict(color='#c084fc', width=3)))
        fig_act.update_layout(**chart_layout_config, title=f"Allure de {activation_type}", xaxis_title="Entrée", yaxis_title="Sortie")
        st.plotly_chart(fig_act, use_container_width=True)