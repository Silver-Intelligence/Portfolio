import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from groq import Groq

st.markdown("""
    <style>
    .ai-card {
        background-color: #ffffff;
        padding: 12px 20px;
        border-radius: 10px;
        border-left: 5px solid #003399;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .ai-text {
        font-size: 0.95em;
        line-height: 1.4;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="Silver AI Intelligence - Predictive Hub", layout="wide")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@st.cache_data
def load_raw_data():
    df_brut = pd.read_csv('data/ai4i2020.csv')

    # 1. Renommage des colonnes (Indispensable pour la cohérence)
    df_brut.columns = ['ID', 'Product ID', 'Type', 'Air_Temp', 'Process_Temp', 
                       'Rotational_Speed', 'Torque', 'Tool_Wear', 'Failure', 
                       'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    # 2. Mapping du Type (Catégorique vers Numérique)
    mapping = {'L': 0, 'M': 1, 'H': 2}
    df_brut['Type'] = df_brut['Type'].map(mapping)

    # 3. FEATURE ENGINEERING (Doit correspondre exactement au X_train)
    # On crée ici les variables que le Pipeline va devoir traiter
    df_brut['Temp_Diff'] = df_brut['Process_Temp'] - df_brut['Air_Temp']
    df_brut['Power'] = df_brut['Torque'] * df_brut['Rotational_Speed']
    df_brut['Strain'] = df_brut['Torque'] * df_brut['Tool_Wear']
    df_brut['Heat_Stress'] = df_brut['Rotational_Speed'] * df_brut['Temp_Diff']
    
    
    return df_brut

df = load_raw_data() 

# --- 2. FONCTIONS CORE ---
def load_ai_assets():
    res = joblib.load('models/silver_core_v1.joblib')
    pipeline = joblib.load('models/pipeline_predictive.pkl') 
    model_multi = joblib.load('models/model_multi_class.pkl')
    reliability_scores = joblib.load('models/fiabilite_parametres.joblib')
    
    return res, pipeline, model_multi, reliability_scores

res, pipeline, model_multi, reliability_scores = load_ai_assets()

print("\n--- CONTENU DU FICHIER EXPORTÉ ---")
print(json.dumps(reliability_scores, indent=4))

@st.cache_data(show_spinner=False)
def get_expert_ai_advice(tp, fp, fn, current_cost, savings, threshold, best_name):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""Expert Silver AI. Analyse technique : {best_name} (Seuil {threshold:.2f}). 
        Données : {tp} pannes détectées, {fn} pannes non détectées, {fp} fausses alertes. 
        Impact financier : Économie de {savings}€. 
        Instructions : Décris uniquement les chiffres. Ne commente pas la qualité du modèle et ne suggère pas d'améliorations et ne pose pas de question. 
        Réponse en 2 phrases max."""
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except:
        return "Analyse stratégique prête."

def get_best_strategy(res, c_unp, c_pre, c_fp):
    thresholds = np.linspace(0.05, 0.9, 100)
    def calc_cost(probs, t):
        y_p = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(res['y_true'], y_p).ravel()
        return (fn * c_unp) + (tp * c_pre) + (fp * c_fp)
    
    costs_p = [calc_cost(res['y_probs'], t) for t in thresholds] 
    
    
    return "Performance", res['y_probs'], thresholds[np.argmin(costs_p)]

with st.sidebar:
    st.header("🎛️ Paramètres Économiques")
    c_unplanned = st.number_input("Coût Panne Imprévue (€)", value=500)
    c_maintenance = st.number_input("Coût Maintenance (€)", value=100)
    c_false_alert = st.number_input("Coût Fausse Alerte (€)", value=50)
    
    best_name, y_probs, best_t = get_best_strategy(res, c_unplanned, c_maintenance, c_false_alert)
    
    st.divider()
    if st.button("🎯 Appliquer le seuil optimal"):
        st.session_state.threshold = best_t
        st.session_state.check_live_monitoring = False
        st.rerun()

    threshold = st.slider("Réglage de la Sensibilité", 0.05, 0.9, step=0.01, key="threshold")
    
    # CALCUL FINAL POUR LA SIDEBAR
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(res['y_true'], y_pred).ravel()
    current_cost = (fn * c_unplanned) + (tp * c_maintenance) + (fp * c_false_alert)
    baseline_cost = (tp + fn) * c_unplanned # Coût si on ne faisait rien (mode réactif)
    total_savings = baseline_cost - current_cost


    st.divider()
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: flex-start; gap: 0px;">
        <p style="font-size: 14px; font-weight: 500; margin: 0px; color: #31333F;">💰 GAIN NET RÉALISÉ</p>
        <p style="font-size: 28px; color: #28a745; margin: 0px; font-family: 'Source Sans Pro', sans-serif;">
            {int(total_savings):,}€
        </p>
    </div>
""".replace(",", " "), unsafe_allow_html=True)
    st.caption("Argent économisé par rapport à une maintenance sans IA.")

# --- 4. AFFICHAGE ---
st.title("🚀 Smart Factory : Pilotage IA")

# Bulle Agent IA Réduite
advice = get_expert_ai_advice(tp, fp, fn, current_cost, total_savings, threshold, best_name)
st.markdown(f"""
    <div class="ai-card">
        <span style="font-size: 1.2em; font-weight: bold; color: #003399;">🤖 Agent IA :</span>
        <span class="ai-text">{advice}</span>
    </div>
""", unsafe_allow_html=True)

# KPIs Principaux
baseline_cost = (tp + fn) * c_unplanned  # Coût si 0 maintenance prédictive
total_savings = baseline_cost - current_cost
# Calcul du % : (Economie / Coût de départ) * 100
pct_gain = (total_savings / baseline_cost * 100) if baseline_cost > 0 else 0

# --- SECTION AFFICHAGE DES KPIs ---
# --- 6. PRÉCISION DU MODÈLE CHOISI ---
st.divider()
st.markdown(f"### 📊 Analyse des performances : Modèle **{best_name}**")
    
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: flex-start;">
            <p style="font-size: 14px; color: rgba(49, 51, 63, 0.6); margin-bottom: 0px;">Pannes Évitées</p>
            <p style="font-size: 42px; color: #28a745; margin-top: -10px; font-weight: 400; font-family: 'Source Sans Pro', sans-serif;">{int(tp)}</p>
        </div>
    """, unsafe_allow_html=True)

with c2: 
    # Delta inversé car on veut que les pannes manquées soient "rouges" si elles augmentent
    st.metric("Pannes Manquées", int(fn))

with c3: 
    # Ici on affiche le coût actuel, avec le % de gain en dessous
    st.metric(
        label="Coût Actuel", 
        value=f"{int(current_cost):,} €".replace(",", " "),
        delta=f"{pct_gain:.1f}% d'économie",
        delta_color="normal" # "normal" = positif en vert, ce qui est parfait pour une économie
    )
    
# Graphiques
st.divider()
col_left, col_right = st.columns([2, 1])

with col_left:
    ts = np.linspace(0.05, 0.9, 50)
    cs = [((confusion_matrix(res['y_true'], (y_probs >= t).astype(int)).ravel()[2] * c_unplanned) + 
           (confusion_matrix(res['y_true'], (y_probs >= t).astype(int)).ravel()[3] * c_maintenance) + 
           (confusion_matrix(res['y_true'], (y_probs >= t).astype(int)).ravel()[1] * c_false_alert)) for t in ts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=cs, name='Coût', line=dict(color='#003399', width=2)))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig.update_layout(title="Courbe d'Optimisation des Coûts", height=350, margin=dict(t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    fig_pie = px.pie(values=[fn*c_unplanned, tp*c_maintenance, fp*c_false_alert], 
                     names=['Pannes', 'Maintenance', 'Alertes'], hole=0.5)
    fig_pie.update_layout(title="Répartition Budget", height=350, margin=dict(t=30, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)
    
st.sidebar.subheader("🚀 Mode Démonstration")
# --- MISE À JOUR DU MENU DE SIMULATION ---
scenario = st.sidebar.selectbox("Simuler un état machine", [
    "Production Normale", 
    "Alerte Surcharge (OSF)",      
    "Alerte Usure Outil (TWF)", 
    "Alerte Surchauffe (HDF)", 
    "Alerte Puissance (PWF)"       
])

import time

# --- LOGIQUE DE DIAGNOSTIC TECHNIQUE ---
def predict_cause_hybrid(row_features, prob_failure, threshold=0.05):
    if prob_failure < threshold:
        return "Normal"
    
    features_multi = [
        'Type', 'Air_Temp', 'Process_Temp', 'Rotational_Speed', 
        'Torque', 'Tool_Wear', 'Temp_Diff', 'Power', 
        'Strain', 'Heat_Stress'
    ]
    

    input_df = pd.DataFrame([row_features])[features_multi]
    
    multi_pred = model_multi.predict(input_df)[0]
    
    mapping_multi = {1: 'TWF', 2: 'HDF', 3: 'PWF', 4: 'OSF', 0: 'RNF'}
    res_ml = mapping_multi.get(multi_pred, "RNF")
    
    if row_features['Tool_Wear'] >= 190: return 'TWF'
    if row_features['Strain'] > 11000: return 'OSF'
        
    return res_ml

def afficher_jauge(prob, threshold):
    # Couleur : Vert si OK, Rouge si au-dessus du seuil
    couleur_barre = "#28a745" if prob < threshold else "#ff4b4b"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': "Probabilité de Panne (%)", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': couleur_barre},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold*100], 'color': 'rgba(40, 167, 69, 0.1)'},
                {'range': [threshold*100, 100], 'color': 'rgba(255, 75, 75, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(t=50, b=0, l=20, r=20))
    return fig

# --- AJOUT MODE USINE (FLOTTE) ---
st.divider()
st.subheader("🏭 Surveillance Globale du Parc (2 900 Unités)")
f1, f2, f3 = st.columns(3)

# Simulation de l'état du parc basée sur les stats réelles
machines_en_alerte = int(2900 * (1 - res['y_probs'].mean()) * 0.05) 
f1.metric("Disponibilité Parc", "98.4%", "+0.2%")
f2.metric("Unités en Alerte", machines_en_alerte, "-2")
f3.metric("Productivité Optimisée", "+12%", "Silver AI")

# --- LOGIQUE DE FLOTTE (Étape 4) ---
def get_fleet_status(df_full, current_pipeline, threshold):
    """Analyse l'ensemble du parc réel (10 000 unités)"""
    
    # 1. On utilise TOUT le dataframe au lieu d'un .sample()
    fleet_data = df_full.copy()
    
    # 2. Préparation des features pour le pipeline
    features_list = ['Type', 'Air_Temp', 'Process_Temp', 'Rotational_Speed', 'Torque', 'Tool_Wear', 'Temp_Diff', 'Power', 'Strain', 'Heat_Stress']
    X_fleet = fleet_data[features_list]
    
    # 3. Calcul des probabilités pour TOUTES les machines
    probs = current_pipeline.predict_proba(X_fleet)[:, 1]
    
    # 4. Identification des unités critiques
    fleet_data['Risk_Score'] = probs
    critical_units = fleet_data[fleet_data['Risk_Score'] >= threshold].copy()
    
    # 5. Score de Santé Global basé sur le parc TOTAL
    total_units = len(fleet_data)
    health_score = 100 - (len(critical_units) / total_units * 100)
    
    # On retourne le score et la liste triée par Risque (décroissant)
    return round(health_score, 2), critical_units.sort_values(by='Risk_Score', ascending=False)

# --- DANS LA SECTION AFFICHAGE ---
st.divider()
health, critical_list = get_fleet_status(df, pipeline, threshold)

# Affichage du Score Global avec une couleur dynamique
color_ghs = "#003399" if health > 95 else "#3366CC" if health > 85 else "#002266"

st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-top: 10px solid {color_ghs}; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin:0; color: #1f1f1f;">🌍 Indice de Santé Global du Parc</h2>
        <p style="font-size: 48px; font-weight: bold; color: {color_ghs}; margin: 0;">{health}%</p>
        <p style="color: #666; font-style: italic;">Supervision Silver AI - 2 900 unités</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

st.write("### 🚨 Top 5 des Unités Critiques")

if not critical_list.empty:
    # 1. On prépare les données pour le tri
    display_fleet = critical_list.copy()
    
    # 2. TRI MULTI-NIVEAUX : 
    # Niveau 1 : Type (2=High, 1=Medium, 0=Low) -> Descendant pour avoir High en premier
    # Niveau 2 : Risk_Score -> Descendant pour avoir le plus gros risque en premier
    display_fleet = display_fleet.sort_values(
        by=['Type', 'Risk_Score'], 
        ascending=[False, False]
    ).head(5)
    
    # 3. Diagnostic IA 
    display_fleet['Diagnostic IA'] = display_fleet.apply(
        lambda row: predict_cause_hybrid(row, row['Risk_Score'], threshold), axis=1
    )
    
    # 4. Traduction pour l'affichage (Mapping)
    type_map = {0: "Low", 1: "Medium", 2: "High"}
    display_fleet['Gamme'] = display_fleet['Type'].map(type_map)
    
    # 5. Formatage du Risque avec strictement 2 chiffres après la virgule
    display_fleet['Risque %'] = display_fleet['Risk_Score'].apply(lambda x: f"{x*100:.2f}%")
    # Plus simplement via f-string pour garantir la consigne :
    display_fleet['Risque'] = display_fleet['Risk_Score'].apply(lambda x: f"{round(x*100, 2):.2f}%")
    
    # 6. Sélection des colonnes finales
    final_table = display_fleet[['Product ID', 'Gamme', 'Diagnostic IA', 'Risque']]
    final_table.columns = ['ID Machine', 'Gamme', 'Diagnostic IA', 'Risque %']
    
    # 7. Affichage
    st.dataframe(final_table, hide_index=True, use_container_width=True)
    
else:
    st.success("✅ Aucune unité critique détectée pour le moment.")

# --- 2. FONCTIONS CŒUR (Définies au début) ---

def ask_ai_chat(prompt, context_data):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        system_prompt = f"""
        Tu es l'expert technique par Silver AI. 
        Ton rôle est purement DESCRIPTIF et FACTUEL.
        
        CONTEXTE DE LA FLOTTE : {context_data}
        
        DIRECTIVES STRICTES :
        0. Ne dis jamais "En tant qu'IA". Tu es un consultant Silver AI.
        1. Analyse les données de manière factuelle : utilise uniquement les chiffres fournis.
        2. Réponds de manière extrêmement concise et stratégique.
        3. Ne pose pas de question en retour.
        4. Si on te pose une question sur les coûts, utilise les chiffres du contexte.
        5. Sois proactif : propose des ajustements de seuil si tu vois que les pertes sont élevées.
        6. FORMATAGE : Ne donne jamais plus de 2 chiffres après la virgule dans tes analyses.
        7. Quand tu parles de gain parle en Euros
        8. Reviens à la ligne à chaque étape de ton analyse
        """
        
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
        messages.append({"role": "user", "content": prompt})
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.3, 
        )
        return chat_completion.choices[0].message.content
    except Exception:
        return "Analyse factuelle prête. Posez votre question."
    
# Préparation d'un contexte pour l'IA
context_ia = {
    "economie": {
        "gain_net": total_savings,
        "cout_actuel": current_cost,
        "pertes_pannes_manquees": fn * c_unplanned,
        "cout_fausses_alertes": fp * c_false_alert
    },
    "technique": {
        "pannes_evitees": tp,
        "pannes_ratees": fn,
        "fiabilite_capteurs": reliability_scores, 
        "seuil_actuel": threshold
    }
}

# --- 3. INITIALISATION DU SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. RÉSERVATION DES ESPACES (Ordre d'affichage immuable) ---
# Ces conteneurs fixent la position des éléments sur la page
container_titre = st.container()
container_surveillance = st.container()
container_audit = st.container()
container_chatbot = st.container()

# --- 5. LOGIQUE DU CHATBOT ---
with container_chatbot:
    st.divider()
    st.subheader("💬 Chatter avec l'Expert Silver AI")
    
    # Zone d'affichage des messages
    chat_placeholder = st.container()
    with chat_placeholder:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Zone de saisie
    if prompt := st.chat_input("Posez une question sur les coûts ou les risques..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_placeholder:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = ask_ai_chat(prompt, context_ia)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- SECTION SURVEILLANCE DYNAMIQUE ---
st.sidebar.divider()

if st.sidebar.checkbox("🚀 Activer Surveillance Temps Réel", key="check_live_monitoring"):
    st.subheader("📡 Flux de données capteurs en direct")
    
    type_flux = st.sidebar.radio("Simulation", ["Normal", "Forcer Anomalie"], key="radio_simulation_type")
    
    # --- 1. RÉSERVATION DES ESPACES ---
    placeholder_jauge = st.empty()
    placeholder_info = st.empty()

    # --- 2. AFFICHAGE DE L'AUDIT TECHNIQUE ---
    st.divider()
    st.subheader("🛡️ Audit de Fiabilité Technique")
    
    data_fiab = [
        {'Type': 'Usure Outil (TWF)', 'Score': reliability_scores.get('TWF', 0)},
        {'Type': 'Thermique (HDF)', 'Score': reliability_scores.get('HDF', 0)},
        {'Type': 'Puissance (PWF)', 'Score': reliability_scores.get('PWF', 0)},
        {'Type': 'Surcharge (OSF)', 'Score': reliability_scores.get('OSF', 0)}
    ]

    df_fiabilite = pd.DataFrame(data_fiab)

    # Création du graphique avec les chiffres REELS de ton joblib
    fig_fiab = px.bar(
        df_fiabilite, 
        x='Score', 
        y='Type', 
        orientation='h', 
        text=df_fiabilite['Score'].apply(lambda x: f'{x}%'),
        color='Score',
        color_continuous_scale=['#ff4b4b', '#ffa500', '#28a745'], 
        range_x=[0, 115] # Marge pour ne pas couper le texte
    )

    fig_fiab.update_layout(
        showlegend=False, 
        height=300, 
        margin=dict(t=20, b=20, l=0, r=0), 
        coloraxis_showscale=False,
        xaxis_title="Taux de détection (%)"
    )
    
    c_graph, c_stat = st.columns([2, 1])
    with c_graph:
        st.plotly_chart(fig_fiab, use_container_width=True)
    with c_stat:
        score_global = reliability_scores.get('GLOBAL', 0)
        st.metric("FIABILITÉ GLOBALE", f"{score_global}%")
        st.write("Détail technique :")
        st.caption(f"Le modèle détecte en moyenne {score_global}% des pannes réelles sur le set de validation.")

    # --- 3. LANCEMENT DE LA BOUCLE DE SIMULATION ---
    for i in range(50):
        # A. SÉLECTION DE LA DONNÉE
        if type_flux == "Normal":
            sample = df[df['Failure'] == 0].sample(1)
        else:
            if "OSF" in scenario: sample = df[df['OSF'] == 1].sample(1)
            elif "HDF" in scenario: sample = df[df['HDF'] == 1].sample(1)
            elif "TWF" in scenario: sample = df[df['TWF'] == 1].sample(1)
            elif "PWF" in scenario: sample = df[df['PWF'] == 1].sample(1)
            else: sample = df[df['Failure'] == 1].sample(1)
                
        machine_id = sample.iloc[0]['Product ID']
        row = sample.iloc[0].copy()

        row['Temp_Diff'] = row['Process_Temp'] - row['Air_Temp']
        row['Power'] = row['Torque'] * row['Rotational_Speed']
        row['Strain'] = row['Torque'] * row['Tool_Wear']
        row['Heat_Stress'] = row['Rotational_Speed'] * row['Temp_Diff']

        features_list = ['Type', 'Air_Temp', 'Process_Temp', 'Rotational_Speed', 'Torque', 'Tool_Wear', 'Temp_Diff', 'Power', 'Strain', 'Heat_Stress']
        input_data = pd.DataFrame([row])[features_list] 
        prob_panne = pipeline.predict_proba(input_data)[0][1]
                
        with placeholder_jauge:
            st.plotly_chart(afficher_jauge(prob_panne, threshold), use_container_width=True, key=f"jauge_live_{i}")
        
        with placeholder_info:
            if prob_panne >= threshold:
                type_panne = predict_cause_hybrid(row, prob_panne, threshold)
                st.error(f"🚨 ALERTE : {machine_id} | Défaut détecté : {type_panne} | Probabilité : {prob_panne:.1%}")
            else:
                st.success(f"✅ Unité {machine_id} : Fonctionnement nominal ({prob_panne:.1%})")
                    
        time.sleep(1.5)

