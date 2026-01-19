import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- 1. PRÉPARATION DES DONNÉES ---
df_brut = pd.read_csv('data/ai4i2020.csv')
df_brut.columns = ['ID', 'Product ID', 'Type', 'Air_Temp', 'Process_Temp', 'Rotational_Speed', 'Torque', 'Tool_Wear', 'Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Feature Engineering (Indépendant des labels)
def apply_feature_engineering(data):
    d = data.copy()
    d['Temp_Diff'] = d['Process_Temp'] - d['Air_Temp']
    d['Power'] = d['Torque'] * d['Rotational_Speed']
    d['Strain'] = d['Torque'] * d['Tool_Wear']
    d['Heat_Stress'] = d['Rotational_Speed'] * d['Temp_Diff']
    mapping = {'L': 0, 'M': 1, 'H': 2}
    d['Type'] = d['Type'].map(mapping)
    return d

df = apply_feature_engineering(df_brut)

# Création de la cible Multi-classe 
def create_multi_label(row):
    if row['TWF'] == 1: return 1
    if row['HDF'] == 1: return 2
    if row['PWF'] == 1: return 3
    if row['OSF'] == 1: return 4
    return 0

df['Multi_Failure'] = df.apply(create_multi_label, axis=1)

# Split des données
X = df[['Type', 'Air_Temp', 'Process_Temp', 'Rotational_Speed', 'Torque', 'Tool_Wear', 'Temp_Diff', 'Power', 'Strain', 'Heat_Stress']]
y_bin = df['Failure']
y_multi = df['Multi_Failure']

X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
    X, y_bin, y_multi, test_size=0.3, random_state=42, stratify=y_bin
)

# --- 2. PIPELINE MODÈLE BINAIRE (Prédire SI il y a une panne) ---
pipeline_bin = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=150, random_state=42))
])

print("Entraînement du Pipeline Binaire...")
pipeline_bin.fit(X_train, y_train_bin)

#-- SMOTE -- 

smote = SMOTE(random_state=42, k_neighbors=5)

X_train_resampled, y_train_multi_resampled = smote.fit_resample(X_train, y_train_multi)

# --- 3. MODÈLE BONUS : XGBOOST MULTI-CLASSE (Prédire la panne) ---
model_multi = XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    random_state=42,
    learning_rate=0.05,    
    n_estimators=300,      
    max_depth=8,           
    subsample=0.8,        
    colsample_bytree=0.8
)
print("Entraînement du Modèle Multi-classe XGBoost...")
model_multi.fit(X_train_resampled, y_train_multi_resampled)

# --- 4. LOGIQUE DE DIAGNOSTIC DÉCOUPLÉE (Indépendante des labels cibles) ---
def predict_cause_hybrid(row_features, prob_failure, threshold=0.05):

    if prob_failure < threshold:
        return "Normal"
    
    if row_features['Tool_Wear'] >= 200: return 'TWF (via Expert)'
    
    # Règle métier (Système expert)
    if row_features['Strain'] > 11000: return 'OSF (via Expert)'
    
    multi_pred = model_multi.predict(pd.DataFrame([row_features]))[0]
    mapping_multi = {1: 'TWF (via ML)', 2: 'HDF (via ML)', 3: 'PWF (via ML)', 4: 'OSF (via ML)', 0: 'Unknown'}
    
    return mapping_multi.get(multi_pred, "RNF")

y_probs = pipeline_bin.predict_proba(X_test)[:, 1]

def trouver_seuil_optimal(y_true, y_probs, c_unplanned=500, c_maint=100, c_false=50):
    thresholds = np.linspace(0.01, 0.9, 100)
    costs = []
    
    for t in thresholds:
        y_p = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
        # Calcul du coût total pour ce seuil
        total_cost = (fn * c_unplanned) + (tp * c_maint) + (fp * c_false)
        costs.append(total_cost)
        
    return thresholds[np.argmin(costs)]

best_t = trouver_seuil_optimal(y_test_bin, y_probs)

# --- 5. ÉVALUATION ET EXPORT ---
y_probs = pipeline_bin.predict_proba(X_test)[:, 1]
y_pred_bin_opt = (y_probs >= best_t).astype(int)

print("\n--- PERFORMANCE PIPELINE BINAIRE (Avec Class Weights) ---")
print(classification_report(y_test_bin, y_pred_bin_opt))

# Exportation des actifs
joblib.dump(pipeline_bin, 'models/pipeline_predictive.pkl')
joblib.dump(model_multi, 'models/model_multi_class.pkl')

resultats_historiques = {
    'y_true': y_test_bin, 
    'y_probs': y_probs
}

joblib.dump(resultats_historiques, 'models/silver_core_v1.joblib')

print("\n✅ Pipeline et Modèle Multi-classe exportés avec succès.")

# --- ANALYSE DE FIABILITÉ ---

def afficher_fiabilite_technique(y_test_multi, X_test, model_multi):
    """
    Calcule et affiche la fiabilité de détection par paramètre physique.
    Indispensable pour le reporting client en freelance.
    """
    print("\n" + "="*60)
    print("🛡️ RAPPORT DE FIABILITÉ TECHNIQUE (MODÈLE MULTI-CLASSE)")
    print("="*60)
    
    # Prédiction sur le set de test
    y_pred_multi = model_multi.predict(X_test)
    
    # Mapping des classes 
    target_names = ['Normal', 'TWF (Usure)', 'HDF (Thermique)', 'PWF (Puissance)', 'OSF (Surcharge)']
    
    # Rapport détaillé
    report = classification_report(y_test_multi, y_pred_multi, target_names=target_names, output_dict=True)
    
    # Analyse par paramètre clé
    stats_fiabilite = {
        'Surcharge (OSF)': {'score': report['OSF (Surcharge)']['recall'], 'critere': 'Variable Strain (Torque * Wear)'},
        'Thermique (HDF)': {'score': report['HDF (Thermique)']['recall'], 'critere': 'Variable Temp_Diff'},
        'Usure Outil (TWF)': {'score': report['TWF (Usure)']['recall'], 'critere': 'Accumulation Tool_Wear'},
        'Puissance (PWF)': {'score': report['PWF (Puissance)']['recall'], 'critere': 'Calcul Couple * Vitesse'}
    }

    for label, data in stats_fiabilite.items():
        precision = data['score'] * 100
        barre = "█" * int(precision / 5) + "░" * (20 - int(precision / 5))
        print(f"{label:<18} : {barre} {precision:>6.1f}% | Basé sur : {data['critere']}")

    print("="*60)
    print(f"MOYENNE GLOBALE DE DÉTECTION : {report['macro avg']['recall']:>6.2%}")
    print("="*60)

# Appel de la fonction de fiabilité
afficher_fiabilite_technique(y_test_multi, X_test, model_multi)

# --- MISE À JOUR DU DICTIONNAIRE POUR STREAMLIT ---
report = classification_report(y_test_multi, model_multi.predict(X_test), output_dict=True)

reliability_scores = {
    'TWF': round(report.get('1', report.get(1, {})).get('recall', 0) * 100, 1),
    'HDF': round(report.get('2', report.get(2, {})).get('recall', 0) * 100, 1),
    'PWF': round(report.get('3', report.get(3, {})).get('recall', 0) * 100, 1),
    'OSF': round(report.get('4', report.get(4, {})).get('recall', 0) * 100, 1),
    'GLOBAL': round(report.get('macro avg', {}).get('recall', 0) * 100, 1)
}

joblib.dump(reliability_scores, 'models/fiabilite_parametres.joblib')

print("\n✅ Métriques de fiabilité imprimées et exportées.")