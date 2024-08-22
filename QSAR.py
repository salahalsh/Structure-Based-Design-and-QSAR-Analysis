import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create a directory for output files
output_dir = 'qsar_model_output'
os.makedirs(output_dir, exist_ok=True)

def calculate_descriptors(smiles):
    """Calculate an extensive set of molecular descriptors for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    descriptors = {}
    
    # Physicochemical properties
    descriptors['MolWt'] = Descriptors.ExactMolWt(mol)
    descriptors['LogP'] = Descriptors.MolLogP(mol)
    descriptors['HBD'] = Descriptors.NumHDonors(mol)
    descriptors['HBA'] = Descriptors.NumHAcceptors(mol)
    descriptors['PSA'] = Descriptors.TPSA(mol)
    descriptors['RotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    descriptors['AromaticRings'] = Descriptors.NumAromaticRings(mol)
    descriptors['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
    descriptors['Rings'] = Descriptors.RingCount(mol)
    descriptors['MolarRefractivity'] = Crippen.MolMR(mol)
    descriptors['FractionCSP3'] = Descriptors.FractionCSP3(mol)
    descriptors['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)
    descriptors['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
    descriptors['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
    descriptors['MaxAbsPartialCharge'] = Descriptors.MaxAbsPartialCharge(mol)
    descriptors['MinAbsPartialCharge'] = Descriptors.MinAbsPartialCharge(mol)

    # Pharmacokinetic properties (simplified predictions)
    # Lipinski's Rule of Five
    mw, logp, hbd, hba = descriptors['MolWt'], descriptors['LogP'], descriptors['HBD'], descriptors['HBA']
    descriptors['Lipinski'] = int((mw <= 500) + (logp <= 5) + (hbd <= 5) + (hba <= 10))
    
    # Veber's Rule
    rotatable_bonds, psa = descriptors['RotatableBonds'], descriptors['PSA']
    descriptors['Veber'] = int((rotatable_bonds <= 10) and (psa <= 140))
    
    # GI absorption (based on Veber's Rule)
    descriptors['GI_absorption'] = 1 / (1 + np.exp(-0.1 * (140 - psa)))
    
    # BBB permeability (based on a simplified model)
    descriptors['BBB_permeant'] = int((logp - (hbd + hba) / 4 - 2) > 0)
    
    # Plasma protein binding (very simplified estimation)
    descriptors['PPB'] = 1 / (1 + np.exp(-0.2 * (logp - 1.5)))
    
    # CYP inhibition (simplified, based on general structural features)
    descriptors['CYP_inhibition'] = int(descriptors['AromaticRings'] > 1 and descriptors['HBA'] > 2)
    
    # P-glycoprotein substrate (simplified prediction)
    descriptors['Pgp_substrate'] = int(mw > 400 and (hbd + hba) > 8)
    
    return descriptors

def prepare_data(csv_file):
    """Prepare data from CSV file containing SMILES and binding energy."""
    df = pd.read_csv(csv_file)
    X = df['SMILES'].apply(calculate_descriptors).apply(pd.Series)
    y = df['Binding_Energy']
    return X, y

def train_model(X, y):
    """Train a Random Forest model for QSAR and perform cross-validation."""
    # Split the data without relying on 'Ligand_Name'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create dummy names for test set (indices as strings)
    names_test = [str(i) for i in range(len(y_test))]
    
    return model, scaler, mse, rmse, mae, r2, cv_rmse, X_test, y_test, y_pred, names_test, X.columns

def plot_feature_importance(model, feature_names):
    """Plot feature importance of the model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    # Export feature importances to CSV
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)

def plot_correlation_heatmap(X):
    """Plot correlation heatmap of features."""
    corr = X.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # Export correlation matrix to CSV
    corr.to_csv(os.path.join(output_dir, 'feature_correlations.csv'))

def plot_predicted_vs_actual(y_test, y_pred, names_test):
    """Plot predicted vs actual values with ligand names."""
    plt.figure(figsize=(12, 10))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    
    for i, name in enumerate(names_test):
        plt.annotate(name, (y_test.iloc[i], y_pred[i]), fontsize=8, alpha=0.7)
    
    plt.xlabel("Actual Binding Energy")
    plt.ylabel("Predicted Binding Energy")
    plt.title("Predicted vs Actual Binding Energy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=300)
    plt.close()
    
    # Export predicted vs actual data to CSV
    pred_vs_actual_df = pd.DataFrame({
        'Ligand_Name': names_test,
        'Actual_Binding_Energy': y_test,
        'Predicted_Binding_Energy': y_pred
    })
    pred_vs_actual_df.to_csv(os.path.join(output_dir, 'predicted_vs_actual.csv'), index=False)

def plot_residuals(y_test, y_pred, names_test):
    """Plot residuals."""
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 10))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    for i, name in enumerate(names_test):
        plt.annotate(name, (y_pred[i], residuals.iloc[i]), fontsize=8, alpha=0.7)
    
    plt.xlabel("Predicted Binding Energy")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), dpi=300)
    plt.close()

def export_model_summary(model, mse, rmse, mae, r2, cv_rmse, feature_names):
    """Export a summary of the model and its performance."""
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        f.write("QSAR Model Summary\n")
        f.write("==================\n\n")
        f.write(f"Model Type: Random Forest Regressor\n")
        f.write(f"Number of Trees: {model.n_estimators}\n")
        f.write(f"Number of Features: {len(feature_names)}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"  Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"  R-squared (R2) Score: {r2:.4f}\n")
        f.write(f"  Cross-Validation RMSE (mean ± std): {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}\n\n")
        f.write("Top 10 Important Features:\n")
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

def main():
    # Load and prepare data
    X, y = prepare_data('ligand_data.csv')
    
    # Check if 'Ligand_Name' is in X before dropping
    if 'Ligand_Name' in X.columns:
        X_for_correlation = X.drop('Ligand_Name', axis=1)
        names = X['Ligand_Name']
    else:
        X_for_correlation = X
        names = None
    
    # Plot correlation heatmap
    plot_correlation_heatmap(X_for_correlation)
    
    # Train model
    model, scaler, mse, rmse, mae, r2, cv_rmse, X_test, y_test, y_pred, names_test, feature_names = train_model(X, y)
    
    # Export model summary
    export_model_summary(model, mse, rmse, mae, r2, cv_rmse, feature_names)
    
    # Plot feature importance
    plot_feature_importance(model, feature_names)
    
    # Plot predicted vs actual
    plot_predicted_vs_actual(y_test, y_pred, names_test if names_test is not None else range(len(y_test)))
    
    # Plot residuals
    plot_residuals(y_test, y_pred, names_test if names_test is not None else range(len(y_test)))
    
    # Save the model and scaler
    joblib.dump(model, os.path.join(output_dir, 'qsar_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'qsar_scaler.joblib'))
    
    print(f"All outputs have been saved to the '{output_dir}' directory.")

    # Example prediction for a new compound
    new_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin as an example
    new_descriptors = calculate_descriptors(new_smiles)
    new_X = pd.DataFrame([new_descriptors])
    new_X_scaled = scaler.transform(new_X)
    predicted_binding_energy = model.predict(new_X_scaled)[0]
    
    print(f"\nPredicted Binding Energy for {new_smiles}: {predicted_binding_energy:.2f}")

if __name__ == "__main__":
    main()