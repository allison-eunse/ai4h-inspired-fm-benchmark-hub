import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_external_integrations():
    """Check if external repositories are correctly installed."""
    integrations = {}
    
    # Check Caduceus
    try:
        sys.path.append(os.path.join(os.getcwd(), 'external/caduceus'))
        from caduceus.modeling_caduceus import CaduceusForMaskedLM
        integrations['caduceus'] = "Available"
    except ImportError as e:
        if "mamba_ssm" in str(e):
            integrations['caduceus'] = "Partial (Missing mamba-ssm - CUDA required)"
        else:
            integrations['caduceus'] = f"Not found ({e})"
    except Exception as e:
        integrations['caduceus'] = f"Error ({e})"
        
    # Check Bagel
    try:
        sys.path.append(os.path.join(os.getcwd(), 'external/bagel'))
        import inferencer
        integrations['bagel'] = "Available"
    except ImportError:
        integrations['bagel'] = "Not found"
        
    # Check Other Integrations
    repo_list = [
        'brainlm', 'dnabert2', 'evo2', 'generator', 'swift', 
        'titan', 'me-lamma', 'brainjepa', 'brainharmony', 
        'hyena', 'brainmt', 'M3FM', 'MoT'
    ]
    
    for repo in repo_list:
        path = os.path.join(os.getcwd(), f'external/{repo}')
        if os.path.exists(path):
            integrations[repo] = "Found (Folder exists)"
        else:
            integrations[repo] = "Not Found"

    return integrations

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def mock_data_loading(n_subjects=100, n_genes=38, gene_dim=256, n_rois=176):
    """
    Mock data loading since we don't have access to the full UKB dataset.
    Generates synthetic embeddings to demonstrate the pipeline.
    """
    logging.info("Generating synthetic data for initial analysis...")
    
    # Synthetic Genetics Embeddings (N, 38 * 256)
    # We simulate 38 genes, each with a 256-dim embedding
    X_genetics = np.random.randn(n_subjects, n_genes * gene_dim)
    
    # Synthetic sMRI Features (N, 176)
    X_smri = np.random.randn(n_subjects, n_rois)
    
    # Synthetic Metadata
    metadata = pd.DataFrame({
        'subject_id': [f'sub-{i:04d}' for i in range(n_subjects)],
        'site_scanner': np.random.choice(['SiteA', 'SiteB', 'SiteC'], n_subjects),
        'mdd_diagnosis': np.random.choice([0, 1], n_subjects),
        'age': np.random.normal(60, 10, n_subjects),
        'sex': np.random.choice(['M', 'F'], n_subjects)
    })
    
    return X_genetics, X_smri, metadata

def preprocess_features(X, n_components=50):
    """
    Standardize and reduce dimensionality using PCA.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca

def run_cca_analysis(X_a, X_b, n_components=10):
    """
    Run Canonical Correlation Analysis.
    """
    cca = CCA(n_components=n_components)
    cca.fit(X_a, X_b)
    X_a_c, X_b_c = cca.transform(X_a, X_b)
    
    # Calculate canonical correlations
    # Correlation between the corresponding canonical variates
    corrs = [np.corrcoef(X_a_c[:, i], X_b_c[:, i])[0, 1] for i in range(n_components)]
    
    return corrs, cca

def main():
    config_path = "experiments/01_cca_gene_smri.yaml"
    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)
    logging.info(f"Starting Experiment: {config['experiment_name']}")
    
    # Check Integrations
    integrations = check_external_integrations()
    logging.info(f"External Integrations: {integrations}")

    # 1. Load Data (Mocked for now)
    # In a real run, this would load from config['datasets']['genetics'] and ['smri']
    X_genes, X_smri, metadata = mock_data_loading()
    
    # 2. Preprocessing
    logging.info("Preprocessing features...")
    # Project to 512 dim as per config (using 50 here for toy example stability)
    target_dim = min(X_genes.shape[0], 50) 
    X_genes_pca = preprocess_features(X_genes, n_components=target_dim)
    X_smri_pca = preprocess_features(X_smri, n_components=target_dim)
    
    # 3. Cross-Validation & CCA
    cv = StratifiedGroupKFold(n_splits=config['cross_validation']['k_folds'])
    
    fold_results = []
    
    groups = metadata['site_scanner']
    y = metadata['mdd_diagnosis']
    
    logging.info(f"Running {config['cross_validation']['k_folds']}-fold CV...")
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_genes_pca, y, groups=groups)):
        X_g_train, X_g_test = X_genes_pca[train_idx], X_genes_pca[test_idx]
        X_s_train, X_s_test = X_smri_pca[train_idx], X_smri_pca[test_idx]
        
        # Fit CCA on train
        corrs, model = run_cca_analysis(X_g_train, X_s_train, n_components=config['analysis']['n_components'])
        
        logging.info(f"Fold {fold+1}: Top Correlation = {corrs[0]:.3f}")
        fold_results.append(corrs)
        
    avg_corrs = np.mean(fold_results, axis=0)
    logging.info("Analysis Complete.")
    logging.info(f"Average Canonical Correlations: {avg_corrs}")
    
    # 4. Permutation Test (Simplified)
    logging.info("Running Permutation Test (First Component)...")
    n_perms = 100
    perm_scores = []
    for _ in range(n_perms):
        # Shuffle subjects in one modality
        X_s_perm = np.random.permutation(X_smri_pca)
        c, _ = run_cca_analysis(X_genes_pca, X_s_perm, n_components=1)
        perm_scores.append(c[0])
    
    p_value = (np.sum(np.array(perm_scores) >= avg_corrs[0]) + 1) / (n_perms + 1)
    logging.info(f"Permutation p-value for 1st component: {p_value:.4f}")

if __name__ == "__main__":
    main()

