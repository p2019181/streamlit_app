import streamlit as st
import scanpy as sc
import scanpy.external as sce
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import hdf5plugin
import random

np.random.seed(42)
random.seed(42)

plt.style.use('default')

st.set_page_config(page_title="Single-Cell Analysis App", layout="wide")

st.title("🔬 Single-Cell RNA-Seq Analysis App")

# --- Sidebar Παράμετροι για την Προεπεξεργασία των δεδομένων---
st.sidebar.header("⚙️ Ρυθμίσεις Παραμέτρων")

min_genes = st.sidebar.slider("Ελάχιστα γονίδια ανά κύτταρο", 0, 1800, 600)
min_cells = st.sidebar.slider("Ελάχιστα κύτταρα ανά γονίδιο", 0, 50, 3)
min_disp = st.sidebar.slider("Minimum Dispersion", 0.1, 5.0, 0.5)
max_mean = st.sidebar.slider("Maximum Mean Expression", 1.0, 5.0, 3.0)

# --- Load data ---
@st.cache_resource
def load_data():
    adata = sc.read("data/pancreas_data.h5ad")
    return adata.copy()

original_adata = load_data()

if "adata" not in st.session_state:
    st.session_state["adata"] = original_adata.copy()



tab1, tab2, tab3, tab4 = st.tabs(["About Us","Dataset","Preprocessing", "Differential Expression"])
# --- About Us ---
with tab1:
    st.markdown("<h3 style='text-align: center;'>ΙΟΝΙΟ ΠΑΝΕΠΙΣΤΗΜΙΟ</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.image("apps/IU.png", width=250)

    st.markdown("<h4 style='text-align: center;'>ΤΜΗΜΑ ΠΛΗΡΟΦΟΡΙΚΗΣ</h4> <br><br><br>", unsafe_allow_html=True)

    st.subheader(" Ποιοι είμαστε;")
    st.write("Είμαστε μια ομάδα προγραμματιστών του Τμήματος Πληροφορικής του Ιόνιου Πανεπιστημίου που αναπτύσει εργαλεία για την ανάλυση δεδομένων RNA-Seq.")
    st.write("Προγραματιστής είναι ο Μαριόλ Μιέστρι Π2019181")
    st.write("Αυτή η εφαρμογή έγινε στα πλαίσια του μαθήματος ΤΕΧΝΟΛΟΓΙΑ ΛΟΓΙΣΜΙΚΟΥ με επιβλέποντα καθηγητή Άρη Βραχάτη.")
    st.write("Σχεδιάστηκε για να διευκολύνει την ανάλυση δεδομένων single-cell RNA-Seq.")
    st.write("Η εφαρμογή μας επιτρέπει στους χρήστες να εκτελούν προεπεξεργασία, ανάλυση και οπτικοποίηση δεδομένων RNA-Seq.")

# --- Dataset ---
with tab2:
    st.write(st.session_state["adata"].obs)
    st.write(f"📊 Dataset με {st.session_state['adata'].n_obs} κύτταρα και {st.session_state['adata'].n_vars} γονίδια")

# --- Preprocessing ---
with tab3:
    st.subheader("🔧 Προεπεξεργασία")
    dim_reduction_method = st.selectbox("Επιλογή Dimensionality Reduction", ["UMAP", "t-SNE"], key="pre_dim")# Επιλογή μεθόδου για Dimensionality Reduction UMAP ή t-SNE
    if st.button("🚀 Εκτέλεση Preprocessing"):
        adata = original_adata.copy()
        sc.pp.filter_cells(adata, min_genes=min_genes)# Φιλραρισμα κυττάρων με βάση το ελάχιστο αριθμό γονιδίων που έχει δώσει ο χρήστης 
        sc.pp.filter_genes(adata, min_cells=min_cells)# Φιλτραρισμα γονιδίων με βάση το ελάχιστο αριθμό κυττάρων που έχει δώσει ο χρήστης
        adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(('ERCC', 'MT-', 'mt-'))]].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=max_mean, min_disp=min_disp)# Υπολογισμός γονιδίων υψηλής μεταβλητότητας
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, random_state=42)
        sc.tl.umap(adata, random_state=42)
        sc.tl.tsne(adata, random_state=42)

        st.session_state["adata"] = adata
        st.success("✅ Preprocessing ολοκληρώθηκε!")

        # Επιλογή Dimensionality Reduction μετά το Preprocessing
        st.subheader("📈 Visualization μετά το Preprocessing")
        
        # Οπτικοποίηση των αποτελεσμάτων με UMAP ή t-SNE
        if dim_reduction_method == "UMAP":
            sc.pl.umap(adata, color=['celltype', 'batch'], legend_fontsize=10)
        elif dim_reduction_method == "t-SNE":
            sc.pl.tsne(adata, color=['celltype', 'batch'], legend_fontsize=10)
        st.pyplot(plt.gcf())
        plt.clf()
    
    # --- Harmony Integration ---
    st.subheader("🔄 Harmony Integration")
    # Επιλογή μεθόδου για Dimensionality Reduction UMAP ή t-SNE μετά το Harmony Integration
    dim_reduction_method_harmony = st.selectbox("Επιλογή Dimensionality Reduction μετά το Harmony", ["UMAP", "t-SNE"], key="harmony_dim")
    
    if st.button("✨ Εκτέλεση Harmony Integration"):
        adata = st.session_state["adata"].copy()
        
        if "X_pca" not in adata.obsm:
            st.warning("🚨 Δεν βρέθηκε PCA. Υπολογίζω τώρα...")
            sc.pp.pca(adata)
        
        if np.isnan(adata.obsm["X_pca"]).any():
            st.error("🚨 Τα δεδομένα PCA περιέχουν NaNs. Δεν μπορεί να γίνει Harmony.")
            st.stop()
        # Εκτέλεση Harmony Integration
        sce.pp.harmony_integrate(adata, 'batch')
        sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    
        if dim_reduction_method_harmony == "UMAP":
            sc.tl.umap(adata, random_state=42)
        elif dim_reduction_method_harmony == "t-SNE":
            sc.tl.tsne(adata, use_rep="X_pca_harmony", random_state=42)
        
        st.session_state["adata"] = adata  # Ανανέωση του επεξεργασμένου μετά το Harmony

        st.success("✅ Harmony Integration ολοκληρώθηκε!")
        st.subheader("📈 Visualization μετά το Harmony Integration")
        # Οπτικοποίηση των αποτελεσμάτων με UMAP ή t-SNE μετά το Harmony Integration
        if dim_reduction_method_harmony == "UMAP":
            sc.pl.umap(adata, color=['batch', 'celltype'], legend_fontsize=10)
        elif dim_reduction_method_harmony == "t-SNE":
            sc.pl.tsne(adata, color=['batch', 'celltype'], legend_fontsize=10)
        st.pyplot(plt.gcf())
        plt.clf()

    # --- Find Clusters ---
    st.subheader("🧬 Clustering")
    resolution = st.slider("Resolution", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    dim_reduction_method_clustering = st.selectbox("Επιλογή Dimensionality Reduction για Clustering", ["UMAP", "t-SNE"], key="clustering_dim")
    if st.button("🔍 Εκτέλεση Clustering"):
        adata = st.session_state["adata"].copy()
        # Έλεγχος αν υπάρχει ήδη υπολογισμένος γράφος γειτονικών κυττάρων
        if "neighbors" not in adata.uns:
            st.warning("🚨 Δεν βρέθηκε γράφος γειτονικών κυττάρων. Υπολογίζεται τώρα...")
            sc.pp.neighbors(adata, use_rep="X_pca_harmony")

        sc.tl.leiden(adata, resolution=resolution, random_state=42)
        cluster_key = "leiden"
        st.session_state["adata"] = adata  # Ενημέρωση με clustering αποτέλεσμα
        st.success(f"✅ Clustering με Leiden ολοκληρώθηκε!")

        # Οπτικοποίηση των clusters
        st.subheader("📊 Visualization Clusters")
        if dim_reduction_method_clustering == "UMAP" and "X_umap" in adata.obsm:
            sc.pl.umap(adata, color=[cluster_key], legend_fontsize=10)
        elif dim_reduction_method_clustering == "t-SNE" and "X_tsne" in adata.obsm:
            sc.pl.tsne(adata, color=[cluster_key], legend_fontsize=10)
        else:
            st.warning("Δεν βρέθηκε Dimensionality Reduction για Visualization.")
        st.pyplot(plt.gcf())
        plt.clf()

# --- Differential Expression ---
with tab4:
    st.subheader("📈 Differential Expression Analysis")
    # Επιλογή μεθόδου (Wilcoxon, t-test) από το χρήστη
    de_method = st.selectbox("Μέθοδος Ανάλυσης", ["wilcoxon", "t-test"])
    plot_type = st.selectbox("Επιλογή γραφήματος", ["Volcano", "DotPlot"])# Επιλογή τύπου γραφήματος (Volcano ή DotPlot) από το χρήστη
    
    if "disease" in original_adata.obs.columns:
        disease_groups = original_adata.obs["disease"].unique().tolist()
    else:
        st.warning("Το adata.obs δεν περιέχει στήλη 'disease'. Δεν μπορεί να γίνει DE analysis.")
        st.stop()
    # 
    if st.button("📈 Εκτέλεση DE Analysis"):
        sc.tl.rank_genes_groups(
            original_adata,
            groupby='disease', # the column in adata.obs
            method=de_method, # Μέθοδος ανάλυσης (Wilcoxon ή t-test)
            groups=['case'], # test 'case' vs 'control'
            reference='control', # control group
            use_raw=False
        )

        if plot_type == "Volcano":
            deg_result = original_adata.uns["rank_genes_groups"]
            degs_df = pd.DataFrame({
                "genes": deg_result["names"]["case"],
                "pvals": deg_result["pvals"]["case"],
                "pvals_adj": deg_result["pvals_adj"]["case"],
                "logfoldchanges": deg_result["logfoldchanges"]["case"],
            })

            degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"])
            # Add a column for differential expression classification
            degs_df["diffexpressed"] = "NS"
            degs_df.loc[(degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "UP"
            degs_df.loc[(degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "DOWN"

            # Select top downregulated genes (prioritize by highest significance, then most negative log2FC)
            top_downregulated = degs_df[degs_df["diffexpressed"] == "DOWN"]
            top_downregulated = top_downregulated.sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, True]).head(20)

            # Select top upregulated genes (prioritize by highest significance, then most positive log2FC)
            top_upregulated = degs_df[degs_df["diffexpressed"] == "UP"]
            top_upregulated = top_upregulated.sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, False]).head(81)

            # Combine top genes
            top_genes_combined = pd.concat([top_downregulated["genes"], top_upregulated["genes"]])
            df_annotated = degs_df[degs_df["genes"].isin(top_genes_combined)]

            st.subheader("📄 Top DEGs")
            st.dataframe(degs_df)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=degs_df, x="logfoldchanges", y="neg_log10_pval", hue="diffexpressed",
                            palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
                            alpha=0.7, edgecolor=None)
            plt.axhline(y=-np.log10(0.05), color='gray', linestyle='dashed')
            plt.axvline(x=-1, color='gray', linestyle='dashed')
            plt.axvline(x=1, color='gray', linestyle='dashed')
            plt.xlim(-11, 11)
            plt.xlabel("log2 Fold Change", fontsize=14)
            plt.ylabel("-log10 p-value", fontsize=14)
            plt.title(f"Volcano Plot: {'case'} vs {'control'} ({de_method})", fontsize=16)
            plt.legend(title="Expression", loc="upper right")
            st.pyplot(plt.gcf())
            plt.clf()
            
        elif plot_type == "DotPlot":
            st.subheader("🔵 DotPlot των κορυφαίων DEGs")
            sc.pl.rank_genes_groups_dotplot(original_adata, n_genes=20)
            st.pyplot(plt.gcf())
            plt.clf()
