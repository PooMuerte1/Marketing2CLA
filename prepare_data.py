import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from stepmix import StepMix
import os

REFERENCE_DATE = pd.Timestamp('2026-06-01')
RANDOM_STATE = 42

print("Loading customers.csv and orders.csv...")
df_customers = pd.read_csv('customers.csv')
df_orders = pd.read_csv('orders.csv')

# 1. AGGREGATE PROMOTIONAL METRICS FROM ORDERS
print("Aggregating discount metrics per customer...")
cust_orders = df_orders.groupby('customer_id').agg(
    orders_with_discount=('discount_pct', lambda x: (x > 0).sum()),
    avg_discount_pct=('discount_pct', 'mean'),
    total_discount_usd=('discount_amount_usd', 'sum')
).reset_index()

# Merge into full customers dataframe first so we don't lose anything
df_customers = pd.merge(df_customers, cust_orders, on='customer_id', how='left')
df_customers['orders_with_discount'] = df_customers['orders_with_discount'].fillna(0)
df_customers['avg_discount_pct'] = df_customers['avg_discount_pct'].fillna(0)
df_customers['total_discount_usd'] = df_customers['total_discount_usd'].fillna(0)
df_customers['discount_ratio'] = df_customers['orders_with_discount'] / df_customers['total_orders']
df_customers['discount_ratio'] = df_customers['discount_ratio'].fillna(0)

# 2. PRE-PROCESS AND FIT K-MEANS ON RFM (MATCHING THE IPYNB EXACTLY)
print("Fitting K-Means on RFM exactly as in the notebook (full 8,000 customers)...")

columnas_rfm = ['days_since_last_purchase', 'total_orders', 'total_spend_usd']
columnas_log = ['recency_log', 'frequency_log', 'monetary_log']

# Log transform
df_customers[columnas_log] = np.log1p(df_customers[columnas_rfm])

# Z-score standardization
scaler = StandardScaler()
columnas_finales = ['recency_final', 'frequency_final', 'monetary_final']
df_customers[columnas_finales] = scaler.fit_transform(df_customers[columnas_log])

# KMeans K=3
from sklearn.cluster import KMeans
X_rfm = df_customers[columnas_finales]
kmeans_rfm = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
raw_labels = kmeans_rfm.fit_predict(X_rfm)

# Align cluster labels with the notebook exactly using spend rank:
# Notebook has:
# Cluster 0: High Spend ($2514.8) -> Highest mean spend
# Cluster 1: Medium Spend ($1641.6) -> Second highest mean spend
# Cluster 2: Low Spend ($286.3) -> Lowest mean spend
df_customers['temp_label'] = raw_labels
spend_means = df_customers.groupby('temp_label')['total_spend_usd'].mean().sort_values(ascending=False)

# Mapping: highest spend -> 0, second highest -> 1, lowest -> 2
label_mapping = {
    spend_means.index[0]: 0,
    spend_means.index[1]: 1,
    spend_means.index[2]: 2
}

df_customers['Cluster_RFM'] = df_customers['temp_label'].map(label_mapping)
df_customers.drop(columns=['temp_label'], inplace=True)

# 3. FILTER ACTIVE CUSTOMERS (CHURNED == 0) FOR DEMOGRAPHICS LCA
print("Filtering active customers (churned == 0) for demographics...")
df_active = df_customers[df_customers['churned'] == 0].copy().reset_index(drop=True)
print(f"Active customers: {df_active.shape[0]}")

# Create categorical bins for RFM (to display discretized bar charts in the app)
df_active['recency_cat'] = pd.cut(
    df_active['days_since_last_purchase'],
    bins=[-1, 30, 75, 999],
    labels=['Activo', 'En Riesgo', 'Dormido']
).astype(str)

df_active['frequency_cat'] = pd.cut(
    df_active['total_orders'],
    bins=[0, 1, 5, 999],
    labels=['Compra Unica', 'Ocasional', 'Frecuente']
).astype(str)

tertiles = df_active['total_spend_usd'].quantile([0.33, 0.66]).values
df_active['monetary_cat'] = pd.cut(
    df_active['total_spend_usd'],
    bins=[-1, tertiles[0], tertiles[1], 9999999],
    labels=['Gasto Bajo', 'Gasto Medio', 'Gasto Alto']
).astype(str)

# 4. PRE-PROCESS AND FIT LCA ON DEMOGRAPHICS
print("Preparing Demographics LCA features...")
df_active['registration_date'] = pd.to_datetime(df_active['registration_date'])
df_active['account_age_days'] = (REFERENCE_DATE - df_active['registration_date']).dt.days

df_active['age_group'] = pd.cut(
    df_active['age'],
    bins=[17, 25, 35, 45, 55, 75],
    labels=['18-25', '26-35', '36-45', '46-55', '56+']
).astype(str)

region_map = {
    'United States':'Norteamérica','Canada':'Norteamérica','Mexico':'Norteamérica',
    'United Kingdom':'Europa','Germany':'Europa','France':'Europa',
    'Spain':'Europa','Italy':'Europa','Netherlands':'Europa',
    'Australia':'Oceanía','New Zealand':'Oceanía',
    'India':'Asia','Japan':'Asia','China':'Asia',
    'South Korea':'Asia','Singapore':'Asia',
    'Brazil':'Latinoamérica','Argentina':'Latinoamérica','Chile':'Latinoamérica',
    'South Africa':'África'
}
df_active['region'] = df_active['country'].map(region_map).fillna('Otros')

CAT_COLS = [
    'gender', 'age_group', 'region', 'membership_tier',
    'preferred_device', 'acquisition_channel', 'preferred_category'
]

# Encode
X_parts_socio = []
for col in CAT_COLS:
    le = LabelEncoder()
    X_parts_socio.append(le.fit_transform(df_active[col]).reshape(-1, 1))

X_lca = np.hstack(X_parts_socio).astype(float)
MEASUREMENT_SOCIO = {col: {'model': 'categorical', 'n_columns': 1} for col in CAT_COLS}

print("Fitting StepMix LCA on Demographics...")
lca_model = StepMix(
    n_components=3,
    measurement=MEASUREMENT_SOCIO,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=500,
    verbose=0
)
lca_model.fit(X_lca)

df_active['Cluster_LCA'] = lca_model.predict(X_lca)
df_active['prob_max_clase'] = lca_model.predict_proba(X_lca).max(axis=1)

# Save segmented customers
output_path = 'segmented_customers.csv'
df_active.to_csv(output_path, index=False)
print(f"\nSaved segmented dataset to {output_path} successfully!")
