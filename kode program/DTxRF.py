import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. LOAD DATA
# ==============================
data = pd.read_csv("dataset_clean.csv")
slope = pd.read_csv("dataset_slope.csv")

# ==============================
# 2. CLEANING
# ==============================
data = data.dropna()

# ==============================
# 3. SAMAKAN KOORDINAT
# ==============================
data['X'] = data['X'].round(5)
data['Y'] = data['Y'].round(5)

slope['X'] = slope['X'].round(5)
slope['Y'] = slope['Y'].round(5)

# ==============================
# 4. MERGE SLOPE
# ==============================
data = pd.merge(data, slope[['X','Y','slope']], on=['X','Y'], how='inner')
data['slope'] = data['slope_y']
data = data.drop(columns=['slope_x', 'slope_y'])

# ==============================
# 5. PERBAIKAN DATA
# ==============================
if data['rain'].max() < 100:
    data['rain'] = data['rain'] * 1000

# ==============================
# 6. LABEL (RULE-BASED)
# ==============================
def klasifikasi(row):
    skor = []

    suhu = (row['tmin'] + row['tmax']) / 2
    skor.append(1 if 24 <= suhu <= 28 else 2 if 22 <= suhu < 24 or 28 < suhu <= 31 else 3 if 20 <= suhu < 22 or 31 < suhu <= 33 else 4)

    skor.append(1 if 1700 <= row['rain'] <= 3000 else 2 if 1450 <= row['rain'] < 1700 or row['rain'] > 3000 else 3 if 1250 <= row['rain'] < 1450 else 4)

    skor.append(1 if 5.0 <= row['ph'] <= 6.5 else 2 if 4.2 <= row['ph'] < 5.0 or 6.5 < row['ph'] <= 7.0 else 3)

    skor.append(1 if row['cec'] > 16 else 2 if 5 <= row['cec'] <= 16 else 3)

    skor.append(1 if row['slope'] < 8 else 2 if row['slope'] <= 15 else 3 if row['slope'] <= 30 else 4)

    return ['S1','S2','S3','N'][max(skor)-1]

data['label'] = data.apply(klasifikasi, axis=1)
print("\nDistribusi Kelas:")
print(data['label'].value_counts())

print("\nPersentase Kelas:")
print(round(data['label'].value_counts(normalize=True)*100,2))

# ==============================
# DISTRIBUSI KELAS
# ==============================

distribusi = data['label'].value_counts()

distribusi_df = pd.DataFrame({
    'Kelas': distribusi.index,
    'Jumlah': distribusi.values
})

distribusi_df.to_csv(
    'distribusi_kelas.csv',
    index=False
)

print("\nDistribusi Kelas:")
print(distribusi_df)
# ==============================
# PROPORSI KELAS
# ==============================

proporsi = round(
    data['label'].value_counts(normalize=True)*100,
    2
)

proporsi_df = pd.DataFrame({
    'Kelas': proporsi.index,
    'Persentase': proporsi.values
})

proporsi_df.to_csv(
    'proporsi_kelas.csv',
    index=False
)

print("\nProporsi Kelas:")
print(proporsi_df)

# ==============================
# GRAFIK DISTRIBUSI KELAS
# ==============================

kelas = ['S1', 'S2', 'S3', 'N']
jumlah = [397, 1664, 501, 170]

warna = [
    '#2E7D32',  # hijau
    '#F9A825',  # kuning
    '#EF6C00',  # oranye
    '#C62828'   # merah
]

plt.figure(figsize=(9,5))

bars = plt.bar(
    kelas,
    jumlah,
    color=warna,
    width=0.6
)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 20,
        f'{int(height):,}',
        ha='center',
        fontsize=10,
        fontweight='bold'
    )

plt.title(
    'Distribusi Kelas Kesesuaian Lahan',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Kelas')
plt.ylabel('Jumlah Data')

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    'distribusi_kelas.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ==============================
# PROPORSI KELAS (DONUT CHART)
# ==============================

kelas = ['S1', 'S2', 'S3', 'N']
persentase = [14.53, 60.91, 18.34, 6.22]

warna = [
    '#2E7D32',  # hijau (S1)
    '#F9A825',  # kuning (S2)
    '#EF6C00',  # oranye (S3)
    '#C62828'   # merah (N)
]

plt.figure(figsize=(7,7))

wedges, texts, autotexts = plt.pie(
    persentase,
    labels=kelas,
    colors=warna,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.78,
    textprops={
        'fontsize': 11,
        'fontweight': 'bold'
    }
)

# lubang donut
centre_circle = plt.Circle(
    (0,0),
    0.55,
    fc='white'
)

plt.gca().add_artist(centre_circle)

# teks tengah
plt.text(
    0,
    0,
    '2732\nData',
    ha='center',
    va='center',
    fontsize=14,
    fontweight='bold'
)

plt.title(
    'Proporsi Kelas Kesesuaian Lahan',
    fontsize=14,
    fontweight='bold',
    pad=20
)

plt.tight_layout()

plt.savefig(
    'proporsi_kelas.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ==============================
# 7. MACHINE LEARNING
# ==============================
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# fitur & label
data['temperature'] = (
    data['tmax'] + data['tmin']
) / 2
X = data[['dem','bulk','cec','clay',
          'landcover','ph','prec',
          'rain','soc',
          'temperature',
          'slope']]
y = data['label']

# ==============================
# SCALING (WAJIB DI SINI)
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# SPLIT DATA (1x SAJA)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# MODEL
# ==============================
dt = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=20,
    min_samples_split=40,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# training
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Hitung accuracy training dan testing
train_acc_dt = dt.score(X_train, y_train)
test_acc_dt = dt.score(X_test, y_test)

train_acc_rf = rf.score(X_train, y_train)
test_acc_rf = rf.score(X_test, y_test)

print("=== Decision Tree ===")
print("Train Accuracy:", train_acc_dt)
print("Test Accuracy :", test_acc_dt)

print("\n=== Random Forest ===")
print("Train Accuracy:", train_acc_rf)
print("Test Accuracy :", test_acc_rf)

# prediksi
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

# ==============================
# AKURASI
# ==============================
print("Accuracy Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Accuracy Random Forest:", accuracy_score(y_test, y_pred_rf))

print("\n=== Classification Report DT ===")
print(classification_report(y_test, y_pred_dt))

print("\n=== Classification Report Random Forest ===")
print(classification_report(y_test, y_pred_rf))

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("\n=== METRIK DT ===")
print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_dt,
        average='weighted'
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred_dt,
        average='weighted'
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        y_pred_dt,
        average='weighted'
    )
)

print("\n=== METRIK RF ===")
print(
    "Precision:",
    precision_score(
        y_test,
        y_pred_rf,
        average='weighted'
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        y_pred_rf,
        average='weighted'
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        y_pred_rf,
        average='weighted'
    )
)

# ==============================
# CROSS VALIDATION
# ==============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_dt = cross_val_score(dt, X_scaled, y, cv=cv)
cv_rf = cross_val_score(rf, X_scaled, y, cv=cv)

print("\nNilai CV Decision Tree:")
print(cv_dt)

print("Decision Tree CV Mean:", cv_dt.mean())
print("Decision Tree CV Std:", cv_dt.std())

print("\nNilai CV Random Forest:")
print(cv_rf)

print("Random Forest CV Mean:", cv_rf.mean())
print("Random Forest CV Std:", cv_rf.std())

# ==============================
# BAR CHART
# ==============================
plt.figure()
plt.bar(['Decision Tree','Random Forest'],
        [cv_dt.mean(), cv_rf.mean()])
plt.ylim(0,1)
plt.title('Perbandingan Akurasi Model (Cross Validation)')
plt.savefig('accuracy_compare.png', dpi=300)
plt.show()

# ==============================
# CONFUSION MATRIX
# ==============================
labels = ['N', 'S1', 'S2', 'S3']

cm = confusion_matrix(y_test, y_pred_dt, labels=labels)

fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("cm_dt.png", dpi=300)
plt.show()

# ==============================
# NORMALIZED MATRIX
# ==============================
cm_norm = confusion_matrix(y_test, y_pred_dt, labels=labels, normalize='true')

fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
disp.plot(ax=ax, values_format=".2f")

plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("cm_dt_norm.png", dpi=300)
plt.show()

# ==============================
# CONFUSION MATRIX TABEL
# ==============================
labels = ['N', 'S1', 'S2', 'S3']

cm = confusion_matrix(y_test, y_pred_dt, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print("\nConfusion Matrix Table:\n")
print(cm_df)

cm_df.to_csv("cm_dt.csv")

# ==============================
# CONFUSION MATRIX - RF
# ==============================
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels)

fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=labels)
disp.plot(ax=ax)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("cm_rf.png", dpi=300)
plt.show()

# ==============================
# NORMALIZED MATRIX - RF
# ==============================
cm_rf_norm = confusion_matrix(
    y_test,
    y_pred_rf,
    labels=labels,
    normalize='true'
)

fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf_norm, display_labels=labels)
disp.plot(ax=ax, values_format=".2f")

plt.title("Normalized Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("cm_rf_norm.png", dpi=300)
plt.show()

# ==============================
# CONFUSION MATRIX TABLE - RF
# ==============================
cm_rf_df = pd.DataFrame(cm_rf,
                        index=labels,
                        columns=labels)

print("\nConfusion Matrix RF:\n")
print(cm_rf_df)

cm_rf_df.to_csv("cm_rf.csv")

print("\n=== Perbandingan Error ===")
print("DT Error:", 1 - accuracy_score(y_test, y_pred_dt))
print("RF Error:", 1 - accuracy_score(y_test, y_pred_rf))

# ==============================
# SIMPAN MATRIX
# ==============================
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv("cm_dt.csv")

# ==============================
# FEATURE IMPORTANCE
# ==============================
importance = dt.feature_importances_
df_imp = pd.DataFrame({'Fitur': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=True)

plt.figure()
plt.barh(df_imp['Fitur'], df_imp['Importance'])
plt.title("Feature Importance - Decision Tree")
plt.savefig("fi.png", dpi=300)
plt.show()

# ==============================
# FEATURE IMPORTANCE RANDOM FOREST
# ==============================
importance_rf = rf.feature_importances_

df_imp_rf = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': importance_rf
}).sort_values(by='Importance', ascending=True)

print("\nFeature Importance Random Forest:")
print(df_imp_rf.sort_values(by='Importance', ascending=False))

plt.figure(figsize=(8,6))
plt.barh(df_imp_rf['Fitur'], df_imp_rf['Importance'])
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("fi_rf.png", dpi=300)
plt.show()