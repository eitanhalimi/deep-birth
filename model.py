import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectKBest, f_classif

# 1. טעינת הקבצים
try:
    df_features = pd.read_csv('classic_ml_features.csv')
    df_meta = pd.read_csv('tpehg_metadata.csv')
    print("Files loaded successfully!")
except FileNotFoundError:
    print("Error: Files not found. Check your folder.")

# 2. מיזוג והנדסת פיצ'רים (סנכרון רחמי)
df = pd.concat([df_features, df_meta.drop('Label', axis=1, errors='ignore')], axis=1)

# יצירת יחסים בין ערוצים - זה עוזר למודל להבין אם הרחם עובד בתיאום
df['freq_ratio_31'] = df['ch3_med_freq'] / (df['ch1_med_freq'] + 1e-5)
df['var_diff_31'] = df['ch3_var'] - df['ch1_var']
df['rms_total'] = df['ch1_rms'] + df['ch2_rms'] + df['ch3_rms']

# 3. ניקוי בטוח - רק עמודות מספריות
# כאן אנחנו פותרים את שגיאת ה-TypeError
df_numeric = df.select_dtypes(include=[np.number])
# מחשבים ממוצע רק לעמודות המספריות וממלאים חסרים
df_numeric = df_numeric.fillna(df_numeric.mean())

# הפרדה ל-X ו-y (מניעת Leakage)
columns_to_drop = ['Label', 'Gestation', 'RecID']
X_raw = df_numeric.drop(columns=columns_to_drop, errors='ignore')
y = df_numeric['Label']

# 4. בחירת תכונות (Feature Selection) - צמצום רעשים קליני
# אנחנו בוחרים את 15 הפיצ'רים שהכי משפיעים סטטיסטית
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X_raw, y)
selected_names = X_raw.columns[selector.get_support()]
print(f"Selected Clinical Features: {list(selected_names)}")

# 5. חלוקה וסילום
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. איזון נתונים (SMOTE-Tomek)
print("Applying SMOTE-Tomek for cleaner separation...")
smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train_scaled, y_train)

# 7. אימון מודל XGBoost מכויל
print("Training Clinical XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=3,          # עצים פשוטים יותר מונעים הצמדות לרעש (Overfitting)
    learning_rate=0.03,
    scale_pos_weight=6,   # דגש חזק מאוד על זיהוי המחלקה הקטנה (לידה מוקדמת)
    subsample=0.8,        # שימוש בחלק מהדאטה בכל עץ ליציבות
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_res, y_train_res)

# 8. אופטימיזציה של סף ההחלטה (Threshold)
probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.1 # הורדת הסף ל-0.3 כדי לתפוס יותר לידות מוקדמות
predictions = (probs >= threshold).astype(int)

# 9. תוצאות
print(f"\n--- Final Clinical Model (Threshold: {threshold}) ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))