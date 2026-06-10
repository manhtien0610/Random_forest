import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.tree import plot_tree

# Đọc dữ liệu
df = pd.read_csv(r'D:\ttnt\NormalData.csv')

print(df.head())
print(df.columns)

# Mã hóa dữ liệu
le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

print(df.head())

# Tách dữ liệu
inputs = df.drop('LUNG_CANCER', axis=1)
target = df['LUNG_CANCER']

# Chia tập train/test
X_train, X_test, Y_train, Y_test = train_test_split(
    inputs,
    target,
    test_size=0.3,
    random_state=42
)

# Random Forest
clf = RandomForestClassifier(
    n_estimators=100,
    max_features=4,
    min_samples_split=2,
    max_depth=20,
    random_state=42
)

clf.fit(X_train, Y_train)

# Dự đoán
Y_pred = clf.predict(X_test)

print("Kết quả dự đoán:")
print(Y_pred)

# Đánh giá mô hình
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

# Feature Importance
feature_importances = pd.Series(
    clf.feature_importances_,
    index=inputs.columns
).sort_values(ascending=False)

print("\nMức độ quan trọng của thuộc tính:")
print(feature_importances)

plt.figure(figsize=(10, 5))
feature_importances.plot.bar()
plt.title("Feature Importance")
plt.show()

# Vẽ 2 cây đầu tiên
trees = clf.estimators_[:2]

plt.figure(figsize=(20, 10))

for i, tree in enumerate(trees, start=1):
    plt.subplot(2, 1, i)
    plot_tree(
        tree,
        feature_names=inputs.columns,
        filled=True,
        max_depth=3,
        fontsize=8
    )
    plt.title(f"Cây quyết định {i}")

plt.tight_layout()
plt.show()

# ROC Curve
Y_pred_proba = clf.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# So sánh AUC theo số cây
train_auc_scores = []
test_auc_scores = []

n_estimators_range = range(10, 300, 10)

for n in n_estimators_range:

    model = RandomForestClassifier(
        n_estimators=n,
        random_state=0
    )

    model.fit(X_train, Y_train)

    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    train_auc_scores.append(
        roc_auc_score(Y_train, train_pred)
    )

    test_auc_scores.append(
        roc_auc_score(Y_test, test_pred)
    )

plt.figure(figsize=(10, 6))
plt.plot(
    n_estimators_range,
    train_auc_scores,
    marker='o',
    label='Train AUC'
)

plt.plot(
    n_estimators_range,
    test_auc_scores,
    marker='o',
    label='Test AUC'
)

plt.xlabel("n_estimators")
plt.ylabel("AUC")
plt.title("AUC theo số lượng cây")
plt.legend()
plt.show()

# Dự đoán bệnh nhân mới
new_patient = [[
    0, 1, 1, 0, 1,
    1, 0, 1, 0, 1,
    1, 1, 1, 1, 1
]]

new_patient_df = pd.DataFrame(
    new_patient,
    columns=inputs.columns
)

prediction = clf.predict(new_patient_df)

if prediction[0] == 1:
    print("Bệnh nhân có nguy cơ mắc ung thư phổi.")
else:
    print("Bệnh nhân không có nguy cơ mắc ung thư phổi.")