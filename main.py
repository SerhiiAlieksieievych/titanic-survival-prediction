# Імпорт бібліотек
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Завантаження датасету
titanic = sns.load_dataset('titanic')

# Очищення даних
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])

# Видалення непотрібних колонок
titanic.drop(['deck', 'embark_town', 'alive', 'who', 'adult_male', 'alone', 'class'], axis=1, inplace=True)

# Кодування змінних
titanic['sex'] = LabelEncoder().fit_transform(titanic['sex'])
titanic = pd.get_dummies(titanic, columns=['embarked'], drop_first=True)

# Інженерія ознак
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
titanic['is_alone'] = 0
titanic.loc[titanic['family_size'] == 1, 'is_alone'] = 1

# Розвідувальний аналіз даних
plt.figure(figsize=(12, 8))
sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm')
plt.title('Кореляційна матриця')
plt.show()

# Підготовка до моделювання
X = titanic.drop(['survived'], axis=1)
y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Моделі для навчання
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC()
}

# Навчання та оцінка моделей
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {acc:.4f}')

# Тонка настройка Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Оцінка найкращої моделі
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Найкращі параметри:", grid.best_params_)
print("Точність:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-міра:", f1_score(y_test, y_pred))
print("Звіт класифікації:\n", classification_report(y_test, y_pred))

# Матриця помилок
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Матриця помилок')
plt.xlabel('Прогноз')
plt.ylabel('Реальні значення')
plt.show()
