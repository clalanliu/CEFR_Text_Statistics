## How to Classify Text into 5 CEFR Levels
We use word frequency and numbers of CEFR vocabularies to classify.

### Inspect
```
	word_freq_std word_freq_mean cefr_A1 cefr_A2   cefr_B1      cefr_B2     cefr_C1
label							
A1	0.000284	0.000170	30.559028	8.288194	4.597222	3.142361	1.750000
A2	0.000325	0.000189	67.643382	20.150735	11.915441	8.742647	3.459559
B1	0.000367	0.000219	102.736585	41.473171	25.707317	23.253659	8.424390
B2	0.000357	0.000209	107.332168	51.674825	32.884615	33.482517	14.517483
C1	0.000328	0.000180	132.556017	76.406639	50.850622	52.912863	23.531120
```

### Results
RandomForest
```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

X = df.drop('label', axis=1)
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Initialize and train the XGBoost classifier
#classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
```

Relax the criteria: count predictions that are within ±1 of the actual label as correct:
Accuracy: 0.916