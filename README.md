## How to Classify Text into 5 CEFR Levels
We use word frequency and numbers of CEFR vocabularies to classify.
Dataset: https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts

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
Accuracy: 0.54
Classification Report:
              precision    recall  f1-score   support

          A1       0.74      0.84      0.78        67
          A2       0.49      0.52      0.50        52
          B1       0.37      0.31      0.33        36
          B2       0.40      0.43      0.41        54
          C1       0.51      0.43      0.47        51
          C2       0.58      0.56      0.57        39

    accuracy                           0.54       299
   macro avg       0.51      0.51      0.51       299
weighted avg       0.53      0.54      0.53       299
```

Relax the criteria: count predictions that are within ±1 of the actual label as correct:
Accuracy: 0.916