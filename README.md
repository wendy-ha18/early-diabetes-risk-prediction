# Early Diabetes Risk Prediction In Females By Using Classification Machine Learning Model 
# Technologies and resources
- Python 3.6
- Analysis libraries: numpy, pandas
- Visualization libraries: matplotlib, seaborn
- Data modelling packages:

<table>
	<tr>
	    <th> Scikit learn package</th>
	    <th> Related functions</th>
	    <th> Using purpose</th>  
	</tr >
  <tr >
	    <td>sklearn.preprocessing</td>
	    <td>MinMaxScaler()</td>
	    <td>For Feature Scalling</td>
	</tr>
	<tr >
	    <td rowspan="3">sklearn.model_selection</td>
	    <td>train_test_split()</td>
	    <td>For splitting data into train set and test set </td>
	</tr>
	<tr>
	    <td>GridSearchCV()</td>
	    <td>For hyperparameter opimization</td>
	</tr>
  <tr>
	    <td>KFold()</td>
	    <td>For running Kfold cross validation</td>
	</tr>
  <tr >
	    <td>sklearn.neighbors</td>
	    <td>KneighborsClassifier()</td>
	    <td>For running K-Nearest Neighbors agorithm </td>
	</tr>
   <tr >
	    <td>sklearn.tree</td>
	    <td>RandomForestClassifier()</td>
	    <td>For running Random Forest agorithm </td>
	</tr>
   <tr >
	    <td>sklearn.ensemble</td>
	    <td>DecisionTreeClassifier()</td>
	    <td>For running Decision Tree agorithm </td>
	</tr>
  <tr >
	    <td rowspan="3">sklearn.metrics</td>
	    <td>accuracy_score </td>
	    <td>For finding Accuracy Score of a classification model</td>
	</tr>
  <tr>
	    <td>confusion_matrix()</td>
	    <td>For finding Confusion Matrix of a classification model</td>
	</tr>
  <tr>
	    <td>classification_report()</td>
	    <td>For retrieving a classification model's Precision, Recall, and F1 Score report</td>
	</tr>
 </table> 
  
- Author: Wendy Ha
- Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
## Objectives
Diabetes, according to the World Health Organization, is a chronic disease that occurs when blood glucose levels are abnormally high, posing a risk to several body organs such as the heart, eyes, kidneys, and lungs (World Health Organization 2020). There is currently no cure for diabetes, but it is possible to predict an individual's likelihood of developing the disease using medical indicators such as blood pressure, glucose levels, insulin levels and genetics.
<br/><br/>
Recognizing the importance of early diabetes diagnosis, this project utilised the Pima Indians diabetic dataset from the machine learning repository at the University of California, Irvine to develop a supervised classification machine learning model. Through the application of data science, this initiative aims to assist medical practitioners in increasing the life expectancy of women in the community.
## Data preprocessing process
### Handling Null data by approriate Mean and Median values
- Checking data distribution by histogram

![image](https://user-images.githubusercontent.com/90888090/175816349-7a62a024-5638-462e-bb6f-f2007b5ee1aa.png)
>**Findings:** <br/>
The histogram above indicates that <b>Glucose</b> and <b>BloodPressure</b> have roughly normal distributions; hence, null values in these columns can be replaced with the <b>MEAN</b>.<br/><br/>
For <b>Insulin, SkinThickness, and BMI,</b> the data skew to the left is excessively pronounced; hence, the <b>MEDIAN</b> should be used to replace null values.

```ruby
# Filling null values in 'Insulin', 'SkinThickness', 'BMI' columns with the MEDIAN.
for col in ['Insulin', 'SkinThickness', 'BMI']:
    asm2_data[col] = asm2_data[col].fillna(asm2_data[col].median())
#asm2_data['Insulin', 'SkinThickness', 'BMI'] = asm2_data['Insulin'].fillna(asm2_data['Insulin'].median()) 
```

```ruby
# Filling null values in Glucose, BloodPressure columns with the MEAN.
for col in ['Glucose', 'BloodPressure']:
    asm2_data[col] = asm2_data[col].fillna(asm2_data[col].mean())
```

- The project data set no longer contains missing values.

![image](https://user-images.githubusercontent.com/90888090/175816435-36b4b377-b4b6-4074-8e3f-e0707f96ebe0.png)
### Detect Outliers
![image](https://user-images.githubusercontent.com/90888090/175816478-5c3eb440-73b3-45fc-bfb8-147c1ce44fae.png)

>**Findings:**
<br/><b>Insulin, DiabetesPedigreeFunction, and SkinThickness</b> are the features with the greatest number of outliers among the eight independent columns.<br/><br/>
Outliers can influence the accuracy of several classification models, such as KNN and Logistic Regression, hence they will be preserved.

### Conclusions
In the end of the data preprocessing process, the <b>project_data</b> has been modified so that:
- It no longer contains missing value.
- It has been decided to keep outliers.

This data set is now ready for the model development process.
## Data Exploration
### Explore distribution of Non-Diabetic (0) and Diabetes (1) data in each independent parameter

![image](https://user-images.githubusercontent.com/90888090/175816728-9fa516e5-bc46-40e8-baf4-d1185f134fed.png)

### Relationship between features

![image](https://user-images.githubusercontent.com/90888090/175816765-d345f6ef-5819-4bc2-b0d7-f04266557b0a.png)

## Formulate Hypothesis
### Which variables increase the likelihood of diabetes in women?
Glucose, BMI, Pregnancies, Diabetes Pedigree Function and Age are primary medical indicators that have a positive correlation with the development of diabetes. This can be discussed in detail as follows:
- In terms of Glucose: As explained by Australia Department of Health, during the process of eating, a portion of the food's sugar is absorbed by the human body. If this amount of sugar is not metabolised and instead builds up excessively in the blood, a situation known as hyperglycemia develops, leading to diabetes (Australia Government - Department of Health 2019).
- In terms of Body Max Index (BMI): As the level of fat in the body increases, tissues and cells become more resistant and limit the production of the hormone insulin, hence increasing the risk of diabetes (Diabetes Australia 2020). This is evident from the BMI displot illustrated in section 2.1, which classifies women with a BMI 25-29.9  𝑘𝑔/𝑚2  as "overweight" or "pre-obese", with a BMI 30.0-34.9  𝑘𝑔/𝑚2  as "obesity class 1", with a BMI 35.0-39.9  𝑘𝑔/𝑚2  as "obesity class 2", and with a BMI 40  𝑘𝑔/𝑚2  or more as "severe obesity" (Diabetes Australia 2020). All individuals in these high BMI groups were diagnosed with diabetes.
- In terms of Diabetes Pedigree Function: People with a family history of diabetes, especially a parent, are more likely to inherit the diabetes-causing genes (Diabetes Australia 2020).
- In terms of Age: As people get older, their cells become insulin-resistant and the pancreas produces less insulin than when they were younger. Consequently, diabetes is common among the elderly (Diabetes Australia 2020).
- In terms of Pregnancies: During pregnancy, the placenta produces a hormone that aids in the development of the foetus but restricts insulin production by the pancreas. Moreover, the blood glucose levels of pregnant women are frequently high, and they gain weight as well. This results in gestational diabetes, a form of diabetes that occurs during pregnancy. After a woman is diagnosed with gestational diabetes, the risk of having diabetes later in her life is extremely high (Diabetes Australia 2020).


On the other hand, blood pressure, insulin, and skin thickness also affect the likelihood of developing diabetes, but not significantly.
### How does pregnancy cause detrimental effects to women clinical parameters?
- Insulin, Skin Thickness, and Diabetes Pedigree Function all correlate negatively with Pregnancies. In particular, during pregnancy, insulin levels in female's body frequently decline because of a hormone generated by the placenta. Additionally, as the size of the foetus increases, the skin of pregnant women grows thinner. The more the number of pregnancies a woman has, the less elastic her skin gets (Australia Government - Department of Health 2019).

- Furthermore, as previously mentioned, some women do not have diabetes before they get pregnant; they develop gestational diabetes only during pregnancy, and the majority of these women return to being nondiabetic after giving birth (Diabetes Australia 2020). Therefore, the inverse correlation between Pregnancies and Diabetes Pedigree Function implies that not all examined women in the Pima diabetes data set contain the hereditary diabetes gene.

## Data Modelling

![image](https://user-images.githubusercontent.com/90888090/175817240-a495dae4-88b0-4ca6-b68d-e47761ac01f4.png)

### Building Base Models (Reusable functions) for repeated tasks
- Base function to find best value of the hyperparameter
```ruby
def tune_hyperparameter (model, x_train, y_train, isKnn = True):
    print ('Find best value for hyperparameter by GridSearchCV')
    print(15*'-')
    if isKnn == True:
        para = {'n_neighbors':np.arange(1,11)}
    else:
        para = {'max_depth':np.arange(1,11)}
    model_cv= GridSearchCV(model,para, cv=10)
    model_cv.fit(x_train, y_train)   
    print("- Best cv_scores:" + str(model_cv.best_score_))
    print("- Best parameter: " + str(model_cv.best_params_)) 
```

- Base function to evaluate performance of each chosen classification model
```ruby
def model_evaluation(model, x_train, y_train, x_test, y_test, train = True):
    #Train set
    if train == True:
        y_train_pred = model.predict(x_train)
        print("\n"+35*'*'+"\n")
        print ('\033[1m TRAIN RESULT  \033[0m'+"\n")
        print (f"1/ Confusion Matrix: \n {confusion_matrix(y_train, y_train_pred)}")
        print(f"2/ Accuracy Score: {accuracy_score(y_train, y_train_pred)}")
        print(f"3/ F1 Score: {f1_score(y_train, y_train_pred, average='weighted')}")
        print(f"4/ Classification Report: \n {classification_report(y_train, y_train_pred)}")
    #Test set
    if train == False:
        y_test_pred = model.predict(x_test)
        print("\n" + 35*'*'+"\n")
        print ('\033[1m TEST RESULT  \033[0m'+"\n")
        print (f"1/ Confusion Matrix: \n {confusion_matrix(y_test, y_test_pred)}")
        print(f"2/ Accuracy Score: {accuracy_score(y_test, y_test_pred)}")
        print(f"3/ F1 Score: {f1_score(y_test, y_test_pred, average='weighted')}")
        print(f"4/ Classification Report: \n {classification_report(y_test, y_test_pred)}")
```

- Base function to compare performance of 02 chosen classification models
```ruby
def model_comparision(model_1, model_2, x_test, y_test):
    y_test_pred1 = model_1.predict(x_test)
    y_test_pred2 = model_2.predict(x_test)
    as_1 = accuracy_score(y_test, y_test_pred1)
    as_2 = accuracy_score(y_test, y_test_pred2)
    f1_sc1=f1_score(y_test, y_test_pred1, average='weighted')
    f1_sc2=f1_score(y_test, y_test_pred2, average='weighted')
    model_cp = {}
    model_cp['K-nearest Neighbors Classifier'] = [as_1, f1_sc1]
    model_cp['Decision Tree Classifier'] = [as_2, f1_sc2]
    model_cp_df = pd.DataFrame.from_dict(model_cp).T
    model_cp_df.columns = ['Accuracy', 'F1 Score']
    return model_cp_df
```

- Base function to visualise comparison metrics of 02 chosen classification model
```ruby
def model_compare_visual (data):
    data.plot(kind="barh", figsize=(12, 10))
    plt.title("Classification Models Comparison for splitting dataset ", fontsize=18, pad=20)
    plt.xlabel("Metrics", fontsize=13)
    plt.ylabel("Type of Classification Model", fontsize=13)
    plt.show()
```

## Applying Base model for data partitioning: 80% for training and 20% for testing
### Model 1: K-nearest Neighbors Algorithm
- *Find best value of the hyperparameter "n_neighbors"*
```ruby
knn_core1 = KNeighborsClassifier()
tune_hyperparameter (knn_core1, X_train1, y_train1, True)
```
- *Evaluate K-nearest Neighbors Algorithm Performance*

![image](https://user-images.githubusercontent.com/90888090/175817392-592f08ae-76b5-4626-9724-6c553cc1979e.png)
### Model 2: Decision Tree Algorithm
- *Find best value of the hyperparameter "max_depth"*
```ruby
dt_core1 = DecisionTreeClassifier()
tune_hyperparameter (dt_core1 , X_train1, y_train1, False)
```
- *Evaluate Decision Tree Algorithm Performance*

![image](https://user-images.githubusercontent.com/90888090/175817503-959ca586-885b-42e2-a6c7-2df9d8490ced.png)
### Models comparison
- *Comparing metrics approach*

![image](https://user-images.githubusercontent.com/90888090/175817640-d501e079-e1da-4da7-bd14-a2a469a6070e.png)
- *Visualizing performance's difference between 02 models*

![image](https://user-images.githubusercontent.com/90888090/175817580-4e826dac-d0b4-4fec-bce2-6e2f3f70f21e.png)
