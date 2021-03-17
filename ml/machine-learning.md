# <font color = "blue"> Machine Learning in Python </font>


> Exploration Notes

By: Anoushkrit Goel
On: March 8, 2021

- [Machine Learning in Python](#machine-learning-in-python)
  - [Part 0 : Getting Started](#part-0--getting-started)
  - [Part 1: Data Pre-processing](#part-1-data-pre-processing)
  - [Part 2: Models of Regression](#part-2-models-of-regression)
    - [Simple Linear Regression](#simple-linear-regression)
    - [Multiple Regression](#multiple-regression)
      - [Understanding P-Value](#understanding-p-value)
    - [Model Building](#model-building)
    - [Polynomial Regression](#polynomial-regression)
    - [Support Vector for Regression (SVR)](#support-vector-for-regression-svr)
    - [Decision Tree Regression](#decision-tree-regression)
    - [Random Forest Regression](#random-forest-regression)
    - [R Squared (Goodness of fit)](#r-squared-goodness-of-fit)
    - [Adjusted $R^2$](#adjusted-r2)
    - [Model Selection (Regression)](#model-selection-regression)
    - [Regularization](#regularization)
  - [Part 3: Models of Classification](#part-3-models-of-classification)
  - [Part 4: Clustering](#part-4-clustering)
  - [Part 5: Associate Rule Learning](#part-5-associate-rule-learning)
  - [Part 6: Reinforcement Learning](#part-6-reinforcement-learning)
  - [Part 7: Natural Language Processing](#part-7-natural-language-processing)
    - [Bag of Words](#bag-of-words)
  - [Part 8: Deep Learning](#part-8-deep-learning)
  - [Part 9: Dimensionality Reduction](#part-9-dimensionality-reduction)
  - [Part 10: Model Selection & Boosting](#part-10-model-selection--boosting)
  - [Appendix](#appendix)

## Part 0 : Getting Started


## Part 1: Data Pre-processing 
Processes studied in Part 1:

1.	Importing Libraries
2.	Importing Datasets
    1. Using **iloc** function to iterate over the indexes of the columns which will then allow us to pick specific columns for making the X and Y( Dependent Variable)
3.	Imputing the Missing Values
    1. 1% of the missing data won’t have that

```python
from sklearn.impute import Simpleimputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])  # X[rows, columns]
imputer.transform(X[:, 1:3])

```
4. Encoding Categorical Data

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers =[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough' )
X = ct.fit_transform(X)

# For forcing this to a numpy array 

X = np.array(ct.fit_transform(X))

## Encoding Dependent Variables with Label Encoding 

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
y = le.fit_transform(y)
```

5.	Splitting the Dataset into training and testing
> Splitting the data should happen before Feature Scaling because the Test set won't be provided in the form of scaled features.

```python
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0,2, random_state = 1 # fixing the random state so that we can get the exactly same split)

```
6.	Feature Scaling

**Standardisation**

$x_{stand} = (x - mean(x)) / standard deviation (x)$ 

```python 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler 
X_train = sc.fit_transform(X_train[:, 3:4])

## Do the same for X_test
```
1. Feature Scaling should not be applied to the dummy variables for example the one-hot encoded values.
2. Also, can be said that apply Feature Scaling only to your numerical values.
   
**Other types of Standardisation**


**Normalisation**

$x_{norm} = {x- min(x)}/max(x) - min(x)$


**Object Oriented Programming**
1.	Class
2.	Object 
3.	Method

## Part 2: Models of Regression

### Simple Linear Regression

For predicting all continuous numbers

$y = b_0 + b_1 * x_1$

> $y$ is the Dependent Variable
> , $b$ is the coefficient ( $b_0$ is the coefficient of $x_0$) 
>,  $x$ is the Independent Variable 


**Ordinary Least Square**


$\Sigma (y^h_i - y_i)^2$

```python
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].

## Conversion from pd.Dataframe to list values

```

```python

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

[Code](https://drive.google.com/drive/folders/10uenHNwaxw9hA-3C9OlSydTAlxw2b-2I)

### Multiple Regression 

$y = b_0 + b_1 * x_1 + b_2 * x_2 + . . . . + b_n * x_n$

> All the $x$ values are Independent Variables

**<font color = 'red'>Assumptions of Linear Regression </font>**

1. Linearity
2. Homoscedasticity
3. Multivariate Normality
4. Independence of errors
5. Lack of Multicollinearity

**Dummy Variable Trap**

Always omit one Dummy Variable because 

#### Understanding P-Value

$H_0$:Null Hypothesis 
$H_1$:Alternate Hypothesis


In null hypothesis significance testing, the p-value is the probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct.

![P-value](https://scientistseessquirrel.files.wordpress.com/2015/02/p-value_in_statistical_significance_testing-svg.png?w=624)

The **p value** is the evidence against a null hypothesis. The smaller the p-value, the stronger the evidence that you should reject the null hypothesis. P values are expressed as decimals although it may be easier to understand what they are if you convert them to a percentage. For example, a p value of 0.0254 is 2.54%.

### Model Building

5 methods of building models

1. All-in 
   1. Throw in all the parameters or all the use cases.
   2. Does / Doesn't have prior knowledge
   3. You have to use some of the variables
   4. Preparing for the Backward Elimination
2. Backward Elimination 
   1. Select a Significance Level to stay in the model (SL = 0.05)
   2. Fit the full model with all possible predictors 
   3. Consider the predictor with the highest P-value. If P (P-value) > SL (Significance Level), go to Step 4, otherwise go to FIN
   4. Remove the predictor 
   5. Fit the model without this variable
3. Forward Selection 
   1. Select the Significance Level to enter the model (eg, SL = 0.05)
   2. Fit all simple regression model $y ~ x_n$. Select the one with the lowest P-value
   3. Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
   4. Consider the predictor with the **lowest** P-value. If P< SL, go to Step 3, otherwise go to FIN : Keep the previous model
4. Bidirectional Elimination 
   1. Select a significance level to enter and to stay in the model eg:SLENTER = 0.05, SLSTAY = 0.05 
   2. Perform the next step of Forward Selection (new variables must have: P < SLENTER to enter)
   3. Perform ALL steps of **Backward Elimination** old variables must have P< SLSTAY to stay
   4. No new variables can enter and no old variables can exit. 
   5. FIN
   6. > Backward Elimination is not valid in **Python** because the sklearn library automatically takes care of selecting the statistically significant features when training the model to make accurate predictions.
5. Score Comparison (All Possible Models)
   1. Compare the models that come out based on multiple score parameters.
   2. Select a criterion of goodness of fit (eg: Akaike Criterion)
   3. Construct ALL Possible Regression Models: $2^N - 1$ total combinations
   4. Select the one with the best criterion.
   5. FIN


2,3,4 also known as **Step wise Regression **

> In the **LinearRegression()** sklearn takes care of the Dummy Variable Trap.

**BONUS** 

**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array. Simply put:

$1, 0, 0, 160000, 130000, 300000 \rightarrow \textrm{scalars}$

$[1, 0, 0, 160000, 130000, 300000] \rightarrow \textrm{1D array}$

$[[1, 0, 0, 160000, 130000, 300000]] \rightarrow \textrm{2D array}$

**Important note 2:** Notice also that the "California" state was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the second row of the matrix of features X, "California" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, not the last three ones, because the dummy variables are always created in the first columns.

```javascript


```


```python 

## For predicting only a single value using the regression model 
## Note: that a 2 D array has to be supplied for this to work 


print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

## For printing the coefficients

print(regressor.coef_)
print(regressor.intercept_)

```


**Important Note: 3** To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values.

### Polynomial Regression


![Polynomial Regression](https://static.javatpoint.com/tutorial/machine-learning/images/machine-learning-polynomial-regression.png)

$y = b_0 + b_1*x_1^1 + b_2*x_1^2 + b_3*x_1^3 .... b_n*x_1^n$

Different powers of the same feature are used as multiple features for the problem. 
Consider this code snippet for the understanding

```python 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
## Creating those multiple features (which are power functions of a specific feature)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```




### Support Vector for Regression (SVR)

<img src="https://i.pinimg.com/736x/3b/c5/3c/3bc53ced72ff0ecf225c948a75bc481a.jpg" width="500">

Contains a $\epsilon$ insensitie tube in which do not care about the errors in that tube. 

$1/2 \mod\mod {w^2} + C\Sigma(\Epsilon_i + \Epsilon_i^*) \rightarrow min$

[Support Vector Kernels](https://data-flair.training/blogs/svm-kernel-functions/)

**Gaussian RBF Kernel**


<img src="https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/4356651d-2606-4b23-b053-60647095485d/dea2832c-c54a-4951-a9a2-0c67fe8457df/images/screenshot.jpg" width="300"/>

```python
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

## Generally for predicting we would do 

regressor.predict([[6.5]])
## but this requires transformed input and inverse_transformed output

## Inverse scaled transform 
## Transforming the input
sc_X.transform([[6.5]])

sc_y.inverse_transform(regressor.predict(sc_X.transform(([[6.5]]))

```


### Decision Tree Regression 

[Decision Tree Regression](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html)

**Intuition**

<img src = "https://gdcoder.com/content/images/2019/05/Screen-Shot-2019-05-17-at-00.09.26.png">

> **Feature Scaling** is not a requirement for the Decision Tree

That's because you know the predictions from a **decision tree regression** or **random forest regression** model are resulting from successive splits of the data, different nodes of your tree and therefore there are not some equations like with the previous models and that's why of course no feature scaling is needed to you know split the different values of your feature into these different categories leading to different predictions we can still do this with the original scale of your features even if your features take different ranges of values.

```python 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
```


### Random Forest Regression

Random Forest Intuition 

1. Pick at random K data points from the Training Set 
2. Buiild the Decision Tree associated to these K data points.
3. Choose the number of Ntree of trees you want to build and repeat STEP 1 & 2 
4. For a new data point, make each one of your Ntree tree predict the value of Y to for the data point in question, and assign the new data point the average across all of the predicted Y values. 


```python 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

## n_estimators is the no. of trees.
```

### R Squared (Goodness of fit)

<img src = "https://miro.medium.com/max/2812/1*_HbrAW-tMRBli6ASD5Bttw.png" width = "500" name = "R Squared">
$R \epsilon [0,1]$

Closer the R squared values is to 1, the better the regression model is performing.

![](https://quantifyinghealth.com/wp-content/uploads/2020/04/plots-with-different-correlation-coefficients.png)

Problem is R squared is that the value of R-squared is always increasing regardless of whether you add a new variable or not. Because the model will pick up some random correlation that will either increase the R squared or it will remain the same.

### Adjusted $R^2$  


<img src = "https://lh3.googleusercontent.com/yDM9AYoupBqM3N7gBG1mzQ2i_SbQy-uQmBL3WgxzhJ_MwVNzMWUuYp1HCGTdMeYqPohWqwnFi72fOmQi632nCWaz2ToZL2CuBy5kLWW-tJ0-Oe1PTrTyH2H3B5DNnoUL8exH1Bev" width = "500" name = "Adjusted R Squared">

$p \rightarrow number of regressors$

$n \rightarrow sample size$


Penalizes the independent variables that don't help your model

### Model Selection (Regression)

However, the $r^2$ score can be calculated for every regression model and the performance of each model can also be evaluated. But there are some parameters that are NOT learned during the training of the model. They are called hyperparameters


| Regression Models | Pros | Cons |
|-------------------|------|------|
|  Linear Regression| Works on any size of dataset, gives informations about relevance of features| The Linear Regression Assumptions|
| Polynomial Regression| Works on any size of dataset, works very well on non linear problems| Need to choose the right polynomial degree for a good bias/variance tradeoff|
| SVR | Easily adaptable, works very well on non linear problems, not biased by outliers| Compulsory to apply feature scaling, not well known, more difficult to understand| 
| Decision Tree Regression|Interpretability, no need for feature scaling, works on both linear / nonlinear problems | Poor results on too small datasets, overfitting can easily occur| 
| Random Forest Regression | Powerful and accurate, good performance on many problems, including non linear | No interpretability, overfitting can easily occur, need to choose the number of trees| 




### Regularization

When in NO Regularization

[Ridge and Lasso Regression: L1 and L2 Regularization](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b)


1. Ridge Regression (L1)
      > In ridge regression, the cost function is altered by adding a penalty equivalent to square of the magnitude of the coefficients.

![Ridge Regression](https://miro.medium.com/max/875/1*hAGhQehrqAmT1pvz3q4t8Q.png)

2. Lasso Regression (L2)
      > *The only difference is instead of taking the square of the coefficients, magnitudes are taken into account*. This type of regularization (L1) can lead to zero coefficients i.e. some of the features are completely neglected for the evaluation of output. So Lasso regression not only helps in reducing over-fitting but it can help us in feature selection.

![Lasso Regression](https://miro.medium.com/max/875/1*P5Lq5mAi4WAch7oIeiS3WA.png)

3. Elastic Net 
## Part 3: Models of Classification 

### Logistic Regression

Logistic Regression is just Linear Regresion where 

$y = b_0 + b_1*x \rightarrow Linear Regression$

$p = 1/{1 + e^{-y}} \rightarrow Sigmoid   Function$

$ln(p/{1-p}) = b_0 + b_1*x \rightarrow Logistic Regression$ 


### K-Nearest Neighbors (K-NN)
### Support Vector Machine (SVM)
### Kernel SVM
### Naive Bayes
### Decision Tree Classification
### Random Forest Classification
 
## Part 4: Clustering 
 
## Part 5: Associate Rule Learning 
 
## Part 6: Reinforcement Learning
 
## Part 7: Natural Language Processing 

### Bag of Words 

Collection of words

For example having a 20,000 elements long vector to signify the sentences and words.

 
## Part 8: Deep Learning 
 
## Part 9: Dimensionality Reduction 
 
## Part 10: Model Selection & Boosting

## Appendix 

1. [Codes and Datasets](https://drive.google.com/drive/folders/1OFNnrHRZPZ3unWdErjLHod8Ibv2FfG1d?usp=sharing)
2. 
