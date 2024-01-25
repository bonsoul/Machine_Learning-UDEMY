# Regression
This notebook outlines the process of training and evaluating a regression model, focusing on linking data features to predicted variables, and adjusting parameters for optimization.

## Data Exploration

The purpose is to comprehend the links between qualities, specifically any association between features and predicted labels. 
This includes detecting and fixing data problems (e.g. missing values, mistakes, or outliers) and creating new feature columns by modifying or merging existing ones.

The data consists of the following columns:

    instant: A unique row identifier
    dteday: The date on which the data was observed;
    season: A numerically encoded value indicating the season
    yr: The year of the study in which the observation was made 
    mnth: The calendar month in which the observation was made 
    holiday: A binary value indicating whether or not the observation was made on a public holiday)
    weekday: The day of the week on which the observation was made 
    workingday: A binary value indicating whether or not the day is a working day 
    weathersit: A categorical value indicating the weather situation 
    temp: The temperature in celsius 
    atemp: The apparent ("feels-like") temperature in celsius (normalized)
    hum: The humidity level 
    windspeed: The windspeed 
    rentals: The number of bicycle rentals recorded.

In this dataset, rentals represents the label (the y value) we must train our model to predict. The other columns are potential features (x values).
descriptive statistics using the describe () function.

Visual examination of numerical feaatures - Numeric features appear to be more normal distributed, whereas category features have a more or less uniform distribution. 

Next, we want to anticipate the correlations between the features and the rentals label.Generate scatter plots for numeric features to show the intersection of feature and label values.

## Train a Regression Model
The dataset is divided into numpy arrays X and Y, where X_train and Y_train represent the feature values and labels, respectively.
The data is then separated into two subsets: training and a smaller "hold-back" set for validation. The LinearRegression estimator trains a linear regression model by concentrating on the linear relationship between the X and Y labels.


    X_train: The feature values we'll use to train the model
    y_train: The corresponding labels we'll use to train the model
    X_test: The feature values we'll use to validate the model
    y_test: The corresponding labels we'll use to validate the model
## Evaluate the Trained Model
To assess the model's efficacy, three metrics are used: Mean Square Error (MSE), Root Mean Square Error (RMSE), and Coefficient of Determination (R-squared). The smaller the number, the better the model fits. The RMSE shows the average number of rentals for which forecasts are incorrect, and R2 represents the variation that the model can explain.



