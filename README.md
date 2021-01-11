# Machine-Learning-Projects

## Ozone Level Detection

#### Description: 

The purpose of this study was to help the Texas Commission on Environmental Quality (TCEQ) to build highly accurate ozone level alarm forecasting models for the Houston area. Therefore, I need to identify which one of the five selected statistical models has the best performance on detecting ozone level.

The data was collected from 1998 to 2004 at the Houston, Galveston and Brazoria area. It is a binary classification dataset, which contains more than 2000 observations and over 70 parameters (features). The dataset contains temperature, wind speed, wind direction, humidity level measured at different times throughout the day. 

Key characteristics of this problem that are challenging and interesting include:

    1. the dataset is sparse (72 features, and 2% or 5% positives depending on the criteria of "ozone days")
    
    2. evolving over time from year to year
    
    3. limited in collected data size (7 years or around 2500 data entries)
    
    4. contains a large number of irrelevant features
    
    5. biased in terms of "sample selection bias"
    
    6. the true model is stochastic as a function of measurable factors. 


#### Steps:

a. Determine the dataset is banlanced or not. If not, 
b. For each n(learn) ∈ {0.5n, 0.9n}, I repeated the following steps 100 times: 
 
    1. Randomly split the dataset into two mutually exclusive datasets D(validation) and D(learn) with size n(validation) and n(learn) such that n(learn) + n(validation) = n. 
    2. Use Dlearn to fit random forrest, radial svm, logistic, logistic lasso, and logistic ridge. 
    3. Tune the hyper parameters (of radial svm, logistic lasso, and logistic ridge) using 10-fold CV. 
    4. For each estimated model calculate the 
        1)training rate
        2) minimum cross-validation rate
        3) test misclassification error rate
 
