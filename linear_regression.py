import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


# predict Electrical Energy Output as a function of Temperature, Ambient Pressure, Relative Humidity, and Exhaust Vacuum
# pandas dataframe des valeurs
filename = 'combined_cycle_power_plant.xlsx'
my_data = pd.read_excel(filename, sheet_name=1, names=['AT', 'V', 'AP', 'RH', 'PE'], engine='openpyxl', dtype=float)
W = np.zeros([1, 5])
alpha = 0.01
iters = 1000


# normalization of data points
def normalize(df):
    for column in df:
        minimum = min(df[column])                                           # valeur minimale
        maximum = max(df[column])                                           # valeur maximale
        df[column] = (df[column] - minimum) / (maximum - minimum)


# compute cost
def computeCost(X,y,W):
    to_be_summed = np.power((y-(X @ W.T)),2)
    return np.sum(to_be_summed)/(len(X))


# gradient descent linear regression
def gradientDescent(X,y,W,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        W1=W.T
        A= y - X @ W1
        B= X.T @ A
        B1=(-2*alpha/len(X)) * B
        W1 = W1 - B1
        W=W1.T
        cost[i] = computeCost(X, y, W)
        # print(cost[i])
    return W,cost


# check accuracy of predicted data using Loss Functions
def error(ytest, ypred):
    r2 = r2_score(ytest, ypred)
    print("R-squared: ", r2)
    mse = mean_squared_error(ytest, ypred)
    print("Mean Squared Error: ", mse)
    print("Root Mean Square Error (RMSE): ", np.sqrt(mse))
    median_error = median_absolute_error(ytest, ypred)
    print("Median Absolute Error: ", median_error)
    mean_error = mean_absolute_percentage_error(ytest, ypred)
    print("Mean Absolute Percentage Error: ", mean_error)


# pearson and spearman correlation coefficients results interpretations
def correlation_interpretation(correlation_coefficient, p_value):
    if correlation_coefficient > 0.8:
        print("Very strong positive correlation.")
    elif correlation_coefficient > 0.6:
        print("Strong positive correlation.")
    elif correlation_coefficient > 0.4:
        print("Moderate positive correlation.")
    elif correlation_coefficient > 0.2:
        print("Weak positive correlation.")
    elif correlation_coefficient > -0.2:
        print("No correlation.")
    elif correlation_coefficient > -0.4:
        print("Weak negative correlation.")
    elif correlation_coefficient > -0.6:
        print("Moderate negative correlation.")
    elif correlation_coefficient > -0.8:
        print("Strong negative correlation.")
    elif correlation_coefficient > -1:
        print("Very strong negative correlation.")

    if p_value < 0.001:
        print("Forte certitude des résultats.")
    elif p_value < 0.05:
        print("Moyenne certitude des résultats.")
    elif p_value < 0.1:
        print("Faible certitude des résultats.")
    else:
        print("Aucune certitude des résultats.")


# 1) Représentations graphiques des différentes relations entre les variables explicatives et la variable d’intérêt
for column in my_data:
    if column == 'PE': continue                                             # can't plot PE vs. PE
    my_data.plot(x=column, y='PE', linestyle="None", marker='d', markersize=0.2)
    plt.xlabel(column)
    plt.ylabel('Net hourly electrical energy output, in MW')
    plt.title(f"PE vs. {column}")
    # plt.show()                                                              # plot graphs of variables

    print(f"Data: PE vs. {column}")
    pearson_coefficient, p_value = stats.pearsonr(my_data[f'{column}'], my_data['PE'])
    print(f"Pearson correlation coefficient, Pearson p-value: {pearson_coefficient, p_value}")
    correlation_interpretation(pearson_coefficient, p_value)
    spearman_coefficient, p_value = stats.spearmanr(my_data[f'{column}'], my_data['PE'])
    print(f"Spearman correlation coefficient, Spearman p-value: {spearman_coefficient, p_value}")
    correlation_interpretation(spearman_coefficient, p_value)
    print()


# 2) Normalisation des données
normalize(my_data)
X = my_data.iloc[1:, 0:-1].values                                            # variables explicatives
ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)
y = my_data.iloc[1:, -1:].values                                             # variable d’intérêt


# 3) Division du dataset en deux ensembles aléatoires, pour entrainement et validation
X, y = shuffle(X, y)                                                                        # shuffle X and y datasets
x_training, x_validation, y_training, y_validation = train_test_split(X, y, test_size=.2)   # split datasets in two


# 4) Regression linéaire multiple et evaluation de l'erreur
print('Multiple Linear Regression')
# Ajouter un epsilon sur les gradients
g,cost = gradientDescent(x_training, y_training, W, iters, alpha)
print(f"Weights of variables: {g}")
finalCost = computeCost(x_training, y_training, g)
print(f"Final Cost: {finalCost}")
fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Itérations')
ax.set_ylabel('Cost Function')
ax.set_title("Gradient Descent: Erreur vs. Nombre d'itérations")
print()


# 5) Regression descente de gradient stochastique et evaluation de l'erreur
print('Stochastic Linear Regression')
model = LinearRegression(fit_intercept=True).fit(x_training, y_training)

# Return the mean accuracy on the given test data
stochastic_score = model.score(x_training, y_training)
print("Stochastic Model coefficient of determination:", stochastic_score)

# predict test data by using the trained model
y_prediction = model.predict(x_validation)
error(y_validation, y_prediction)
print()


# 6) Regression Ridge et evaluation de l'erreur
print('Ridge Regression')
model = Ridge(alpha=0.000001, max_iter=1000)           # alpha value has the biggest impact on accuracy score
model.fit(x_training, y_training)                      # low accuracy score when alpha too high
ridge_score = model.score(x_training, y_training)      # accuracy plateaus (no increase) when alpha decreased past 10^-6
print("Ridge Model coefficient of determination:", ridge_score)
y_prediction = model.predict(x_validation)
error(y_validation, y_prediction)

plt.show()
