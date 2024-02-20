import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, silhouette_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
from itertools import combinations
from scipy import stats
from collections import defaultdict
import csv


##### VISUALIZATION

# CATEGORICAL BAR CHARTS
def bar_charts_categorical(df, feature, target):
    '''
    Generate categorical bar charts for a feature against a target variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.

    feature : str
        The categorical feature column name for which bar charts are generated.

    target : str
        The target variable column name used for comparison.

    Returns:
    --------
    None
    '''
    # Create a contingency table using crosstab
    cont_tab = pd.crosstab(df[feature], df[target], margins=True)
    # Extract categories from the index of the contingency table excluding the 'All' row
    categories = cont_tab.index[:-1]
        
    # Create a figure to hold subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot for frequency bar chart
    plt.subplot(121)
    # Plot bars for each category for both target values ('$y_i=0$', '$y_i=1$')
    p1 = plt.bar(categories, cont_tab.iloc[:-1, 0].values, 0.55, color="powderblue")
    p2 = plt.bar(categories, cont_tab.iloc[:-1, 1].values, 0.55, bottom=cont_tab.iloc[:-1, 0], color="cadetblue")
    # Add legend for the bars
    plt.legend((p2[0], p1[0]), ('$y_i=1$', '$y_i=0$'))
    plt.title("Frequency bar chart")
    plt.xlabel(feature)
    plt.ylabel("$Frequency$")

    # Calculate observed proportions for each category and target value
    obs_pct = np.array([np.divide(cont_tab.iloc[:-1, 0].values, cont_tab.iloc[:-1, 2].values), 
                        np.divide(cont_tab.iloc[:-1, 1].values, cont_tab.iloc[:-1, 2].values)])
      
    # Subplot for proportion bar chart
    plt.subplot(122)
    # Plot bars representing observed proportions for each category and target value
    p1 = plt.bar(categories, obs_pct[0], 0.55, color="snow")
    p2 = plt.bar(categories, obs_pct[1], 0.55, bottom=obs_pct[0], color="rosybrown")
    # Add legend for the bars
    plt.legend((p2[0], p1[0]), ('$y_i=1$', '$y_i=0$'))
    plt.title("Proportion bar chart")
    plt.xlabel(feature)
    plt.ylabel("$p$")

    # Show the plot
    plt.show()


# PROPERTIES OF PLOTS
def set_plot_properties(ax, x_label, y_label, y_lim=[]):
    '''
    Set properties of a plot axis.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].

    Returns:
        None
    '''
    ax.set_xlabel(x_label)  # Set the label for the x-axis
    ax.set_ylabel(y_label)  # Set the label for the y-axis
    if len(y_lim) != 0:
        ax.set_ylim(y_lim)  # Set the limits for the y-axis if provided


# BAR CHART
def plot_bar_chart(ax, data, variable, x_label, y_label='Count', y_lim=[], legend=[], color='cadetblue', annotate=False):
    '''
    Plot a bar chart based on the values of a variable in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].
        legend (list, optional): The legend labels. Defaults to [].
        color (str, optional): The color of the bars. Defaults to 'cadetblue'.
        annotate (bool, optional): Flag to annotate the bars with their values. Defaults to False.

    Returns:
        None
    '''
    counts = data[variable].value_counts()  # Count the occurrences of each value in the variable
    x = counts.index
    y = counts.values

    ax.bar(x, y, color=color)  # Plot the bar chart with specified color
    ax.set_xticks(x)  # Set the x-axis tick positions
    if len(legend) != 0:
        ax.set_xticklabels(legend)  # Set the x-axis tick labels if provided

    if annotate:
        for i, v in enumerate(y):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=12)  # Annotate the bars with their values

    set_plot_properties(ax, x_label, y_label, y_lim)  # Set plot properties using helper function


# PIE CHART
def plot_pie_chart(data, variable, colors, labels=None, legend=[], autopct='%1.1f%%'):
    '''
    Plot a pie chart based on the values of a variable in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        colors (list): The colors for each pie slice.
        labels (list, optional): The labels for each pie slice. Defaults to None.
        legend (list, optional): The legend labels. Defaults to [].
        autopct (str, optional): The format for autopct labels. Defaults to '%1.1f%%'.

    Returns:
        None
    '''
    counts = data[variable].value_counts()  # Count the occurrences of each value in the variable

    # Plot the pie chart with specified parameters
    plt.pie(counts, colors=colors, labels=labels, startangle=90, autopct=autopct, textprops={'fontsize': 25})
    
    if len(legend) != 0:
        plt.legend(legend, fontsize=16, bbox_to_anchor=(0.7, 0.9))  # Add a legend if provided
    
    plt.show()  # Display the pie chart


# HISTOGRAM
def plot_histogram(ax, data, variable, x_label, y_label='Count', color='rosybrown'):
    '''
    Plot a histogram based on the values of a variable in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        color (str, optional): The color of the histogram bars. Defaults to 'cadetblue'.

    Returns:
        None
    '''
    plt.hist(data[variable], bins=50, color=color)  # Plot the histogram using 50 bins

    set_plot_properties(ax, x_label, y_label)  # Set plot properties using helper function


# CORRELATION MATRIX
def plot_correlation_matrix(data, method):
    '''
    Plot a correlation matrix heatmap based on the given data.

    Args:
        data (pandas.DataFrame): The input data for calculating correlations.
        method (str): The correlation method to use.

    Returns:
        None
    '''
    corr = data.corr(method=method)  # Calculate the correlation matrix using the specified method

    mask = np.tri(*corr.shape, k=0, dtype=bool)  # Create a mask to hide the upper triangle of the matrix
    corr.where(mask, np.NaN, inplace=True)  # Set the upper triangle values to NaN

    plt.figure(figsize=(30, 15))  # Adjust the width and height of the heatmap as desired

    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                annot=True,
                vmin=-1, vmax=1,
                cmap=sns.diverging_palette(220, 10, n=20))  # Plot the correlation matrix heatmap
    

##### DATA PREPROCESSING

# DATA TYPES
def datatype_distinction(data):
    '''
    Distinguishes between the numerical and categorical columns in a DataFrame.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame.

    Returns:
    --------
    numerical : pandas.DataFrame
        DataFrame containing only numerical columns.

    categorical : pandas.DataFrame
        DataFrame containing only categorical columns.
    '''
    # Select numerical columns using select_dtypes with np.number
    numerical = data.select_dtypes(include=np.number).copy()
    
    # Select categorical columns by excluding numerical types
    categorical = data.select_dtypes(exclude=np.number).copy()
    
    return numerical, categorical


# DATA TRANSFORMATION
def transformation(technique, data, column_transformer=False):
    '''
    Applies the specified transformation technique to the DataFrame.

    Parameters:
    -----------
    technique : object
        The transformation technique (e.g., from Scikit-learn) to be applied.

    data : pandas.DataFrame
        The input DataFrame to be transformed.

    column_transformer : bool, optional (default=False)
        Flag to indicate if a column transformer is used for custom column names.

    Returns:
    --------
    data_transformed : pandas.DataFrame
        Transformed DataFrame.

    Notes:
    ------
    - If column_transformer is False, the columns in the transformed DataFrame
      will retain the original column names.
    - If column_transformer is True, the method assumes that technique has a
      get_feature_names_out() method and uses it to get feature names for the
      transformed data, otherwise retains the original column names.
    '''
    # Apply the specified transformation technique to the data
    data_transformed = technique.transform(data)
    
    # Create a DataFrame from the transformed data
    data_transformed = pd.DataFrame(
        data_transformed,
        index=data.index,
        columns=technique.get_feature_names_out() if column_transformer else data.columns
    )
    
    return data_transformed


def data_transform(technique, X_train, X_val=None, column_transformer=False):
    '''
    Fits a data transformation technique on the training data and applies the transformation 
    to both the training and validation data.

    Parameters:
    -----------
    technique : object
        The data transformation technique (e.g., from Scikit-learn) to be applied.

    X_train : pandas.DataFrame or array-like
        The training data to fit the transformation technique and transform.

    X_val : pandas.DataFrame or array-like, optional (default=None)
        The validation data to be transformed.

    column_transformer : bool, optional (default=False)
        Flag to indicate if a column transformer is used for custom column names.

    Returns:
    --------
    X_train_transformed : pandas.DataFrame
        Transformed training data.

    X_val_transformed : pandas.DataFrame or None
        Transformed validation data. None if X_val is None.

    Notes:
    ------
    - Fits the transformation technique on the training data (X_train).
    - Applies the fitted transformation to X_train and optionally to X_val if provided.
    '''
    # Fit the transformation technique on the training data
    technique.fit(X_train)
    
    # Apply transformation to the training data
    X_train_transformed = transformation(technique, X_train, column_transformer)
    
    # Apply transformation to the validation data if provided
    X_val_transformed = None
    if X_val is not None:
        X_val_transformed = transformation(technique, X_val, column_transformer)
        
    return X_train_transformed, X_val_transformed


# IMPUTING MISSING VALUES
def knn_imputer_best_k(data, k_min, k_max, weights='distance'):
    '''
    Determines the best K value for KNNImputer by optimizing silhouette scores.

    Parameters:
    -----------
    data : pandas.DataFrame or array-like
        The dataset for imputation and clustering.

    k_min : int
        The minimum value of K for KNN.

    k_max : int
        The maximum value of K for KNN.

    weights : str, optional (default='distance')
        Weight function used in prediction for KNNImputer.

    Returns:
    --------
    best_k : int
        The best K value determined by maximizing silhouette scores.

    Notes:
    ------
    - Uses KNNImputer for missing value imputation.
    - Utilizes KMeans for clustering after imputation.
    - Calculates silhouette scores for different K values.
    - Determines the best K value based on the highest silhouette score.
    '''
    # Initialize KNNImputer and KMeans
    knn_imputer = KNNImputer(weights=weights)
    kmeans = KMeans()

    # Function to calculate mean silhouette score for a given K value
    def cv_silhouette_score(k):
        knn_imputer.set_params(n_neighbors=k)
        scores = []
        for _ in range(10):  # Adjust folds as needed
            X_train, X_test = train_test_split(data, train_size=0.9, random_state=42)
            X_train_imputed = knn_imputer.fit_transform(X_train)
            X_test_imputed = knn_imputer.transform(X_test)
            kmeans.set_params(n_clusters=k)
            kmeans.fit(X_train_imputed)
            labels = kmeans.predict(X_test_imputed)
            scores.append(silhouette_score(X_test_imputed, labels))
        return np.mean(scores)

    # Calculate silhouette scores in parallel for different K values
    results = Parallel(n_jobs=-1)(delayed(cv_silhouette_score)(k) for k in range(k_min, k_max + 1))
    
    # Determine the best K value with the highest silhouette score
    best_k = range(k_min, k_max + 1)[np.argmax(results)]
    print('Best K value:', best_k)
    return best_k


##### FEATURE SELECTION

# CHI-SQUARE
def TestIndependence(X, y, var, alpha=0.05):
    '''
    Test the independence of a categorical variable with respect to the target variable.

    Parameters:
    -----------
    X : pandas.Series or array-like
        The independent categorical variable.

    y : pandas.Series or array-like
        The target variable.

    var : str
        The name of the variable being tested for importance.

    alpha : float, optional (default=0.05)
        The significance level for the test.

    Returns:
    --------
    None

    Notes:
    ------
    - Performs a chi-squared test of independence between X and y.
    - Compares the p-value with the significance level (alpha).
    - Prints whether the variable is important for prediction or not based on p-value.
    '''
    # Create a contingency table of observed frequencies
    dfObserved = pd.crosstab(y, X)
    
    # Perform chi-squared test and retrieve test statistics
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    
    # Create a DataFrame of expected frequencies
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index=dfObserved.index)
    
    # Determine the importance of the variable based on the p-value
    if p < alpha:
        result = "{0} is IMPORTANT for Prediction".format(var)
    else:
        result = "{0} is NOT an important predictor. (Discard {0} from model)".format(var)
    
    # Print the result
    print(result)


# TREE-BASED
def tree_based_method(X, y, threshold='median'):
    '''
    Perform feature selection using the Extra Trees Classifier method.

    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        The feature matrix.

    y : pandas.Series or array-like
        The target variable.

    threshold : str or float, optional (default='median')
        The feature importance threshold to select features.

    Returns:
    --------
    None

    Notes:
    ------
    - Uses Extra Trees Classifier for feature selection based on feature importances.
    - Prints the names of selected features based on the specified threshold.
    - Does not modify the original data.
    - Reference: https://scikit-learn.org/stable/modules/feature_selection.html
    '''
    # Initialize Extra Trees Classifier
    rf_model = ExtraTreesClassifier(n_estimators=100, random_state=42)

    # Fit the model to your data
    rf_model.fit(X, y)

    # Create a feature selector based on feature importances
    feature_selector = SelectFromModel(rf_model, prefit=True, threshold=threshold)

    # Get the selected feature indices
    selected_indices = feature_selector.get_support(indices=True)

    # Get the column names of selected features
    selected_feature_names = X.columns[selected_indices]

    # Print the names of selected features
    print("\nSelected features:")
    for feature in selected_feature_names:
        print('->', feature, end='\n')


# VARIABLE INFLATION FACTOR
def vif(X):
    '''
    Calculate Variance Inflation Factor (VIF) for each feature to detect multicollinearity.

    Parameters:
    -----------
    X : pandas.DataFrame or array-like
        The feature matrix.

    Returns:
    --------
    None

    Notes:
    ------
    - Calculates VIF for each feature in the given feature matrix.
    - Identifies variables with moderate and high multicollinearity based on a threshold.
    - Prints the variables with moderate and high multicollinearity.
    '''
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Features"] = X.columns
    vif_data["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Set the threshold for high multicollinearity
    threshold = 5

    # Print variables with moderate multicollinearity
    moderate_multicollinearity = vif_data[vif_data["VIF Factor"] <= threshold].sort_values(by='VIF Factor', ascending=True)
    print("Variables with Moderate Multicollinearity:")
    print(moderate_multicollinearity)

    # Print variables with high multicollinearity
    print("\nVariables with High Multicollinearity (discard):")
    print(vif_data[vif_data["VIF Factor"] > threshold].sort_values(by='VIF Factor', ascending=True))


# EXHAUSTIVE
def evaluate_subset(X_train_scaled, X_test_scaled, y_train, y_test):
    '''
    Evaluate a subset of features using Logistic Regression and F1 score.

    Parameters:
    -----------
    X_train_scaled : pandas.DataFrame
        Training set features.

    X_test_scaled : pandas.DataFrame
        Test set features.

    y_train : pandas.Series
        Training set target.

    y_test : pandas.Series
        Test set target.

    Returns:
    --------
    float
        F1 score for the subset of features.
    '''
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)


def exhaustive_method_parallel(X, y, scaler=None):
    '''
    Perform an exhaustive feature selection using all possible feature combinations in parallel.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    scaler : Scaler object, optional (default=None)
        Scaler for feature scaling.

    Returns:
    --------
    list
        Best subset of features based on F1 score.
    '''
    best_score = 0
    best_subset = None
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    def evaluate_subset_wrapper(subset, scaler=None):
        # Evaluate F1 score for the subset of features
        X_subset = X[list(subset)]
        
        scores = []
        for train_index, test_index in skf.split(X_subset, y):
            X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if scaler is not None:
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            score = evaluate_subset(X_train, X_test, y_train, y_test)
            scores.append(score)

        return np.mean(scores), subset

    # Generate all possible combinations of features and evaluate their performance
    for k in range(1, len(X.columns) + 1):
        feature_combinations = list(combinations(X.columns, k))
        results = Parallel(n_jobs=-1)(delayed(evaluate_subset_wrapper)(subset, scaler) for subset in feature_combinations)

        for mean_score, subset in results:
            if mean_score > best_score:
                best_score = mean_score
                best_subset = subset
    
    return best_subset


def exhaustive_method(X, y, iterations, scaler=None):
    '''
    Perform exhaustive feature selection over multiple iterations.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    iterations : int
        Number of iterations for the exhaustive search.

    scaler : Scaler object, optional (default=None)
        Scaler for feature scaling.

    Returns:
    --------
    defaultdict
        Number of occurrences of each variable in the best subsets obtained across iterations.
    '''
    variable_occurrences = defaultdict(int)

    for _ in range(iterations):
        best_subset = exhaustive_method_parallel(X, y, scaler)
        # Increment count for each variable in the best subset
        for var in best_subset:
            variable_occurrences[var] += 1

    return variable_occurrences


# LASSO
def lasso_method(X, y):
    '''
    Perform feature selection using Lasso regression and visualize feature importance.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    Returns:
    --------
    None

    Notes:
    ------
    - Fits a LassoCV model to perform feature selection.
    - Plots and visualizes feature importance based on Lasso coefficients.
    '''
    # Fit LassoCV model
    reg = LassoCV()
    reg.fit(X, y)
    
    # Extract feature coefficients and index them with column names
    coef = pd.Series(reg.coef_, index=X.columns)
    
    # Sort coefficients by magnitude
    imp_coef = coef.sort_values()
    
    # Plot feature importance using horizontal bar plot
    imp_coef.plot(kind="barh", color='gold')
    plt.title("Feature importance using Lasso Model")
    plt.xlabel("Coefficient magnitude")
    plt.ylabel("Features")
    plt.show()


# GENERAL FEATURE SELECTION
def feature_selection(X, y, iterations=100, scaler=None):
    '''
    Perform feature selection using various methods and print/visualize results.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    iterations : int, optional (default=100)
        Number of iterations for exhaustive feature selection.

    scaler : Scaler object, optional (default=None)
        Scaler for feature scaling.

    Returns:
    --------
    None

    Notes:
    ------
    - Utilizes multiple feature selection methods to analyze and print results.
    - Includes univariate analysis, tree-based method, correlation analysis, VIF calculation,
      exhaustive feature selection, and Lasso-based feature selection.
    '''
    # Scale the features if scaler is provided
    if scaler is not None:
        X_scaled = data_transform(scaler, X)[0]
    else:
        X_scaled = X

    # Print univariate variable analysis (variances)
    print('\n\nUNIVARIATE VARIABLES:')
    variables_variance = X_scaled.var()
    print(variables_variance)

    # Print tree-based feature selection results
    print('\n\nTREE-BASED:')
    tree_based_method(X_scaled, y)

    # Plot correlation matrix
    print('\n\nCORRELATIONS:')
    plot_correlation_matrix(X_scaled.join(y), 'spearman')
    plt.show()

    # Calculate and print VIF
    print('\n\nVIF:')
    vif(X)

    # Perform and print exhaustive feature selection
    print('\n\nEXHAUSTIVE:')
    occurrences = exhaustive_method(X, y, iterations, scaler)
    print("Variable Occurrences:")
    for var, count in occurrences.items():
        print(f"-> {var}: {count}")    

    # Perform Lasso-based feature selection and visualize results
    print('\n\nLASSO:')
    lasso_method(X_scaled, y)
    plt.show()


# PRINCIPAL COMPONENT ANALYSIS
def pc_analysis(X_train, var_threshold=0.8):
    '''
    Perform Principal Component Analysis (PCA) and retain components based on variance threshold.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training data for PCA.

    var_threshold : float, optional (default=0.8)
        Variance threshold to retain principal components.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing principal components that satisfy the variance threshold.

    Notes:
    ------
    - Computes principal components from the input training data.
    - Plots the cumulative explained variance ratio and a threshold line.
    - Retains principal components that explain variance above the specified threshold.
    '''
    n_columns = len(X_train.columns)
    
    # Perform PCA
    pca = PCA(n_components=n_columns)
    components = pca.fit_transform(X_train)

    # Plot cumulative explained variance ratio
    plt.plot(pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.axhline(y=var_threshold, color='r', linestyle='-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
    plt.show()

    # Determine the number of components that reach the variance threshold
    num_components = (pca.explained_variance_ratio_.cumsum() >= var_threshold).argmax() + 1

    # Create DataFrame containing principal components meeting the variance threshold
    pc = pd.DataFrame(components, 
                  index=X_train.index, 
                  columns=[f'PC{i}' for i in range(n_columns)]
                  ).iloc[:, :num_components]
    
    return pc


##### MODEL EVALUATION
def model_evaluator(model, parameters, X, y, scaler=None, log=False):
    '''
    Evaluates a model using GridSearchCV to find the best parameters and their performance.

    Args:
    - model: Machine learning estimator (e.g., classifier or regressor)
    - parameters: Dictionary of parameters for the model
    - X: Input features (DataFrame or array-like)
    - y: Target variable (Series or array-like)
    - scaler: Scaler object (e.g., StandardScaler, MinMaxScaler), default=None
    - log: Boolean indicating whether to log results to a CSV file, default=False

    Returns:
    - None

    Prints the best parameters and best F1 score achieved by the model.

    If log=True, appends the model's best parameters, best accuracy score, and feature columns to 'record.csv'.
    '''
    # Create steps for the Pipeline, including the estimator (model) and optionally a scaler
    steps = [('estimator', model)]
    
    if scaler is not None:
        steps.insert(0, ('scaler', scaler))

    # Create a Pipeline with the defined steps
    pipeline = Pipeline(steps)

    # Set up GridSearchCV to find the best parameters for the model
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=parameters,
                               scoring='f1',
                               cv=10
                               # n_jobs=-1
                               )

    # Fit the GridSearchCV object to find the best parameters
    grid_search.fit(X, y)
    
    # Get the best parameters and best accuracy score
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    # Print the best parameters and best accuracy score
    print('Best parameters: {}'.format(best_parameters))
    print('Best score: {}'.format(best_accuracy))
    
    # Log results to a CSV file if log=True
    if log:
        with open('record.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([X.columns, model, parameters, best_accuracy])



def avg_score(model, X, y, scaler=None): 
    '''
    Calculate the average F1 score for a given model using cross-validation.

    Parameters:
    -----------
    model : sklearn model object
        The model to evaluate.

    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    scaler : Scaler object, optional (default=None)
        Scaler for feature scaling.

    Returns:
    --------
    str
        A string containing the average F1 score +/- its standard deviation for train and test sets.

    Notes:
    ------
    - Utilizes Stratified K-Fold cross-validation with 10 splits.
    - Computes F1 score for train and test sets and calculates their average and standard deviation.
    '''
    # Apply k-fold cross-validation
    skf = StratifiedKFold(n_splits=10)

    # Create lists to store the results from different folds
    score_train = []
    score_test = []

    for train_index, val_index in skf.split(X, y):
        # Get the indexes of the observations assigned for each partition
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        if scaler is not None:
            # Fit and transform scaler on training data
            scaling = scaler.fit(X_train)
            X_train = scaling.transform(X_train)
            # Transform validation data using the scaler fitted on training data
            X_val = scaling.transform(X_val)
        
        # Fit the model to the training data
        model.fit(X_train, y_train)
        
        # Calculate F1 score for train and test sets
        value_train = f1_score(y_train, model.predict(X_train))
        value_test = f1_score(y_val, model.predict(X_val))
        
        # Append the F1 scores
        score_train.append(value_train)
        score_test.append(value_test)
 
    # Calculate the average and the standard deviation for train and test F1 scores
    avg_train = round(np.mean(score_train), 3)
    avg_test = round(np.mean(score_test), 3)
    std_train = round(np.std(score_train), 2)
    std_test = round(np.std(score_test), 2)
    
    # Format and return the results as a string
    return (
        str(avg_train) + '+/-' + str(std_train),
        str(avg_test) + '+/-' + str(std_test)
    )


##### DIMENSIONALITY REDUCTION
def visualize_dimensionality_reduction(transformation, targets):
    '''
    Visualize the dimensionality reduction results using a scatter plot.

    Args:
        transformation (numpy.ndarray): The transformed data points after dimensionality reduction.
        targets (numpy.ndarray or list): The target labels or cluster assignments.

    Returns:
        None
    '''
    # Convert object labels to categorical variables
    labels, targets_categorical = np.unique(targets, return_inverse=True)

    # Create a scatter plot of the t-SNE output
    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=0, vmax=len(labels) - 1)
    plt.scatter(transformation[:, 0], transformation[:, 1], c=targets_categorical, cmap=cmap, norm=norm)

    # Create a legend with the class labels and corresponding colors
    handles = [plt.scatter([], [], c=cmap(norm(i)), label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles, title='Clusters')

    # Display the plot
    plt.show()