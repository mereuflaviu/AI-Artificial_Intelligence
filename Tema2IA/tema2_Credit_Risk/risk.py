import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the new datasets
credit_risk_test = pd.read_csv('credit_risk_test.csv')
credit_risk_train = pd.read_csv('credit_risk_train.csv')
credit_risk_full = pd.read_csv('credit_risk_full.csv')

# Display basic information about the datasets
def display_info(df, name):
    print(f"{name} Dataset Information:")
    df.info()
    print(f"First few rows of the {name} Dataset:")
    print(df.head())

# Function to plot histogram for the 'loan_approval_status' column
def plot_approval_histogram(df, name):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['loan_approval_status'], kde=False)
    plt.title(f'Histogram for Loan Approval Status ({name} Dataset)')
    plt.xlabel('Loan Approval Status')
    plt.ylabel('Frequency')
    plt.savefig(f'loan_approval_histogram_{name}.png')
    plt.show()
    print(f"\nHistogram for Loan Approval Status ({name} Dataset): This histogram shows the frequency distribution of the 'loan_approval_status' variable. It helps in understanding the class balance of the target variable.\n")

# Descriptive statistics for continuous numeric attributes
def descriptive_statistics(df, name):
    desc_stats = df.describe()
    print(f"Descriptive Statistics for Continuous Numeric Attributes ({name} Dataset):")
    print(desc_stats)
    desc_stats.to_csv(f'descriptive_statistics_{name}.csv')
    print(f"\nDescriptive Statistics for Continuous Numeric Attributes ({name} Dataset): This table provides a summary of the central tendency, dispersion, and shape of the dataset’s distribution.\n")

# Boxplot for continuous numeric attributes
def plot_boxplots(df, columns, name):
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot for {column} ({name} Dataset)')
        plt.savefig(f'boxplot_{column}_{name}.png')
        plt.show()
        print(f"\nBoxplot for {column} ({name} Dataset): This boxplot shows the distribution of the data based on a five-number summary: minimum, first quartile, median, third quartile, and maximum. It is useful for detecting outliers.\n")

# Histograms for categorical and ordinal attributes
def plot_histograms(df, columns, name):
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=False)
        plt.title(f'Histogram for {column} ({name} Dataset)')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f'histogram_{column}_{name}.png')
        plt.show()
        print(f"\nHistogram for {column} ({name} Dataset): This histogram shows the frequency distribution of the categorical/ordinal variable. It helps in understanding the distribution and frequency of each category.\n")

# Class balance analysis
def class_balance(df, target_column, name):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df[target_column])
    plt.title(f'Class Balance for {target_column} ({name} Dataset)')
    plt.xlabel(target_column)
    plt.ylabel('Frequency')
    plt.savefig(f'class_balance_{target_column}_{name}.png')
    plt.show()
    print(f"\nClass Balance for {target_column} ({name} Dataset): This countplot shows the frequency of each class in the target variable. It helps in understanding if the dataset is balanced or imbalanced.\n")

# Correlation analysis between numeric attributes
def numeric_correlation(df, name):
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Correlation Matrix for Numeric Attributes ({name} Dataset)')
    plt.savefig(f'numeric_correlation_{name}.png')
    plt.show()
    print(f"\nCorrelation Matrix for Numeric Attributes ({name} Dataset): This heatmap shows the correlation coefficients between pairs of numeric attributes. It helps in understanding the strength and direction of the relationship between variables.\n")

# Correlation analysis between categorical attributes using Chi-Square test
def categorical_correlation(df, cols, name):
    results = {}
    for col1 in cols:
        for col2 in cols:
            if col1 != col2:
                table = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, ex = chi2_contingency(table)
                results[(col1, col2)] = p
    return results

def plot_categorical_correlation(correlation_results, name):
    for key, p_value in correlation_results.items():
        if p_value < 0.05:
            print(f'Significant correlation between {key[0]} and {key[1]} ({name} Dataset) with p-value: {p_value}')
        else:
            print(f'No significant correlation between {key[0]} and {key[1]} ({name} Dataset) with p-value: {p_value}')

if __name__ == "__main__":
    datasets = {'Test': credit_risk_test, 'Train': credit_risk_train, 'Full': credit_risk_full}

    for name, df in datasets.items():
        display_info(df, name)
        plot_approval_histogram(df, name)
        
        # Perform descriptive statistics
        descriptive_statistics(df, name)
        
        # List of continuous numeric columns
        numeric_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col != 'loan_approval_status']
        plot_boxplots(df, numeric_columns, name)
        
        # List of categorical and ordinal columns
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        plot_histograms(df, categorical_columns, name)

        # Perform class balance analysis
        class_balance(df, 'loan_approval_status', name)
        
        # Perform numeric correlation analysis
        numeric_correlation(df, name)
        
        # Perform categorical correlation analysis
        correlation_results = categorical_correlation(df, categorical_columns, name)
        plot_categorical_correlation(correlation_results, name)
