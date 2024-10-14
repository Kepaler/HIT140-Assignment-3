import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load datasets
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# List of well-being score variables in dataset3
wellbeing_vars = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Sum the well-being scores for each participant
dataset3['sum_wellbeing'] = dataset3[wellbeing_vars].sum(axis=1)

# Select relevant columns from dataset3
dataset3_reduced = dataset3[['ID', 'sum_wellbeing']]

# Merge the summed well-being scores with dataset2 on 'ID'
merged_data = pd.merge(dataset2, dataset3_reduced, on='ID')

# Prepare the data for regression
screen_time_vars = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
merged_data['average_screen_time'] = merged_data[screen_time_vars].mean(axis=1)

X = merged_data[['average_screen_time']]
y = merged_data['sum_wellbeing']

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict well-being scores
predictions = model.predict(X)

# Add predictions to the dataframe
merged_data['predicted_well_being'] = predictions

# Select only the columns of interest
result = merged_data[['ID', 'average_screen_time', 'predicted_well_being']]

# Print the full list
print(result)

# Save the result to a CSV file
result.to_csv('predicted_wellbeing.csv', index=False)

# Plotting the results
plt.figure(figsize=(10,6))
sns.scatterplot(x='average_screen_time', y='sum_wellbeing', data=merged_data, label='Actual Well-Being', color='blue', alpha=0.3)
sns.lineplot(x='average_screen_time', y='predicted_well_being', data=merged_data, label='Predicted Well-Being', color='red')

plt.title('Linear Regression: Screen Time vs Well-Being')
plt.xlabel('Average Screen Time')
plt.ylabel('Well-Being Score')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()