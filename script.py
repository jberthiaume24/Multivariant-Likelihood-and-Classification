import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data from CSV
data = pd.read_csv('data.csv')

# Load train and test data from .npz files
train_data = np.load('train.npz', allow_pickle=True)
test_data = np.load('test.npz', allow_pickle=True)

# Extract data from the .npz files
X_train = train_data['X_train']
y_train = train_data['y_train']
X_test = test_data['X_test']
y_test = test_data['y_test']

# Convert the DataFrame to a numpy array
dataValues = data.values

# Split data into classes
c0 = dataValues[0:30, 0:2]  # Class 0 data
c1 = dataValues[30:60, 0:2] # Class 1 data
c2 = dataValues[60:100, 0:2] # Class 2 data

# Verify shapes of the class data
print(f"Shape of c0: {c0.shape}")  # Should be (30, 2)
print(f"Shape of c1: {c1.shape}")  # Should be (30, 2)
print(f"Shape of c2: {c2.shape}")  # Should be (40, 2)

# Step 1: Generate scatter plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(c0[:, 0], c0[:, 1], edgecolors='r', label='Class 0')
ax1.scatter(c1[:, 0], c1[:, 1], edgecolors='g', label='Class 1')
ax1.scatter(c2[:, 0], c2[:, 1], edgecolors='b', label='Class 2')
ax1.legend()
plt.savefig('scatterplot.png')
plt.show()

# Step 2: Estimate prior probabilities
num_c0 = len(c0)
num_c1 = len(c1)
num_c2 = len(c2)
N = len(dataValues)

P_C0 = num_c0 / N
P_C1 = num_c1 / N
P_C2 = num_c2 / N

print(f"P(C0) = {P_C0:.4f}")
print(f"P(C1) = {P_C1:.4f}")
print(f"P(C2) = {P_C2:.4f}")

# Step 3: Estimate mean and covariance
mean_c0 = np.mean(c0, axis=0)
cov_c0 = np.cov(c0, rowvar=False)

mean_c1 = np.mean(c1, axis=0)
cov_c1 = np.cov(c1, rowvar=False)

mean_c2 = np.mean(c2, axis=0)
cov_c2 = np.cov(c2, rowvar=False)

print(f"Mean of Category 0: {mean_c0}")
print(f"Covariance of Category 0: \n{cov_c0}")

print(f"Mean of Category 1: {mean_c1}")
print(f"Covariance of Category 1: \n{cov_c1}")

print(f"Mean of Category 2: {mean_c2}")
print(f"Covariance of Category 2: \n{cov_c2}")

# Define grid for likelihood plots
x_min, x_max = dataValues[:, 0].min() - 1, dataValues[:, 0].max() + 1
y_min, y_max = dataValues[:, 1].min() - 1, dataValues[:, 1].max() + 1

x, y = np.meshgrid(np.arange(x_min, x_max, 0.01),
                   np.arange(y_min, y_max, 0.01))
pos = np.dstack((x, y))

# Compute likelihoods
rv0 = multivariate_normal(mean_c0, cov_c0)
rv1 = multivariate_normal(mean_c1, cov_c1)
rv2 = multivariate_normal(mean_c2, cov_c2)

# Create plot
ax2 = plt.subplot()

# Plot contours for class 0
contour0 = ax2.contourf(x, y, P_C0 * rv0.pdf(pos), alpha=0.6, cmap='Blues', levels=30)

# Plot contours for class 1
contour1 = ax2.contourf(x, y, P_C1 * rv1.pdf(pos), alpha=0.6, cmap='Reds', levels=30)

# Plot contours for class 2
contour2 = ax2.contourf(x, y, P_C2 * rv2.pdf(pos), alpha=0.6, cmap='Greens', levels=30)

# Add contour lines on top
ax2.contour(x, y, P_C0 * rv0.pdf(pos), levels=10, colors='blue', linestyles='solid', linewidths=1)
ax2.contour(x, y, P_C1 * rv1.pdf(pos), levels=10, colors='red', linestyles='solid', linewidths=1)
ax2.contour(x, y, P_C2 * rv2.pdf(pos), levels=10, colors='green', linestyles='solid', linewidths=1)

# Scatter plot data
ax2.scatter(c0[:,0], c0[:,1], color='b', label='Class 0', edgecolor='k')
ax2.scatter(c1[:,0], c1[:,1], color='r', label='Class 1', edgecolor='k')
ax2.scatter(c2[:,0], c2[:,1], color='g', label='Class 2', edgecolor='k')

# Add labels, title, and legend
ax2.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Likelihood Function with Contour Plot')
plt.savefig('likelihood.png')
plt.show()

# Step 4: Classify entire dataset and compute confusion matrix
p0 = P_C0 * rv0.pdf(dataValues[:,0:2])
p1 = P_C1 * rv1.pdf(dataValues[:,0:2])
p2 = P_C2 * rv2.pdf(dataValues[:,0:2])

possibilities = np.c_[p0, p1, p2]
predic = np.argmax(possibilities, axis=1)

accuracy = np.mean(predic == dataValues[:,2])
print(f'The classification accuracy on the entire set is: {accuracy:.4f}')

cm = confusion_matrix(dataValues[:,2], predic)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.savefig('ConfusionMatrix.png')
plt.show()

# Step 5: Classification using train and test data
num_c0_train = np.sum(y_train == 0)
num_c1_train = np.sum(y_train == 1)
num_c2_train = np.sum(y_train == 2)
N_train = len(y_train)

P_C0_train = num_c0_train / N_train
P_C1_train = num_c1_train / N_train
P_C2_train = num_c2_train / N_train

mean_c0_train = np.mean(X_train[y_train == 0], axis=0)
cov_c0_train = np.cov(X_train[y_train == 0], rowvar=False)

mean_c1_train = np.mean(X_train[y_train == 1], axis=0)
cov_c1_train = np.cov(X_train[y_train == 1], rowvar=False)

mean_c2_train = np.mean(X_train[y_train == 2], axis=0)
cov_c2_train = np.cov(X_train[y_train == 2], rowvar=False)

# Multivariate normal distributions
rv0_train = multivariate_normal(mean_c0_train, cov_c0_train)
rv1_train = multivariate_normal(mean_c1_train, cov_c1_train)
rv2_train = multivariate_normal(mean_c2_train, cov_c2_train)

# Classify test data
p0_test = P_C0_train * rv0_train.pdf(X_test)
p1_test = P_C1_train * rv1_train.pdf(X_test)
p2_test = P_C2_train * rv2_train.pdf(X_test)

possibilities_test = np.c_[p0_test, p1_test, p2_test]
predic_test = np.argmax(possibilities_test, axis=1)

accuracy_test = np.mean(predic_test == y_test)
print(f'The classification accuracy on the test set is: {accuracy_test:.4f}')

cm_test = confusion_matrix(y_test, predic_test)
cm_display_test = ConfusionMatrixDisplay(cm_test).plot()
plt.savefig('ConfusionMatrixTesting.png')
plt.show()

# Step 6: Identify the top 5 points with the largest gi values
g0_train = P_C0_train * rv0_train.pdf(X_train)
g1_train = P_C1_train * rv1_train.pdf(X_train)
g2_train = P_C2_train * rv2_train.pdf(X_train)

top5_indices_c0 = np.argsort(g0_train)[-5:]
top5_indices_c1 = np.argsort(g1_train)[-5:]
top5_indices_c2 = np.argsort(g2_train)[-5:]

top5_points_c0 = X_train[top5_indices_c0]
top5_points_c1 = X_train[top5_indices_c1]
top5_points_c2 = X_train[top5_indices_c2]

print('----- Largest five points in class 0 -----')
print(top5_points_c0)

print('----- Largest five points in class 1 -----')
print(top5_points_c1)

print('----- Largest five points in class 2 -----')
print(top5_points_c2)

# Plot the top 5 points with largest gi values
fig2, ax3 = plt.subplots()

# Scatter plot for entire dataset with class colors
colors = {0: 'blue', 1: 'red', 2: 'green'}
for class_label in [0, 1, 2]:
    ax3.scatter(X_train[y_train == class_label, 0], X_train[y_train == class_label, 1], 
                color=colors[class_label], label=f'Class {class_label}', s=20, alpha=0.6)

# Highlight top 5 points for each class
ax3.scatter(top5_points_c0[:, 0], top5_points_c0[:, 1], color='darkblue', edgecolor='k', label='Top 5 Class 0', s=80, marker='o')
ax3.scatter(top5_points_c1[:, 0], top5_points_c1[:, 1], color='darkred', edgecolor='k', label='Top 5 Class 1', s=80, marker='o')
ax3.scatter(top5_points_c2[:, 0], top5_points_c2[:, 1], color='darkgreen', edgecolor='k', label='Top 5 Class 2', s=80, marker='o')

# Add labels, title, and legend
ax3.set_xlabel('X1')
ax3.set_ylabel('X2')
ax3.set_title('Top 5 Points with Largest gi Values')
ax3.legend()
plt.savefig('gi.png')
plt.show()
