import cvxpy as cp
import numpy as np
import csv
import matplotlib.pyplot as plt
#example for one problem using cvxpy
# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)
# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)
# The optimal objective value is returned by ‘prob.solve()‘.
result = prob.solve()
# The optimal value for x is stored in ‘x.value‘.
#print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# ‘constraint.dual_value‘.
#print(constraints[0].dual_value)

#Exercise one
# Reading csv file for male data
#male_index,male_bmi,male_stature = []
#female_index, female_bmi, female_stature = []
male_train = []
female_train = []
with open("male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    csv_file.readline()
    for row in reader:
        male_train.append(row[1:])
#        male_index.append(row[0])
#        male_bmi.append(row[1])
#        male_stature.append(row[2])

#print(male_index)
#print(male_bmi)
#print(male_stature)
# Add your code here to process the data into usable form
csv_file.close()
# Reading csv file for female data
with open("female_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    csv_file.readline()
    for row in reader:
        female_train.append(row[1:])
      #  female_index.append(row[0])
      #  female_bmi.append(row[1])
      #  female_stature.append(row[2])
#print(female_index)
#print(female_bmi)
#print(female_stature)
#print(male_train)
#print(female_train)
# Add your code here to process the data into usable form
csv_file.close()


#Exercise 2 c
n_male = len(male_train)
n_female = len(female_train)

#get b: 1 for male, -1 for female
class_one = np.ones((n_male))
class_mone = np.ones((n_female))* (-1)
y = (class_one, class_mone)
b = np.concatenate(y)
#print(b)
#get A
#Pre_A = np.array(np.concatenate((male_train,female_train)))
Pre_A = np.array((male_train + female_train), dtype=float)
#print(Pre_A)
ones = np.ones(((n_male + n_female), 1))
#print(ones)
A = np.concatenate((Pre_A, ones),axis=1)


#calculate the weight vector
A_T = np.transpose(A)
ATA = np.dot(A_T,A)
invA = np.linalg.inv(ATA)
invAAT = np.dot(invA, A_T)
w_vector = np.dot(invAAT, b)
print(w_vector)
#Answer: [-1.23396767e-02  6.67486843e-03 -1.07017505e+01]

#Exercise 2d
x = cp.Variable(len(A[0]))
objective = cp.Minimize(cp.sum_squares(A*x - b))
#constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective)
# The optimal objective value is returned by ‘prob.solve()‘.
result = prob.solve()
print(x.value)
#print(constraints[0].dual_value)


#Exercise 3
male_t = np.array(male_train,dtype=float)
fmale_t = np.array(female_train,dtype=float)
male_x = male_t[:,0]
male_y = male_t[:,1]
fmale_x = fmale_t[:,0]
fmale_y = fmale_t[:,1]
plt.scatter(male_x, male_y)
plt.scatter(fmale_x,fmale_y)
#plt.show()

#point range from 10 to 90
boundary_x = [10,90]
#w_vector[0] -> theta1, w_vector[1] -> theta2 ->w_vector[2]-> theta0
boundary_y = [-w_vector[2]/w_vector[1] - w_vector[0]* 10 / w_vector[1],-w_vector[2]/w_vector[1] - w_vector[0]* 90 / w_vector[1]]
plt.plot(boundary_x,boundary_y)
plt.show()

#exercise3_b
female_test = []
male_test = []
sm = 0
sf = 0
with open("male_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    csv_file.readline()
    for row in reader:
        if (float(w_vector[0])* float(row[1]) + float(w_vector[1])*float(row[2]) + float(w_vector[2])) > 0:
            sm = sm + 1
        male_test.append(row[1:])
csv_file.close()
with open("female_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    csv_file.readline()
    for row in reader:
        if (float(w_vector[0])* float(row[1]) + float(w_vector[1])*float(row[2]) + float(w_vector[2])) < 0:
            sf = sf + 1
        female_test.append(row[1:])
csv_file.close()

total_s = sm + sf
total_t = len(female_test) + len(male_test)
print(total_s, total_t)
s_rate = total_s / total_t
print(s_rate)

#0.8393213572854291   for the successful rate 83.932%


#Exercise 4

#a

lambda_list = np.arange(0.1, 10, 0.1)
w_vlist = []
#divide the stature__mm value by 100 to reduce the numerical error
A[:,1] = A[:,1]/100
for lam in lambda_list:
# Solve the regularized least-squares problem depending on lambda
    x_temp = cp.Variable(len(A[0]))
    objective = cp.Minimize(cp.sum_squares(A * x_temp - b) + lam*cp.sum_squares(x_temp))
    #constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective)
# The optimal objective value is returned by ‘prob.solve()‘.
    result = prob.solve()
    w_vlist.append(x_temp.value)
# Plot the decision boundaries with theta_{lambda=0.1, 1.1, ..., 9.1}
x_min = 10
x_max = 90
x = np.linspace(x_min, x_max, 200)
legend_str = []
plt.figure(figsize=(15,7.5))
#print(w_vlist)
male_x = male_t[:,0]
male_y = male_t[:,1] / 100
fmale_x = fmale_t[:,0]
fmale_y = fmale_t[:,1] / 100
plt.scatter(male_x, male_y)
plt.scatter(fmale_x, fmale_y)
for i in range(len(lambda_list))[0::10]:
    y = -w_vlist[i][2] / w_vlist[i][1] - w_vlist[i][0] * x / w_vlist[i][1]
 #   y = INTERCEPT(lambda_list[i]) + SLOPE(lambda_list[i])*x
    plt.plot(x, y.T)
    legend_str.append('$\lambda = $' + str(lambda_list[i]))

plt.legend(legend_str)
plt.xlabel('BMI')
plt.ylabel('Stature')
plt.savefig('exercise4.png')

#Exercise 4 (a) (ii)
#when regularization parameter increase, the regularization fuction which is the 2nd term increases. The first term decreases.


