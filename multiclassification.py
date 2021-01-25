import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# There are 5000 training examples in ex3data1.mat,
# where each training example is a 20 pixel by 20 pixel grayscale image of the digit.
# Each pixel is represented by a floating point number
# indicating the grayscale intensity at that location.

filename = "ex3data1.mat"
mat = scipy.io.loadmat(filename)
X = mat["X"]
y = mat["y"].flatten()
y = np.where(y == 10, 0, y)




def display_data(X):
    # X is a matrix, each row represent an example
    # 20 * 20 pixel img will have 400 column

    # cal example width and height
    n = X.shape[1]
    example_width = round(np.sqrt(n))
    example_height = round(n / example_width)

    pad = 1
    # cal row and col for example
    m = X.shape[0]
    row = int(np.floor(np.sqrt(m)))
    col = int(np.ceil(m / row))
    # init display array to -1 including pad
    disp_row = pad + row * (example_height + pad)
    disp_col = pad + col * (example_width + pad)
    disp_array = - np.ones([disp_row, disp_col])

    for i in range(m):
        max_val = max(abs(X[i, :]))
        cur_row = i // col
        cur_col = i % col
        row_idx = pad + cur_row * (example_height + pad)
        col_idx = pad + cur_col * (example_width + pad)
        disp_array[row_idx:row_idx + example_height, col_idx:col_idx + example_width] = X[i, :].reshape(example_width,
                                                                                                        example_height) / max_val

    plt.imshow(disp_array.T, cmap="gray")
    plt.show()

    return disp_array


# we need to transport the matrix using scipy.io.loadmat()
# random = X[4000,:]
# plt.imshow(random.reshape(20,20).T)

# random pick 100 example from the data
# random_idx = np.random.permutation(X.shape[0])
# display_data(X[random_idx[:100], :])


def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g


def compute_cost(theta, X, y, lamb):
    # X(m * n) theta(n,) h(m,)
    h = sigmoid(X.dot(theta))
    m = X.shape[0]
    # y(m,)
    # y.T.dot(np.log(h))  == y.dot(np.log(h)) when y and h are both 0 dimension
    J = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + (lamb / m / 2) * np.sum(np.power(theta[1:], 2))
    return J


def compute_gradient(theta, X, y, lamb):
    # X(m * n) theta(n,) h(m,) y(m,)
    h = sigmoid(X.dot(theta))
    m = X.shape[0]
    # grad(n,)    h-y(m,)  X(m,n)
    grad = 1 / m * (h - y).T.dot(X)
    # not regulization the theta0
    temp = theta
    temp[0] = 0
    grad = grad + lamb / m * temp
    return grad


# reshape(m,n) is different in python and matlab
# matlab.reshape = python.reshape(order="F")
# test_input_arr = np.asarray([x / 10 for x in range(1, 16)])
# test_X = test_input_arr.reshape(5, 3, order="F")
# test_ones = np.ones([test_X.shape[0], 1])
# print(test_input_arr)
# print(test_input_arr.reshape(5,3))
# print(test_X)

# test_X = np.hstack([test_ones, test_X])
# test_y = np.asarray([1, 0, 1, 0, 1])
#
# lamb = 3
# theta = np.asarray([-2, -1, 1, 2])


# 2.534819396109744  for cost
# print(compute_cost(theta,test_X,test_y,lamb))
# [ 0.14656137 -0.54855841  0.72472227  1.39800296] for gradient
# print(compute_gradient(theta,test_X,test_y,lamb))

# cost function and compute gradient are to serve the fmin function, so no need to add ones
# one_vs_all is going to train the model using fmin, so need to add ones to raw data
def one_vs_all(X, y, num_label, lamb):
    m = X.shape[0]
    ones = np.ones([m, 1])
    # X(m * n+1)
    X = np.hstack([ones, X])
    # each row is the theta for label
    # row = num_label    column = features = X.shape[1](include the theta0)
    # theta[:,i] (n+1,1)
    all_theta = np.zeros([X.shape[1], num_label])

    # number start from 0 to 9
    for num in range(num_label):
        # convert y from label into [0,1] array
        # y become shape (5000,1) when it has dimension
        # the subtract h-y will become (5000,5000)
        # need to flatten the y here
        # remeber not to change the y value here   y = (y == num).astype(int).reshape(-1) will change the y value
        label = (y == num).astype(int).reshape(-1)
        theta = opt.fmin_cg(compute_cost, all_theta[:, num], fprime=compute_gradient, args=(X, label, lamb))
        all_theta[:, num] = theta


    return all_theta


# test one_vs_all
num_label = 10
lamb = 0.1
all_theta = one_vs_all(X, y, num_label, lamb)



# print(all_theta[:,0].shape)

# use X to dot all_theta, the max value's corresponding index is the predict label
def predict_one_vs_all(X, all_theta):
    # add ones to X    X(5000,401)
    m = X.shape[0]
    ones = np.ones([m, 1])
    X = np.hstack([ones, X])
    # all_theta (401,10)   predict result(5000,10)
    result = np.dot(X,all_theta)
    # axis 0 find the vertical max index
    # axis 1 find the horizontal max index
    predict = np.argmax(result, axis=1)
    return predict


predict = predict_one_vs_all(X, all_theta)
correctness = sum(predict == y) / len(predict) * 100
print(correctness)
# print(predict.shape)
