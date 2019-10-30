# Marius Orehovschi
# S19
# Project 2: Data Management
# CS 251

# Modified as part of CS 251 Project 7 (Spring 2019)
import math
import random
import numpy as np
import scipy.stats
import data
import pcadata
import scipy.cluster.vq as vq

from data import *
from decimal import Decimal


def data_range(data, headerList):
    # returns data range in parameter numpy matrix
    mat = data.get_columns(headerList)

    # initially assign both minimum and maximum to first element in matrix
    minVal = mat[0,0]
    maxVal = mat[0,0]

    # iterate over each matrix element and assign accordingly
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j] < minVal:
                minVal = mat[i,j]

            if mat[i,j] > maxVal:
                maxVal = mat[i,j]

    return tuple((minVal, maxVal))


def mean(data, headerList):
    # returns a list of means of each column corresponding to headers in parameter header list
    # (in parameter data)

    means = []

    # iterate over each header
    for header in headerList:
        mean = np.mean(data.get_column(header))
        means.append(mean)  # append to mean list the mean of every column

    return means


def stdev(data, headerList):
    # returns a list of standard deviations of each column corresponding to headers in parameter header list
    # (in parameter data)

    stdevs = []

    # iterate over each header
    for header in headerList:
        stdev = np.std(data.get_column(header))
        stdevs.append(stdev)  # append to stdevs list the standard deviation of every column

    return stdevs


def normalize_columns_separately(data, headerList=[None]):
    # returns a normalized matrix created from columns corresponding to each header in
    # parameter headerList where each column is normalize separately

    # if header list unspecified, use all
    if headerList[0] is None:
        headerList = data.get_headers()

    # create an empty container matrix that has the shape of the transpose of our final matrix
    container = np.zeros(shape=(len(headerList), data.get_num_points()))

    for i in range(len(headerList)):
        mat = data.get_column(headerList[i])

        # use data_range() to find min and max
        min = data_range(data, headerList[i:(i+1)])[0]
        max = data_range(data, headerList[i:(i+1)])[1]

        # transpose by -min and divide by extent
        mat = mat - min
        if (max - min) != 0:
            mat = mat / (max - min)

        # transpose the column matrix before adding to container
        container[i] = mat.T

    # return transposed container
    return container.T


def normalize_columns_together(data, headerList):
    # returns a normalized matrix created from columns corresponding to each header in
    # parameter headerList

    mat = data.get_columns(headerList)

    # use data_range() function to compute min and max
    min = data_range(data, headerList)[0]
    max = data_range(data, headerList)[1]

    # transpose by negative min and divide by extent
    mat = mat - min
    mat = mat / (max - min)

    return mat


def single_linear_regression(data_obj,ind_var, dep_var):
    # takes in a data object and the names of the columns corresponding to
    # independent and dependent variables and performs single linear regression;
    # returns a tuple with: slope, y-intercept, correlation coefficient, p-value,
    # standard error of estimated gradient, a tuple with the minimum and maximum
    # in the independent variable, and a tuple with the minimum and maximum in the
    # dependent variable

    variables = data_obj.get_columns([ind_var, dep_var])

    reg = scipy.stats.linregress(variables)

    output = (reg[0],  # slope
              reg[1],  # intercept
              reg[2],  # rvalue
              reg[3],  # pvalue
              reg[4],  # standard error
              data_range(data_obj, [ind_var]),  # tuple of min & max of independent column
              data_range(data_obj, [dep_var]))  # tuple of min & max of dependent column

    print('single linear regression results:')
    print(f'independent: {ind_var}, dependent: {dep_var}')
    print(f"intercept: {reg[1]} \nslope: {reg[0]} \nr: {reg[2]} \np-value: {reg[3]} \nstandard error: {reg[4]}")

    return output


def linear_regression(d, ind, dep, verbose=False):
    """computes linear regression using dataset d for independent variables ind
    (as list) and dependent variable dep"""

    d: data.Data

    # assign to y the column of data for the dependent variable
    y = d.get_column(dep)
    # assign to A the columns of data for the independent variables
    A = d.get_columns(ind)
    # add column of 1's at beginning
    A = np.hstack((np.zeros(shape=(d.get_num_points(), 1))+1, A))

    # assign to AAinv the inverse of the result of multiplying A by its transpose
    AAinv = np.linalg.inv(np.dot(A.T, A))

    # solve for y = Ab, where A is a matrix of the independent data, b is the set of
    # unknowns as a column vector, and y is the dependent column of data
    x = np.linalg.lstsq(A, y, rcond=None)

    # assign to b the first element of solution to the best fit regression
    b = x[0]

    # assign to N the number of data points (rows in y)
    N = d.get_num_points()

    # assign to C the number of coefficients (rows in b)
    C = np.shape(b)[0]

    # number of degrees of freedom of the error
    df_e = N-C

    # number of degrees of freedom of the model fit
    df_r = C-1

    # difference between predicted and actual value
    error = y - np.dot(A, b)

    # sum of the squares of the errors computed in the prior step, divided by the
    # number of degrees of freedom of the error.
    sse = np.dot(error.T, error) / df_e

    # the square root of the diagonals of the sum-squared error multiplied by the
    # inverse covariance matrix of the data.
    stderr = np.sqrt( np.diagonal( sse[0, 0] * AAinv ) )

    # t-statistic
    t = b.T / stderr

    #  probability of the coefficient indicating a random relationship (slope = 0)
    p = 2*(1 - scipy.stats.t.cdf(abs(t), df_e))

    # coefficient indicating the quality of the fit.
    r2 = 1 - error.var() / y.var()

    if verbose:
        print('linear regression results: ')
        print('intercept: ', b[0])
        print('slopes: ', b[1:])
        print('standard error: ', stderr)
        print('R^2: ', r2)
        print('t: ', t)
        print('p-value: ', p)

    # return regression values
    return b, sse, r2, t, p

def test_linear_regression(filename, ind, dep):
    """tests linear regression on parameter filename;"""
    """takes in a list of independent variable names and a dependent variable;"""
    """prints regression results (designed for multiple regression with two independent variables)"""

    datafile = data.Data(filename)

    regress = linear_regression(datafile, ind, dep)

    print("m0 = {:.3f}, m1 = {:.3f}, b = {:.3f}, sse = {:.3f},".format(
                                                regress[0][1, 0], regress[0][2,0], regress[0][0,0], regress[1][0,0]))

    print("R2 = {:.3f}".format(regress[2]),
          "t = [{:.3f}, {:.3f}, {:.3f}], ".format(regress[3][0, 1], regress[3][0, 2], regress[3][0, 0]),
          "p = [{:.3f}, {:.3f}, {:.3f}] ".format(regress[4][0, 1], regress[4][0, 2], regress[4][0, 0]))


# This version uses SVD
def pca(d, headers, normalize=True):
    """
    performs a principal component analysis on columns corresponding to parameter headers
    in parameter d (data object);
    optional argument normalize determines if columns are normalized separately
    """

    # cannot do PCA with just one variable
    if len(headers) == 1:
        return

    if normalize:
        A = normalize_columns_separately(d, headers)
    else:
        A = d.get_data(headers)

    # center the data
    m = np.mean(A, axis=0)
    D = A - m

    # perform SVD
    U, S, V = np.linalg.svd(D, full_matrices=False)

    # get eigenvalues from S
    evalues = (S * S) / (d.get_num_points() - 1)

    evecs = V

    # project the data onto the eigenvectors
    projected_data = np.dot(V, D.T).T

    # create and return a PCA data object
    return pcadata.PCAData(projected_data, evecs, evalues, m, headers)


def kmeans_numpy(d, headers, K, whiten=True):
    """
    Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes, and representation error.
    """

    A = d.get_numeric_matrix()
    W = vq.whiten(A)

    codebook, bookerror = vq.kmeans(W, K)
    codes, error = vq.vq(W, codebook)

    return codebook, codes, error


def kmeans_init(A, K):
    """
    Selects K random rows from the data matrix A and returns them as a matrix
    """

    indices = set()

    # add random integers until set is of desired size
    while len(indices) < K:
        indices.add(random.randint(0, A.shape[0]-1))

    returnable = np.zeros(shape=(K, A.shape[1]))

    # add to returnable K random unique rows of A
    for i in range(K):
        returnable[i] = A[indices.pop()]

    return returnable

def kmeans_classify(A, codebook, p=2):
    """
    Given a data matrix A and a set of means in the codebook
    Returns a matrix of the id of the closest mean to each point
    Returns a matrix of the sum-squared distance between the closest mean and each point
    Optional parameter p: p in L^p norm
    """
    closestIDs = np.zeros(shape=(A.shape[0], 1), dtype=int)
    closestDists = np.zeros(shape=(A.shape[0], 1))

    for i in range(A.shape[0]):
        diffs = codebook - A[i, :]  # calculate the difference from point to each cluster mean
        diffs = np.power(diffs, p)  # square differences

        # turn differences into L^p distances (calculated across columns)
        dists = np.sum(np.absolute(diffs), axis=1)
        dists = np.power(dists, 1.0/p)

        # store the ID of the closest cluster mean and the distance from it
        closestIDs[i, 0] = np.argmin(dists, axis=0)
        closestDists[i, 0] = dists[closestIDs[i, 0]]

    return closestIDs, closestDists

def kmeans_algorithm(A, means, min_change=1e-7, max_iterations=100, p=2):
    """
    Given a data matrix A and a set of K initial means, compute the optimal
    cluster means for the data and an ID and an error for each data point;
    optional parameter p: p in L^p norm
    """
    # set up some useful constants
    D = means.shape[1]    # number of dimensions
    K = means.shape[0]    # number of clusters
    N = A.shape[0]        # number of data points

    # iterate no more than max_iterations
    for i in range(max_iterations):

        # calculate the codes by calling kemans_classify
        codes = kmeans_classify(A, means, p)
        # codes[j,0] is the id of the closest mean to point j

        # initialize newmeans to a zero matrix identical in size to means and cast to matrix
        newmeans = np.zeros(shape=means.shape)
        newmeans = np.asmatrix(newmeans)

        # stores how many points get assigned to each mean
        counts = np.zeros(shape=(K, 1))

        # for the number of data points
        for j in range(A.shape[0]):
            # add to the closest mean (row codes[j,0] of newmeans) the jth row of A
            newmeans[codes[0][j, 0]] += A[j]
            # add one to the corresponding count for the closest mean
            counts[codes[0][j, 0]] += 1

        # finish calculating the means, taking into account possible zero counts
        for i in range(K):
        # for the number of clusters K
            # if counts is not zero, divide the mean by its count
            if counts[i, 0] != 0:
                newmeans[i] = newmeans[i] / counts[i, 0]
            # else pick a random data point to be the new cluster mean
            else:
                newmeans[i] = A[random.randint(0, A.shape[0])]

        # test if the change is small enough and exit if it is
        diff = np.sum(np.square(means - newmeans))
        means = newmeans
        if diff < min_change:
            break

    # call kmeans_classify one more time with the final means
    codes, errors = kmeans_classify(A, means, p)

    # return the means, codes, and errors
    return means, codes, errors


def kmeans(d, headers, K, whiten=True):
    '''Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes and representation errors.
    '''

    # assign to A the result getting the data given the headers
    A = d.get_data(headers)
    # if whiten is True
    if whiten:
        # store column means and standard deviations
        m = np.mean(A, axis=0)
        stdevs = np.std(A, axis=0)

        # center the data
        A = A - m

        W = vq.whiten(A)
    # else
    else:
        # assign to W the matrix A
        W = A

    # assign to codebook the result of calling kmeans_init with W and K
    codebook = kmeans_init(W, K)

    # assign to codebook, codes, errors, the result of calling kmeans_algorithm with W and codebook
    codebook, codes, errors = kmeans_algorithm(W,codebook)

    # if data was whitened before performing kmeans
    if whiten:

        # multiply each column by the original column standard deviation and add original column mean
        for i in range(codebook.shape[1]):
            codebook[:, i] = codebook[:, i] * stdevs[0, i] + m[0, i]

    # return the codebook, codes, and representation error
    return codebook, codes, errors

def kmeans_new(A, K, whiten=True, p=2):
    """
    performs K means clustering on A
    :param A: data matrix with data points as rows and features as columns
    :param K: number of cluster centers
    :param whiten: (optional) if True, centers data and divides columns by standard deviation before analysis
    :param p: (optional) p in L^p norm to be used – default is Euclidean
    :return: codebook of cluster centers, codes mapping each point to nearest cluster center, representation errors
    """

    if whiten:
        # store column means and standard deviations
        m = np.mean(A, axis=0)
        stdevs = np.std(A, axis=0)

        # center the data
        A = A - m
        W = vq.whiten(A)
    else:
        # assign to W the matrix A
        W = A

    # assign to codebook the result of calling kmeans_init with W and K
    codebook = kmeans_init(W, K)

    # assign to codebook, codes, errors, the result of calling kmeans_algorithm with W and codebook
    codebook, codes, errors = kmeans_algorithm(W, codebook, p)

    # if data was whitened before performing kmeans
    if whiten:
        for i in range(A.shape[1]):
            # if any column has standard deviation 0
            if np.count_nonzero(stdevs[:, i]) == 0:
                # make it 1 so it does not influence the next calculation
                stdevs[:, i] += 1

        # multiply each column by the original column standard deviation and add original column mean
        for i in range(codebook.shape[1]):
            codebook[:, i] = codebook[:, i] * stdevs[0, i] + m[0, i]

    # return the codebook, codes, and representation error
    return codebook, codes, errors

def kmeans_quality(errors, K):
    """
    computes the decription length (a measure of quality of a clustering algorithm
    :param errors: numpy matrix of errors produced by clustering algorithm
    :param K: number of clusters
    :return: float description length
    """

    squared = np.square(errors)
    sum = np.sum(squared)

    return sum + 0.5*K*np.log2(errors.shape[0])

def getCumulativeValueList(values):
    """
    computes a numpy matrix of cumulative values
    :param evalues: numpy matrix of values
    :return: numpy matrix of cumulative values
    """

    if (type(values) is not np.matrix) and (type(values) is not np.ndarray):
        raise ValueError("Parameter values must be numpy matrix or numpy ndarray")

    # make an empty matrix
    cumulativeValues = np.zeros(shape=values.shape)

    # for each column, add its own value and of the cells before it
    for i in range(values.shape[0]):
        if i == 0:
            cumulativeValues[i] = values[i]
            continue

        cumulativeValues[i] = values[i] + cumulativeValues[i - 1]

    # divide by the sum (the last entry)
    cumulativeValues = cumulativeValues / cumulativeValues[-1]

    return cumulativeValues

def digit_precision(input):
    """
    finds out the precision of a floating point number;
    if input is 0, precision is considered 0.1
    :param input: (float) number to find the precision of
    :return: (float) smallest increment
    """

    if input % 10 == 0:
        exponentof10 = 0

        # divide by 10 until it is not divisible by 10
        while input % 10 == 0:
            input /= 10
            exponentof10 += 1

        return math.pow(10, exponentof10)

    # if input not divisible by 10
    instring = str(input)

    # count how many digits after dor
    idx = instring.find('.')
    after_dot = len(instring) - idx - 1
    print(after_dot)

    # precision equals 10 raised to the power of negative how many digits after dot
    return math.pow(10, -after_dot)


def avg_precision(cmtx):
    """
    calculates precision of a confusion matrix
    :param cmtx: confusion matrix
    :return: (float) average precision
    """

    # used to store precision of each class
    container = np.zeros(shape=(1, cmtx.shape[1]))

    # calculate true positives divided by all positives
    for j in range(cmtx.shape[1]):
        if np.sum(cmtx[:, j]) == 0:
            container[0, j] = 0
            continue

        container[0, j] = cmtx[j, j] / np.sum(cmtx[:, j])

    # return the average class precision
    return np.mean(container)


def avg_recall(cmtx):
    """
    calculates recall of a confusion matrix
    :param cmtx: confusion matrix
    :return: (float) average recall
    """

    # used to store precision of each class
    container = np.zeros(shape=(cmtx.shape[0], 1))

    # calculate true positives divided by all positives
    for i in range(cmtx.shape[0]):
        if np.sum(cmtx[i, :]) == 0:
            container[i, 0] = 0
            continue

        container[i, 0] = cmtx[i, i] / np.sum(cmtx[i, :])

    # return the average class precision
    return np.mean(container)

def f1_score(cmtx):
    """
    calculates f1 score of a confusion matrix using average class recall and average class precision
    :param cmtx: confusion matrix
    :return: (float) f1 score
    """

    precision = avg_precision(cmtx)
    recall = avg_recall(cmtx)

    if precision == 0 and recall == 0:
        return 0.0

    return 2 * precision * recall / (precision+recall)


def main():
    # # test for data_range()
    # dataset = Data('testdata3.csv')
    # mat = dataset.get_numeric_matrix()
    # print('range: ', data_range(dataset, ['headers', 'spaces']))  # should print 'range: (1, 10)'
    #
    # # test for mean()
    # print('means: ', mean(dataset, ['headers', 'spaces']))  # should print 'means: [5.0, 6.0]'
    #
    # # test for stdev()
    # # should print out 'st devs:  [3.265986323710904, 3.265986323710904]'
    # print('st devs: ', stdev(dataset, ['headers', 'spaces']))
    #
    # print('normalized matrix: \n', normalize_columns_together(dataset, ['headers', 'spaces']))
    #
    # print('normalize matrix (by column): \n', normalize_columns_separately(dataset, ['headers', 'spaces']))
    #
    # print("\nRegressing\n")
    # single_linear_regression(dataset,'headers', 'spaces')
    #
    # print('\nTesting clean')
    # test_linear_regression('data-clean.csv', ['X0','X1'], 'Y')
    #
    # print('\nTesting good')
    # test_linear_regression('data-good.csv', ['X0', 'X1'], 'Y')
    #
    # print('\nTesting noisy')
    # test_linear_regression('data-noisy.csv', ['X0', 'X1'], 'Y')
    #
    # print('\nTesting bodyfat')
    # bodydata = data.Data('body_fat_addedtype.csv')
    # linear_regression(bodydata, ['weight'], 'bodyfat', verbose=True)
    #
    # print('\nTesting bodyfat, multiple variables')
    # bodydata = data.Data('body_fat_addedtype.csv')
    # linear_regression(bodydata, ['weight', 'abdomen'], 'bodyfat', verbose=True)
    #
    # print('\nPCA-ing')
    # pcaable = Data('testdata3.csv')
    # pca(pcaable, ['headers', 'spaces', 'bad'], False)

    print('\nK Means quality test')
    print(kmeans_quality(np.matrix([[2], [3], [4]]), 3))


if __name__ == "__main__":
    main()