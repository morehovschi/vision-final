"""
    Marius Orehovschi
    S19
    Project 2: Data Management
    CS 251
"""

# Modified as part of CS 251 Project 6: Principal Component Analysis (Spring 2019)

import numpy as np
import data
import operator
import analysis


class PCAData(data.Data):

    def __init__(self, projectedData, evecs, evalues, dataMeans, headerList, filename=None):

        # make sure parameter projected data is a matrix
        if (type(projectedData) is not np.matrix) and (type(projectedData) is not np.ndarray):
            raise ValueError("Parameter projected data must be a numpy matrix or a numpy ndarray")

        # make sure parameter projected data is numeric
        if not np.issubdtype(projectedData.dtype, np.number):
            raise ValueError("PCAData object cannot hold non-numeric data")

        if filename:
            super().__init__(filename)
        else:
            super().__init__()

        self.projectedData = projectedData
        self.evecs = evecs
        self.evalues = evalues
        self.dataMeans = dataMeans

        self.org_headers = headerList

        ''' 
        temporary PCA header container;
        helper function just assigns names to PCA components according to the amount of variance
        in the data each dimension contributes 
        '''

        headers = makePCAheaders(evalues)

        # store headers in a permanent map
        for i in range( np.shape(evalues)[0] ):
            self.headers[headers[i]] = i+1  # map indices start at 1 (to accommodate the structure of a data.Data obj)

        # all variables are numeric (exception is thrown before this otherwise)
        for i in range( np.shape(evalues)[0] ):
            self.types.append('numeric')

        self.numericData = np.matrix(projectedData)

    '''start getters'''
    def get_eigenvalues(self):
        return self.evalues

    def get_eigenvectors(self):
        return self.evecs

    def get_original_means(self):
        return self.dataMeans

    def get_original_headers(self):
        return self.org_headers
    '''end getters'''

    def write_cumulative_evalues(self, filename):
        """
        write to a new file each pca header and the amount of variance its column and the previous ones
        contribute to
        """
        f = open(filename, 'w')

        lines = []

        cumulative_evalues = analysis.getCumulativeValueList(self.evalues)

        for i in range(len(self.evalues)):
            f.write("%d. evalue: %d, cumulative: %.3f\n" % (i, self.evalues[i], cumulative_evalues[i]))

        f.close()

def makePCAheaders(evalues):
    """
    make a list of headers for the eigenvalues where the positions are the same as
    in parameter evalues and the name represents the rank in terms of data variance
    in the particular component (e.g. PCA00 is the component that accounts for the
    highest amount of the variance and so on)
    """

    # make an empty list of tuples of the same size as parameter evalues
    evalueRanks = [None] * np.shape(evalues)[0]

    # initialize each list element with position and eigenvalue
    for i in range(np.shape(evalues)[0]):
        evalueRanks[i] = (i, evalues[i])

    # sort by value
    sortedEvalueRanks = sorted(evalueRanks, key=lambda tup: (tup[1]), reverse=True)

    headers = [""] * np.shape(evalues)[0]

    # make a list of header names where the order of evalues is preserved
    # but the index signifies percentage of variance of corresponding evalue
    for i in range(len(sortedEvalueRanks)):
        num = sortedEvalueRanks[i][0]

        headers[i] = "pca" + str(int(num / 10)) + str(num % 10)

    return headers


def main():

    pca = PCAData(np.matrix([[0]]), np.matrix([[0]]), np.matrix([[0]]), np.matrix([[0]]), 'A')

    print(pca.get_data(pca.get_headers()))

    print(pca.get_types())


if __name__ == "__main__":
    main()
