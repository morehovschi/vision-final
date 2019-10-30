 """
    Marius Orehovschi
    S19
    Project 2: Data Management
    CS 251
"""

# Modified as part of CS 251 Project 4 (Spring 2019)
import random

import numpy as np
import csv
import warnings
import sys

# ignore warnings caused by 'rU' flag in open()
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Data:

    def __init__(self, filename=None):

        # dictionary mapping a header to a hash code whose sign represents
        # whether variable is numeric (+ for numeric and - for non-numeric)
        # and absolute value -- variable's column index in corresponding matrix
        self.headers = {}

        self.types = []

        # list of numeric data that will be converted to numeric matrix
        self.numericData = []

        # list of non-numeric data to be converted to non-numeric matrix
        self.nonNumericData = []

        if filename:
            self.read(filename)

    def read(self, filename=None):
        # reads CSV files and separates numeric data from non-numeric

        filePath = open(filename, 'rU')

        if filePath is None:
            raise ValueError()

        reader = csv.reader(filePath)

        # helper that initially stores all headers
        headers = []

        # helper to keep track of the first 2 lines
        lineCounter = 1

        # temporary list of helper values where sign corresponds to
        # numeric or non-numeric type (+ for numeric, - for non-numeric)
        # and absolute value corresponds to column index in appropriate matrix
        # (numeric or non-numeric matrix)
        separator = []

        for line in reader:

            # if first line
            if lineCounter == 1:

                # line elements are simply names of headers
                headers = line
                lineCounter += 1
                continue

            # if second line
            if lineCounter == 2:

                # save line elements as the data types
                self.types = line

                # helpers that count column indices for each of the 2 matrices
                # (numeric and non-numeric)
                numericCounter = 1
                nonNumericCounter = -1

                for i in range( len(line) ):

                    # if line[i] is the word numeric, give hash code a positive value;
                    if line[i].lower().strip() == "numeric":
                        # clean string inside square brackets
                        self.headers[ headers[i].lower().strip() ] = numericCounter
                        numericCounter += 1

                    # if line[i] is the word non-numeric, give hash code a negative value
                    else:
                        # clean string inside square brackets
                        self.headers[ headers[i].lower().strip() ] = nonNumericCounter
                        nonNumericCounter -= 1

                # store dictionary values as list (so we don't have to do the list operation
                # every time we need to use the values)
                separator = list(self.headers.values())

                # indicate that we're done the second line
                lineCounter += 1
                continue

            # the self.headers dictionary now contains the name of each variable,
            # its type, and its corresponding column index in the appropriate matrix

            # helpers that separate data into numeric and non-numeric
            numericLine = []
            nonNumericLine = []

            # iterate over each element in line
            for i in range(len(line)):

                # if element's index corresponds to positive value in separator list
                if separator[i] > 0:
                    # it is numeric; add it to numeric value list in line
                    numericLine.append(line[i])

                else:
                    # it is non-numeric; add it to non-numeric value list in line
                    nonNumericLine.append(line[i])

            # add line lists to greater data object lists
            self.numericData.append(numericLine)
            self.nonNumericData.append(nonNumericLine)

        # convert numeric data list to numpy matrix
        self.numericData = np.matrix(self.numericData, dtype=float)

        # convert non-numeric data list to numpy matrix
        self.nonNumericData = np.matrix(self.nonNumericData, dtype=object)

    # start accessors
    def get_headers(self):
        return list(self.headers.keys())

    def get_types(self):
        return self.types

    def get_num_dimensions(self):
        # returns sum of numbers of columns in numeric and non-numeric matrix
        num_dimensions = 0

        if type(self.numericData) != list:
            num_dimensions += np.shape(self.numericData)[1]

        if type(self.nonNumericData) != list:
            num_dimensions += np.shape(self.nonNumericData)[1]

        return num_dimensions

    def get_num_points(self):
        # returns number of rows in numeric matrix (same for non-numeric matrix)
        return self.numericData.shape[0]

    def get_row(self, rowIndex, numeric=None):
        # by default: returns parameter row in data set
        # boolean parameter numeric determines whether row in numeric
        # or non-numeric matrix will be returned

        if numeric is None:
            returnable = []

            # iterate over list of all headers, numeric and non-numeric
            for header in list(self.headers.keys()):
                # add value corresponding to current header and parameter rowIndex
                returnable.append(self.get_value(header, rowIndex))

            return returnable

        # if boolean numeric specified, return row from corresponding matrix
        elif numeric:
            return self.numericData[rowIndex, :]
        else:
            return self.nonNumericData[rowIndex, :]

    def get_value(self, header, rowIndex):
        # returns value with parameter rowIdx in parameter header column

        # if header does not exist in list, return
        if self.headers.get(header) is None:
            return

        # if the corresponding dictionary index is >0, value is numeric
        if self.headers.get(header) > 0:
            # column idx is 1 less than the corresponding dictionary value of
            # parameter header (because dictionary indices start at 1)
            colIdx = self.headers.get(header) - 1
            return self.numericData[rowIndex, colIdx]

        # if dictionary value <0, value is non-numeric
        else:
            # column index's absolute value is 1 less than dictionary index
            colIdx = self.headers.get(header) + 1
            return self.nonNumericData[rowIndex, colIdx]

    def get_column(self, header):
        # returns the column corresponding to parameter header as np matrix;
        # if header not in self.headers, returns a column of zeros
        # NOTE: returning non-numeric data not supported yet

        # if header not in header list
        if header.lower().strip() not in self.headers:
            # return a column of zeros of appropriate height
            returnable = np.zeros(shape=(self.get_num_points(), 1))
            return returnable

        index = self.headers.get(header.lower().strip())

        '''
        hash code returned by dictionary is 1 greater (in absolute value) than the corresponding 
        index in numeric matrix; positive index corresponds to numeric value, negative to non numeric
        '''
        if index>0:
            index -= 1
            numeric = True
        else:
            index = -(index + 1)
            numeric = False

        if not numeric:
            return self.nonNumericData[:, index]

        return self.numericData[:, index]

    def get_columns(self, headers, rowIdxList=None):
        # returns a numpy matrix with the columns corresponding to parameter headers
        # optional parameter rowIdxList specifies which rows to be returned

        # assign column corresponding to first parameter header
        mat = self.get_column(headers[0].lower().strip())  # clean string with lower() and strip()

        # add a column corresponding to a new header each iteration
        for i in range(1, len(headers)):
            mat = np.hstack((mat, self.get_column(headers[i].lower().strip())))  # clean string with lower() and strip()

        # if user provided list with row indices
        if rowIdxList is not None:
            returnable = mat[rowIdxList[0], :]

            # stack rows with parameter indices from mat
            for i in range(1, len(rowIdxList)):
                returnable = np.vstack((returnable, mat[rowIdxList[i], :]))

            # return matrix that contains only the desired rows
            return returnable

        # if user did not provide idx list, return the entire matrix
        else:
            return mat

    def get_data(self, headers=None, rowIdxList=None):
        """
        same as get_columns; implemented for compatibility reasons;
        """

        # if headers not specified, return all columns
        if headers is None:
            return self.get_columns(self.get_headers(), rowIdxList)

        return self.get_columns(headers, rowIdxList)

    def get_numeric_matrix(self):
        return self.numericData

    def get_non_numeric_matrix(self):
        return self.nonNumericData
    # end accessors

    def addColumn(self, header, type, values):
        # adds a column to the data set
        # NOTE: only works on numeric data
        if type.lower().strip() == "numeric":
            self.headers[ header.lower().strip() ] = self.numericData.shape[1] + 1
            self.types.append(type.lower().strip())

            self.numericData = np.hstack((self.numericData, values))

    def write(self, filename, headers=None):
        """
        writes columns specified by parameter headers to parameter filename
        :param headers: headers of columns to be written; writes all columns when None
        :return:
        """

        # if headers unspecified, use all
        if headers is None:
            headers = list(self.headers.keys())
            types = self.types
            datamat = self.numericData

        else:
            # only add the types of the headers in parameter headers
            indices = []

            for i in range(len(headers)):
                indices.append(self.headers[headers[i].lower().strip()] - 1)

                # correct for non-numeric indices (negative indices)
                for j in range(len(indices)):
                    if indices[j] < 0:
                        indices[j] = abs(indices[j]+2)

            types = []

            for i in range(len(headers)):
                types.append(self.types[indices[i]])

            # store data as matrix
            datamat = self.get_columns(headers)

        f = open(filename, 'w')

        # write headers as first line
        for i in range(len(headers)-1):
            f.write(headers[i]+',')
        f.write(headers[len(headers)-1]+'\n')  # last one gets a '\n' instead of a comma

        # write types as second line
        for i in range(len(headers)-1):
            f.write(types[i] + ',')
        f.write(types[len(headers) - 1] + '\n')  # last one gets a '\n' instead of a comma

        # write the rest of the data
        for i in range(datamat.shape[0]):
            for j in range(datamat.shape[1]-1):
                f.write(str(datamat[i, j]) + ',')

            f.write(str(datamat[i, datamat.shape[1]-1]) + '\n')

        f.close()

    def __str__(self):
        # returns string representation of the whole data set

        returnable = ''

        # get list of headers
        headers = list(self.headers.keys())

        # append each header and comma
        for i in range(self.get_num_dimensions()):
            returnable += headers[i] + ', '
        # remove last comma and whitespace and add newline character
        returnable = returnable[:-2] + '\n'

        # append each data type and comma
        for i in range(self.get_num_dimensions()):
            returnable += self.types[i] + ', '
        # remove last comma and whitespace and add newline character
        returnable = returnable[:-2] + '\n'

        # for each observation
        for i in range(self.get_num_points()):
            line = ''

            # add corresponding in row value for each header
            for header in headers:
                line += str(self.get_value(header, i)) + ', '

            # remove last comma and whitespace and add newline character
            line = line[:-2] + '\n'
            returnable += line

        return returnable


def numericString(string):
    """returns true if parameter string can be converted to number"""

    try:
        dummy = float(string)
        return True

    except ValueError:
        return False


def addTypes(csvFile):
    """"adds types on the second line of parameter csv file;
        only adds 'numeric' or 'non numeric'"""

    # get file lines
    with open(csvFile, 'r') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)

    types = []

    # loop through second line of file
    for dataPoint in lines[1]:
        # if able to read as float, variable numeric
        if numericString(dataPoint):
            types.append('numeric')
        else:
            types.append('non numeric')

    # new list where new line files are stored
    newFileLines = [lines[0], types]

    for i in range(1, len(lines)):
        newFileLines.append(lines[i])

    readFile.close()

    with open(csvFile[:-4] + "_addedtype.csv", 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(newFileLines)

    writeFile.close()

def split_train_test(csv_filename):
    """
    splits parameter csv file into a train and a test set using a randomized 70-30 split
    """

    source = Data(csv_filename)

    train_indices = list(range(source.get_num_points()))
    random.shuffle(train_indices)

    test_indices = []

    # while ratio of test to train indices is less than 3:7
    while (len(test_indices)*1.0 / len(train_indices)) < 0.42:
        # pop from train to test
        test_indices.append(train_indices.pop())

    # assign rows of data matrices with train_indices
    num_train_data = source.get_numeric_matrix()[train_indices, :]
    non_train_data = source.get_non_numeric_matrix()[train_indices, :]

    # assign rows of data matrices with test_indices
    num_test_data = source.get_numeric_matrix()[test_indices, :]
    non_test_data = source.get_non_numeric_matrix()[test_indices, :]

    # make train data object
    traindata = Data()
    traindata.headers = source.headers.copy()
    traindata.types = source.types.copy()
    traindata.numericData = num_train_data
    traindata.nonNumericData = non_train_data

    # make test data object
    testdata = Data()
    testdata.headers = source.headers.copy()
    testdata.types = source.types.copy()
    testdata.numericData = num_test_data
    testdata.nonNumericData = non_test_data

    train_filename = csv_filename[:-4] + "_train.csv"
    test_filename = csv_filename[:-4] + "_test.csv"

    # write data objects to files
    traindata.write(train_filename)
    testdata.write(test_filename)

def main(argv):

    # addTypes('absenteeism.csv')

    # dobj = Data('bogus.csv')
    #
    # dobj.write('newbogus.csv', ['a', 'd', 'b'])

    if len(argv) > 1:
        split_train_test(argv[1])

    else:
        addTypes('responses_original_clean.csv')


if __name__ == "__main__":
    main(sys.argv)
