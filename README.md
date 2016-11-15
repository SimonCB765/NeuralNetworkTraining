# Table of Contents

1. [Overview](#overview)
2. [Data Processing](#data-processing)
    1. [Usage](#data-proc-usage)
    2. [Configuration File Formats](#data-proc-configuration-file-formats)

<a name="overview"></a>
# Overview

General description of the features, packages and use case scenarios.

<a name="data-processing"></a>
# Data Processing

<a name="data-proc-usage"></a>
## Usage

<a name="data-proc-configuration-file-formats"></a>
## Configuration File Formats

should be a dictionary for target data and one for training data (target data in separate file), for the sequences the output and input are assumed to be in the same order (i.e. input in row 1 has its output as row 1 in target file)

needs things like number of examples per file from the medical record analysis repository

- ColumnsToIgnore - list of the columns to not include in the data (i.e. stick the ID column in this to not keep it as data), strings will be interpreted as column names and integers as column indices (0 based), names not matchig column headings and indices too large are just ignored (with a warning output)
- HeaderPresent - true or false, header is taken to be first line in file with one column name per data column
- IDColumn - number for a column index (0 based) or string for column name (must have a header present), assumed to be no ID column, strings not matching a column heading and indices to large cause the program to abort and output a warning

### Vector Example

### Sequence Example
