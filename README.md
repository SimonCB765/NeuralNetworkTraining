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

See the following links for information on JSON schema:

- http://json-schema.org/
- https://spacetelescope.github.io/understanding-json-schema/
- https://spacetelescope.github.io/understanding-json-schema/UnderstandingJSONSchema.pdf

Required items in the schema will be specified as required and have no default defined
Items that need a value but can have sensible defaults defined have defaults defined. These are:

- whether a header is present
- the separator used in the dataset
- the train, test and validation fractions

example, target, output and overwrite are not in configuration file but instead are command line parameters

for the sequences the output and input are assumed to be in the same order (i.e. input in row 1 has its output as row 1 in target file)

Anything in the configuration file will be carried out, so if you don't want to prepare the data, but do want to train, then don't have any data preparation parameters.

Numeric indexing is 0-based

the example ID 'variable' is always ignored in the final data representation

Sharding is randomised in the sense that it is not split perfectly. So if you say 70/15/15 then each example is given a 0.7 chance
of going into train, 0.15 into test and 0.15 into validation

if you do supply the fractions (rather than relying on the defaults) then there must be at least a training fraction provided
also if you only supply a subset (like only supply the train and test) then no examples will go to validation in this case

Training fraction takes precendence, then test then validation. So if fractions are:

- train - T_f
- test - E_f
- validation - V_f

and dataset contains N examples, then the dataset sizes will be:

- train = T_n = T * N
- test = E_n = min((N - T_n) * E_f, N * E_f)
- validation = V_n = min((N - T_n - E_n) * V_f, N * V_f)

There can be examples that are left out of all three datasets, e.g. if T_f = 0.5, E_f = 0.2 and V_f = 0.2.


For default extraction references are expected to be in the form of : "ReffedElement": {"$ref": "#/definitions/VariableIndexer"}
Putting a reference directly into an object is possible, e.g.:
    "Schema": {
        "description": "Variables to convert to a one-of-(C-1) encoding, where C is the number of categories.",
        "type": "object",
        "$ref": "#/definitions/VariableIndexer"
    }
rather than:
    "Schema": {
        "description": "Variables to convert to a one-of-(C-1) encoding, where C is the number of categories.",
        "type": "object",
        "ReffedElement": {"$ref": "#/definitions/VariableIndexer"}
    }
This is not really recommended as you will potentially cause key clashes that are resolved by overwriting key-value pairs in the
original element with the referenced key-value pair

Regular expression variales names are intended to be matched from the beginning of the variable's name.

The min max method implemented scales to be between -1 and 1

There should be no overlap between the variables specified as needing normalising or the ones needing normalising and the ones to ignore

- Normalising + how to do it when you've got a dataset too large to fit in memory
    - http://cs231n.github.io/neural-networks-2/
	- https://visualstudiomagazine.com/articles/2014/01/01/how-to-standardize-data-for-neural-networks.aspx
	- LeCun efficient backprop (http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
	- http://www.faqs.org/faqs/ai-faq/neural-nets/part2/
	- Categorical variables - http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-7.html
	- Basically, standardise to avoid saturation with big values and also to get the points clustered around the origin as if you initialise biases (and weights) with small random values all possible hyperplanes are likely to be very close to the origin, so you don't miss the data cloud if you standardise
	    - Similar reason for making binary data -1/+1 not 0/+1
	- Numeric independent (done only on training portion and then applied to test and validation)
		- min/max scaling
		- normalising
		- decorrelating
		- whitening
    - Categorical independent and dependent (done on entire dataset)
        - one hot coding with -1/+1 for each category
        - one hot coding with -1/+1 for each category -1 (so N-1 variable for N categories)
    
		

DataFormat - stitch rows of the input file together for sequences, treat rows individually for Vector data

- ColumnsToIgnore - list of the columns to not include in the data (i.e. stick the ID column in this to not keep it as data), strings will be interpreted as column names and integers as column indices (0 based), names not matchig column headings and indices too large are just ignored (with a warning output)
- HeaderPresent - true or false, header is taken to be first line in file with one column name per data column
- IDColumn - number for a column index (0 based) or string for column name (must have a header present), assumed to be no ID column, strings not matching a column heading and indices to large cause the program to abort and output a warning

### Vector Example

### Sequence Example
