# PMC-StructuredAbstracts-Dataset
This repository contains code to produce the PMC Structured Abstracts dataset described in the paper *[Structured Summarization of Academic Publications](https://arxiv.org/abs/1905.07695)*. The code reads the .nxml files from the PMC-OA collection that can be found [here](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist) processes it and exports them into binary format.

# Instructions
1. Download and extract the nxml format data files from the PMC website.

2. Process the data running the following: 
```
python data_processing.py -i /path/to/the/top/directory/of/the/data/files -o /path/to/the/output/directory -np 1000
```

**Warning:** Processing the whole dataset will take quite a long time (approx. 10 hours on a 16 core machine).

# Output files
After running the processing you will end up with the following files:
* A vocab file with the words in the vocabulary plus some special tokens. *
* A directory with the pmc ids of the files included in the training, validation and test set. *
* Three directories, namely train, val, test with the .bin data files. *

# Reading the .bin files
