# DSLES-SQL


## Dataset
To reproduce the results reported in the paper, we provide the Spider dataset of the official dataset and the processed .json files used in methods.

## Models
We also offer the original models which are mentioned in the paper and the sql_schema.json file which is used for fine-tuning. To reproduce the results, you should load fine tuned model.

## Setup

To run this project, use the following commands:

```
$ echo "Start installing pip_requirements.txt"
$ pip3 install -r pip_requirements.txt
$ echo "Start installing conda_requirements.txt"
$ conda install --file conda_requirements.txt
$ python3 DSLES-SQL.py --dataset ./data/spider/ --output predicted_sql.txt
$ echo "Finished running DSLES-SQL.py"
```
## Citation 

``` 
corresponding author: Kangping Wang
Key Laboratory of Symbolic Computation and Knowledge Engineering of the Ministry of Education
College of Computer Science and Technology
Jilin University
Changchun, China
wangkp@jlu.edu.cn

In submission
 
```

