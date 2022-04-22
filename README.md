# Lyra
Lyra: A Benchmark for Turducken-Style Code Generation

Paper link: http://arxiv.org/abs/2108.12144

# Dataset

The dataset used in our paper is under `lyra_dataset` folder.

# Leaderboard

## English Comments
| Model | BLEU | Code Executable | AST Matching in Base Language | AST Exact Matching |
| :---: | :---:        | :---:   | :---:        | :---:      |
| Transformer | 47.05 | 23 | 4 | 1.5 | 
| CodeBERT | 56.72 | 51 | 8.5 | 4.5 | 
| GraphCodeBERT | 58.61  | 46 | 12.5 | 6 |
| GPT | 67.29 | 88 | 24.5 | 21.5 | 
| CodeGPT | 65.96 | 93 | 23.5 | 21 | 
| CodeGPT-Adapted | 66.5 |  92 | 29 | 25.5 | 

## Chinese Comments
| Model | BLEU | Code Executable | AST Matching in Base Language | AST Exact Matching |
| :---: | :---:        | :---:   | :---:        | :---:      |
| Transformer | 45.84 | 21.5 | 3 | 0.5 | 
| GPT | 66 | 92 | 22 | 20.5 | 
| CodeGPT | 64.88 | 91 | 26 | 24 | 
| CodeGPT-Adapted |66.37 | 96 | 24.5 | 23 | 

# Traing and Testing of Baseline

**training**

command: `python -m run train --param-path params.json`

- params.json is the file with all hyperparameters

Note: 
In our experiments, we use the wandb to track the model training process. You can remove all operations about wandb in the "run.py" file to make the model code run. Or you can create a "code4sql" project according to the operation tips on the wandb website so that you can also view the running process of the model on the wandb platform.

**testing**

command: `python -m run decode models/train_xxxxxxxx/model`

- train_xxxxxxxx is the folder of trained model.


# Details of Annotation Guidelines

## 1.Guidelines for Code Modification

Most of the function blocks extracted from Github are incomplete or incorrect. There are two purposes for our code modification process. First, make the code snippet independent of the project, and can be run as an independent function block, that is to ensure the correctness of the code snippet. Second, to facilitate the subsequent annotation work, it is required to simplify the modified code snippet and remove the irrelevant statements. The specific guidelines are as follows.

### 1.1 Purpose 1: Correctness
- Remove project-related information and use parameters to express specific variables or classes outside the function block.
- Keep the SQL statements whose operations are only related to SELECT keyword, without INSERT, UPDATE, and CREATE keywords.
- Return the built-in Python data objects (list, dict,...)  after executing the SQL statement, instead of objects of other classes.
- Ensure that there are no undefined variables in the function. Add, delete and modify function parameters are allowed.
- Allow multiple SQL operations within a single function.
- Allows related operations in the flash framework, as well as some simple arithmetic operations.

### 1.2 Purpose 2: Clarity
- Simplify the complex variable names (more than 30 characters). This requires the annotator to simplify according to the functionality of the source code and the meaning of the original function name.
- Remove redundant information that does not affect the functionality of the source code, such as redundant spaces, line breaks and comments.
- Focus on the part of the program where Python code embedded with SQL statements. Try to remove Python code that has nothing to do with SQL operations.

### 1.3 Quality Checking of Code Snippets
We use Pylint (https://www.pylint.org), a Python static code analysis tool, to check the code snippet. 

## 2.Guidelines for Comments Annotation

### 2.1 Principle of Annotation

The problem needs to be solved in comments annotation: assuming that you are not familiar with writing the Python code to manipulate the database, how to give some description to express the code snippet you want to generate? 
Basic requirements of comments: when programmers see comments, they can write code with the same functionality as ground truth code snippet (as part of quality checking)
- Annotation needs to be correct (the programmer can write the code with the same functionality based on the code comment)
- Annotation needs to be diverse (like human natural language)
- Annotation needs to be clarity (no ambiguity confuses programmers)

### 2.2 Content of Annotation

From the perspective of variables in code

- Temporary variables are no need to describe in comments annotation.
- All parameters of the function should be included in the annotation, and each parameter should be marked with a special symbol $.
- Import functions are generally not indicated in annotations. If rare methods or classes are imported, they need to be described directly in the annotation.
- Built-in methods are generally not directly reflected in annotations, but similarly, rare methods need to be described in annotations.

From the perspective of the specific content of the code
- SQL statement description needs to be complete, including table name, query column name, and condition.
- The simple code block (2-3 lines) can be described uniformly.
- The nesting relationship between different program branches needs to be described as clearly as possible.
- Different execution methods of SQL statements need to be distinguished in the annotation.

### 2.3 Quality Checking of Comments
- Check that all parameters are included in the comment annotation
- Sample check whether the code with the same functionality can be written according to the comment annotation.
