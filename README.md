# Submission Instructions

Students should submit their assignments only using the files provided. While most problems need students to write the library functions in Python script, the final problem on MNIST classifier performance evaluation should be written in the Jupyter notebook provided, which is labeled as "HW1_5.ipynb". Students are not allowed to use any deep learning libraries (e.g. PyTorch, Tensorflow, Keras, CAFFE, Jax and other similar packages). The problem statement can be found in this [link](https://sid-nadendla.github.io/teaching/SP2022_MLCV/HWs/HW-1.pdf).

# Instructions for Cloning hw1 folder

Students will be given access to the Git repository as 'developers'. As a result, they can clone the master branch and submit their respective assignments by following the procedure given below:

## Execute once, to clone a repository:
```
$ git clone https://git-classes.mst.edu/2022-SP-CS6406-101-102/<repository_name>.git
```

## Execute as many times as you like from within the directory/repository you cloned to your hard drive (just an example):
```
# To check the status of your repository:
$ git status

# To stage/add a file:
$ git add *.py *.pdf *.md

# To add a folder:
$ git add SUBDIRECTORY/*

# To commit changes and document them:
$ git commit -m "Informative description of the commit"

# To submit your assignments:
$ git push
```


## Do not add:
Compiled or generated files like *.out, *.log, *.syntex.gz, *.bib, your executable files, etc. Put the name of these files in a text file named .gitignore

If you see your changes reflected on the git-classes site, you have submitted successfully.

## Useful links:
[Git Cheatsheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf)

[Videos on Git basics](https://git-scm.com/videos)

## Implementation
Each models NN1 and NN2 contains layers and W. You can store weights in layers.W or W.

## Run test cases

# To run the test cases for activations.py file, run the following command and all tests should pass
$python test_activations.py

```
                TEST_SIGMOID_1 : True
                TEST_SIGMOID_2 : True
           TEST_SIGMOID_GRAD_1 : True
           TEST_SIGMOID_GRAD_2 : True
                TEST_SOFTMAX_1 : True
           TEST_SOFTMAX_GRAD_1 : True
                   TEST_TANH_1 : True
              TEST_TANH_GRAD_1 : True
                     TEST_RELU : True
                TEST_RELU_GRAD : True


TEST_ACTIVATIONS : True
```

# To run the test cases for basis.py file, run the following command and all tests should pass
$python test_basis.py

```
                 TEST_LINEAR_1 : True
            TEST_LINEAR_GRAD_1 : True
                 TEST_RADIAL_1 : True
            TEST_RADIAL_GRAD_1 : True


TEST_BASIS : True
```

# To run the test cases for losses.py file, run the following command and all tests should pass
$python test_losses.py

```
                 TEST_L2 : True
                TEST_L2_GRAD_1 : True


TEST_LOSSES : True
```
NOTE: Test cases are to be added for cross entrophy and cross entrophy grad.

