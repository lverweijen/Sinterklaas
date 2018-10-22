Quick and dirty script to accomplish a real life problem of grouping children together.

# Functionality
- 250 to 300 children need to be split in 29 groups
- Each group can have 8 to 10 children
- Children in a group should have similar age
- Each groups should have the same number of boys and girls as long as the ages within a group don't differ too much
- Some children must be grouped together
- Every group should have 1 or 2 leaders
- Younger groups should have 2 leaders and older groups 1
- Some leaders should be grouped together
- Some leaders and children should be grouped together

# Solution
The solution is accomplished by use of [cvxpy](https://www.cvxpy.org/citing/index.html) and [ecos](https://www.embotech.com/ECOS)

# Installation instruction

```
# 1. Download and install Anaconda from https://www.anaconda.com/download/#linux
$ bash Miniconda3-latest-Linux-x86_64.sh
$ conda env create -n sinterklaas --file requirements.txt
$ conda activate sinterklaas
```

