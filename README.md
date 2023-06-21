# Helicobacter_prognosis
Helicobacter_prognosis is a representation learning-based tool that aims to predict the prognosis of the infected individuals by machine learning on medical high-throughput data that can be collected non-invasively. Inspired by [DeepMicro](https://www.nature.com/articles/s41598-020-63159-5).


## Quick Setup Guide

**Step 1:** Change the current working directory to the location where you want to install `Helicobacter_prognosis`.

**Step 2:** Clone the repository using git command.
```
git clone https://github.com/tuncK/Helicobacter_prognosis
cd Helicobacter_prognosis
```

**Step 3:** Create a virtual environment using conda.
```
conda create --name hp python=3.11
```

**Step 4:** Activate the created virtual environment.
```
conda activate hp
```

**Step 5:** Install required packages
```
pip install hpbandster==0.7.4 keras==2.12.0 matplotlib==3.7.1 numpy==1.24.3 pandas==2.0.2 scikit-learn==1.2.2 scikit-optimize==0.9.0 scipy==1.10.1
```

**Step 6:** Install tensorflow.
* If your machine is *not* equipped with GPU, install tensorflow CPU version
```
pip install tensorflow-cpu==2.12.0
```
* If it is equipped with GPU, then install tensorflow GPU version
```
pip install tensorflow==2.12.0
```

