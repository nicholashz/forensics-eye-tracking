## Forensics Eye-Tracking

This readme contains two sections. The first uses python to recreate many of the figures in the paper, and also self-documents the columns in the data tables in the data folder.

The second uses matlab to reproduce the TECA model correspondence estimates.


### Install python3
https://www.python.org/downloads/

Ensure python3 is working
```
$ python3 --version
```

### Verify data
#### 1. Verify the data directory in the same folder as traditional and notebooks within the PythonCodeForVisualizations folder

#### 2. Find TrialStats and OK_Fixations csv files into the data directory
The ```clean_data.ipynb``` notebook looks for the files at:
```
data/CwCeTrialStats_20200324.csv
```
and
```
data/CwCe_OK_Fixations_20180703.csv
```
These should already exist.

### Set up virtual environment
```
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip3 install --upgrade pip
(venv) $ pip3 install -r requirements.txt
```

### Clean data
The script to clean data (and create some features) is a Jupyter notebook.
#### 1. Open a terminal, activate venv, start the notebook:
```
$ source venv/bin/activate  (only if not already active from above)
(venv) $ jupyter notebook
```

#### 2. Open an internet browser and go to localhost:8888

#### 3. In Jupyter in the web browser, open "notebooks/clean_data.ipynb".

#### 4. Click the double-forward arrow button to run the whole notebook (rerunning if necessary). This creates a file called "clean_data.csv" in the "data" directory. You may need to click the 'Not Trusted button' to get it to trust the file.


### Create visualizations
#### 1. Run the script to produce plots
```
(venv) $ ./traditional/create_visualizations.py
```
Everything will be created in the "visualizations" directory. The code to create the visualizations makes the data in the csv files self-documenting.


## TECA model
The TECA model implementation is the "correspondence_matching" directory, but it should be verified before using.

#### 1. Cluster the fixations on each image
```
$ ./correspondence_matching/cluster_fixations.py --deciding_only --bandwidth 66
```

```--deciding_only``` is a flag to only use deciding (AKA detailing) fixations
```--bandwidth``` is the bandwidth for the Mean Shift clustering

#### 2. Create transition matrices
```
$ ./correspondence_matching/temp_seq.py --deciding_only --bandwidth 66 --weight_clusters
```

```--deciding_only``` and ```--bandwidth``` args should match from the clustering
```--weight_clusters``` should be added to increment the transition by 1 every time (as opposed to just setting it to 1 if it's looked at at all)

#### 3. Match the clusters
```
$ ./correspondence_matching/match_clusters.py --deciding_only --bandwidth 66 --weight_clusters --algo "greedy" --threshold 0.3
```

```--deciding_only```, ```--bandwidth```, and ```--weight_clusters``` should be used the same as previous steps.
```--algo```is either "greedy" or "hungarian"
```--threshold``` is threshold to cut out "weak" links



##  Two separate team members implemented the TECA model for redundancy. A separate implementation is in Matlab. Files for the TECA model in Matlab are found in MatlabCodeForTecaModel

The individual trial data is a variant of CwCeTrialStats_20200324.csv and AllFixationsByImageAndExaminer.mat is a processed version of CwCe_OK_Fixations_20180703.csv restricted to just the comparison phase eye gaze data.

In matlab, run
```
doFindCorrespondencesFromClusteredFixations.m
```
There may be slight variations in the two implementations, but the overall data patterns remain robust across implementations and reasonable parameter choices.

