--ORISIS OF AI
--Documentation

Important Instructions:
* Use Google Colab(Or Jupyter Notebook)
* For Testing Open main.ipynb
* Only Neccesary File for a user is MAIN.ipynb

All files in the folder are well documented. They are partitioned into sections.

Files in this Folder
1 MAIN.ipynb
2 Clustering.ipynb
3 InitialApproach.ipynb
4 Holdout.ipynb
5 DictionaryCreation.ipynb
6 TrainingDataTransformation.ipynb

Main.ipynb - 
Contains main code for getting predictions of the given LAS file. Is capable of testing a file or deploying a file.
It is well documented for better understanding.
Contains 4 sections:
    1. Prerequisites
    2. Main class
    3. Deploy a Single File
    4. Deploy Multiple Files

Clustering.ipynb - 
Kmeans clustering is used. Latitudes and longitudes in the header info of LAS files is used to cluster files either into region 1 or 2. Centers are already taken and put in the main.ipynb  main class.

InitialApproach.ipynb-
Contains the code for initital approach we had.

Holdout.ipynb-
To have confidence on our strategy this file was necessary.

DictionaryCreation.ipynb - 
Dictionary was created using this file.

TrainingDataTransformation.ipynb - 
Training data was already created using this file which is later used in main.ipynb file.









 