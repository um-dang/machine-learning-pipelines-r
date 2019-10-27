# Predict colorectal cancer using microbiome data

Our goal is to try to predict whether someone has colorectal cancer based on the abundance of different bacteria in their stool microbiome.

Feaures: 16S rRNA gene abundances clustered to OTUs (represent bacterial abundances)
Label: Health outcome (whether the patient has colorectal cancer or not)
Classification algorithm: Random forest

- Data from: https://github.com/SchlossLab/Sze_CRCMetaAnalysis_mBio_2018
- Modified from: https://github.com/BTopcuoglu/MachineLearning/blob/master/code/learning/model_pipeline.R
- Further reading on this project: https://www.biorxiv.org/content/10.1101/816090v1

Credit: Thank you Zena Lapp for your live-coding scripts. 

1. First thing we do is download the dataset: There are 2 ways of doing this:
      - Clone this repository on your terminal in Mac or Git Bash on Windows if you have it.
      
      ```
      git clone https://github.com/um-dang/machine-learning-pipelines-r.git
      ```
      
      - Download the data here: https://tinyurl.com/yyqywozj

2. Create a folder in your Documents directory called `machine-learninig-pipelines-r`. Then within that folder, create another folder called `data`. Move the data.tsv file you downloaded into `data` folder.

3. Open RStudio. Go to `File` tab, click on `New Project`, create a project on `Existing Directory`, navigate to `machine-learninig-pipelines-r` directory and start the new project. Now you can open a New R script clicking on the green plus in RStudio. 

4. First we will load packages. If you haven't installed the packages before, please go to your RStudio console:

  ```install.packages('randomForest')```
  
  ```install.packages('caret')```

  ```install.packages('tidyverse')```

If you already installed these all you have to type  now is:

  ```
  library(randomForest)
  ```

__5. We are now ready to read in our data.__

```
data = read.delim('../data/data.tsv'
```

__6. Explore the data:__

```
data[1:5,1:5]
```

__7. Learn about the data:__

- The rows are different samples, each from a different person
- The first column is whether the person has cancer or not
    - 0 means no cancer, 1 means cancer
    - This is the **label**
- The rest of the columns are abundance of different OTUs
    - OTU stands for operational taxonomic unit, which is kind of like a species
    - Scaled to be between 0 and 1
    - These are the **features** we will use to classify/predict the label
  
How many samples do we have? How many features?

```
table(data$cancer)
```

__8. Do we have any missing data?__

```
# check to see if there's any missing data
sum(is.na(data))
```
Since we don't have any missing data, we don't have to remove any of the samples. 

__9. Split data into train and test set:__

The next step is to split the data into a training set (80% of the data) and a test set (20% of the data). We will make a random forest model using the training set and then test the model using the test set.

Why are we doing this? Because to have a reliable model, we need to follow the ML pipeline seen in Figure 1.

![Machine Learning Pipeline](Figure_1.pdf)

  - Random forest needs the label to be a factor if we want to do classification modeling. 
    We are classifying `having cancer` or `not having cancer`.

    ```
    # change the label to a factor (categorical variable) instead of a character 
    data$cancer = as.factor(data$cancer)
    ```

   - Randomly order samples. 
   ```
   random_ordered <- data[sample(nrow(data)),]
   ```

   - Determine the number of training samples
   ```
  number_training_samples <- ceiling(nrow(random_ordered) * 0.8)
  ```

   - Create training set:
   ```
   train <- random_ordered[1:number_training_samples,]
   ```

  - Create testing set
  ```
  test <- random_ordered[(number_training_samples + 1):nrow(random_ordered),]
  ```
  
  
