# Survivor Predictions 

## Project Overview 

One of my favorite things about pursuing a career in data science is its wide-reaching applicability to all areas of life, including my personal interests both academic and otherwise. As I've developed expertise in predictive modeling techniques, there's been a potential application lingering in the back of my mind: my Survivor fantasy draft. 

When a new Survivor season is slated to air each fall and spring, a few former coworkers and I draft players to our teams for a fantasy-football-style competition (using a point system we developed and maintain ourselves, in true quant researcher fashion). Before the season starts, there is limited information available on the players; we haven't yet seen them on the island in Fiji, strategizing with other players, trying to win competitions, or pleading their case at Tribal Council. Instead, we have just a basic bio on the players, with demographic information like age, hometown, and occupation. It's not much, but I am always looking for the edge when it comes to crafting a promising team of players, and I've often wondered about the predictive power that this pre-season demographic data would have in forecasting how players will perform once on the island. 

In this repo, I have built an end-to-end ML pipeline where I ingest and clean Survivor pre-season data, conduct preliminary exploratory data analysis, engineer features I think could be predictive, train several types of regression and classification models, and evaluate the performance of these models in predicting Survivor player performance on out-of-sample test data (i.e. the most recent handful of Survivor seasons, withheld from model training). This was a fun way to explore whether I can improve my drafting prospects before the next season begins while simultaneously getting in some modeling practice. I outline steps below to run this repo locally with adjustable model hyperparameters. Feel free to play around yourself! 

## Repo Structure  

survivor-predictions/
│
├── README.md                 # Project overview, methodology, and findings
├── requirements.txt          # Python dependencies 
│
├── data/
│   ├── raw/                  # Original raw player data CSV
│   └── processed/            # Cleaned datasets with variables for modeling
│
├── notebooks/
│   ├── eda.ipynb             # Initial data exploration, distribution checks, and visualization
│
├── main.py                   # Run full pipeline 
│
├── src/                      # Modular source code package
│   ├── __init__.py
│   ├── config.py             # Specifies global constants, data file, and model types and hyperparameters 
│   ├── ingest.py             # Data ingestion, merging, and cleaning
│   ├── engineer.py           # Custom feature engineering (e.g. relative age, industry clusters)
│   ├── split.py              # Split data frame into train and test data 
│   ├── train.py              # Pipeline construction, training, and RidgeCV tuning
│   └── evaluate.py           # Evaluation metrics, ROC-AUC curves, and coefficient extraction
│
└── outputs/
    ├── figures/              # Saved plots (e.g., feature importance charts)
    ├── metrics/              # Saved model evaluation metrics


## To Use Pipeline

Follow these steps to run this pipeline:

#### 1. Clone this repo to your desktop.

#### 2. Download the dataset as an Excel file here: https://github.com/doehm/survivoR/tree/master. 

#### 3. Save the file survivoR.xlsx in a folder within /data/raw in the repo. 

#### 4. Ensure you have all requirements in requirements.txt installed in your environment. 

#### 5. Update config.py with your desired models and hyperparameters if you wish to update what I've used. 

#### 6. Run main.py within the repo folder in your terminal.


## Data Identification, Ingestion, & Cleaning

The data I used for this project is courtesy of a group of very kind Survivor fans who compiled raw data on all Survivor player demographics and game outcomes. 

After downloading and ingesting the data, I did the following as part of the cleaning process:
- Explored the data and checked for missing values. Luckily, this data is very complete. 
- Limited to data on U.S. seasons of Survivor. I am interested in the predictive power of these models on U.S. Survivor specifically, so that is my focus. Additionally, international versions of Survivor would likely see different impacts of demographic variables on outcomes, and also involve different game mechanics and strategy that also influence player performance. 
- Included returning players in the dataset. If a player goes on Survivor more than once, their performance will vary and they may also experience changes to some demographics (e.g. age, occupation). Therefore, I decided to keep each observation as a player in a given season, not one person overall. 

## Feature Selection & Engineering

### Outcomes

I identified and engineered two outcomes to represent a player's success in the Survivor game, one numerical outcome for regression models and one binary outcome for binary classification models. 

- I constructed a "placement score" as the regression outcome of interest. I created this metric by taking each player's raw place (e.g. in an 18 player season, place 1 = winner and place 18 = first voted out) and calculating how far they lasted into the season as a percentage of the total number of players in their season (e.g. in that same season, the first person voted out has a placement score of .056 and the winner has a placement score of 1.0). This standardizes players' game placements across seasons, given that there is variation in the total number of players per season. This outcome would allow direct comparison of players' predicted placements based on their demographic data and an exploration of how a different demographic feature is predicted to affect placement score on average.  
-  I constructed a "made merge" variable as the classification outcome of interest. This represents whether a given player made the merge/jury stage of their season, which is an indication that the player has achieved some success in the game and reached the second stage of play, as they have survived the initial portion of the game with several competing tribes and will at least get to vote for the eventual winner as part of the jury. I anticipated that this outcome would be easier to predict than exact placement and still helpful in demonstrating whether a player has been somewhat successful. The two classes are also relatively even, with 61% of all players making the merge phase and 39% getting eliminated pre-merge. 

### Predictors 

- Age: I took raw player age and constructed a relative age metric that measures the player's age relative to the average age on their cast (e.g. if the average age is 35, a 40 year old player has a relative age of 5). I felt this may be more predictive of game success than absolute age, because players who are outliers age-wise may struggle socially and with challenges compared to those closer to average on their cast. 
- Gender: I used a binary male/female variable for gender. There are two non-binary players, but both identified as female at the time they were on the show, so I chose to use that gender for modeling purposes. Each cast is equally divided into male and female players, so there is no need to measure relative to the total cast. 
- Race/ethnicity: I chose to use a binary variable for whether an individual is BIPOC (Black, Indigenous, and People of Color) or not to represent race/ethnicity. There is also categorical race/ethnicity data, but I wanted to avoid splitting into smaller categories that may not have much statistical power. 
- Industry: I grouped job titles into industry categories that I felt may similarly predict player performance based on my domain knowledge of the game. For example, broadly, players with a physical/athletic profession will likely perform well in challenges, players with a service profession like hairdresser or waitress often do well socially, and players with a job like lawyer may be perceived as smart, well-spoken, and not in need of the monetary prize, which can affect them negatively in the game. 
- Region: I grouped states into a categorical variable for geographic region. The area of the country players are from can affect what they have in common with others, how they're perceived, and sometimes personality traits, all of which could theoretically affect their in-game performance. 
- New Era: The 10 most recent seasons of the show (Survivor 41-50) are classified as the "New Era", which involves new game twists and advantages as well as a CBS mandate that at least half the cast be BIPOC. I created a binary variable for whether a season is part of the New Era, because I think the Era could impact how predictors affect player outcomes (for example, ideally BIPOC players may do better when there are more players with similar identities, as opposed to being in a small minority).   

## EDA & Visualization 

I conducted exploratory data analysis in a Jupyter notebook to get a sense for the relationships between my predictors and outcomes. The correlation coefficients between any particular predictor and outcome were relatively low. When looking at the predictor-outcome relationships graphically, I did not notice any linear or parametric nonlinear relationships of note. For example, I would've expected a potential parabolic relationship between age and player success, in which case I would've added age and age-squared to my regressions, but the data was too noisy to notice such a pattern. 

Given the noise between the predictors and placement score outcome, I also visualized some breakdowns of Survivor winners by demographic predictor to explore any descriptive relationships. There were several findings of note, including that men and players in their 20s and 30s win most often, which is in line with my knowledge of the show. This doesn't reflect a causal relationship, but is an interesting historical trend. 

## Modeling 

### Train-test split
When deciding on a train-test split of the data for model training and validation, I wanted to ensure data from the same seasons were siloed in one part of the process or the other, and I don't want chronologically later seasons being trained on to predict the outcomes of earlier seasons. I also wanted some New Era seasons in the training set to capture some of the differences that could've occurred, rather than training on only Old Era Seasons and validating on only New Era seasons, which would've happened with an 80/20 train-test split. I chose to use Seasons 1-45 as my training data set and 46-50 as my testing data set, even though this leaves a relatively small amount of data for testing. 

### Regression Models 
My regression models look at the power of the pre-game demographic variables in predicting player's relative numerical placement (through placement score). 

#### Null model
I began with a null model to serve as a baseline for comparison against my models that include predictors. I wanted to compare the predictive power of the other models to this one to see if these fitted models actually do a better job at predicting player performance than a model that selects the average placement for all players across all categories of the demographic variables. This helped me determine if any of these models would actually be a helpful tool to use to gain an advantage in predicting Survivor placement.   

#### Linear regression
I use a linear regression to see the predictive power of assuming a linear relationship between the predictors and outcome. This serves as a starting point to see the benefit of a straightfoward and easily interpretable model. 

#### Ridge regression
I also test a ridge regression, which imposes a penalty on variable coefficients to avoid overfitting the model to the relationship in the training data, shrinking the coefficients towards zero unless they provide enough explanatory power. I tune for the optimal alpha value as part of the model training to determine how large the penalty should be for the best model. 

#### Random Forest
After my linear models, I chose to test a random forest model to capture any potential nonlinearity in the predictor-outcome relationships as well as any interactions between the features (for example, maybe age has a differing effect on performance for men vs. women, or maybe race has a differing effect on performance in the New Era seasons vs. Old). I could've included interaction terms in my linear models, but those specifications across several variables could've gotten out of hand quickly. I chose a random forest over a single decision tree to limit overfitting. 

#### XGBoost
I also test XGBoost as an additional regression model that can capture nonlinearity. While random forest aims to create good predictions through ensemble learning across a number of trees, XGBoost aims to increase predictive power by fitting sequentially on the error remaining from each previous tree. 

#### Results 

The error (MAE) for the linear and ridge regression models was only marginally better than the null model without predictors (.248 and .249 vs. .25). Additionally, Ridge regression hyperparameter tuning chose the largest possible alpha (1,000) as the optimal one, meaning the coefficients were shrunk very close to zero. In effect, these linear models don't have much predictive power at all in using the demographic features to estimate player placement when compared to simply choosing average placement for every player. While random forest and XGBoost are more equipped to capture nonlinear and interaction relationships, they actually performed worse than the null model at predicting player placement. Similar error rates in both ensemble methods provide evidence that there errors are a result of noise in the data that prevents us from predicting player placement based on this pre-game data, rather than a matter of model selection or hyperparameter tuning. 

Graphs of the actual vs. predicted placement score for the random forest and XGBoost models showed all predictions hovered between .4 and .8, showing the uncertainty and tendency to trend towards the average placement. The Ridge regression graph shows all predictions around .5, given that the coefficients were shrunk very close to zero, nearly matching the null model. Most linear regression predictions also hover around the .4 to .6 range, with a few towards .8. None of these models are helpful in estimating the placement of a particular player; rather, they tend towards the null model predictions of .5 for every player. 

While these models aren't ultimately helpful at predicting placement overall, feature importance is still interesting to look at. Relative age was the most important feature for all of the models, overwhelmingly so for the random forest. As expected, the linear relationship between relative age and placement is negative. Ultimately, this relationship still does not have enough predictive power to overcome the variation in data that's attributed to in-game events or other randomness. 

With regression models largely unsuccessful, I hoped utilizing classification models with a binary outcome variable might be more successful. This way, I am not looking to estimate a numerical placement, but rather predict whether a player made it into the jury/merge stage of the game or not. This would still be helpful information for my purposes of gaining some insight into expected player success, and predicting exact placement doesn't matter as much to me as getting a broad sense of how they might perform. 

### Binary Classification Models and Results

The equivalent of a "null" model that I will compare my classification models is simply assigning every player to the majority class. This data doesn't have a huge imbalance, but 60% of players in the data made the merge phase, and 40% did not. Therefore, a model that assigns all players to the majority class (made_merge = 1) has 60% accuracy, and I'm looking for my classification models to perform better than that. 

#### Logistic Regression
The logistic regression model aims to predict the likelihood that a given player makes the merge. Like linear regression, it assumes a linear relationship between the predictors and outcomes, and doesn't account for nonlinearity or interactions unless interaction terms are added to the model.

#### Random Forest and XGBoost
I once again use random forest and XGBoost to try to capture nonlinearity and interactive relationships that likely exist between the predictors and in their relationship to the binary outcome. 
 
The XGBoost model performs the best of these three classification models, with a modest accuracy of 61.5% compared to 60.4% accuracy choosing majority class for all. The logistic regression and random forest models both have accuracies worse than the majority class baseline, with 53% and 56% respectively. Relative age continues to have the most feature importance, followed closely by industry, new era, gender, bipoc, and region for the XGBoost model. 

### Takeaways 

Ultimately, these regression and classification models do not help me better predict how a given player will perform before the season starts, with only the XGBoost classification model seeing a slight (1 percentage point) increase in accuracy of whether a player makes the merge over the baseline. While I'm not any closer to winning my fantasy pool by masterfully drafting players pre-season, this is still helpful information on the game. The lack of predictive power across all models suggests that the majority of variation in player performance depends on what players do in the game itself, rather than demographic characteristics outside of the game—which is a good thing. This doesn't mean there haven't been certain trends historically between demographics and performance (women, people of color, and older people have won less historically, for example), but using these demographics can't substantially improve prediction for future seasons, at least when training models on the first 45 seasons of Survivor and testing accuracy on the most recent 5. Maybe this will change as more "New Era" seasons air. 

### Potential Next Steps

There are some potential next steps to explore this topic further in an attempt to better my chances at predicting player success pre-game:
- Wait for more "New Era" seasons to provide additional data with this change in the game and demographics 
- Use NLP on pre-game player interviews to find out more information on how they plan to play the game and use that for prediction. However, I ultimately think my own perception of these interviews with the knowledge I have may outperform an NLP model, so I hesitate to take the time to create it. 
- Incorporate some in-game data to update predictions as the season airs. This wouldn't be useful for strictly pre-game purposes, but could make me better at determining player success during the season. 