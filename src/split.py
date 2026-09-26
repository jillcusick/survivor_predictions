
from .config import SPLIT_SEASON

def split_data(df, split_season=SPLIT_SEASON):
    """ Splits the Survivor cast data into training and testing sets chronologically by season. """
    
    # split the data into training and testing sets based on the split season above
    train_df = df[df['season'] <= split_season]
    test_df = df[df['season'] > split_season]
    
    # keep only feature columns for X  
    feature_cols = ["rel_age", "gender", "region", "industry", "new_era", "bipoc"]
    
    # prepare training data subset
    X_train = train_df[feature_cols]
    y_train_class = train_df["made_merge"]
    y_train_reg = train_df["place_score"]

    # prepare testing data subset
    X_test = test_df[feature_cols]
    y_test_class = test_df["made_merge"]
    y_test_reg = test_df["place_score"]

    # return the split datasets
    return (
        X_train,
        X_test,
        y_train_class,
        y_test_class,
        y_train_reg,
        y_test_reg,
    )
    