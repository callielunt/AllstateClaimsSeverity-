# Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(themis)
library(discrim)
library(bonsai)
library(lightgbm)

# Read in Data
train <- vroom("train.csv")
test <- vroom("test.csv")


# Recipe
my_recipe <- recipe(loss~ ., data = train) %>%
  step_mutate_at(matches("^cat\\d+$"), fn = as.factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(loss)) %>% 
  step_normalize(all_numeric_predictors())


## Model
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("regression")

## or BART
# bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
#   set_engine("dbarts") %>% # might need to install
#   set_mode("regression")

#Workflow
boosted_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(boost_model)

## CV tune, finalize

## Set up a grid of tuning values
tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

## Set up K-fold CV
folds <- vfold_cv(train, v = 3, repeats = 1)

## Run the CV
CV_results <- boosted_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mae),
            control=control_grid(verbose=TRUE))


## Find best tuning parameters
bestTune <- CV_results %>% 
  select_best(metric = "mae")

## Finalize workflow and predict 
final_wf <-
  boosted_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = train)

## Predict
# make predictions
predictions <- predict(final_wf,
                       new_data = test)

submission <- predictions |>
  rename(loss = .pred) %>% 
  select(loss) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, loss)  # just keep datetime and prediction value


# Write out the file to submit to Kaggle
vroom_write(x= submission, file = "./boostedagain15.csv", delim = ",")

stopCluster(clusters)