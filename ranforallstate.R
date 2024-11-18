library(tidyverse)
library(tidymodels)
library(glmnet)
library(vroom)
library(rpart)
library(ranger)
library(embed)
library(kknn)
library(themis)
install.packages("lme4")
library(lme4)
train <- vroom("train.csv")
test <- vroom("test.csv")
#head(train)
#glimpse(train)


recipe <-recipe(loss~ ., data = train) %>%
  step_mutate_at(matches("^cat\\d+$"), fn = as.factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss))

my_mod_rf <- rand_forest(mtry = tune(),
                         min_n = tune(),
                         trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

#Workflow
randfor_wf <- workflow() %>% 
  add_recipe(recipe) %>% 
  add_model(my_mod_rf)


## Set up a grid of tuning values
grid_of_tuning_params_rf <- grid_regular(mtry(range = c(1, 10)),
                                         min_n(),
                                         levels = 3)

## Set up K-fold CV
folds_rf <- vfold_cv(train, v = 4, repeats = 1)

## Find best tuning parameters
CV_results_rf <- randfor_wf %>% 
  tune_grid(resamples = folds_rf,
            grid = grid_of_tuning_params_rf,
            metrics = metric_set(mae),
            control=control_grid(verbose=TRUE))

bestTune_rf <- CV_results_rf %>% 
  select_best(metric = "mae")

## Finalize workflow and predict 
final_wf_rf <-
  randfor_wf %>% 
  finalize_workflow(bestTune_rf) %>% 
  fit(data = train)

## Predict
predictions_rf <- final_wf_rf %>% predict(new_data = test)


submission_rf <- predictions_rf |>
  rename(loss = .pred) %>% 
  select(loss) %>% 
  bind_cols(.,test) |> # bind preditions with test data
  select(id, loss) 
# Write out the file to submit to Kaggle
vroom_write(x= submission_rf, file = "./rfallstate.csv", delim = ",")



