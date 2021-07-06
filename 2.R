library(tidymodels)
library(tidyverse)
library(recipes)
setwd("D:/monash/etc3250ML/project")
set.seed(32505250)
mydat <- read_csv("student_training_release.csv")
mymodel <- boost_tree(mtry = tune(),
                      trees = tune(),
                      tree_depth = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

mycv <- vfold_cv(mydat, v = 3, strata = "cause")
myrecipe <- recipe(mydat, formula = cause ~ ws + rf + se + maxt + mint + day + FOR_TYPE + COVER + HEIGHT) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_knn(all_predictors(), neighbors = tune()) %>%
  step_dummy(all_nominal_predictors())

myworkflow <- workflow() %>%
  add_recipe(myrecipe) %>%
  add_model(mymodel)

mygrid <- expand.grid(mtry = c(5, 7, 9),
                      neighbors = c(3, 5),
                      trees = seq(200, 1000, 200),
                      tree_depth = c(5, 7, 9))

mycontrol <- control_grid(verbose = TRUE)

grid_result <- myworkflow %>%
  tune_grid(resamples = mycv, grid = mygrid, control = mycontrol)

autoplot(grid_result)

mybestpara <- select_best(grid_result, metric = "accuracy")

myworkflow <- myworkflow %>%
  finalize_workflow(mybestpara)

myworkflow <- myworkflow %>%
  fit(data = mydat)

preddat <- read.csv("student_predict_x_release.csv")

mypred <- myworkflow %>%
  predict(preddat)

data.frame(Category = mypred$.pred_class) %>%
  mutate(Id = 1:n()) %>%
  select(Id, Category) %>%
  write_csv("__test__3.csv")
