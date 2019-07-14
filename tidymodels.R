man_Data <- readr::read_csv('https://www.jaredlander.com/data/manhattan_Train.csv')

##create training data and validation

install.packages("tidyverse")
install.packages("rsample")

library(tidyverse)
library(rsample)

set.seed(123)
man_split <- initial_split (data = man_Data,prop = 0.9, strata = 'TotalValue')

man_split

man_train <- training(man_split)
man_test <- testing(man_split)

install.packages("parsnip")
library(parsnip)

#clean the data a bit

library(recipes)

base_formula <- TotalValue ~ FireService + ZoneDist1 + ZoneDist2 + 
            Class + OwnerType + LotArea +
            BldgArea + NumFloors + UnitsTotal
base_formula

mod1 <- lm(base_formula, data = man_train)
summary(mod1)

install.packages("coefplot")
library(coefplot)
coefplot(mod1, sort = 'magnitude')
coefplot(mod1, sort = 'magnitude',lwdInner = 2, lwdOuter =1) #the effect of each variable on total value, the x axis is impact
##the confidence intervals (the staright lines) are covering 0 meaning these variables are not statistically significant
##but doesn't mean you should take out those variables

base_recipe <- recipe(base_formula, data = man_train)
base_recipe

##standardiz numeric variables, get rid of no variance and log transform, subtract the means

man_recipe <- base_recipe %>%
    step_nzv(all_predictors()) %>%
    step_log(TotalValue, base = 10) %>%
    step_center(all_numeric(),-all_outcomes()) ##unnecessary for sgboost
    step_scale(all_numeric(), -all_outcomes())
    step_other(all_nominal()) %>%
    step_dummy(all_nominal(), one_hot = TRUE) %>%  ##one hot = true means don't drop the baseline, the %>% is a pip 
      ##that lets you pass the function without providing an argument
man_recipe
    
man_prepped <- man_recipe %>% prep(data = man_train, retain = TRUE)
man_prepped

train_baked <- man_prepped %>% juice()
test_baked <- man_prepped %>% bake(new_data = man_test)

##prepping is where we do the calculations like mean and stdev
##bake is where you use your prepped recipe on a dataset, subtract the calculations from the prepped dataset
    #in this case the prepped data is train not test, test data is not in the object of train data
##juice is where you get the baked data from the training data that we already transformed because the test data is not
    #in the same object with the training data
##so use bake when the data has been prepped, juice when the data has not been prepped but we don't want to repeat the same transformation
##retain has to be TRUE for juice to work

##scaling means remove the stdev from each observation and divide by the original value, centering meaning remove the mean from
  #each observation. Standardization (z score) includes both scaling and centering.

##Fit the model
install.packages("xgboost")
library(xgboost)
xgmod <- boost_tree(mode = 'regression') %>% set_engine('xgboost') #another most famous boost engine is C5.0
xgmod

mod2 <- xgmod %>% 
    fit(TotalValue ~ ., data = train_baked)
    
mod2$fit ##don't care about other stuff but fit

install.packages("DiagrammeR")
library(DiagrammeR)
mod2$fit %>% xgb.plot.multi.trees()  ##tree plot, useless, can't interpret

mod2$fit %>% xgb.importance(model=.) %>% 
  `[`(1:15) %>%         ##only use the first 15 columns from the data table
  xgb.plot.importance() ##tells us which variable is more important in impacting
                          #the outcome
mod3 <- boost_tree(mode = 'regression', trees = 130, tree_depth = 4, learn_rate = 0.25) %>%
set_engine('xgboost') %>% 
  fit(TotalValue ~ ., data = train_baked)

preds3 <- predict(mod3, new_data = test_baked)
preds3

truepred3 <- test_baked %>% 
    dplyr::select(TotalValue) %>% 
    dplyr::bind_cols(preds3)

install.packages("yardstick")
library(yardstick)

rmse(truepred3, truth = TotalValue, estimate = .pred)
