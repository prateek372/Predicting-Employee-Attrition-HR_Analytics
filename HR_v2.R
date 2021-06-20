setwd('C:/Users/pradn/OneDrive/Documents/uc 2021/COURSES/1.2 stat comps/data/')
getwd()

library(readxl)
library(glue)
library(scales)
library(mltools)
library(data.table)
library(reshape2)
library(tidyverse)
library(heatmaply)
library(Rose)
suppressMessages(library(randomForest))
suppressMessages(library(caret))
suppressMessages(library(gbm))
suppressMessages(library(survival))
suppressMessages(library(pROC))
suppressMessages(library(rpart))
library("rpart.plot")
install.packages("ROSE")
library(ROSE)

#reading the raw file
df<- read.csv(glue(getwd(),'/WA_Fn-UseC_-HR-Employee-Attrition.csv'))
head(df)

#feature engineering 

##Feature 1: Annual salary
df1 <- df %>%mutate(AnnualSalary = MonthlyIncome * 12)

#dataframe for average salary
annualSalary <- df1 %>% 
                    group_by(JobLevel) %>%
                    summarize(avg_salary = mean(AnnualSalary), .groups = 'drop')

#plot of average salary
plt_annualSalary <-ggplot(data=annualSalary, aes(x=JobLevel, y=avg_salary)) +
                  geom_bar(stat="identity")


##Feature 2: Loyalty Index
df1<- df %>%
      mutate(loyalty_index = (TotalWorkingYears/ (NumCompaniesWorked + 1))) 



# Density graph for Loyalty Index
plt_loyalty_index <- ggplot(data=df1) + 
                     geom_density(aes(x = loyalty_index)) + 
                     xlab("Loyalty Index") + 
                     ylab("Density") + 
                     theme(text = element_text(size = 15))
  
#Gender and role wise Stacked Bar chart for the Loyalty Index   
loyalty_index_jobLevel <- df1 %>% 
                          group_by(JobLevel, Gender) %>%
                          summarize(avg_loyalty_index = mean(loyalty_index), .groups = 'drop')

plt_loyalty_index_jobLevel <- ggplot(data=loyalty_index_jobLevel , aes(x=JobLevel, y=avg_loyalty_index, fill=Gender)) +
                              geom_bar(stat="identity")+
                              geom_text(aes(label=round(avg_loyalty_index, digits = 2)), position = "stack", hjust = 0.3, vjust = 3,color="white", size=3.5)+
                              theme_minimal()

#Compensation Ratio - Comparison to Mean and Median Pay

#Adding column median and mean payment and ratio of mean and median Compensation Ratio 
df1 <- df %>%
       group_by(JobLevel) %>% 
       mutate(median_comp = median(MonthlyIncome), 
       mean_comp = mean(MonthlyIncome), 
       compa_median_ratio = (MonthlyIncome/ median_comp),
      compa_mean_ratio = (MonthlyIncome/ mean_comp))

#Above-Below mean and median pay? Adding columns
df1 <- df1 %>%
        mutate(compa_median_level = ifelse( compa_median_ratio > 1, 1, 0),
        compa_mean_level = ifelse( compa_mean_ratio > 1, 1, 0)) 

#plot desnity for monthly income
plt_income_density <- df1 %>%
                      ggplot() + geom_density(aes(x = MonthlyIncome, fill = JobLevel)) + theme(text = element_text(size = 15)) + 
                      ggtitle("Distribution of Monthly Income by Job Level") + 
                      scale_fill_brewer(palette = "Paired")

#stacked bar chart for gender and role wise Compensation Ratio 
compa_ratio_jobLevel <- df1 %>% 
                        group_by(JobLevel, Gender) %>%
                        summarize(avg_compa_mean_level = mean(compa_mean_ratio),
                        avg_compa_median_level = mean(compa_median_ratio),
                        .groups = 'drop')


plt_compa_median_jobLevel <- ggplot(data=compa_ratio_jobLevel , aes(x=JobLevel, y=avg_compa_mean_level, fill=Gender)) +
                      geom_bar(stat="identity")+
                      geom_text(aes(label=round(avg_compa_mean_level, digits = 2)), position = "stack", hjust = 0.3, vjust = 3,color="white", size=3.5)+
                      theme_minimal()
 

write.csv(df1,"hr_v1.csv")


## Transformations 


#removing redundant columns (Single valued)
df1 <- df1 %>% 
  select(-c("Over18", "StandardHours", "EmployeeCount"))

#distinct values in each column
df %>% summarise_all(list(~n_distinct(.)))

#factor : from char to factor 
# unclassing the data = converting to numeric value
# rescaling between 0 and 1
df1$Gender <- rescale(unclass(factor(df$Gender))) #2 unique values : factor+ unclass + rescale 
df1$OverTime <- rescale(unclass(factor(df$OverTime))) #2 unique values : factor+ unclass + rescale 
df1$Attrition<-rescale(unclass(factor(df$Attrition))) #2 unique values : factor+ unclass + rescale 


df1$MaritalStatus <- one_hot(as.data.table(factor(df$MaritalStatus))) #3 unique values: onehot+factor
df1$Department <-  one_hot(as.data.table(factor(df$Department))) #3 unique values : onehot+factor
df1$BusinessTravel <- one_hot(as.data.table(factor(df$BusinessTravel ))) #3 unique values : onehot+factor
df1$RelationshipSatisfaction<-one_hot(as.data.table(factor(df$RelationshipSatisfaction)))#4 unique values : onehot+factor
df1 <- df1 %>% 
       select(-c("EducationField", "JobRole"))

#df1$EducationField <- rescale(unclass(factor(df$EducationField))) #6 unique values
#df1$JobRole <- rescale(unclass(factor(df$JobRole))) #9 unique values
 

#df2= all numerical featues dataframe
df2 <-df1[, c(1:30,35)] 
rescaleDF <- function(x) (x-min(x))/(max(x) - min(x)) 
  
#  df3: non-numreric
df3 <- df2 %>% select_if(negate(is.numeric))

#df4 : numeric+ rescaled to 0 to 1 
df4 <- df2 %>% select_if((is.numeric))
df4 <- df4 %>% 
        select(-c( "JobLevel"))
df4 <- data.frame(lapply(df4, rescaleDF))

#Tranformed data set
df5<- cbind(df3,df4)
df5 <- df5 %>% 
    select(-c( "JobLevel...1"))
df5 <- as.data.table(df5)

#writing scaled   
write.csv(df5,"hr_v2.csv")


#class imbalance
prop.table(table(df5$Attrition)) # proportion is 0.83 to 0.16, hence class needs to be balanced 

set.seed(123)

#creation of train and testing dataset
n <- nrow(df5)
rnd <- sample(n, n * .70)
train <- df5[rnd,]
test <- df5[-rnd,]



# Modeling 
set.seed(123)
#decision tree: using unbalanced data
dtreepr <- rpart(Attrition ~., data = train)
preds <- predict(dtreepr, test, type = "vector")
rocv <- roc(as.numeric(test$Attrition), as.numeric(preds))
rocv$auc #0.7007

#plot
rpart.plot(dtreepr, 
           type = 1,  
           tweak = 0.9, 
           fallen.leaves = F)

set.seed(123)
# smapling : under+over
data_balanced_both <- ovun.sample(Attrition ~ ., data = train, method = "both", seed = 1)$data
table(data_balanced_both$Attrition)
##accuracy by sampling
dtreepr_both <- rpart(Attrition ~., data = data_balanced_both)
preds_both <- predict(dtreepr_both, test, type = "vector")
rocv_both <- roc(as.numeric(test$Attrition), as.numeric(preds_both))
rocv_both$auc #0.6529
#plot
rpart.plot(dtreepr_both, 
           type = 1,  
           tweak = 0.9, 
           fallen.leaves = F)
set.seed(123)
#ROSE
data_balanced_rose <- ROSE(Attrition ~ ., data = df5, seed = 1)$data
table(data_balanced_rose$Attrition)
##accuracy by sampling
dtreepr_ROSE <- rpart(Attrition ~., data = data_balanced_rose)
preds_rose <- predict(dtreepr_ROSE,  newdata=test)
rocv_rose <- roc(as.numeric(test$Attrition), as.numeric(preds_rose))
rocv_rose$auc #0.7277
#plot
rpart.plot(dtreepr_ROSE, 
           type = 1,  
           tweak = 0.9, 
           fallen.leaves = F)


#rose sampling method is providing highest accuracy 



set.seed(123)
# Random forest plain data

fit.forest <- randomForest(Attrition ~., data = train)
rfpreds <- predict(fit.forest, test, type = "class")

rocrf <- roc(as.numeric(test$Attrition), as.numeric(rfpreds))
rocrf$auc #Area under the curve: 0.8003

set.seed(123)
# Random forest rose
fit.forest_rose <- randomForest(Attrition ~., data = data_balanced_rose)
rfpreds_rose <- predict(fit.forest_rose, test, type = "class")
rocrf_rose <- roc(as.numeric(test$Attrition), as.numeric(rfpreds_rose))
rocrf_rose$auc #Area under the curve: 0.8379




#roc curve
plot(rocrf, ylim = c(0,1), print.thres = T, print.thres.cex = 0.8, main = "ROC curves", col = "salmon")
plot(rocrf_rose, ylim = c(0,1), print.thres = T, print.thres.cex = 0.8,  add = TRUE, col = "darkolivegreen")

# Plotting Feature Importance 
varImpPlot(fit.forest, sort=TRUE, n.var=min(30, nrow(importance(fit.forest))),
           type=NULL, class=NULL, scale=TRUE, 
           main=deparse( substitute(x))) 


imp <-data.frame(importance(fit.forest))
imp.colnames
write.csv(imp,"featureImportance.csv")


set.seed(123)

#creation of train and testing dataset

df6 <-df5 %>%
  select(MonthlyIncome,ï..Age,DailyRate,OverTime,TotalWorkingYears,MonthlyRate,EmployeeNumber,HourlyRate,DistanceFromHome,YearsAtCompany,WorkLifeBalance,StockOptionLevel,PercentSalaryHike,NumCompaniesWorked,JobSatisfaction,EnvironmentSatisfaction,YearsWithCurrManager,JobInvolvement,TrainingTimesLastYear,YearsSinceLastPromotion,JobLevel...6,YearsInCurrentRole,Education)

set.seed(123)
n <- nrow(df6)
rnd <- sample(n, n * .70)
train <- df5[rnd,]
test <- df5[-rnd,]


set.seed(123)
data_balanced_rose_limited <- ROSE(Attrition ~ ., data = df6, seed = 1)$data
table(data_balanced_rose_limited$Attrition)

set.seed(123)
# Random forest rose
fit.forest_rose_limited <- randomForest(Attrition ~., data = data_balanced_rose_limited)
rfpreds_rose_limited <- predict(fit.forest_rose_limited, test, type = "class")
rocrf_rose_limited <- roc(as.numeric(test$Attrition), as.numeric(rfpreds_rose_limited))
rocrf_rose_limited$auc #Area under the curve: 0.7733
