
survey_results <- read.csv("/Users/abdallah/Desktop/Kings College Project/Code/results/survey_results.csv")



# Evaluating the MGS
  # all of it:
df_compl <- survey_results[c(3:nrow(survey_results)),c(24:38)] # indexing all responses to MGS


names(survey_results)

names(df_compl)

values <- c()
for (column in names(df_compl)){
  df_compl[[column]] <- as.numeric(df_compl[[column]])
  values <- c(values,(df_compl[[column]]))
}
values <- na.omit(values)




  # Means of each item
colMeans(df_compl, na.rm = TRUE) # N = 21
values <- unname(rowMeans(df_compl, na.rm = TRUE))
values <- na.omit(values)

  # Perform the one-sided t-test
hypothesized_mean <- 4 # Hypothetical population mean (the value you want to compare against)
result <- t.test(values, mu = hypothesized_mean, alternative = "greater") # Perform a one-sided t-test for less than

print(result) # Print the test result


result$statistic
result$p.value
result$conf.int

  # Confidence interval
df <- 19
alpha <- 0.05
critical_t_value <- qt(1 - alpha, df) # Critical t-value for a one-sided test
print(critical_t_value) # Print the critical t-value
mean(values) - (1.73*result$stderr)
mean(values) + (1.73*result$stderr)

  # Calculate Cohen's d
mean_ratings <- mean(values)
sd_ratings <- sd(values)
cohen_d <- (mean_ratings - 4) / sd_ratings
print(cohen_d)


  # The first dimension
df1 <- df_compl[c(1:5)]
mean(unname(colMeans(df1, na.rm = TRUE)))

  # The second dimension
df2 <- df_compl[c(6:10)]
mean(unname(colMeans(df2, na.rm = TRUE)))

  # The third dimension
df3 <- df_compl[c(11:15)]
mean(unname(colMeans(df3, na.rm = TRUE)))




# exploratory analysis
  # Occupations
survey_results$job_title

survey_results$age_cohort
table(survey_results$age_cohort[3:36])
  # 

  # comment at the end:
survey_results[c(3:nrow(survey_results)),c(39)] # indexing all responses to MGS
