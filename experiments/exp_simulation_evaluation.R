# http://www.sthda.com/english/wiki/chi-square-test-of-independence-in-r
# https://alanarnholt.github.io/PDS-Bookdown2/post-hoc-tests-1.html

exp2_simulation_results_for_chi_square <- read.csv("/Users/abdallah/Desktop/Kings College Project/Code/results/exp2_simulation_results_for_chi_square.csv")


exp2_simulation_results_for_chi_square = subset(exp2_simulation_results_for_chi_square, select = -c(X) )
exp2_simulation_results_for_chi_square




table(exp2_simulation_results_for_chi_square$tool, exp2_simulation_results_for_chi_square$classification)
test <- chisq.test(table(exp2_simulation_results_for_chi_square$tool, exp2_simulation_results_for_chi_square$classification))
test # DF are (r-1)*(c-1) which is 3*1 in this case

test$statistic # test statistic
test$p.value # p-value



df_sim <- table(exp2_simulation_results_for_chi_square$tool, exp2_simulation_results_for_chi_square$classification)

############
file_path <- "http://www.sthda.com/sthda/RDoc/data/housetasks.txt"
housetasks <- read.delim(file_path, row.names = 1)
chisq <- chisq.test(housetasks)
chisq
test
#################

# Post-hoc comparisons
df_sim[c(2,3),] # LIME vs SHAP
post_hoc1 <- chisq.test(df_sim[c(2,3),], correct = FALSE)
post_hoc1$statistic
post_hoc1$p.value


df_sim[c(3,4),] # SHAP vs SVSL
post_hoc2 <- chisq.test(df_sim[c(3,4),], correct = FALSE)
post_hoc2$statistic
post_hoc2$p.value*6 # Times 6 to correct for 6 comparisons


df_sim[c(2,4),] # LIME vs SVSL
post_hoc3 <- chisq.test(df_sim[c(2,4),], correct = FALSE)
post_hoc3$statistic
post_hoc3$p.value*6


df_sim[c(1,2),] # ChoquEx vs LIME
post_hoc4 <- chisq.test(df_sim[c(1,2),], correct = FALSE)
post_hoc4$statistic
post_hoc4$p.value*6


df_sim[c(1,3),] # ChoquEx vs survSHAP
post_hoc5 <- chisq.test(df_sim[c(1,3),], correct = FALSE)
post_hoc5$statistic
post_hoc5$p.value*6


df_sim[c(1,4),] # ChoquEx vs SVSL
post_hoc6 <- chisq.test(df_sim[c(1,4),], correct = FALSE)
post_hoc6$statistic
post_hoc6$p.value*6


# assumption checks: https://www.statology.org/chi-square-test-assumptions/

test$expected # Chi-square test should only be applied when expected frequency of any cell is at least 5.
  # observations are independent, both IVs are categorical, cells are mutually exclusive
