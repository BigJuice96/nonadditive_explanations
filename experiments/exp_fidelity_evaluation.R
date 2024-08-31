library(dplyr)

fidelity_results <- read.csv("/Users/abdallah/Desktop/Kings College Project/Code/results/fidelity/results_fidelity.csv")


# Calculate ANOVA

  # Create DF with factors
fidelity <- c(fidelity_results$SVM_10, fidelity_results$survLIME_10, fidelity_results$survSHAP_10)
group <- c(replicate(100, "SVM_10"), replicate(100, "survLIME_10"), replicate(100, "survSHAP_10"))
df <- data.frame(fidelity, group)

df$group <- as.factor(df$group)
df$group <- ordered(df$group,
                         levels = c("survLIME_10", "survSHAP_10", "SVM_10"))

  # Summary stats
group_by(df, group) %>%
  summarise(
    count = n(),
    mean = mean(fidelity, na.rm = TRUE),
    sd = sd(fidelity, na.rm = TRUE)
  )


  # Compute the analysis of variance
res.aov <- aov(fidelity ~ group, data = df)
  # Summary of the analysis
summary(res.aov)

# Assumption checks
hist(df$fidelity) # doesn’t look very normally distributed: can also create a Q-Q plot to get another look 

  #create Q-Q plot to compare this dataset to a theoretical normal distribution 
qqnorm(res.aov$residuals)

  #add straight diagonal line to plot
qqline(res.aov$residuals) # Seems to fall along the diagonal in the center of distr. (tails matter less)


  #equal variance: create box plots that show distribution of weight loss for each group
boxplot(fidelity ~ group, xlab='group', ylab='fidelity', data=df) # Variance seems equal across groups



# Plot CIs
fidelity_ci <- read.csv("/Users/abdallah/Desktop/Kings College Project/Code/results/fidelity/fidelity_confidence_intervals_k10.csv")
fidelity_ci
fidelity_ci = subset(fidelity_ci, select = -c(x) )

ggplot(fidelity_ci, aes(x.1, y)) +        # ggplot2 plot with confidence intervals
  xlab("Tool") + ylab("Fidelity") +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width=0.5,
                size=0.5,)

ggsave("fidelity_ci_k10.pdf", device="pdf", width=1000, height=1200, units="px")

