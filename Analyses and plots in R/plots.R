library(ggplot2)
library(DALEX)
library(ggpubr)
library(latex2exp)




library(ggplot2)
library(latex2exp)
importance_labels <- c("5th", "4th", "3rd", "2nd", "1st")

# PVI Confidence Intervals

PVI_ci <- read.csv("/Users/abdallah/Desktop/Kings College Project/Code/results/PVI_confidence_intervals.csv")
PVI_ci = subset(PVI_ci, select = -c(X) )

ggplot(PVI_ci, aes(x, y)) +        # ggplot2 plot with confidence intervals
  xlab("Feature") + ylab("Permutation Feature Importance") +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width=0.5,
                size=0.5,)


ggsave("exp2_rsf_PVI_dataset0.pdf", device="pdf", width=1000, height=1200, units="px")



# dataset 0
make_factors <- function(data){
  data$importance_ranking <- factor(data$importance_ranking, levels=sort(unique(data$importance_ranking), decreasing=TRUE))
  data$variable <- factor(data$variable, levels=rev(c("x4", "x3", "x5", "x1", "x2")))
  data
}

barplot_variable_ranking_dataset0 <- function(data, title="", ytitle=""){
  ggplot(data, aes(fill=variable, y=value, x=importance_ranking)) +
    geom_bar(position="stack", stat="identity", color="white", size=0.2, width=0.8, 
             orientation="x") +
    scale_fill_manual("Global ranking\nof variable importance",
      values=c("#ae2c87", "#ffa58c", "#8bdcbe", "#46BAC2", "#4378bf"),
                      limits=c("x4", "x3", "x5", "x1", "x2"), 
                      labels=c(TeX("$x^{(4)}$"), 
                               TeX("$x^{(3)}$"), 
                               TeX("$x^{(5)}$"), 
                               TeX("$x^{(1)}$"), 
                               TeX("$x^{(2)}$"))) + 
    theme_minimal() +
    scale_y_continuous(expand = c(0, 0), 
                       breaks=c(seq(0, 100, 10), 102), 
                       minor_breaks = seq(5, 105, 5)) +
    scale_x_discrete(labels = importance_labels) +
    coord_flip() +
    geom_text(aes(label = value), 
              position = position_stack(vjust = 0.5), size = 3) + 
    labs(x=ytitle, y="Number of observations", title=title) +
    theme(plot.title = element_text(hjust = 0.5), axis.title.x = element_blank()) +
    guides(fill = guide_legend(title.position = "left", title.hjust = 0.5, title.vjust = 0.5))
}


## Refer to my data
setwd("/Users/AboodVU/Library/Mobile Documents/com~apple~CloudDocs/PhD Symbolic AI/Publication - MSc/nonadditive_explanations-master/Results")

# SurvLIME
exp2_survLIME_PLOT <- read.csv("exp2_SurvLIMEorderings_PLOT07_04_2025_22:22:58.csv")
exp2_survLIME_PLOT = subset(exp2_survLIME_PLOT, select = -c(X) )
exp2_survLIME_PLOT <- make_factors((exp2_survLIME_PLOT))
p1_SurvLIME <- barplot_variable_ranking_dataset0(exp2_survLIME_PLOT, ytitle="Importance ranking", title="SurvLIME")
p1_SurvLIME


# SurvSHAP
exp2_survSHAP_PLOT.csv <- read.csv("exp2_SurvSHAPorderings_PLOT07_04_2025_22:22:58.csv")
exp2_survSHAP_PLOT.csv = subset(exp2_survSHAP_PLOT.csv, select = -c(X) )
exp2_survSHAP_PLOT.csv <- make_factors((exp2_survSHAP_PLOT.csv))
p2_SurvSHAP <- barplot_variable_ranking_dataset0(exp2_survSHAP_PLOT.csv, title="SurvSHAP(t)")
p2_SurvSHAP


# MLEX
exp2_MLeX_PLOT <- read.csv("exp2_SurvMLeXorderings_PLOT07_04_2025_22:22:58.csv")
exp2_MLeX_PLOT = subset(exp2_MLeX_PLOT, select = -c(X) )
exp2_MLeX_PLOT <- make_factors((exp2_MLeX_PLOT))
p3_MLeX <- barplot_variable_ranking_dataset0(exp2_MLeX_PLOT, title="MLeX")
p3_MLeX

# SurvChoquEx
exp2_SurvChoquEx_PLOT <- read.csv("exp2_SurvChoquExorderings_PLOT07_04_2025_22:22:58.csv")
exp2_SurvChoquEx_PLOT = subset(exp2_SurvChoquEx_PLOT, select = -c(X) )
exp2_SurvChoquEx_PLOT <- make_factors((exp2_SurvChoquEx_PLOT))
p4_SurvChoquEx <- barplot_variable_ranking_dataset0(exp2_SurvChoquEx_PLOT, title="SurvChoquEx")
p4_SurvChoquEx

Sys.time()



p <- ggarrange(p1_SurvLIME, p2_SurvSHAP, p3_MLeX, p4_SurvChoquEx, ncol=2, nrow=2, common.legend = TRUE, legend="bottom") +
  theme(plot.margin = margin(0.1,0.2,0.1,0.1, "cm")) 
annotate_figure(p, )
ggsave("Simulation_Result.pdf", device="pdf", width=2600, height=800, units="px")




