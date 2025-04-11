library(randomForestSRC)


data(veteran, package = "randomForestSRC")
veteran

elsa_data <- read.csv("/Users/abdallah/Desktop/Kings College Project/Code/results/case_study/selected_elsa_df.csv")

v.obj <- rfsrc(Surv(time,dem_hse_w8)~., elsa_data, nsplit = 10, ntree = 100)

## get partial effect of age on mortality
partial.obj <- partial(v.obj,
                       partial.type = "mort",
                       partial.xvar = "age",
                       partial.values = v.obj$xvar$age,
                       partial.time = v.obj$time.interest)
pdta <- get.partial.plot.data(partial.obj)

names(elsa_data)

## plot partial effect of age on mortality    
plot(lowess(pdta$x, pdta$yhat, f = 1/3),
     type = "l", xlab = "age", ylab = "adjusted mortality")

## partial effects of n_chronic_diseases score on survival
n_chronic_diseases <- quantile(v.obj$xvar$n_chronic_diseases)
partial.obj <- partial(v.obj,
                       partial.type = "surv",
                       partial.xvar = "n_chronic_diseases",
                       partial.values = n_chronic_diseases,
                       partial.time = v.obj$time.interest)
pdta <- get.partial.plot.data(partial.obj)

## plot partial effect of n_chronic_diseases on survival
matplot(pdta$partial.time, t(pdta$yhat), type = "l", lty = 1,
        xlab = "time", ylab = "n_chronic_diseases adjusted survival")
legend("topright", legend = paste0("n_chronic_diseases = ", n_chronic_diseases), fill = 1:5)



























## get partial effect of social participation on mortality
partial.obj <- partial(v.obj,
                       partial.type = "mort",
                       partial.xvar = "social",
                       partial.values = v.obj$xvar$social,
                       partial.time = v.obj$time.interest)
pdta <- get.partial.plot.data(partial.obj)

names(elsa_data)

## plot partial effect of social participation on mortality    
plot(lowess(pdta$x, pdta$yhat, f = 1/3),
     type = "l", xlab = "Social Participation", ylab = "Adjusted Survival")




## get partial effect of depression on mortality
partial.obj <- partial(v.obj,
                       partial.type = "mort",
                       partial.xvar = "cesd_sc",
                       partial.values = v.obj$xvar$cesd_sc,
                       partial.time = v.obj$time.interest)
pdta <- get.partial.plot.data(partial.obj)

names(elsa_data)
elsa_data$procspeed1

## plot partial effect of social participation on mortality    
plot(lowess(pdta$x, pdta$yhat, f = 1/3),
     type = "l", xlab = "Symptoms of Depression", ylab = "Adjusted Mortality")




## get partial effect of n_neuropsychiatric_symptoms on mortality
partial.obj <- partial(v.obj,
                       partial.type = "mort",
                       partial.xvar = "n_neuropsychiatric_symptoms",
                       partial.values = v.obj$xvar$n_neuropsychiatric_symptoms,
                       partial.time = v.obj$time.interest)
pdta <- get.partial.plot.data(partial.obj)

names(elsa_data)
elsa_data$procspeed1

## plot partial effect of social participation on mortality    
plot(lowess(pdta$x, pdta$yhat, f = 1/3),
     type = "l", xlab = "Neuropsychiatric Symptoms", ylab = "Adjusted Mortality")






## partial effects of processing speed on survival
procspeed1 <- quantile(v.obj$xvar$procspeed1)
partial.obj <- partial(v.obj,
                       partial.type = "surv",
                       partial.xvar = "procspeed1",
                       partial.values = procspeed1,
                       partial.time = v.obj$time.interest)
pdta <- get.partial.plot.data(partial.obj)

## plot partial effect of procspeed1 on survival
matplot(pdta$partial.time, t(pdta$yhat), type = "l", lty = 1,
        xlab = "Time in Years", ylab = "Adjusted Survival by Processing Speed")
legend("bottomleft", legend = paste0("percentile = ", names(procspeed1)), fill = 1:5)




