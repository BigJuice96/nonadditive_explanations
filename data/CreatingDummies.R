
library(tidyverse)
library(fastDummies)
library(tableone)
library(expss) # for adding lables to variables 
library(crosstable)
library(readxl)


data <- read.csv("/Users/olesyaa/Library/CloudStorage/OneDrive-King'sCollegeLondon/KCL/LIST of STUDIES/Dementia/Data prepation ELSA Dementia/DATA/Data_ELSA_Combined_dementia_all_NA_OA.csv", 
         header=T, stringsAsFactors = T)

## Code numeric variables as numeric and facror variables as factors 
# Should be factors: 
data <- data%>%
  mutate_at(c("death18fu", "dem_hse_w8",
              "scorg1",  "scorga2","scorg3", "scorg4","scorg5" ,"scorg6", "scorg7" , "scorg8"),
            function(., na.rm=F)
              as.factor(.))

# These variables should be numeric 
data <- data%>%
  mutate_at(c("r1retage", "r1retage" , "dhnch" ,"hebck" ,
              "hekne", "hefet", "hehip","deathy"
             ),
            function(., na.rm=F)
              as.numeric(.))






# Create dummies for factors with more than 2 levels 
dummies_d <- data%>%
  select(c(wlth_gp,edqual,asoccls,hepaa,heala,hebal,hediz,wselfd)) %>%
  fastDummies::dummy_cols(remove_first_dummy = TRUE)%>%
  select(-c(wlth_gp,edqual,asoccls,hepaa,heala,hebal,hediz,wselfd,
            contains("_NA"))) %>%
  mutate_all(
    function(., na.rm=F)
      as.factor(.)
  )
  
data <- cbind(data, dummies_d) %>%
  select(-c(wlth_gp,edqual,asoccls,hepaa,heala,hebal,hediz,wselfd))


# Provide an overview of the final data:
print(CreateTableOne(vars = names(data[,-c(1)]), 
                     strata = c("dem_hse_w8"),
                     includeNA=F,
                     addOverall=T,
                     data = data[,-c(1)]))


# Save file 
write.csv(data, "/Users/olesyaa/Library/CloudStorage/OneDrive-King'sCollegeLondon/KCL/LIST of STUDIES/Dementia/Data prepation ELSA Dementia/DATA/Data_ELSA_Combined_dementia_all_NA_someDummiesOA.csv", 
          row.names = FALSE)



d_to_compare <- read.csv("/Users/olesyaa/Library/CloudStorage/OneDrive-King'sCollegeLondon/KCL/LIST of STUDIES/Dementia/Data prepation ELSA Dementia/DATA/dummied_and_trimmed_data.csv")

