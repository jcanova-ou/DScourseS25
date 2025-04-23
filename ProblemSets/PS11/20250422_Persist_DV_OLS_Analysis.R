library(tidyverse)
library(modelsummary)
library(ggplot2)
library(dplyr)
library(readxl)


#-------------------------------
# Import Data
#-------------------------------
NASO_persist <- read_excel("C:/Users/cano0017/Desktop/OU/Working Papers/2 -Commitment Conundrum (Persistent Hybrid Ent)/Data/Processed Data/20250422_2023_NASO_persist_DV.xlsx", 
                           skip = 1)
View(NASO_persist)
dim(NASO_persist)
#-------------------------------
#Variable name Mappings (Reference)
#-------------------------------
#3	heckle
#4	criticize
#30	active
#47	enough_money
#52	IC
#53	Age
#54	Gender
#60	Race
#66	weekly_games
#67	training_matches
#71	main_level
#72	Self_Rank
#73	Confidence
#73a	Confidence_reversed
#75	sport_structure
#75a	
#75b	
#75c	
#75d	
#75e	
#75f	
#75g	
#75h	
#75_Count	social_count
#89	educ
#90	work-related
#91	occupation
#92	industry_role
#93	organization
#94	employment
#95	first_officiating_year
#95a	Yrs_reffing
#99	first_officiating_age
#100	first_officiating_level
#106	Future
#107	Current
#108	direction
#116	marital_status
#117	kids
#118	family_damage
#119	social_network


#Filter Questions
#30 - Are you currently officiating (as a referee, umpire, etc.)?	(Yes = 1, No = 0, Blank (omitted 1176))


#Controls
#52 - Independent Contractor (Y/N)
#53 - Age in years
#54 - Gender (M-1/F-0/Other excluded)
#108 - Where would you consider yourself in your officiating career?	("Climbing the ladder" = 1, "Satisfied at my current level" = 2, 
                                                                        #"Looking to work at a lower level than I once did" = 3, 
                                                                        #"Ready to retire" = 4, "Have retired" = 5)
#117 - How many children do you have?	"Greater than 5" = 6 (423 instances)
#60 -	Which categories best describe you? Mark all that apply.	(White, Hispanic, Black, Asian, American Indian, Pacific Islander, Middle Eastern, Mixed Race, Other)
#89 -	What is the highest degree or level of school you have completed? If currently enrolled, mark the previous or highest degree received.	
      #"No diploma" = 1, "High school graduate - high school diploma or the equivalent (for example: GED)" = 2, "Some college credit, but no degree" = 3, 
      #"Associate degree (for example, AA, AS)" = 4, "Bachelorâ€™s degree (for example: BA, AB, BS)" = 5, 
      #"Post-graduate study without a Masterâ€™s degree" = 6, "Masterâ€™s degree (for example: MA, MS, MEng, MEd, MSW, MBA)" = 7, 
      #"Professional degree (for example: MD, DDS, DVM, LLB, JD)" = 8, "Doctorate degree (for example: PhD, EdD)" = 9


#Dependent Variable
#95a - Years as a referee


#Independent Variables
  #Autonomy
     #47	If your household income is/were high enough to not need the income from your officiating roles, do/would you continue in these capacities?	
              #(Yes = 1, No = 0, Blank or not sure (omitted 11726))
     #66	On average, how many games or matches do you officiate per week in season?	
             #(Continuous (8 or more set to 8 (3763 instances)))
  
  #Competence ESE
      #72	How would you rank yourself in comparison with other officials at your primary level of each sport? 1st-5th percentile is the best.	
              #"Top 1st-5th percentile" = 7, "6th-15th percentile" = 6, "16th-25th percentile" = 5, "26th-50th percentile" = 4, 
              #"26th-50th percentile" = 3, "51st-75th percentile" = 2, "76th-100th percentile" = 1
  
       #73a	How would you describe your level of confidence/certainty regarding decisions while officiating each sport? 1 is the highest level of confidence possible, while 10 is the lowest.	(Reverse coded)
  

  #Connection
      #3	As a fan, do you ever heckle officials?	Yes = 1, No = 0, Blank (omitted 221)
      #4	Do you ever publicly criticize other officials?	Yes = 1, No = 0, Blank (omitted 74)
      #75_Count	How many social opportunities exist in your sport's formal structure?	
              #Count of: "Officiating Coordinators / Assignors", "Trainers / Instructors", "Assessors", "Formal assessment process", 
              #"Assigned Mentors", "Classroom-training sessions", "Field-training sessions", "Meetings" ("Formal assessment process", 
              #"Centralized online portal / site for officials" excluded because they don't have to be social)
      #118	Has officiating hurt your personal relationships with family members?	Yes= 1, No = 0, Not Sure (Omitted 1605 instances)
      #119	Has officiating hurt or helped your social network?	Helped = 1, Hurt = 0, Not Sure (Omitted 5001 instances)


#-------------------------------
#Clean out 99s
#-------------------------------
# Replace 99 with NA
NASO_persist_clean <- NASO_persist
NASO_persist_clean[NASO_persist_clean == 99] <- NA
view(NASO_persist_clean)
dim(NASO_persist_clean)

#-------------------------------
#Variable Name Proxy Coding
#-------------------------------
controls <- "Yrs_reffing~ IC + Age + Gender + direction + kids + educ"
autonomy_IV <- paste(controls, "enough_money", "weekly_games", sep = " + ")
Competence_ESE_IV <- paste(controls, "Self_Rank", "Confidence_reversed", sep = " + ")
Connection_IV <- paste(controls, "heckle", "criticize", "social_count", "family_damage", "social_network", sep = " + ")

#-------------------------------
#Filtering
#-------------------------------
# Now filter where active = 1
NASO_Clean_Active <- NASO_persist_clean[NASO_persist_clean$active == 1, ]
dim(NASO_Clean_Active)

#-------------------------------
# Descriptive Statistics
#-------------------------------
# Correlation Table
correlation_table <- NASO_Clean_Active %>%
  select("IC","Age", "Gender","direction","kids", "educ", "enough_money", "weekly_games", 
         "Self_Rank", "Confidence_reversed","heckle", "criticize", "social_count", 
         "family_damage", "social_network") %>%
  cor(use = "pairwise.complete.obs")
cor(correlation_table)
datasummary_correlation("IC","Age", "Gender","direction","kids", "educ", "enough_money", "weekly_games", 
                        "Self_Rank", "Confidence_reversed","heckle", "criticize", "social_count", 
                        "family_damage", "social_network")


# Descriptive Statistics
datasummary_skim(NASO_Clean_Active, histogram = FALSE,fmt = 2)



#-------------------------------
# Models
#-------------------------------
Model_Controls <- lm(controls, data = NASO_Clean_Active)
Model_autonomy <- lm(autonomy_IV, data = NASO_Clean_Active)
Model_Competence_ESE <- lm(Competence_ESE_IV, data = NASO_Clean_Active)
Model_Connection <- lm(Connection_IV, data = NASO_Clean_Active)

modelsummary(list("Model 1 Controls"=Model_Controls, "Model 2 Autonomy" = Model_autonomy, "Model 3 Competence" = Model_Competence_ESE, 
                  "Model 4 Connection" = Model_Connection), stars = T)


#-------------------------------
#Variable Name Proxy Coding (stripped down models)
#-------------------------------
controls_slim <- "Yrs_reffing~ IC + Age + Gender + direction + kids + educ"
autonomy_IV_slim <- paste(controls_slim, "enough_money", sep = " + ")
Competence_ESE_IV_slim <- paste(controls_slim, "Self_Rank", "Confidence_reversed", sep = " + ")
Connection_IV_slim <- paste(controls_slim, "heckle","social_count","family_damage", sep = " + ")
Connection_IV_slim_test <- paste(controls_slim, "heckle","social_count", sep = " + ")
test<- lm(Connection_IV_slim_test, data = NASO_Clean_Active)
modelsummary(test, stars = T)


#-------------------------------
# Models (stripped down)
#-------------------------------
Model_Controls_slim <- lm(controls_slim, data = NASO_Clean_Active)
Model_autonomy_slim <- lm(autonomy_IV_slim, data = NASO_Clean_Active)
Model_Competence_ESE_slim <- lm(Competence_ESE_IV_slim, data = NASO_Clean_Active)
Model_Connection_slim <- lm(Connection_IV_slim, data = NASO_Clean_Active)

modelsummary(list("Model 1 Controls slim"=Model_Controls_slim, "Model 2 Autonomy slim" = Model_autonomy_slim, "Model 3 Competence slim" = Model_Competence_ESE_slim, 
                  "Model 4 Connection slim" = Model_Connection_slim), stars = T)
