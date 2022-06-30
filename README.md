# Applied-Statistics-R

The synthesis of heterogenous data and the development of software - is a combination now recognized as data science. Therefore, the field of Applied Statistics has a key role in our understanding of data relationships and associations by testing statistical theorems and econometric theory which can support the broader scope of data science in explaining uncertainty. 

## [A]. Linear and Generalized Regression Models

### Linear Models

A model formula in R has the following form:

$$ \mathsf{response} \ \sim \ \mathsf{linear} \ \mathsf{predictor}$$ 

where the response corresponds to the response (or dependent) variable and the linear predictor corresponds to the set of explanatory variables. 

- **summary:** Produces and print a summary of the fit including parameter estimates along with their standard errors and p-values. 
- **coefficients:** Extracts the parameter estimates from the model fit. 
- **family:** The assumed distribution of the response and the relationship between the left and right-hand sides of the model. 

### Example 1

Consider the [R](https://www.r-project.org/) dataset ['plasma'](https://rdrr.io/cran/gamlss.data/man/plasma.html). Based on the variables of the dataset, with 'retplasma' as the response variable apply the forward model selection method. 

**Data Description**. Observational studies have suggested that low dietary intake or low plasma concentrations of retinol, beta-carotene, or other carotenoids might be associated with increased risk of developing certain types of cancer. The particular, cross-sectional study investigates the relationship between personal characteristics and dietary factors, and plasma concentrations of retinol, beta-carotene and other carotenoids (see, [Harell (2022)](https://hbiostat.org/data/repo/plasma.html)). 

```R

install.packages("CAMAN")
install.packages("gdata")
install.packages("readxl")

library("CAMAN")
library("gdata")
library("readxl")

data <- read_excel("plasma.xlsx")

age      <- data$age
sex      <- data$sex
smokstat <- data$smokstat
quetelet <- data$quetelet
vituse   <- data$vituse
calories <- data$calories
fat      <- data$fat
fiber    <- data$fiber
alcohol  <- data$alcohol 
cholesterol <- data$cholesterol
betadiet    <- data$betadiet
retdiet     <- data$retdiet
betaplasma  <- data$betaplasma
retplasma   <- data$retplasma 

mydata <- data
mydata <- data.frame(mydata)

null <- lm(retplasma ~ 1, data=mydata )
full <- formula( lm(retplasma ~.,data=mydata) )

fwd.model <- step( null, direction='forward', scope=full )

```

### Generalized Linear Models

```R

> glm( formula, family = binomial(link = probit) )

```
### Example 2

Consider the implementation of a Generalized Linear Model to the [R](https://www.r-project.org/) dataset 'Anorexia'.




