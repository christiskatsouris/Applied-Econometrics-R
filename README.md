# Applied-Statistics-R

The synthesis of heterogenous data and the development of software - is a combination now recognized as data science. Therefore, the field of Applied Statistics has a key role in our understanding of data relationships and associations by testing statistical theorems and econometric theory which can support the broader scope of data science in explaining uncertainty. 

# [A]. Linear and Generalized Linear Models

## [A1]. Linear Models

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

## [A2]. Generalized Linear Models

In GLM we consider two important ingredients:

- Linear predictor: to model linear relationships. 

$$ \eta_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + ... + \beta_p x_{pi}.$$

- Link function: to link the linear predictor to the Binomial probability. Various link functions can be employed, however for binomial response, the most commonly used link function is Logit, which is defined as below

$$ \eta = log \left(  \frac{  p }{ 1 - p }  \right).$$



```R

> glm( formula, family = binomial(link = probit) )

```
### Example 2

Consider the implementation of a Generalized Linear Model to the [R](https://www.r-project.org/) dataset 'Anorexia'.

# [B]. Sequence Analysis and Bionomial GLM

Sequence Analysis is a non-parametric technique particularly useful for statistical inference with longitudinal data of employment and work-family related trajectories. Such data are commonly used in Labour Economics, Social Statistics and Demography and the main scope is to find statistical significant covariates that explain the particular data topologies across time. Although the presence of time-varying covariates requires additional regularity conditions, the use of sequence analysis along with the implementation of a Binomial GLM provides a methodology for analysing the trajectories of such Survey Study participants for static data structures (such as a particular cross-sectional or wave dataset, that is, a follow-up study for a given period of time.).     

```R

install.packages("TraMineR")
install.packages("cluster")

library(TraMineR)
library(cluster)


```

# [C]. Proportional Hazard Model


