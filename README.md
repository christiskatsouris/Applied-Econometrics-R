# Applied-Statistics-R

The synthesis of heterogenous data and the development of software - is a combination now recognized as data science. Therefore, the field of Applied Statistics has a key role in our understanding of data relationships and associations by testing statistical theorems and econometric theory which can support the broader scope of data science in explaining uncertainty. 

In this teaching page of Applied Statistics using R, we present three main applications which are commonly presented in empirical economic and finance studies, that is, (i) the use of GLMs for the analysis of Binary Data, (ii) the implementation of Sequence Analysis for evaluating employment and/or work trajectories and (iii) statistical inference based on the Proportional Hazard Model which is useful for modelling the probability of default in various settings.

# [A]. Linear and Generalized Linear Models

## [A1]. Linear Models

A model formula in R has the following form:

$$ \mathsf{response} \ \sim \ \mathsf{linear} \ \mathsf{predictor}$$ 

where the response corresponds to the response (or dependent) variable and the linear predictor corresponds to the set of explanatory variables. 

- **summary:** Produces and print a summary of the fit including parameter estimates along with their standard errors and p-values. 
- **coefficients:** Extracts the parameter estimates from the model fit. 
- **family:** The assumed distribution of the response and the relationship between the left and right-hand sides of the model. 


### Example 1 

Consider the [swiss fertility data-set](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/swiss.html) in R. 


### Example 2

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

Overall in a Binamial GLM the response variable yi, for i = 1,...,n, is assumed to be Binomially distributed where n is fixed and represents independent trials. In other words, for Binary Data we have that yi = 1 or 0 where pi represents the probability of success. 

$$Y_i \sim \mathsf{Binomial} \left( n_i, p_i \right), \ \ \ \text{with} \ \ \ \mathbb{P} \left( Y_i = y_i \right) = n_i C_{y_i} p^{y_i} ( 1 - p_i )^{ n_i - y_i }.$$

Thus, the fundamental difference between Logistic Regression and Multiple Linear Regression, (as in classical econometric applications), is that the response is Binomially distributed. Then, the probability pi is modelled as a linear combination of p covariates. 

Therefore, in Generalized Linear Models (GLM), such as the Logistic Regression, we consider two important ingredients:

- Linear predictor: to model linear relationships. 

$$ \eta_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + ... + \beta_p x_{pi}.$$

- Link function: to link the linear predictor to the Binomial probability. Various link functions can be employed, however for binomial response, the most commonly used link function is Logit, which is defined as below

$$ \eta = log \left(  \frac{  p }{ 1 - p }  \right).$$

Other link functions used in the Logistic Regression (Binomial GLM) of binomial response are: (i) the Probit Link function and (ii) the Complementary Log-Log function which are defined respectively below

$$\eta_i = \Phi^{-1} (p) \ \ \ \text{and} \ \ \ \eta_i = \mathsf{log} \left( - \mathsf{log} ( 1 - p_i ) \right). $$

```R

> glm( formula, family = binomial(link = probit) )

```

For example, if the logit link function is employed then it holds that 

$$p_i = \frac{ e^{\eta_i} }{ 1 + e^{\eta_i} } \ \ \ \text{and} \ \ \ 1 - p_i = \frac{ 1 }{ 1 + e^{\eta_i}  },$$

and the log-likelihood function becomes

$$\ell( \beta | \boldsymbol{y} ) = \sum_{i=1}^n \bigg[  y_i \eta_i - \eta_i \mathsf{log} \left( 1 + e^{\eta_i} \right) + \mathsf{log} \left( n_i C_{y_i}  \right)  \bigg]  .$$

### Example 3

Consider the implementation of a suitable Generalized Linear Model to the [R](https://www.r-project.org/) dataset ['Anorexia'](https://rdrr.io/cran/PairedData/man/Anorexia.html).

```R

library(MASS) # For the anorexia data set
library(reshape2)

# Load the anorexia data set
data(anorexia)

# Give each person a unique ID
anorexia$ID <- as.factor(1:nrow(anorexia))

# Calculate mean, standard deviation, and count
library(plyr)
anorexia_summary <- ddply(anorexia, c("Treatment", "Time"), summarise,
                          mean = mean(Weight),
                          sd   = sd(Weight),
                          n    = length(Treatment))

names(anorexia_summary)[names(anorexia_summary) == "mean"] <- "Weight"

```

### Example 4 

Consider a high-dimensional Binomial GLM with a response variable being 'attrition' which represents a binary response, that is, a binary variable indicating whether a survey participant has droped from the follow-up study. 

```R

y <- attritionGroup1

## lasso selection
fit <-  glmnet(X, y, family = "binomial")
plot(fit, xvar = "dev", label= TRUE)
predict(fit, newx = X[1:100,], type = "class", s = c(0.05,0.01))

cvfit <-  cv.glmnet(X, y, family = "binomial", type.measure = "class")
plot(cvfit)

cvfit$lambda.min

lasso_log    <- glmnet(X, as.factor(y), alpha = 1, family = "binomial", lambda = cvfit$lambda.min) 
lassosel_log <- coef(lasso_log)[coef(lasso_log)@i+1,]

## return subset selected
out <- list()
out$selected$cvlasso_log <- names(lassosel_log)
out$selected             <- lapply(out$selected, function(x) x[x!="(Intercept)"])

```

## Remarks: 

1. An important aspect after model fitting is model checking and diagnostics. Briefly speaking model checking is about investigating that the assumptions of the model are in agreement with the data. 

2. In practise model diagnostics implies checking for misspecification which includes for instance: 

(i)   Testing the independence and constant variance of the errors. 

(ii)  Testing the linearity of the conditional mean function. 

(iii) Testing for multicollinearity.  

(iv)  Residual Analysis and Outlier Detection. 

3. In order to avoid having correlated explanatory variables (the problem of multicollinearity in classical econometric applications), we need to consider a subset of variables, that is, which the model selection step. Econometric theory can provide some indication when considering such causal analyses. In applied statistics, model selection is often done via the following methodologies: 

(i) Automatic variable selection (such as backward elimintation, forward selection and step-wise selection among others).

(ii) Criterion-based variable selection (such as using the AIC or the Mallow's criterion). An interesting model selection methodology is the method proposed by Hansen et al. (2011), so-called the ['model confidence set'](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA5771). 


# [B]. Sequence Analysis and Logistic Regression

Sequence Analysis is a non-parametric technique particularly useful for statistical inference with longitudinal data of employment and work-family related trajectories. Such data are commonly used in Labour Economics, Social Statistics and Demography and the main scope is to find statistical significant covariates that explain the particular data topologies across time. Although the presence of time-varying covariates requires additional regularity conditions, the use of sequence analysis along with the implementation of a Binomial GLM provides a possible methodology for analysing the trajectories of such Survey Study participants for static data structures (such as a particular cross-sectional or wave dataset, that is, a follow-up study for a given period of time).     

### Example 5

Consider a cross-sectional dataset which includes the employment trajectories of a group of survey participants. Implement the Sequence Analysis methodology as described in the R package ['TraMineR'](http://traminer.unige.ch/index.shtml). 

```R

install.packages("TraMineR")
install.packages("cluster")

library(TraMineR)
library(cluster)

## STEP 1: Variable Extraction and Data Transformations

input  <- read_dta("Longitudinal_2014.dta") 
mydata <- data.frame(input)
start  <- which(colnames(mydata)=="jan2011")
end    <- which(colnames(mydata)=="dec2014")

mydata$countrycode   <- as.factor(mydata$countrycode)
welfarestate         <- countrycode

mydata$male          <- as.factor(mydata$male)
mydata$education2011 <- as.factor(mydata$education2011)
mydata$urban2011     <- as.factor(mydata$urban2011)

## Employment trajectories
mvad.alphab <- c("EFT","EPT","SEM","UNEMPLED","STUDENT", "DOMESTIC", "INACTIVE","MILITARY", "NA")
mydata.seq  <- seqdef(mydata, var=start:end, states=mvad.alphab, labels=c("EFT","EPT","SEM","UNEMPLED","STUDENT","DOMESTIC","INACTIVE","MILITARY","NA"),xtstep = 9)
seqiplot( mydata.seq, title = "Index plot (first 10 sequences)",withlegend = TRUE )

## STEP 2: COMPUTE PAIRWISE OPTIMAL MATCHING (OM) DISTANCES

mvad.om1 <- seqdist(mydata.seq, method = "OM", indel = 1, sm = "TRATE",with.missing = TRUE)

## STEP 3: AGGLOMERATIVE HIERARCHICAL CLUSTERING  

clusterward <- agnes(mvad.om1, diss = TRUE, method = "ward")
mvad.cl4    <- cutree(clusterward, k = 4)
cl4.lab     <- factor(mvad.cl4, labels = paste("Cluster", 1:4))

## STEP 4: VISUALIZE THE CLUSTER PATTERNS 

seqdplot(mydata.seq, group = cl4.lab, border=NA)

# MEAN TIME SPENT IN EACH STATE BY GENDER
seqmtplot( mydata.seq, group = mydata$male, title = "Mean time" )
seqfplot( mydata.seq, group = Cluster3, pbarw = T )


## STEP 5: LONGITUDINAL ENTROPY

par(mfrow = c(1, 2))
entropies <- seqient(mydata.seq)
hist(entropies)

Turbulence <- seqST(mydata.seq)
hist(Turbulence)

summary(entropies)
summary(Turbulence)

tr <- seqtrate(mydata.seq)
round(tr, 2)

submat   <- seqsubm(mydata.seq, method = "TRATE")
dist.om1 <- seqdist(mydata.seq, method = "OM", indel = 1,sm = submat)

## step 6: BINIMIAL GLM IMPLEMENTATION

mb4 <- (cl4.lab == "Cluster 1")

# model 1
model1 <- glm( mb4 ~ male + age + education2011 + ... ,  data = mydata, family = "binomial" )
summary(model1)

```

## References

- Aisenbrey, S., & Fasang, A. (2017). The interplay of work and family trajectories over the life course: Germany and the United States in comparison. American Journal of Sociology, 122(5), 1448-1484.
- Gabadinho, A., Ritschard, G., Mueller, N. S., & Studer, M. (2011). Analyzing and visualizing state sequences in R with TraMineR. Journal of statistical software, 40(4), 1-37.
- Studer, M., & Ritschard, G. (2016). What matters in differences between life trajectories: A comparative review of sequence dissimilarity measures. Journal of the Royal Statistical Society: Series A (Statistics in Society), 179(2), 481-511.
- Studer, M., Ritschard, G., Gabadinho, A., & Müller, N. S. (2011). Discrepancy analysis of state sequences. Sociological Methods & Research, 40(3), 471-510.

# [C]. Proportional Hazard Model

Next we focus on a different application of Applied Statistics in economics and finance, namely the use of the proportional hazard regression model in related empirical studies. In particular, we are interested to model the probability of default (e.g., which is useful for credit scoring or risk management purposes). By definition, the hazard rate is given by 
$$h(t) = \underset{ \delta t \to 0 }{ \mathsf{lim} } = \frac{ \mathbb{P} \left( t \leq T \leq t + \delta t | T \geq t \right) }{ \delta t  }.$$

Then, the probability of survival at time t is given by

$$S(t) = P \left( T \geq t \right) = \mathsf{exp} \left( -  \int_0^t h(u) du \right).$$

For example, the survival function gives the probability an account has not defaulted by some time t after the account has been opended. The probability of default can be affected by macroeconomic conditions such as (i) bank interest rates, (ii) the unemployement index and (iii) the customer earnings. These covariates can be employed as explanatory variables in a Proportional Hazard Model. More specifically, the Cox Hazard specification form is defined as below 
$$h( t, x(t), \beta ) = h_0(t) . \mathsf{exp} \left( \beta. x(t) \right).$$

Then, the log-likelihood function can be computed as
$$\ell_p( \beta ) = \prod_{i=1}^n \left[ \frac{ \mathsf{exp} \left( \beta . x_i(t_i )  \right) }{ \sum_{ j \in R(t_i)} \mathsf{exp} \left( \beta . x_j(t_i )  \right)  } \right]^{c_i}.$$

## References

- Chen, S. (2019). Quantile regression for duration models with time-varying regressors. Journal of Econometrics, 209(1), 1-17.
- Crook, J., & Bellotti, T. (2010). Time varying and dynamic models for default risk in consumer loans. Journal of the Royal Statistical Society: Series A (Statistics in Society), 173(2), 283-305.
- Duffie, D., Eckner, A., Horel, G., & Saita, L. (2009). Frailty correlated default. The Journal of Finance, 64(5), 2089-2123.
- Hahn, J. (1994). The efficiency bound of the mixed proportional hazard model. The Review of Economic Studies, 61(4), 607-629. 

# Further Reading

[1] Zeileis, C. K. A. (2008). Applied Econometrics with R. Springer: New York, NY, USA.

[2] Dunn, P. K., & Smyth, G. K. (2018). Generalized linear models with examples in R (Vol. 53). New York: Springer. 

[3] Chen, X. (2007). Large sample sieve estimation of semi-nonparametric models. Handbook of econometrics, 6, 5549-5632.

 
