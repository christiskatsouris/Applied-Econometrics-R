# Applied-Econometrics-R


A teaching page presenting various aspects related to Applied Econometrics Using R (Drafted July 2022).

# Introduction

The synthesis of heterogenous data and the development of software (or coding procedures), is a combination which is now recognized as data science. Moreover, both the fields of Applied Econometrics and Applied Statistics have a key role in sheding light to our understanding of causal relations and associations by testing statistical hypotheses as well as econometric theory which can support the broader scopes of economic and data sciences in explaining uncertainty and decision making. 

In this teaching page we present four main applications which are commonly presented in empirical economic and finance studies, such that:

(i)   Statistical Analysis of Binary Data based on the Generalized Linear Models (GLMs),

(ii)  Sequence Analysis for evaluating employment and/or work trajectories, 

(iii) Average Treatment Effects Estimation for regression models based on conditional mean specification forms, and 

(iv)  Statistical Inference based on the Proportional Hazard Model which is useful for modelling the probability of default in various settings.

# [A]. Linear and Generalized Linear Models

## [A1]. Linear Models

A model formula in R has the following form:

$$ \mathsf{response} \ \sim \ \mathsf{linear} \ \mathsf{predictor}$$ 

where the response corresponds to the response (or dependent) variable and the linear predictor corresponds to the set of explanatory variables. 

- **summary:** Produces and print a summary of the fit including parameter estimates along with their standard errors and p-values. 
- **coefficients:** Extracts the parameter estimates from the model fit. 
- **family:** The assumed distribution of the response and the relationship between the left and right-hand sides of the model. 


### Example 1 

Consider the [swiss fertility data-set](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/swiss.html) in R (see, also this [tutorial](https://rpubs.com/bmedina17/731785)). 

<img src="https://github.com/christiskatsouris/Applied-Statistics-R/blob/main/data/Rplot.jpeg" width="750"/>

```R

## Without removing any influential points we fit the model
## Usually outliers should be removed before fitting the linear regression model

> linear.model <- lm(Fertility ~ . , data = swiss)
> summary( linear.model )

Call:
lm(formula = Fertility ~ ., data = swiss)

Residuals:
     Min       1Q   Median       3Q      Max 
-15.2743  -5.2617   0.5032   4.1198  15.3213 

Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
(Intercept)      66.91518   10.70604   6.250 1.91e-07 ***
Agriculture      -0.17211    0.07030  -2.448  0.01873 *  
Examination      -0.25801    0.25388  -1.016  0.31546    
Education        -0.87094    0.18303  -4.758 2.43e-05 ***
Catholic          0.10412    0.03526   2.953  0.00519 ** 
Infant.Mortality  1.07705    0.38172   2.822  0.00734 ** 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 7.165 on 41 degrees of freedom
Multiple R-squared:  0.7067,	Adjusted R-squared:  0.671 
F-statistic: 19.76 on 5 and 41 DF,  p-value: 5.594e-10

> anova( linear.model )

Analysis of Variance Table

Response: Fertility
                 Df  Sum Sq Mean Sq F value    Pr(>F)    
Agriculture       1  894.84  894.84 17.4288 0.0001515 ***
Examination       1 2210.38 2210.38 43.0516 6.885e-08 ***
Education         1  891.81  891.81 17.3699 0.0001549 ***
Catholic          1  667.13  667.13 12.9937 0.0008387 ***
Infant.Mortality  1  408.75  408.75  7.9612 0.0073357 ** 
Residuals        41 2105.04   51.34                      
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

```

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

- GLMs extend ordinary (continous response) or Gaussian regression models to non-Gaussian response variable distributions. Generally speaking, GLMs can be applied to a variaty of non-Gaussian forms of data such as Binary Data (or proportions), Categorical Data, Count Data and so on. For this course, we focus on Binary data (binary response variable) and later on to Binary regressors (for the ATE models).   

A Binamial GLM implies that the response variable $y_i$, for i = 1,...,n, is assumed to be Binomially distributed where n is fixed and represents independent trials. In other words, for Binary Data we have that $y_i$ = 1 or 0 where pi represents the probability of success. 

$$Y_i \sim \mathsf{Binomial} \left( n_i, p_i \right), \ \ \ \text{with} \ \ \ \mathbb{P} \left( Y_i = y_i \right) = n_i C_{y_i} p^{y_i} ( 1 - p_i )^{ n_i - y_i }.$$

Thus, the fundamental difference between Logistic Regression and Multiple Linear Regression, (as in classical econometric applications), is that the response is Binomially distributed. Then, the probability pi is modelled as a linear combination of p covariates. 

Therefore, in Generalized Linear Models (GLM), such as the Logistic Regression, we consider two important ingredients:

- $\textbf{Linear predictor:}$ to model linear relationships. 

$$ \eta_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + ... + \beta_p x_{pi}.$$

- $\textbf{Link function:}$ to link the linear predictor to the Binomial probability. Various link functions can be employed, however for binomial response, the most commonly used link function is Logit, which is defined as below

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

3. In order to avoid having correlated explanatory variables (the problem of multicollinearity in classical econometric applications), a commony used approach, especially under the presence of a large number of candidate covariates, is to consider a subset of explantory variables, which is the model selection step. Furthermore, although econometric theory can provide some indication of possible covariates to choose when considering modelling based on causal identification, the model selection step can be implemented using one of the following statistical methodologies: 

(i)  Automatic variable selection (such as backward elimination, forward selection and step-wise selection among others).

(ii) Criterion-based variable selection (such as using the AIC or the C Mallow's criterion). For instance, an interesting model selection methodology is the method proposed by Hansen et al. (2011), so-called ['model confidence set'](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA5771). Further related material on model selection is Section 9.4 (page 221) from Davidson, J. (2000).

> Question: When does an application of a suitable transformation on the response variable contribute to robust statistical inference? 
 
Specifically, this phenomenon refers to the case when the response variable y does not look as a linear function of the predictors. Therefore, an appropriate transformation can be applied in order to restore linearity, or in other words, the Box-Cox family of transformations. In particular, let g be a function belonging in some parametric family indexed by a parameter $\lambda$. Then, the Box-Cox transformation is given by

$$g(y) = \frac{ y^{\lambda} - 1 }{ \lambda }.$$

Then, the MLE methodology can be employed to determine the value of the unknown parameter $\lambda$.

## References

- Box, G. E., & Cox, D. R. (1964). An analysis of transformations. Journal of the Royal Statistical Society: Series B (Methodological), 26(2), 211-243.
- Barro, R. J., & Becker, G. S. (1989). Fertility choice in a model of economic growth. Econometrica: Journal of the Econometric Society, 481-501.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.
- Manski, C. F., & Thompson, T. S. (1989). Estimation of best predictors of binary response. Journal of Econometrics, 40(1), 97-123.

# [B]. Sequence Analysis and Logistic Regression

Sequence Analysis is a non-parametric technique particularly useful for statistical inference with longitudinal data of employment and work-family related trajectories. Such data are commonly used in Labour Economics, Social Statistics and Demography and the main scope is to find statistical significant covariates that explain the particular data topologies across time. Although the presence of time-varying covariates requires additional regularity conditions, the use of sequence analysis along with the implementation of a Logistic Regression (Binomial GLM) provides a possible methodology for analysing the trajectories of such Survey Study participants for static data structures (such as a particular cross-sectional or wave dataset, that is, a follow-up study for a given period of time).     

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
model1 <- glm( mb4 ~ male + age + education + ... ,  data = mydata, family = "binomial" )
summary(model1)

```

## Remarks: 

Various statistics can be employed as dissimilarity measures for comparing the trajectories of sequences. Some of these include:

(a) Longest Common Prefix.

(b) Longest Common Suffix.

(c) Longest Common Subsequence.

(d) Hamming Distance.

Moreover the entropy index (i.e., Gini heterogeneity index) provides a measure of sequence heterogeneity. In other words, it captures the state changes that occur in the sequences of trajectories. Furthermore, the dissimilarity between two sequences can be obtained by considering the necessary operations to transform one sequence into the other. For example, given a longitudinal dataset grouped per European country one might be interested to obtain dissimilarity measures which can provide insights regarding the level of similarity of trajectories of a particular sub-population, such as the young people when followed up. In addition we can construct a logistic regression model across these groups based on a suitable set of covariates in order to obtain statistical evidence explaining the differences of these employment trajectories. 

In the above example implemented in R the clustering of sequences with most similarities based on the OM algorithm has identified four distinct patterns of school to work transitions (i.e., employment trajectories) for the sub-population of young people. Thus, the role of statistical modelling (using Logistic regression) is exactly to provide statistical evidence on how cluster membership depends on certain covariates which capture the main socio-economic characteristics of the particular population.   

In terms of economic theory the above statistical analysis can be expanded in order to address further economic policy research questions, such as: 
- How do monetary shocks affect the transition to the different employment states and in particular in relation to the group clustering obtained from the algorithmic procedures of Sequence Analysis?
- Which macroeconomic factors have explanatory power in explaining these employment transitions? 

In general a macroeconomist might be interested to investigate the impact of various macroeconomic shifts (i.e., structural changes in the economic sence) on individual mobility processes (or latent processes) such as employment trajectories. Neverthless, these particular research questions are beyond the scope of this teaching page, as our main focus is to introduce certain econometric methods and statistical procedures which are particularly useful for related applied and theoretical studies.   

## References

- Aisenbrey, S., & Fasang, A. (2017). The interplay of work and family trajectories over the life course: Germany and the United States in comparison. American Journal of Sociology, 122(5), 1448-1484.
- Bester, C. A., Conley, T. G., & Hansen, C. B. (2011). Inference with dependent data using cluster covariance estimators. Journal of Econometrics, 165(2), 137-151.
- Gabadinho, A., Ritschard, G., Mueller, N. S., & Studer, M. (2011). Analyzing and visualizing state sequences in R with TraMineR. Journal of statistical software, 40(4), 1-37.
- Hansen, B. E., & Lee, S. (2019). Asymptotic theory for clustered samples. Journal of econometrics, 210(2), 268-290.
- Liang, K. Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. Biometrika, 73(1), 13-22.
- Studer, M., & Ritschard, G. (2016). What matters in differences between life trajectories: A comparative review of sequence dissimilarity measures. Journal of the Royal Statistical Society: Series A (Statistics in Society), 179(2), 481-511.
- Studer, M., Ritschard, G., Gabadinho, A., & Müller, N. S. (2011). Discrepancy analysis of state sequences. Sociological Methods & Research, 40(3), 471-510.

## Further Reading (Advanced Topics)

- Hidden Markov Models (to add references and related R package).
- Joao, I. C., Lucas, A., Schaumburg, J., & Schwaab, B. (2022). Dynamic clustering of multivariate panel data. Journal of Econometrics.


# [C]. Average Treatment Effects Estimation 

There is a growing literature on modelling methodologies for Average Treatment Effects which are particularly useful for evaluating economic and health policy outcomes. An R tutorial can be found [here](https://cran.r-project.org/web/packages/targeted/vignettes/ate.html). 

The Average Treatment Effect is defined as below:

$$ATE = \mathbb{E} [ Y(1) ] - \mathbb{E} [ Y(0) ]. $$

Many factors can influence both the response and the treatment covariate. Overall a related measure of interest is the propensity score which is defined as:

$$\Pi (x) = \mathbb{P} ( D = 1 | X = x ) $$

where we assume that the unconfoundedness condition holds. 

## Remarks: 

Notice that the main difference of ATE regression models in comparision to the Binomial GLM (Logistic Regression) is that we are modelling both a response and an explanatory variable which is binary. Additional covariates can be also incorporated but the use of a Generalized Linear Model is not a suitable modelling approach in this case.      

## References

- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs with experimental data. The American economic review, 604-620.
- Hausman, J. A., & Wise, D. A. (1979). Attrition bias in experimental and panel data: the Gary income maintenance experiment. Econometrica: Journal of the Econometric Society, 455-473.
- Huber, M. (2012). Identification of average treatment effects in social experiments under alternative forms of attrition. Journal of Educational and Behavioral Statistics, 37(3), 443-474.
- Ma, X., & Wang, J. (2020). Robust inference using inverse probability weighting. Journal of the American Statistical Association, 115(532), 1851-1860.
- Katsouris, C. (2021). Treatment effect validation via a permutation test in Stata. [arXiv preprint:2110.12268](https://arxiv.org/abs/2110.12268).


# [D]. Proportional Hazard Regression Model

Next we focus on a different application of Applied Statistics in economics and finance, namely the use of the proportional hazard regression model in related empirical studies. In particular, we are interested to model the probability of default (e.g., which is useful for credit scoring or risk management purposes). By definition, the hazard rate is given by 
$$h(t) = \underset{ \delta t \to 0 }{ \mathsf{lim} } = \frac{ \mathbb{P} \left( t \leq T \leq t + \delta t | T \geq t \right) }{ \delta t  }.$$

Then, the probability of survival at time t is given by

$$S(t) = P \left( T \geq t \right) = \mathsf{exp} \left( - \int_0^t h(u) du \right).$$

For example, the survival function gives the probability a portfolio account has not defaulted by some time t after the account has been opended. The probability of default can be affected by macroeconomic conditions such as (i) bank interest rates, (ii) the unemployement rate and (iii) the customer earnings. These covariates can be employed as explanatory variables in a Proportional Hazard Regression Model and account for heterogeneity. 

More specifically, the Cox Hazard specification form is defined as below 
$$h( t, x(t), \beta ) = h_0(t) . \mathsf{exp} \left( \beta^{\prime} . x(t) \right).$$

Then, the log-likelihood function can be computed as
$$\ell_p( \beta ) = \prod_{i=1}^n \left[ \frac{ \mathsf{exp} \left( \beta^{\prime} . x_i(t_i )  \right) }{ \sum_{ j \in R(t_i)} \mathsf{exp} \left( \beta^{\prime} . x_j(t_i )  \right)  } \right]^{c_i}.$$

Furthermore, frailty models which are an extension to the standard survival models (such as the Proportional Hazard model) are random effects models for time-to-event data used to account for unobserved heterogeneity and dependence. 

### Example 6

Consider the R packages ['survival'](https://cran.r-project.org/web/packages/survival/index.html) and ['dynfrail'](https://www.rdocumentation.org/packages/dynfrail/versions/0.5.2/topics/dynfrail) that provide implementations of the Proportional Hazard Model (see also this [tutorial](https://data.princeton.edu/pop509/frailtyr)) and the Dynamic Frailty Model. Begin by deriving the maximum likelihood function for both the hazard proportional model and the frailty model focusing on both the theoretical estimation as well as the computational aspects to that. Consider the simplest model specification initially before adding exogenous covariates to the functional form.  

```R

# Install R packages 
install.packages("survival")
library(survival)

install.packages("dynfrail")
library(dynfrail)

```

## Remarks: 

Some important concepts in Survival Analysis include the following: 

- Cencoring: This terminology implies that the endpoint is not observed for all subjects (e.g., the patient is still alive at the time of analysis, or a portfolio account has not been defaulted at the 'observation' time. Furthermore, there are two types of cencoring: (i) right cencoring and (ii) left cencoring. For instance, the second case which is less frequent implies that the survival times are known only to be less that some value t*. Another interesting example of left cencoring is a portfolio of loas that were previously extended past their original maturities, which could represent distressed loans where the extension was part of a loss mitigation strategy adopted by the lender (see, ['From Originiation to Renegotiation'](https://link.springer.com/article/10.1007/s11146-016-9548-1)).    

- The underline theoretical underpinnings of Survival models are important for understanding how statistical inference can be conducted. Firstly, the actual survival time t must be independent of any mechanism that causes that individual's time to be censored at c < t. In other words, the prognosis for individual alive at time t (think for example the prognosis of a portfolio account not to be defaulted at time t) should not be affected by cencoring at t. Secondly, how censoring is applied is important for the validity of statistical assumptions and the robustness of the results to hold. In particular, a censored patient is representative of those at risk (or a company/portfolio account which is about to default) if censoring happens either at random or at fixed time of analysis. On the other hand, when censoring occurs due to cessation of treatment for example, which results to the deterioration of a condition then the results from the particular patient are not representative of those at risk. In other words, censoring should be independent of illnes and/or tratement.   

> Question: What about attrition which appears in regression models of Average Treatment Effects? Is attrition and censoring the same thing? What are the econometric strategies we need to take into account when considering these two different modelling methodologies?  

- Notice that we mainly consider the implementation of the Proportional Hazard Models such as those of Frailty Models for applications in finance (e.g., retail or behavioural finance). In other words, when survival models are based on 'time to default' data (for example for the purpose of credit scoring), then in practise we can predict not just if a borrower will default but when he/she will default. In such modelling environments we basically refer to these econometric specifications as behavioural models of default since the behavioural characteristics of borrowers are used as explanatory variables (regressors) to forecast the probability of default. 


## References

On Credit Scoring:
- Bellotti, T., & Crook, J. (2013). Forecasting and stress testing credit card default using dynamic models. International Journal of Forecasting, 29(4), 563-574.
- Banasik, J., Crook, J. N., & Thomas, L. C. (1999). Not if but when will borrowers default. Journal of the Operational Research Society, 50(12), 1185-1190.
- Crook, J., & Bellotti, T. (2010). Time varying and dynamic models for default risk in consumer loans. Journal of the Royal Statistical Society: Series A (Statistics in Society), 173(2), 283-305.
- Duffie, D., Eckner, A., Horel, G., & Saita, L. (2009). Frailty correlated default. The Journal of Finance, 64(5), 2089-2123.
- Dirick, L., Claeskens, G., & Baesens, B. (2017). Time to default in credit scoring using survival analysis: a benchmark study. Journal of the Operational Research Society, 68(6), 652-665.
- Kiefer, N. M. (2010). Default estimation and expert information. Journal of Business & Economic Statistics, 28(2), 320-328.
- Kiefer, N. M. (2011). Default estimation, correlated defaults, and expert information. Journal of Applied Econometrics, 26(2), 173-192.

On Properties of Hazard functions:
- Chen, S. (2019). Quantile regression for duration models with time-varying regressors. Journal of Econometrics, 209(1), 1-17.
- Hahn, J. (1994). The efficiency bound of the mixed proportional hazard model. The Review of Economic Studies, 61(4), 607-629. 

On Probability Theory:
- Enki, D. G., Noufaily, A., & Farrington, C. P. (2014). A time-varying shared frailty model with application to infectious diseases. The Annals of Applied Statistics, 430-447.
- Gjessing, Håkon K., Odd O. Aalen, and Nils Lid Hjort. "Frailty models based on Lévy processes." Advances in Applied Probability 35.2 (2003): 532-550.
- Singpurwalla, N. D. (1995). Survival in dynamic environments. Statistical science, 86-103.

# Reading List

$\textbf{[1]}$ Zeileis, C. K. A. (2008). Applied Econometrics with R. Springer: New York, NY, USA.

$\textbf{[2]}$  Dunn, P. K., & Smyth, G. K. (2018). Generalized linear models with examples in R (Vol. 53). New York: Springer. 

$\textbf{[3]}$  Millimet, D., Smith, J., & Vytlacil, E. (2008). Modelling and evaluating treatment effects in econometrics. Emerald Group Publishing.

$\textbf{[4]}$  Kleinbaum, D. G., & Klein, M. (2012). Survival analysis: a self-learning text (Vol. 3). New York: Springer.

$\textbf{[5]}$  Aalen, O., Borgan, O., & Gjessing, H. (2008). Survival and event history analysis: a process point of view. Springer Science & Business Media.

$\textbf{[6]}$  Davidson, J. (2000). Econometric theory. John Wiley & Sons.

# Learning Outcomes

1. Understand the basic properties of Generalized Linear Models (Binomial GLM).
2. Be able to obtain the parameters of Binomial GLMs from the exponential family of distributions. 
3. Be able to construct statistical tests and use model diagnostic tools for GLMs.
4. Understand the basic properties of Average Treatment Effect Models. 
5. Be able to apply related econometric tools for ATE estimation such as IPW.  
6. Understand the basic properties of Proportional Hazard Models. 
7. Be able to apply parameter estimation methodologies (MLE, EM etc.)
8. Be able to implement the above modelling techniques to economic, finance and actuarial data. 
9. Be able to use Statistical/Econometric Programming Software such as R, Matlab or Stata. 

# Disclaimer 

The author (Christis G. Katsouris) declares no conflicts of interest. 

The proposed Course Syllabus is currently under development and has not been officially undergone quality checks. All rights reserved.  

Any errors or omissions are the responsibility of the author.

# How to Cite a Website 

See: https://www.mendeley.com/guides/web-citation-guide/ 
