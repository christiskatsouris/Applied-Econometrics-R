# Applied-Econometrics-R


A teaching page presenting various aspects related to Applied Econometrics Using R (Drafted July 2022).

### Course Overview:

The main philosophy with this course is to combine traditional statistical modelling methodologies such as GLMs (Binary data only) with modern econometric specifications suitable for both cross-sectional and time series data. Emphasis with this course is to introduce some important economic and finance applications such as the modelling of employment trajectories as well as the modelling of the probability of corporate/firm default. Furthermore, we introduce state-of-the-art techniques and programming capabilities with R for each topic covered.

# Introduction

The synthesis of heterogenous data and the development of software (or coding procedures), is a combination which is now recognized as data science. Moreover, both the fields of Applied Econometrics and Applied Statistics have a key role in sheding light to our understanding of causal relations and associations by testing statistical hypotheses as well as econometric theory which can support the broader scopes of economic and data sciences in explaining uncertainty and decision making. 

In this teaching page we present four main applications which are commonly presented in empirical economic and finance studies, such that:

$\textbf{(i)}$   Statistical Analysis of Binary Data based on the Generalized Linear Models (GLMs),

$\textbf{(ii)}$  Sequence Analysis for evaluating employment trajectories, 

$\textbf{(iii)}$ Average Treatment Effects Estimation based on conditional mean specification forms,

$\textbf{(iv)}$  Statistical Inference based on the Proportional Hazard Model which is useful for modelling the probability of default in various settings.

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


<p align="center">

<img src="https://github.com/christiskatsouris/Applied-Statistics-R/blob/main/data/Rplot.jpeg" width="750"/>

</p>  

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

## Remarks: 

- Notice that we focus on the Binary GLM which is based on binary data. For instance, when considering other forms of data such as count data which is particularly useful in various actuarial and insurance applications, then we will need to consider the Poisson GLM.
- A Poisson GLM has a different specification in comparison to the Binomial GLM, however it still belongs to the Exponential Family with corresponding parameters such as canonical parameter, link functions which can be mapped (might add these aspects in the future). 

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

$\textbf{(a).}$ An important aspect after model fitting is model checking and diagnostics. Briefly speaking model checking is about investigating that the assumptions of the model are in agreement with the data. 

$\textbf{(b).}$  In practise model diagnostics implies checking for misspecification which includes for instance: 

(i)   Testing the independence and constant variance of the errors. 

(ii)  Testing the linearity of the conditional mean function. 

(iii) Testing for multicollinearity.  

(iv)  Residual Analysis and Outlier Detection. 

In summary the above testing procedures are concerned with tests regarding the violations of commonly used assumptions in linear regression models. Examples include: testing for violations of linearity assumption, testing for violation of constant variance assumption and so on (see,  [Sage et al. (2022)](https://www.tandfonline.com/doi/full/10.1080/00031305.2022.2107568)). Lastly, residual analysis based on fitted linear regression models as well as outlier detection are considered to be classical statistical model diagnostic tools. 

$\textbf{(c).}$  Furthermore, in order to avoid having correlated explanatory variables (the problem of multicollinearity in classical econometric applications), a commony used approach, especially under the presence of a large number of candidate covariates, is to consider a subset of explantory variables, which is the model selection step. Furthermore, although econometric theory can provide some indication of possible covariates to choose when considering modelling based on causal identification, the model selection step can be implemented using one of the following statistical methodologies: 

(i)  Automatic variable selection (such as backward elimination, forward selection and step-wise selection among others).

(ii) Criterion-based variable selection (such as using the AIC or the C Mallow's criterion). For instance, an interesting model selection methodology is the method proposed by Hansen et al. (2011), so-called ['model confidence set'](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA5771). Further related material on model selection is Section 9.4 (page 221) from Davidson, J. (2000).

> Question: When does an application of a suitable transformation on the response variable contribute to robust statistical inference? 
 
Specifically, this phenomenon refers to the case when the response variable y does not look as a linear function of the predictors. Therefore, an appropriate transformation can be applied in order to restore linearity, or in other words, the Box-Cox family of transformations. In particular, let g be a function belonging in some parametric family indexed by a parameter $\lambda$. Then, the Box-Cox transformation is given by

$$g(y) = \frac{ y^{\lambda} - 1 }{ \lambda }.$$

Then, the MLE methodology can be employed to determine the value of the unknown parameter $\lambda$.

- Classical theory for significance testing in linear regression operates on two fixxed nested models. However, in the case of LASSO selection the signicance of the covariates included in the model seems to be cumbersome and many times needs special consideration since it may not be following traditional asymptotic theory. Consider for instance the stepwise regression procedure where we start with the null model and we enter predictors one at a time, at each step choosing the predictor $j$ that gives the largest drop in residual sum of squares

$$ R_j = \frac{ RSS_M -  RSS_{ M \cup \{j\} } }{ \sigma^2 }.$$

- In particular, [Lockhart et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4285373/) introduce the covariance test which provides a way to test for the significance of additional predictors using the LASSO optimization function. Furthermore, the authors in order to construct a test statistic which can be generalized under different model estimations, a strong assumption regarding the hypothesis is made, that is, there is the assumption that the parameter space under the null hypothesis is a random set which can be approximated under suitable distributional assumptions. However, the problem of assessing significance in an adaptive linear model fit by the lasso is a difficult one and different approaches have been taken in order to overcome this computational challenge. 

## References

- Box, G. E., & Cox, D. R. (1964). An analysis of transformations. Journal of the Royal Statistical Society: Series B (Methodological), 26(2), 211-243.
- Barro, R. J., & Becker, G. S. (1989). Fertility choice in a model of economic growth. Econometrica: Journal of the Econometric Society, 481-501.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.
- Lockhart, R., Taylor, J., Tibshirani, R. J., & Tibshirani, R. (2014). A significance test for the lasso. Annals of statistics, 42(2), 413.
- Manski, C. F., & Thompson, T. S. (1989). Estimation of best predictors of binary response. Journal of Econometrics, 40(1), 97-123.
- Sage, A. J., Liu, Y., & Sato, J. (2022). From Black Box to Shining Spotlight: Using Random Forest Prediction Intervals to Illuminate the Impact of Assumptions in Linear Regression. The American Statistician, (just-accepted), 1-26.

# [B]. Sequence Analysis and Logistic Regression

Sequence Analysis is a non-parametric technique particularly useful for statistical inference with longitudinal data of employment and work-family related trajectories. Such data are commonly used in Labour Economics, Social Statistics and Demography and the main scope is to find statistical significant covariates that explain the particular data topologies across time (Katsouris C. and Ierodiakonou C. (2022)). 

Although the presence of time-varying covariates requires additional regularity conditions, the use of sequence analysis along with the implementation of a Logistic Regression (Binomial GLM) provides a possible methodology for analysing the trajectories of such Survey Study participants for static data structures, such as a particular cross-sectional or wave dataset, that is, a follow-up study for a given period of time.     

### Example 5

Consider a cross-sectional dataset which includes the employment trajectories of a group of survey participants.

Before fitting any statistical model to any form of data structure, a good practice is to do some preliminary data analysis in order to understand the nature of the given dataset. An important variable which provides insights regarding the socio-economic status across European countries is the income distribution. Furthermore, there is a vast literature on formal statistical methodologies that can be employed in order to obtain statistical significante evidence of stochastic dominance of cross-country income distributions.

<p align="center">

<img src="https://github.com/christiskatsouris/Applied-Statistics-R/blob/main/data/Income_Distributions.jpg" width="850"/>

</p>  

> Figure above displays the income distributions across main European countries based on reported annual income of the survey participants (whole population, i.e., no age restrictions) for the year 2014. Data resource: EU-SILC survey study. Data analysis is part of work done while being a Research Assistant (Special Scientist) at UCY during the academic year 2016-2017. 

Similar data analysis could be also applied to other important economic metrics such as the aggregate consumption and the aggregate savings density plots (via the use of a suitable proxy variable) across main European economics in the core and/or in the periphery. In particular, the relation of consumption and savings especially during periods of 'structural change' (i.e., by comparing estimates during periods of economic turbulence versus economic recovery) can provide insights regarding economic conditions and thus contribute to economic policy making. In particular, information regarding the behaviour of households can be obtained by considering such statistics during the pandemic. For example, one can say that indeed consumption of households across the Eurozone area has been supported by savings. On the other hand, one can also argue that active labour participants during the pandemic were paid to work from home, which contributed to an increase of electronic commerce (via e-commerce websites such as Amazon (see, Houde et al. (2022)), consequently contributing to a decrease of households savings, when accounting for necessary monetary value corrections. In other words, during periods of structural changes related policies need to have a holistic view of various interdependent effects especially when considering societal structure as well (indicative references include: Johnson (1951) and Direr (2001)).

### References

- Johnson, H. G. (1951). A note on the effect of income redistribution on aggregate consumption with interdependent consumer preferences. Economica, 18(71), 295-297.
- Direr, A. (2001). Interdependent preferences and aggregate saving. Annales d'Economie et de Statistique, 297-308.
- Houde, J. F., Newberry, P., & Seim, K. (2022). Nexus Tax Laws and Economies of Density in E-Commerce: A Study of Amazon’s Fulfillment Center Network. Fortcoming Econometrica. 


Next, we implement the Sequence Analysis methodology as described in the R package ['TraMineR'](http://traminer.unige.ch/index.shtml). 

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

Moreover the entropy index (i.e., Gini heterogeneity index) provides a measure of sequence heterogeneity. In other words, it captures the state changes that occur in the sequences of trajectories. Furthermore, the dissimilarity between two sequences can be obtained by considering the necessary operations to transform one sequence into the other. For example, given a longitudinal dataset grouped per European country one might be interested to obtain dissimilarity measures which can provide insights regarding the level of similarity of trajectories of a particular sub-population, such as the young people when followed up. In addition we can construct a logistic regression model across these groups based on a suitable set of covariates in order to obtain statistical evidence explaining the differences of these employment trajectories. The entropy measure can be computed as below

$$H_t = \sum_{j=1}^q p_{tj} ln (  p_{tj} ),$$

where q = 1,...,7 (the number of different states) and t = 1,...,48 time periods. The (aggregate) entropy statistic provides a statistical measure of the smoothness of transitions between the different states.  

<p align="center">

<img src="https://github.com/christiskatsouris/Applied-Statistics-R/blob/main/data/graphs_entropy_measure.jpg" width="785"/>

</p>  

> The entropy measure can be computed for each individual (shows the transition rates for moving between the employment states). Then, we can plot the histogram for the entropies of all individuals. In particular, there is indication for a proportion of the individuals with entropy rate around 0.3, however we are interested to identify the states of which there is often transitions to/from as well as the characteristics of individuals with frequent transitions. In addition the figure above shows the aggregate entropy measure for main European countries (EU periphery). The estimation of the aggregated entropy measure is based on monthly time-spanned data from the longitudinal wave between the years 2011 to 2014 (48 observations).  

Furthermore, in the above example implemented in R the clustering of sequences with most similarities based on the OM algorithm has identified four distinct patterns of school to work transitions (i.e., employment trajectories) for the sub-population of young people. Thus, the role of statistical modelling (using Logistic regression) is exactly to provide statistical evidence on how cluster membership depends on certain covariates which capture the main socio-economic characteristics of the particular population.   

<p align="center">

<img src="https://github.com/christiskatsouris/Applied-Statistics-R/blob/main/data/Clustering.jpg" width="785"/>

</p>  

> Figure above corresponds to the Longitudinal wave of 2014 for the sub-population with age 18-22. Data resource: EU-SILC survey study. Data analysis is part of work done while being a Research Assistant (Special Scientist) at UCY during the academic year 2016-2017.

```R

> #model 1
> model1 <- glm(mb4 ~ male + age + education2011 + educf2011 + empf_jan2011_n,  data = mydata, family = "binomial")
> summary(model1)

Call:
glm(formula = mb4 ~ male + age + education2011 + educf2011 + 
    empf_jan2011_n, family = "binomial", data = mydata)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.6858  -0.5605  -0.4084  -0.3126   2.8272  

Coefficients:
                 Estimate Std. Error z value Pr(>|z|)    
(Intercept)       1.02233    0.80669   1.267 0.205041    
male1             0.26125    0.12087   2.161 0.030666 *  
age19             0.40290    0.20457   1.969 0.048896 *  
age20             0.72207    0.19677   3.670 0.000243 ***
age21             1.07375    0.19410   5.532 3.16e-08 ***
age22             1.06336    0.20224   5.258 1.46e-07 ***
education2011  2  0.09087    0.68433   0.133 0.894360    
education2011  3 -1.48391    0.63431  -2.339 0.019314 *  
education2011  4 -2.20318    0.75225  -2.929 0.003403 ** 
educf2011  2     -0.95206    0.49244  -1.933 0.053192 .  
educf2011  3     -1.96732    0.47881  -4.109 3.98e-05 ***
educf2011  4     -3.09990    0.52212  -5.937 2.90e-09 ***
empf_jan2011_n1  -0.41636    0.13466  -3.092 0.001988 ** 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 2207.6  on 2769  degrees of freedom
Residual deviance: 1959.9  on 2757  degrees of freedom
  (1372 observations deleted due to missingness)
AIC: 1985.9

Number of Fisher Scoring iterations: 5

```

In terms of economic theory the above statistical analysis can be expanded in order to address further economic policy research questions, such as: 
- How do monetary shocks affect the transition to the different employment states and in particular in relation to the group clustering obtained from the algorithmic procedures of Sequence Analysis?
- Which macroeconomic factors have explanatory power in explaining these employment transitions? 

In general a macroeconomist might be interested to investigate the impact of various macroeconomic shifts (i.e., structural changes in the economic sense) on individual mobility processes (or latent processes) such as employment trajectories. However, these particular research questions are beyond the scope of this teaching page, as our main focus is to introduce certain econometric methods and statistical procedures which are particularly useful for related applied and theoretical studies. 

On the other hand, an econometrician might be interested in the following research questions:

- Applying a suitable statistical procedure (e.g., econometric identification) which can be employed to determine the cluster membership of certain sub-populations, while ensuring robustness to various unobserved effects that can affect inference such as: selection bias, clustered errors, changing dynamics across longitudinal  waves. 

- Another relevant research question is to find suitable mechanisms in order to account for the possible presence of endogenous sorting/oselection, ensuring this way the implementation of robust econometric methodologies and estimation results (see, [Berger, Y. and Patilea, V. (2021)](https://www.sciencedirect.com/science/article/pii/S2452306221001489) and [Horrace et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0304407615001748)).

- Lastly, an implementation of a job-search-matching modeling approach that accommodates such employment trajectories, especially from the viewpoint of cluster membership (see, Joao, I. et al.  (2022)), would be an additional aspect worth investigating further but it would require both a more complex econometric identification and estimation strategy as well as further information from a suitable dataset. 

> According to [Carneiroa et al. (2022)](https://www.sciencedirect.com/science/article/pii/S0304407621002839), the main idea in these models is that a worker’s future earnings and employment prospects will depend on his/her personal characteristics that are transferable across jobs, the job-match specific component of the current job, the job-to-job transitions over the life cycle, and the unemployment shocks. In this framework wage persistence plays an important role in the sense that an individual’s job search aspirations are largely determined by the job-specific component of the current job, which depends on previous wage offers, with job changes induced by offers of higher wages.

## References

- Abbott, A., & Hrycak, A. (1990). Measuring resemblance in sequence data: An optimal matching analysis of musicians' careers. American Journal of Sociology, 96(1), 144-185.
- Aisenbrey, S., & Fasang, A. (2017). The interplay of work and family trajectories over the life course: Germany and the United States in comparison. American Journal of Sociology, 122(5), 1448-1484.
- Berger, Y. G., & Patilea, V. (2021). A semi-parametric empirical likelihood approach for conditional estimating equations under endogenous selection. Econometrics and Statistics.
- Bester, C. A., Conley, T. G., & Hansen, C. B. (2011). Inference with dependent data using cluster covariance estimators. Journal of Econometrics, 165(2), 137-151.
- Carneiro, A., Portugal, P., Raposo, P., & Rodrigues, P. M. (2022). The persistence of wages. Journal of Econometrics.
- Frech, A., & Damaske, S. (2019). Men’s income trajectories and physical and mental health at midlife. American Journal of Sociology, 124(5), 1372-1412.
- Gabadinho, A., Ritschard, G., Mueller, N. S., & Studer, M. (2011). Analyzing and visualizing state sequences in R with TraMineR. Journal of Statistical Software, 40(4), 1-37.
- Gibbons, R. D., & Hedeker, D. (2000). Applications of mixed-effects models in biostatistics. Sankhyā: The Indian Journal of Statistics, Series B, 70-103.
- Horrace, W. C., Liu, X., & Patacchini, E. (2016). Endogenous network production functions with selectivity. Journal of Econometrics, 190(2), 222-232.
- Katsouris C. & Ierodiakonou C. (2022). A Sequence Analysis of Employment Trajectories with Cluster-Based Logistic GLM. Department of Business and Public Administration. University of Cyprus. Working paper.   
- Lersch, P. M., Schulz, W., & Leckie, G. (2020). The variability of occupational attainment: How prestige trajectories diversified within birth cohorts over the twentieth century. American Sociological Review, 85(6), 1084-1116.
- Liang, K. Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. Biometrika, 73(1), 13-22.
- Murphy, K., Murphy, T. B., Piccarreta, R., & Gormley, I. C. (2021). Clustering longitudinal life‐course sequences using mixtures of exponential‐distance models. Journal of the Royal Statistical Society: Series A (Statistics in Society).
- McLeod, J. D., & Fettes, D. L. (2007). Trajectories of failure: The educational careers of children with mental health problems. American Journal of Sociology, 113(3), 653-701.
- Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins. Journal of molecular biology, 48(3), 443-453.
- Stone, J., Netuveli, G., & Blane, D. (2008). Modelling socioeconomic trajectories: An optimal matching approach. International Journal of Sociology and Social Policy.
- Studer, M., & Ritschard, G. (2016). What matters in differences between life trajectories: A comparative review of sequence dissimilarity measures. Journal of the Royal Statistical Society: Series A (Statistics in Society), 179(2), 481-511.
- Studer, M., Ritschard, G., Gabadinho, A., & Müller, N. S. (2011). Discrepancy analysis of state sequences. Sociological Methods & Research, 40(3), 471-510.

## Further Reading (Advanced Topics)

- Hidden Markov Models (to add references and related R package). 
In particular, when considering the modelling aspects of 'Dynamic Employment Trajectories', that is, time-varying to track gradual changes in cluster characteristics over time, then one has to consider modelling the transitions from one cluster to the next through time. A suitable statistical methodology that captures such phenomena is the Hidden Markov Model (HMM). Specifically, under the assumption that the latent (hidden) states evolve over time, then a HMM can characterize these transition dynamics. Although, pragmatically such dynamics would be more challenging to capture from a survey study, as we would need to identify the same study participants from 2 consequentive longitudinal waves, it could still be a possible modelling strategy with fruitful results. 

- Hansen, B. E., & Lee, S. (2019). Asymptotic theory for clustered samples. Journal of Econometrics, 210(2), 268-290.

- Dzemski, A., & Okui, R. (2017). Confidence set for group membership. arXiv preprint arXiv:1801.00332.

- Joao, I. C., Lucas, A., Schaumburg, J., & Schwaab, B. (2022). Dynamic clustering of multivariate panel data. Journal of Econometrics.

- Lumsdaine, R. L., Okui, R., & Wang, W. (2022). Estimation of panel group structure models with structural breaks in group memberships and coefficients. Journal of Econometrics.


# [C]. Average Treatment Effects Estimation 

There is a growing literature on modelling methodologies for Average Treatment Effects which are particularly useful for evaluating economic and health policy interventions/outcomes. Specifically, a treatment effect is how an outcome of interest, such as earnings, is affected by some treatment, such as a job training program. We review the main components of the particular framework with main focus the key aspects of statistical etimation and inference. 

Let D denote a treatment indicator, equal to 1 if the survey participant is treated and 0 otherwise. For example, D = 1 might correspond to enrollement in some training program or to some medical treatment. Then, the Average Treatment Effect is defined as below:

$$ATE = \mathbb{E} [ Y(1) ] - \mathbb{E} [ Y(0) ]. $$

Many factors can influence both the response and the treatment covariate. Overall a related measure of interest is the propensity score which is defined as:

$$\Pi (x) = \mathbb{P} ( D = 1 | X = x ) $$

where we assume that the unconfoundedness condition holds. 

## Remarks: 

- First, notice that the main difference of ATE regression models in comparision to the Binomial GLM (Logistic Regression) is that we are modelling both a response and an explanatory variable which is binary. Furthermore, additional covariates can be also incorporated but the use of a Generalized Linear Model is not a suitable modelling approach in this case.      

- Second, the notion of 'attrition', especially in longitudinal studies requires specific modeling techniques. According to Gibbons and Hedeker (2000), various methods have been proposed to handle missing data in longitudinal studies (Heckman, 1976). Furthermore, these alternative approaches are termed 'selective models' and involve two stages which are either performed separately or iteratively. In particular, the first stage is to develop a predictive model for whether or not a subject drops out, using variables obtained prior to the dropout often the variables measured at baseline. This model of dropout provides a predicted dropout probability or propensity for each subject. Then, these dropout propensity scores are used in the second stage longitudinal data model as covariate to adjust for the potential influence of dropout.    

An R tutorial with implementations of these models can be found [here](https://cran.r-project.org/web/packages/targeted/vignettes/ate.html).

```R

# Examples in R


```


## References

- Conti, G., Mason, G., & Poupakis, S. (2019). The Developmental Origins of Health Inequality. IZA DP 12448 and IFS WP 19-17. In the Oxford Research Encyclopedia of Health Economics (OUP), August 2019.
- Conti, G., Heckman, J. J., & Pinto, R. (2016). The effects of two influential early childhood interventions on health and healthy behaviour. The Economic Journal, 126(596), F28-F65.
- Huber, M. (2012). Identification of average treatment effects in social experiments under alternative forms of attrition. Journal of Educational and Behavioral Statistics, 37(3), 443-474.
- Hausman, J. A., & Wise, D. A. (1979). Attrition bias in experimental and panel data: the Gary income maintenance experiment. Econometrica: Journal of the Econometric Society, 455-473.
- Heckman, J. J. (1976). The common structure of statistical models of truncation, sample selection and limited dependent variables and a simple estimator for such models. Annals of Economic and Social measurement, Volume 5, number 4 (pp. 475-492).
- Katsouris, C. (2021). Treatment effect validation via a permutation test in Stata. [arXiv preprint:2110.12268](https://arxiv.org/abs/2110.12268).
- Ma, X., & Wang, J. (2020). Robust inference using inverse probability weighting. Journal of the American Statistical Association, 115(532), 1851-1860.
- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs with experimental data. The American economic review, 604-620.


## Further Reading (Advanced Topics)

- Chernozhukov, V., Newey, W. K., & Singh, R. (2022). Automatic debiased machine learning of causal and structural effects. Econometrica, 90(3), 967-1027.


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

- Cencoring: This terminology implies that the endpoint is not observed for all subjects (e.g., the patient is still alive at the time of analysis, or a portfolio account has not been defaulted at the 'observation' time. Furthermore, there are two types of cencoring: (i) right cencoring and (ii) left cencoring. For instance, the second case which is less frequent implies that the survival times are known only to be less that some value $t^*$. Another interesting example of left cencoring is a portfolio of loas that were previously extended past their original maturities, which could represent distressed loans where the extension was part of a loss mitigation strategy adopted by the lender (see, ['From Originiation to Renegotiation'](https://link.springer.com/article/10.1007/s11146-016-9548-1)).    

- The underline theoretical underpinnings of Survival models are important for understanding how statistical inference can be conducted. Firstly, the actual survival time t must be independent of any mechanism that causes that individual's time to be censored at $c < t$. In other words, the prognosis for individual alive at time $t$ (think for example the prognosis of a portfolio account not to be defaulted at time $t$) should not be affected by cencoring at $t$. Secondly, how censoring is applied is important for the validity of statistical assumptions and the robustness of the results to hold. In particular, a censored patient is representative of those at risk (or a company/portfolio account which is about to default) if censoring happens either at random or at fixed time of analysis. On the other hand, when censoring occurs due to cessation of treatment for example, which can result to the deterioration of a condition, then the results from the particular patient are not representative of those at risk. In other words, censoring should be independent of illness and/or the treatement to ensure valid statistical inference.   

> Question: What about attrition which appears in regression models of Average Treatment Effects? Is attrition and censoring equivalent from the survey sampling  perspective? What are the econometric strategies we need to consider when accounting for these aspects based on these two different modelling methodologies?  

- Notice that we mainly consider the implementation of the Proportional Hazard Models such as those of Frailty Models for applications in finance (e.g., retail or behavioural finance). In other words, when survival models are based on 'time to default' data (for example for the purpose of credit scoring), then in practise we can predict not just if a borrower will default but when he/she will default. In such modelling environments we basically refer to these econometric specifications as behavioural models of default since the behavioural characteristics of borrowers are used as explanatory variables (regressors) to forecast the probability of default. 

- Lastly the study of censored processes has various important applications from the time series perspective as well, such as the time series modelling of mortality rates, monetrary policy rates close to the zero lower bound etc.  


## References

On Credit Scoring and Default Models:
- Agosto, A., Cavaliere, G., Kristensen, D., & Rahbek, A. (2016). Modeling corporate defaults: Poisson autoregressions with exogenous covariates (PARX). Journal of Empirical Finance, 38, 640-663.
- Brettschneider, J., & Burgess, M. (2017). Using a frailty model to measure the effect of covariates on the disposition effect. Department of Statistics. University of Warwick. Working paper.
- Bellotti, T., & Crook, J. (2013). Forecasting and stress testing credit card default using dynamic models. International Journal of Forecasting, 29(4), 563-574.
- Banasik, J., Crook, J. N., & Thomas, L. C. (1999). Not if but when will borrowers default. Journal of the Operational Research Society, 50(12), 1185-1190.
- Crook, J., & Bellotti, T. (2010). Time varying and dynamic models for default risk in consumer loans. Journal of the Royal Statistical Society: Series A (Statistics in Society), 173(2), 283-305.
- Duffie, D., Eckner, A., Horel, G., & Saita, L. (2009). Frailty correlated default. The Journal of Finance, 64(5), 2089-2123.
- Dirick, L., Claeskens, G., & Baesens, B. (2017). Time to default in credit scoring using survival analysis: a benchmark study. Journal of the Operational Research Society, 68(6), 652-665.
- Kiefer, N. M. (2010). Default estimation and expert information. Journal of Business & Economic Statistics, 28(2), 320-328.
- Kiefer, N. M. (2011). Default estimation, correlated defaults, and expert information. Journal of Applied Econometrics, 26(2), 173-192.

## Further Reading (Advanced Topics)

On Properties of Hazard functions:
- Chen, S. (2019). Quantile regression for duration models with time-varying regressors. Journal of Econometrics, 209(1), 1-17.
- Hahn, J. (1994). The efficiency bound of the mixed proportional hazard model. The Review of Economic Studies, 61(4), 607-629. 

On Probability Theory:
- Enki, D. G., Noufaily, A., & Farrington, C. P. (2014). A time-varying shared frailty model with application to infectious diseases. The Annals of Applied Statistics, 430-447.
- Gjessing, Håkon K., Odd O. Aalen, and Nils Lid Hjort. "Frailty models based on Lévy processes." Advances in Applied Probability 35.2 (2003): 532-550.
- Singpurwalla, N. D. (1995). Survival in dynamic environments. Statistical science, 86-103.

# Reading List

$\textbf{[1]}$  Davidson, J. (2000). Econometric theory. John Wiley & Sons.

$\textbf{[2]}$  Dunn, P. K., & Smyth, G. K. (2018). Generalized linear models with examples in R (Vol. 53). New York: Springer. 

$\textbf{[3]}$ Zeileis, C. K. A. (2008). Applied Econometrics with R. Springer: New York, NY, USA.

$\textbf{[4]}$  Millimet, D., Smith, J., & Vytlacil, E. (2008). Modelling and evaluating treatment effects in econometrics. Emerald Group Publishing.

$\textbf{[5]}$  Aalen, O., Borgan, O., & Gjessing, H. (2008). Survival and event history analysis: a process point of view. Springer Science & Business Media.

$\textbf{[6]}$  Kleinbaum, D. G., & Klein, M. (2012). Survival analysis: a self-learning text (Vol. 3). New York: Springer.

$\textbf{[7]}$ Charu C. Aggarwal (2014). Data Classification: Algorithms and Applications 2014. Data Classification: Algorithms and Applications. IBM T.J Watson Research Center, New York, USA. CRC Press.  

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

Any views and opinions expressed herein are those of the author. The author accepts no liability for any loss or damage a person suffers because that person has directly or indirectly relied on any information found on this website.

# Acknowledgments

The author greatfully acknowledges financial support from Graduate Teaching Assistantships at the School of Economic, Social and Political Sciences of the University of Southampton as well as funding from Research Grants of various interdisciplinary Centers of research excellence based at the University of Cyprus (UCY) as well as at University College London (UCL). Furthermore, the author gratefully acknowledges financial support from the Vice-Chancellor's PhD Scholarship of the University of Southampton, for the duration of the academic years 2018 to 2021. 

If you are interested to collaborate on any of the topics discussed in this teaching page, don't hesitate to contact me at christiskatsouris@gmail.com

# Historical Background

> Standing on the shoulders of giants.
> 
> $\textit{''If I have been able to see further, it was only because I stood on the shoulders of giants."}$
> $- \text{Isaac Newton, 1676}$ 

$\textbf{David Cox}$ (15 July 1924 – 18 January 2022) was a British statistician and educator. His wide-ranging contributions to the field of statistics included introducing logistic regression, the proportional hazards model and the Cox process, a point process named after him. He was a professor of statistics at Birkbeck College, London, Imperial College London and the University of Oxford, and served as Warden of Nuffield College, Oxford. The first recipient of the International Prize in Statistics, he also received the Guy, George Box and Copley medals, as well as a knighthood. (Source: [Wikipedia](https://en.wikipedia.org/wiki/David_Cox_(statistician))). 

$\textbf{Ronald Fisher}$ (17 February 1890 – 29 July 1962) was a British polymath who was active as a mathematician, statistician, biologist, geneticist, and academic. For his work in statistics, he has been described as "a genius who almost single-handedly created the foundations for modern statistical science" and "the single most important figure in 20th century statistics". In genetics, his work used mathematics to combine Mendelian genetics and natural selection; this contributed to the revival of Darwinism in the early 20th-century revision of the theory of evolution known as the modern synthesis. For his contributions to biology, Fisher has been called "the greatest of Darwin’s successors". From 1919, he worked at the Rothamsted Experimental Station for 14 years; there, he analysed its immense body of data from crop experiments since the 1840s, and developed the analysis of variance (ANOVA). He established his reputation there in the following years as a biostatistician. Together with J. B. S. Haldane and Sewall Wright, Fisher is known as one of the three principal founders of population genetics. His contributions to statistics include promoting the method of maximum likelihood and deriving the properties of maximum likelihood estimators, fiducial inference, the derivation of various sampling distributions, founding principles of the design of experiments, and much more. (Source: [Wikipedia](https://en.wikipedia.org/wiki/Ronald_Fisher)).
