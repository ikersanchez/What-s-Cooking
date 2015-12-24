#################################
####### What´s cooking? #########
#################################

#This competition asks you to predict 
#the category of a dish's cuisine given a list of its ingredients

#Categorization accuracy scored = 80.742% (Top 10%)



##Let´s cook
        setwd("Your working directory here")

##Load and reshape data:
        
        library(jsonlite)
        library(reshape2)
        
        
        data1 <- fromJSON("train.json")
        data2 <- fromJSON("test.json")
        data2$cuisine <- rep(NA,nrow(data2))
        data <- rbind(data1,data2)
        
        dataframe <- data.frame(id = data$id)
        dataframe$cuisine <- factor(data$cuisine)
        dataframe$ingredients <- data$ingredients
        x <- do.call(rbind,dataframe$ingredients)#Unlist a list of ingredients
        dataframe$Ingredientes <- NULL
        
        for (i in 1:dim(x)[1]){
                dataframe$Ingredientes[i] <- paste0(unique(x[i,]),collapse = " ")
                
        } #Get unique ingredients for each row -> One word = One ingredient. Not using bigrams.
        
        dataframe$ingredients <- NULL
        str(dataframe,list.len = 1000) #Check


##New variables
        
        NumIngredients <- sapply(data$ingredients,function(x) length(x)) #Number of ingredients
        DiffIngredients <- sapply(data$ingredients,function(x) round(length(x)-mean(NumIngredients),2)) #Distance to the mean number of ingredients

##Bag of words (word = ingredient)
        
        library(tm)
        library(SnowballC)
        library(RTextTools)
        library(Matrix)
        library(data.table)
        
        matrix <- create_matrix(dataframe$Ingredientes, language="english",minWordLength=1,stemWords=TRUE,toLower=TRUE)
        matrix <- as.matrix(matrix)
        
##Add created variables
        
        matrix <- cbind(matrix,NumIngredients)
        matrix <- cbind(matrix,DiffIngredients)

##More Feature extraction
        
        n3 <- 2835-matrix[,2836] #Total Ingredients - Number of Ingredients
        matrix <- cbind(matrix,n3)
        max.item <- max.col(matrix[,1:2835],"last") #Top common ingredients
        max.item2 <- data.table(num=1:49718,max.item)
        max.item2$max.item <- factor(max.item2$max.item)
        max.item2 <- sparse.model.matrix(num ~ .-1 ,data = max.item2)
        matrix <- Matrix(matrix,sparse = T)
        matrix <- cbind(matrix,max.item2)
        
##Modelling
        library(xgboost)
        
        #Split data
        
        matrix.train <- matrix[1:39774,]
        matrix.test <- matrix[39775:49718,]
        rm(matrix)
        
        #Create labels
        
        num.class <- length(levels(factor(data1$cuisine)))
        labels <- factor(data1$cuisine)
        levels(labels) <- 1:num.class
        y <- as.matrix(as.integer(labels)-1) #xgboost [0,20]
        
        #Parameters
        
        param <- list("objective" = "multi:softprob",
                      "num_class" = num.class,
                      "eval_metric"="merror",
                      "colsample_bytree" = 0.4,
                      "eta" = 0.1,
                      "gamma" = 0)
        
        #Cross Validation
        
        set.seed(111)
        nround.cv = 650
        cv <- xgb.cv(param = param,data=matrix.train,label=y,nfold=3,nrounds= nround.cv,predictions = T,verbose = T)
        cv
        min.error <- which.min(cv[,test.merror.mean])
        
        #Final Model
        model1 <- xgboost(param = param,data = matrix.train,label = y,nrounds = min.error)
        

##Predict
        
        pred <- predict(model1,matrix.test)

##Reshape predictions
        
        pred <- matrix(pred,nrow=num.class,ncol = length(pred)/num.class)
        pred <- t(pred)
        pred <- max.col(pred,"last") #Get the column with the highest probability
        pred2 <- factor(pred)
        levels(pred2) <- levels(factor(data1$cuisine))
        head(pred2) #Check

##Submission (single model)
        
        submission <- data.frame(id = data2$id,cuisine = pred2)
        write.csv(submission,"name.csv",row.names = FALSE)
        
##Ensemble
        
        #Xgboost with other parameters and SVM (submit6)

        xgb <- read.csv("maxdepth3.csv")
        submit6 <- read.csv("submit6.csv")
        submission <- read.csv("stack2.csv")
        xgb2 <- read.csv("xgbetamod.csv")
        xgb3 <- read.csv("xgb3.csv")
        
        ensemble <- cbind(xgb,s6= submit6$cuisine)
        ensemble <- cbind(ensemble,sub=submission$cuisine)
        ensemble <- cbind(ensemble,xgb2 = xgb2$cuisine)
        ensemble <- cbind(ensemble,xgb3= xgb3$cuisine)
        
        #Majority of vote
        
        id <- ensemble$id
        ensemble$id<- NULL
        
        #Get the most repeated ingredient among predictions
        EnsembleCuisine <- numeric()
        for (i in 1:dim(ensemble)[1]){
                EnsembleCuisine[i] <- attr(which.max(table(as.character(ensemble[i,]))),"names") 
        } 
        
        EnsembleCuisine <- factor(EnsembleCuisine,levels= c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))
        levels(EnsembleCuisine) <- levels(ensemble$cuisine)
        head(EnsembleCuisine)
        
        #Submission
        
        Final <- data.frame(id = id,cuisine = EnsembleCuisine)
        write.csv(Final,"name.csv",row.names = FALSE)

