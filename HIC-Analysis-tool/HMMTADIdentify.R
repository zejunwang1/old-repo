
###################################################################################
# TAD extraction via "directionality index" and Hidden Markov model

# Reference:
# Topological domains in mammalian genomes identified by analysis of chromatin 
# interactions J. R. Dixon, S. Selvaraj, et al.

# This algorithm firstly computes DI value for each bin, then use Hidden Markov
# and Gaussian Mixture model to identify TADs' boundaries. I use RHmm package to
# implement this algorithm. Ensure package RHmm installed. You can download RHmm
# package at the following link:
# https://cran.r-project.org/src/contrib/Archive/RHmm/
# install it in Linux/Mac by: R CMD INSTALL {package}.tar.gz

# Ensure package mixtools installed. You can install it by the following command
# at R console:
# install.packages("mixtools")

# Implemented by 
# Zejun Wang
# Tsinghua University, Beijing
# mails: xiaolijun_thu@163.com

###################################################################################



HMMTADIdentify <- function(H,Res=0,Window=1000000,InitMethod=1,ModelMethod=1){
  
  ###############################################################
  # H: Input 2D HiC contact matrix
  # Res: Resolution which determines window size
  # Window: DI parameter, default 1M
  # InitMethod: Initialization method
  #             1 - Heuristic initialization
  #             2 - Mixture Gaussian Fit initialization
  #             3 - Random initialization
  #             default 1
  # ModelMethod: Model selection method
  #              1 - AIC score
  #              2 - BIC score
  #              3 - Loglikelihood
  #              default 1
  ###############################################################
  
  if(Res==0){
    stop("lose resolution parameter in HMMTADIdentify function")
  }
  
  if(class(H)!="matrix"){
    warning("input type is not matrix, convert it to matrix")
    H <- as.matrix(H)
  }
  
  if(nrow(H)!=ncol(H)){
    stop("input matrix is not square")
  }
  
  # Ensure matrix be symmetric
  H <- (H+t(H))/2
  l <- nrow(H)
  
  # Compute DI value for each bin
  w <- round(Window / Res)
  DI <- DIndex(H,w)
  DI[is.na(DI)] <- 0
  
  # Remove extreme values at two edges
  DI[DI>max(DI[w:(l-w)])] <- max(DI[w:(l-w)])
  DI[DI<min(DI[w:(l-w)])] <- min(DI[w:(l-w)])
  
  # Hidden Markov model 
  O <- DI              # set DI value as observations
  Q <- 3               # number of states: 3 states
  
  nMix <- 15           # number of Gaussian Mixtures
  AIC <- rep(0,nMix)   # AIC score 
  BIC <- rep(0,nMix)   # BIC score
  LogL <- rep(0,nMix)  # Log-likelihood
  Path <- matrix(0,nrow = nMix,ncol = length(O))    # States 
  
  # Loop for number of Gaussian Mixtures
  for(N in 1:nMix){
    message(paste("Fitting HMM model for ",N," gaussian components...",sep = ""))
    if(InitMethod==1){
      
      # Initial state probabilities
      prob0 <- c(1/3,1/3,1/3)
      
      # Initial transition matrix
      trans0 <- rbind(c(1/3,1/3,1/3),c(1/3,1/3,1/3),c(1/3,1/3,1/3))
      
      if(N==1){
        # Univariate normal distribution
        # Mean 
        mu1 <- max(O) / nMix
        mu2 <- min(O) / nMix
        mu3 <- min(O)
        
        # Variance 
        var1 <- 1
        var2 <- 1
        var3 <- 1
        
        # Distribution set
        dis <- distributionSet("NORMAL",mean = c(mu1,mu2,mu3),var = c(var1,var2,var3))
        # Generate initial point
        init <- HMMSet(prob0,trans0,dis)
        
        # Fit HMM model 
        res <- HMMFit(O,nStates = Q,dis = "NORMAL",control = list(
          init="USER",iter=20,initPoint=init))
      }else{
        # Univariate mixture normal distribution
        # Mean
        mu1 <- seq(N,1,-1)
        mu2 <- seq(N/2-1,-N/2,-1)
        mu3 <- seq(-1,-N,-1)
        
        # Variance
        var1 <- rep(1,N)
        var2 <- rep(1,N)
        var3 <- rep(1,N)
        
        # Mixture weight
        w1 <- rep(1,N)/N
        w2 <- rep(1,N)/N
        w3 <- rep(1,N)/N
        
        # Distribution set
        dis <- distributionSet("MIXTURE",mean = list(mu1,mu2,mu3),var = list(var1,var2,var3),
                               proportion = list(w1,w2,w3))
        # Generate initial point
        initp <- HMMSet(prob0,trans0,dis)
        
        # Fit HMM model
        res <- HMMFit(O,nStates = Q,dis = "MIXTURE",nMixt = N,control = list(
          init="USER",iter=20,initPoint=initp))
      }
    }else if(InitMethod==2){
      
      # Initial state probabilities
      prob0 <- c(1/3,1/3,1/3)
      
      # Initial transition matrix
      trans0 <- rbind(c(1/3,1/3,1/3),c(1/3,1/3,1/3),c(1/3,1/3,1/3))
      
      # Repeat times
      M <- 10
      tempAIC <- Inf
      tempBIC <- Inf
      tempLogL <- -Inf
      
      for(i in 1:M){
        # Mixture Gaussian fit for DI values
        gaussfit <- normalmixEM(O,arbvar = FALSE,epsilon = 1e-4,k = N*Q,maxit = 300)
        st <- sort(gaussfit$mu,decreasing = TRUE,index.return = TRUE)
        mu <- st$x
        w <- gaussfit$lambda[st$ix]
        var <- gaussfit$sigma[st$ix]
        
        # Initial value
        mu1 <- mu[1:N]
        mu2 <- mu[(N+1):(2*N)]
        mu3 <- mu[(2*N+1):(3*N)]
        var1 <- var[1:N]
        var2 <- var[(N+1):(2*N)]
        var3 <- var[(2*N+1):(3*N)]
        
        if(N==1){
          # Initial point 
          dis <- distributionSet("NORMAL",mean = c(mu1,mu2,mu3),var = c(var1,var2,var3))
          initp <- HMMSet(prob0,trans0,dis)
          
          # Fit univariate gaussian HMM model 
          tempRes <- HMMFit(O,nStates = Q,dis = "NORMAL",control = list(init="USER",iter=20,
                                                                    initPoint=initp))
        }else{
          # Initial weights
          w1 <- w[1:N]/sum(w[1:N])
          w2 <- w[(N+1):(2*N)]/sum(w[(N+1):(2*N)])
          w3 <- w[(2*N+1):(3*N)]/sum(w[(2*N+1):(3*N)])
          
          # Initial point
          dis <- distributionSet("MIXTURE",mean = list(mu1,mu2,mu3),var = list(var1,var2,var3),
                                 proportion = list(w1,w2,w3))
          initp <- HMMSet(prob0,trans0,dis)
          
          # Fit mixture gaussian HMM model
          tempRes <- HMMFit(O,nStates = Q,dis = "MIXTURE",nMixt = N,control = list(init="USER",
                        iter=20,initPoint=initp))
        }
        
        if(ModelMethod==1){
          if(!is.na(tempRes$AIC)){
            if(tempRes$AIC<tempAIC){
              res <- tempRes
              tempAIC <- tempRes$AIC
            }
          }
        }else if(ModelMethod==2){
          if(!is.na(tempRes$BIC)){
            if(tempRes$BIC<tempBIC){
              res <- tempRes
              tempBIC <- tempRes$BIC
            }
          }
        }else{
          if(!is.na(tempRes$LLH)){
            if(tempRes$LLH>tempLogL){
              res <- tempRes
              tempLogL <- tempRes$LLH
            }
          }
        }
      }
    }else{
      # Random initialization repeat times
      M <- 20
      tempAIC <- Inf
      tempBIC <- Inf
      tempLogL <- -Inf
      
      for(i in 1:M){
        if(N==1){
          # Univariate normal distribution
          # Random initialization
          tempRes <- HMMFit(O,nStates = Q,dis = "NORMAL",control = list(iter=20))
        }else{
          # Univariate mixture normal distribution
          tempRes <- HMMFit(O,nStates = Q,dis = "MIXTURE",nMixt = N,control = list(iter=20))
        }
        
        if(ModelMethod==1){
          if(!is.na(tempRes$AIC)){
            if(tempRes$AIC<tempAIC){
              res <- tempRes
              tempAIC <- tempRes$AIC
            }
          }
        }else if(ModelMethod==2){
          if(!is.na(tempRes$BIC)){
            if(tempRes$BIC<tempBIC){
              res <- tempRes
              tempBIC <- tempRes$BIC
            }
          }
        }else{
          if(!is.na(tempRes$LLH)){
            if(tempRes$LLH>tempLogL){
              res <- tempRes
              tempLogL <- tempRes$LLH
            }
          }
        }
      }
    }
    
    # Save AIC score 
    if(is.na(res$AIC)){
      AIC[N] <- Inf
    }else{
      AIC[N] <- res$AIC
    }
    
    # Save BIC score
    if(is.na(res$BIC)){
      BIC[N] <- Inf
    }else{
      BIC[N] <- res$BIC
    }
    
    # Log-likelihood
    if(is.na(res$LLH)){
      LogL[N] <- -Inf
    }else{
      LogL[N] <- res$LLH
    }
    
    # Viterbi algorithm 
    if(!is.na(res$LLH)){
      v <- viterbi(res,O)
      Path[N,] <- v$states
    }
  }
  
  # Model Selection
  if(ModelMethod==1){
    ind <- which(AIC==min(AIC))
    if(length(ind)>1){
      temp <- LogL[ind]
      idx <- which(temp==max(temp))
      idx <- idx[1]
      ind <- ind[idx]
    }
  }else if(ModelMethod==2){
    ind <- which(BIC==min(BIC))
    if(length(ind)>1){
      temp <- LogL[ind]
      idx <- which(temp==max(temp))
      idx <- idx[1]
      ind <- ind[idx]
    }
  }else{
    ind <- which(LogL==max(LogL))
    ind <- ind[1]
  }
  message("\nThe optimal number of gaussian components is: ",ind)
  TAD_mark <- Path[ind,]
  
  # Get TAD boundaries
  TAD_boundaries <- TADMarkToBoundaries(TAD_mark)
  return(TAD_boundaries)
}


DIndex <- function(H,w=10){
  
  ###############################################################
  # Compute DI value for each bin
  # H: input HiC data matrix
  # w: window size, default 10 bins
  ###############################################################
  
  N <- nrow(H)
  
  # Downstream reads count value
  B <- rep(0,N)
  subH <- H[1:w,1:w]
  B[1] <- (sum(subH) - sum(diag(subH)))/2
  
  # Compute downstream value for 2:(N-w+1) bin
  for(i in 2:(N-w+1)){
    idx <- (2:w) + (i-2)
    B[i] <- B[i-1] - sum(H[i-1,idx]) + sum(H[i+w-1,idx])
  }
  # Compute downstream value for (N-w+2):N bin
  for(i in (N-w+2):N){
    subH <- H[i:N,i:N]
    B[i] <- (sum(subH) - sum(diag(subH)))/2
  }
  B[N] <- 0
  
  # Upstream reads count value
  A <- rep(0,N)
  A[w:N] <- B[1:(N-w+1)]
  
  # Compute upstream value for 1:(w-1) bin
  for(i in 1:(w-1)){
    subH <- H[1:i,1:i]
    A[i] <- (sum(subH) - sum(diag(subH)))/2
  }
  A[1] <- 0
  
  # Compute DI value
  E <- (A+B)/2
  DI <- (B-A)/(abs(B-A))*((A-E)^2/E+(B-E)^2/E)
  return(DI)
}


TADMarkToBoundaries <- function(TAD_mark){
  
  # Get boundaries from mark
  Labels <- rep(1,length(TAD_mark))
  Labels[1] <- 1
  Alarm_M <- 0
  
  for(i in 2:length(TAD_mark)){
    
    # Continue state
    Cond_cont1 <- TAD_mark[i]==3 & TAD_mark[i-1]==3
    Cond_cont2 <- TAD_mark[i]==2 & TAD_mark[i-1]==3
    Cond_cont3 <- TAD_mark[i]==2 & TAD_mark[i-1]==2
    Cond_cont4 <- !Alarm_M & TAD_mark[i]==3 & TAD_mark[i-1]==2
    Cond_cont5 <- TAD_mark[i]==2 & TAD_mark[i-1]==1
    Cond_cont <- Cond_cont1 | Cond_cont2 | Cond_cont3 | Cond_cont4 | Cond_cont5
    
    # Alarm state 
    Cond_alarm1 <- TAD_mark[i]==1 & TAD_mark[i-1]==3
    Cond_alarm2 <- TAD_mark[i]==1 & TAD_mark[i-1]==2
    Cond_alarm3 <- TAD_mark[i]==1 & TAD_mark[i-1]==1
    Cond_alarm <- Cond_alarm1 | Cond_alarm2 | Cond_alarm3
    
    # End state
    Cond_end1 <- TAD_mark[i]==3 & TAD_mark[i-1]==1
    Cond_end2 <- Alarm_M & (TAD_mark[i]==3 & TAD_mark[i-1]==2)
    Cond_end <- Cond_end1 | Cond_end2
    
    if(Cond_cont){
      Labels[i] <- 2
      next
    }
    if(Cond_alarm){
      Labels[i] <- 2
      Alarm_P <- i
      Alarm_M <- 1
      next
    }
    if(Cond_end){
      Labels[Alarm_P] <- 3
      Labels[i] <- 1
      Alarm_M <- 0
    }
  }
  Labels[length(Labels)] <- 3
  
  Pos <- which(Labels==1)
  Pos <- c(Pos,length(Labels))
  return(Pos)
}













