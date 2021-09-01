
###################################################################################
# HiC normalization method

# These functions implement common normalization methods for HiC contact matrix,
# including Toeplitz Normalization / Distance-based Normalization / 
# Vanilla-Coverage Normalization / Sqrt-VC Normalization / Sinkhorn-Knopp 
# Normalization / Knight-Ruiz Normalization

# Reference:
# 1. Comprehensive Mapping of Long-Range Interactions Reveals Folding Principles of 
#    the Human Genome[J]. Science, 2009, Erez L A, Berkum N L V, Louise W, et al.
# 2. A 3D Map of the Human Genome at Kilobase Resolution Reveals Principles of 
#    Chromatin Looping[J]. Cell, 2014, Rao S P, Huntley M, Durand N, et al.

# Implemented by 
# Zejun Wang
# Tsinghua University, Beijing
# mails: xiaolijun_thu@163.com

###################################################################################

VCNorm <- function(H){
  
  # Vanilla-Coverage Normalization for HiC contact matrix
  # preprocess 
  H <- preprocess(H)
  
  avg <- mean(rowSums(H))
  temp <- H / rowSums(H)
  H <- t(t(temp)/colSums(H))*(avg^2)
  
  # shrink the variation
  # H <- shrinkVariation(H)
  
  return(H)
}

SqrtVCNorm <- function(H){
  
  # Sqrt Vanilla-Coverage Normalization for HiC contact matrix
  # preprocess
  H <- preprocess(H)
  
  avg <- mean(rowSums(H))
  temp <- H / sqrt(rowSums(H))
  H <- t(t(temp)/sqrt(colSums(H)))*avg
  
  # shrink the variation
  # H <- shrinkVariation(H)
  
  return(H)
}

ToeplitzNorm <- function(H,method=1){
  
  # do VC/Sqrt-VC/KR Normalization before Toeplitz Normalization
  # H: input 2D matrix
  # method=1: consider number of diagonal elements including zeros
  # method=2: consider number of diagonal elements not including zeros
  # default 1
  
  # preprocess
  H <- preprocess(H)
  if(max(H)>20){
    H <- shrinkVariation(H)
  }
  
  N <- nrow(H)
  # diagonal summation
  ds <- diagonalSum(H)
  
  if(method==1){
    len <- seq(N,1,-1)
  }else{
    len <- ds$num[N:0]
  }
  
  # diagonal mean value
  dsmean <- ds$vec[N:0] / len
  
  # generate toeplitz matrix
  Tp <- toeplitz(dsmean)
  
  # normalization
  Hn <- H/Tp
  Hn[is.infinite(Hn)] <- 0
  Hn[is.nan(Hn)] <- 0
  
  return(Hn)
}

diagonalSum <- function(H){
  # compute diagonal summation and return a list
  
  N <- nrow(H)
  Hmod <- matrix(0,nrow = N+N-1,ncol = N)
  logVec <- c(rep(0,N-1),rep(1,N),rep(0,N-1))
  indMat <- matrix(rep(1:N,N+N-1),nrow = N+N-1,byrow = TRUE)
  indMat <- indMat + matrix(rep(0:(N+N-2),N),ncol = N)
  logMat <- matrix(logVec[indMat],nrow = N+N-1)
  Hmod[which(logMat==1)] <- H
  
  # diagonal summation
  sumVec <- rowSums(Hmod)
  nonzerosNum <- apply(Hmod,1,numNonzeros)
  
  s <- list()
  s$vec <- sumVec
  s$num <- nonzerosNum
  
  return(s)
}

numNonzeros <- function(x){
  # compute number of non-zeros in vector
  return(length(which(x!=0)))
}

loessBasedNorm <- function(H){
  
  # do VC/Sqrt-VC/KR Normalization before loess-based Normalization
  # similar to Toeplitz Normalization, use loess curve to fit expected values
  
  # preprocess
  H <- preprocess(H)
  if(max(H)>20){
    H <- shrinkVariation(H)
  }
  
  # Diagonal summation
  ds <- diagonalSum(H)
  l <- nrow(H)
  
  avg <- ds$vec[l:0] / seq(l,1,-1)
  dg <- data.frame(seq(1,l,1),avg)
  colnames(dg) <- c("dis","avg")
  
  # Loess fit 
  model <- loess(avg~dis,data = dg,span = 0.1)
  dge <- toeplitz(model$fitted)
  H <- H/dge
  
  return(H)
}

SinkhornKnoppNorm <- function(H,maxIter = 100,tol = 1e-1){
  
  # Sinkhorn-Knopp algorithm to implement matrix balancing
  # repeatedly execute VC algorithm until convergence or reach maximum iterations
  # H: input hic raw contact matrix
  # maxIter: maximum iterations, default 100
  # tol: error tolerance, default 0.1
  
  # preprocess
  H <- preprocess(H)
  l <- nrow(H)
  
  x0 <- rep(1,l)
  delta_x <- rowSums(H)
  k <- 0
  while(var(delta_x)>tol & k<maxIter){
    # number of iterations
    k <- k + 1
    # message(paste("number of iterations= ",k,sep = ""))
    delta_x <- delta_x / mean(delta_x)
    H <- H / delta_x
    H <- t(t(H)/delta_x)
    x <- x0*delta_x
    delta_x <- rowSums(H)
  }
  
  # shrink the variation
  # H <- shrinkVariation(H)
  
  return(H)
}

KnightRuizNorm <- function(H,tol=1e-4,delta1=0.1,delta2=3){
  
  # Knight-Ruiz algorithm to implement matrix balancing
  # a fast balancing algorithm for symmetric matrices
  # firstly attempt to find a vector x, such that diag(x)*A*diag(x) is close to doubly
  # stochastic. H must be symmetric and non-negative.
  
  # H: input hic raw contact matrix
  # tol: error tolerance, default 1e-4
  # delta1/delta2: how close/far balancing vectors can get to/from the edge of the 
  # positive cone. delta1: 0.1 default. delta2: 3 default.
  # return KR normalized hic matrix

  # preprocess
  H <- preprocess(H)
  l <- nrow(H)
  
  # initial vector
  x0 <- rep(1,l)
  # inner stopping criterion parameters
  g <- 0.9
  etamax <- 0.1
  eta <- etamax
  stop_tol <- tol*0.5
  x <- x0
  rt <- tol^2
  v <- as.vector(x*(H%*%x))
  rk <- 1-v
  
  rho_km1 <- sum(rk^2)
  rout <- rho_km1
  rold <- rout
  MVP <- 0
  i <- 0        # outer iterations
  
  # outer iteration loop
  while(rout>rt){
    i <- i + 1
    k <- 0         # inner iterations
    y <- rep(1,l)
    innertol <- max(eta^2*rout,rt)
    
    # inner iteration loop
    while(rho_km1>innertol){
      k <- k + 1
      if(k==1){
        z <- rk/v
        p <- z
        rho_km1 <- sum(rk*z)
      }else{
        beta <- rho_km1/rho_km2
        p <- z + beta*p
      }
      
      # update search direction efficiently
      w <- x*(as.vector(H%*%(x*p))) + v*p
      alpha <- rho_km1/(sum(p*w))
      ap <- alpha*p
      
      # test distance to boundary of cone
      ynew <- y + ap
      if(min(ynew)<=delta1){
        if(delta1==0){
          break
        }
        ind <- which(ap<0)
        gamma <- min((delta1-y[ind])/ap[ind])
        y <- y + gamma*ap
        break
      }
      if(max(ynew)>=delta2){
        ind <- which(ynew>delta2)
        gamma <- min((delta2-y[ind])/ap[ind])
        y <- y + gamma*ap
        break
      }
      y <- ynew
      rk <- rk - alpha*w
      rho_km2 <- rho_km1
      z <- rk/v
      rho_km1 <- sum(rk*z)
    }
    x <- x*y
    v <- x*as.vector(H%*%x)
    rk <- 1 - v
    rho_km1 <- sum(rk^2)
    rout <- rho_km1
    MVP <- MVP + k + 1
    
    # update inner iteration stopping criterion
    rat <- rout/rold
    rold <- rout
    res_norm <- sqrt(rout)
    eta_o <- eta
    eta <- g*rat
    
    if(g*eta_o^2>0.1){
      eta <- max(eta,g*eta_o^2)
    }
    eta <- max(min(eta,etamax),stop_tol/res_norm)
  }
  
  # getting normalized hic matrix
  avg <- sqrt(mean(colSums(H)))
  x <- avg*x
  H <- diag(x)%*%H%*%diag(x)
  
  # shrink the variation
  # H <- shrinkVariation(H)
  
  return(H)
}


preprocess <- function(H){
  
  if(class(H)!="matrix"){
    warning("convert input to a matrix")
    H <- as.matrix(H)
  }
  
  if(nrow(H)!=ncol(H)){
    stop("input matrix is not square")
  }
  
  # ensure matrix be square
  H <- (H + t(H))/2
  
  # remove unmappable region
  idx <- which(colSums(H)==0)
  if(length(idx)!=0){
    H <- H[-idx,]
    H <- H[,-idx]
  }
  return(H)
}

shrinkVariation <- function(H,method=1){
  # shrink hic contact matrix's variation
  # H: input hic contact matrix
  # method: 1-use log function, 2-use sqrt function, default 1
  
  if(method==1){
    # log transition function
    thresh <- log(max(H)) - 1
    H <- pmin(log(ceiling(H)),thresh)
    H[H==-Inf] <- -1
    H <- H + 1
  }else{
    # sqrt transition function
    thresh <- sqrt(sqrt(max(H))) - 1
    H <- pmin(sqrt(sqrt(ceiling(H))),thresh)
  }
  
  return(H)
}

normSum <- function(H){
  # H: input raw HiC contact matrix
  # preprocess
  H <- preprocess(H)
  s0 <- log(colSums(H))

  h <- VCNorm(H)
  s1 <- log(colSums(h))
  
  h <- SqrtVCNorm(H)
  s2 <- log(colSums(h))
  
  h <- SinkhornKnoppNorm(H)
  s3 <- log(colSums(h))
  
  h <- KnightRuizNorm(H)
  s4 <- log(colSums(h))
  
  s <- data.frame(Original=s0,VC=s1,SqrtVC=s2,SK=s3,KR=s4)
  return(s)
}






  