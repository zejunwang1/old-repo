
###################################################################################
# TAD extraction via Spectral Graph method

# Reference:
# Hierachical identification of topological domain via graph Laplacian
# J. Chen, A. O. Hero, I. Rajapakse

# This algorithm can get hierachical domain structures by setting different 
# fiedler value threshold.

# For large-scale matrix eigenvalue decomposition, I recommend using RcppEigen
# instead of eigen function to compute fiedler value and vector. Ensure package
# RcppEigen be installed and eigenSolver.cpp included in path:
# .libPath()/RcppEigen/include/

# Implemented by 
# Zejun Wang
# Tsinghua University, Beijing
# mails: xiaolijun_thu@163.com

###################################################################################


TADLaplace <- function(H,Threshold=0.5,Region=10,Method=1,Flag=2,Norm=1){
  
  ################################################################
  # H: input HiC data matrix
  # Threshold: fiedler value threshold, default 0.5
  # Region: minimum splitting region threshold
  # Method: 1 - split on toeplitz normalized matrix
  #         2 - split on original data matrix
  #         default 1
  # Flag: 1 - compute eigenvalues using RcppEigen
  #       2 - compute eigenvalues using eigen function
  #       default 2
  # Norm: 1 - consider diagonal elements including zeros
  #       2 - consider diagonal elements not including zeros
  #       default 1
  ################################################################
  
  if(class(H)!="matrix"){
    warning("Input type is not matrix, convert it to matrix")
    H <- as.matrix(H)
  }
  
  if(nrow(H)!=ncol(H)){
    stop("Input matrix is not square!")
  }
  
  # Ensure input matrix be symmetric
  H <- (H+t(H))/2
  
  # Remove unmappable region if they are included
  idx <- which(colSums(H)==0)
  if(length(idx)!=0){
    H <- H[-idx,]
    H <- H[,-idx]
  }
  
  # Remove the diagonal elements
  H <- H - diag(diag(H))
  N <- nrow(H)
  
  if(Method==1){
    # Splitting on Toeplitz normalized matrix
    Hn <- ToeplitzNorm(H,Norm)
    
    # Compute fiedler value and vector
    Fd <- FdVectors(Hn,Flag)
    Fdval <- Fd$value
    Fdvec <- Fd$vector
    
    # Position of start domain
    l <- length(Fdvec)
    Pos <- c(1,which((sign(Fdvec[2:l])-sign(Fdvec[1:(l-1)]))!=0)+1,N)
  }else{
    # Splitting directly on original matrix
    Pos <- c(1,N)
  }
  
  # Recursive splitting until convergence
  spa <- rep(0,N)
  spa[Pos] <- 1
  
  for(i in 1:(length(Pos)-1)){
    # Get block range
    idx <- Pos[i]:(Pos[i+1]-1)
    
    # If block size <= Region, stop split
    if(length(idx)>Region){
      # Get sub-matrix
      subH <- H[idx,idx]
      
      # Get fiedler value and vector
      Fd <- FdVectors(subH,Flag)
      Fdval <- Fd$value
      Fdvec <- Fd$vector
      
      # If fiedler value > Threshold, stop split
      if(Fdval<=Threshold){
        sp <- SubSplit(subH,Threshold,Region,Flag)
        # Mark boundaries
        spa[Pos[i]+which(sp>0)-1] <- 1
      }
    }
  }
  TAD_boundaries <- which(spa>0)
  return(TAD_boundaries)
}

ToeplitzNorm <- function(H,Method){
  
  #############################################################
  # H: input HiC data matrix
  # Method: 1 - number of diagonal elements including zeros
  #         2 - number of diagonal elements not including zeros
  #         default 1
  #############################################################
  
  N <- nrow(H)
  
  # Diagonal summation
  ds = diagSum(H)
  if(Method==1){
    len <- seq(N,1,-1)
  }else{
    len <- ds$num[N:0]
  }
  
  # Diagonal mean value
  dsmean <- ds$vec[N:0] / len
  
  # Generate toeplitz matrix
  Tp <- toeplitz(dsmean)
  
  # Normalization
  Hn <- H/Tp
  Hn[is.infinite(Hn)] <- 0
  Hn[is.nan(Hn)] <- 0
  
  return(Hn)
}

diagSum <- function(H){
  
  #############################################################
  # Summation across all separate diagonals of matrix H
  # Return a column vector
  # H: Input 2D matrix
  #############################################################
  
  N <- nrow(H)
  Hmod <- matrix(0,nrow = N+N-1,ncol = N)
  logVec <- c(rep(0,N-1),rep(1,N),rep(0,N-1))
  indMat <- matrix(rep(1:N,N+N-1),nrow = N+N-1,byrow = TRUE)
  indMat <- indMat + matrix(rep(0:(N+N-2),N),ncol = N)
  logMat <- matrix(logVec[indMat],nrow = N+N-1)
  Hmod[which(logMat==1)] <- H
  
  # Diagonal summation
  sumVec <- rowSums(Hmod)
  nonzerosNum <- apply(Hmod,1,numNonzeros)
  
  res <- list()
  res$vec <- sumVec
  res$num <- nonzerosNum
  
  return(res)
}

numNonzeros <- function(x){
  # Compute number of non-zeros in vector
  return(length(which(x!=0)))
}

FdVectors <- function(H,Flag){
  
  #############################################################
  # H: input 2D symmetric matrix
  # Flag: 1 - compute eigenvectors using RcppEigen
  #       2 - compute eigenvectors using eigen function
  #       default 1
  #############################################################
  
  N <- nrow(H)
  
  # Compute Laplace matrix
  dgs <- colSums(H)
  dgs[dgs==0] <- 1
  L <- diag(dgs) - H
  Ln <- (diag(dgs^(-1/2)))%*%L%*%(diag(dgs^(-1/2)))
  Ln <- (Ln+t(Ln))/2
  
  # Eigenvalue decomposition
  if(Flag==1){
    # Ensure package Rcpp and RcppEigen installed
    require(Rcpp)
    require(RcppEigen)
    
    # Ensure eigenSolver.cpp is included in libPaths/RcppEigen/include/
    path <- .libPaths()
    path <- paste(path,"/RcppEigen/include/eigenSolver.cpp",sep = "")
    
    # Source eigenSolver.cpp
    Rcpp::sourceCpp(path)
    
    # Compute fiedler value and vector
    eig <- getEigenVectors(Ln)
    Fiedler <- list(value = eig$eigenvalues[2],vector = eig$eigenvectors[,2])
  }else{
    # Use eigen function, slow for large matrix
    eig <- eigen(Ln,symmetric = TRUE)
    l <- length(eig$values)
    Fiedler <- list(value = eig$values[l-1],vector = eig$vectors[,l-1])
  }
  return(Fiedler)
}

standard <- function(x){
  # Normalize each row/column
  return(x/sqrt(sum(x)))
}

SubSplit <- function(H,Threshold,Region,Flag){
  # Recursively split a connection matrix via Fiedler value and vector
  Fd <- FdVectors(H,Flag)
  Fdval <- Fd$value
  Fdvec <- Fd$vector
  l <- length(Fdvec)
  
  # If fiedler value > Threshold, stop split
  if(Fdval>Threshold){
    sp <- 1
  }else{
    N <- nrow(H)
    # Position of sign change (sub-block starting)
    Pos <- c(1,which((sign(Fdvec[2:l])-sign(Fdvec[1:(l-1)]))!=0)+1,N)
    sp <- rep(0,N)
    sp[Pos] <- 1
    
    # For each sub-block
    for(i in 1:(length(Pos)-1)){
      idx <- Pos[i]:(Pos[i+1]-1)
      
      # Split sub-block
      if(length(idx)>Region){
        sp1 <- SubSplit(H[idx,idx],Threshold,Region,Flag)
        # Mark boundaries
        sp[Pos[i]+which(sp1>0)-1] <- 1
      }
    }
  }
  return(sp)
}











