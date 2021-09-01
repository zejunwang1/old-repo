
###################################################################################
# TAD extraction via dynamic programming

# Reference:
# Multiscale Identification of Topological Domain in Chromatin. 
# D. Filippova, R. Patro, G. Duggal, C. Kingsford

# This algorithm use dynamic programming to identify TADs. Because multiple loops 
# in R is too slow, I use cpp function to speed up. Ensure package Rcpp installed.

# Implemented by
# Zejun Wang
# Tsinghua University, Beijing
# mails: xiaolijun_thu@163.com

###################################################################################


DPTADIdentify <- function(H,g=1){
  
  ##################################################################
  # H: input 2D HiC contact matrix
  # gamma: resolution factor, default 1
  ##################################################################
  
  if(class(H)!="matrix"){
    warning("Input type is not matrix, convert it to matrix")
    H <- as.matrix(H)
  }
  
  if(nrow(H)!=ncol(H)){
    stop("Input matrix is not square!")
  }
  
  # Ensure matrix be symmetric
  H <- (H+t(H))/2
  l <- nrow(H)
  
  # Calculate mean scaled density
  # Use the following cpp function to compute mean density quickly
  cppFunction('
    NumericVector MeanDensityCompute(NumericMatrix H,double gamma){
      int l = H.nrow();
      NumericVector mus(l-1);
      double s,s_pre,s_all,temp1,temp2;
      for(int i=1;i<l-1;i++){
        s = 0;
        for(int x=1;x<=i+1;x++){
          for(int y=x+1;y<=i+1;y++){
            s = s + H(x-1,y-1);
          }
        }
        s_all = s;
        s_pre = s;
        for(int j=2;j<=l-i;j++){
          temp1 = 0;
          for(int k=j;k<=i+j-1;k++){
            temp1 = temp1 + H(j-2,k-1);
          }
          temp2 = 0;
          for(int k=j;k<=i+j-1;k++){
            temp2 = temp2 + H(j+i-1,k-1);
          }
          s = s_pre - temp1 + temp2;
          s_all = s_all + s;
          s_pre = s;
        }
        mus(i-1) = s_all/(pow(i,gamma))/(l-i);
      }
      s_all = 0;
      for(int i=1;i<=l;i++){
        for(int j=i+1;j<=l;j++){
          s_all = s_all + H(i-1,j-1);
        }
      }
      mus(l-2) = s_all/(pow(l-1,gamma));
      return mus;
    }
  ')
  mus <- MeanDensityCompute(H,g)
  
  
  # Fast Dynamic Programming
  # Use the following cpp function to implement quick dynamic programming
  cppFunction('
    NumericVector DPFindPosition(NumericMatrix H,NumericVector mus,double gamma){
      int l = H.nrow();
      NumericVector opt(l+1);
      NumericVector pos(l+1);
      double sd;
      for(int i=3;i<=l;i++){
        NumericVector opt_candidate(i-1);
        for(int j=2;j<i;j++){
          sd = 0;
          for(int x=j;x<=i;x++){
            for(int y=x+1;y<=i;y++){
              sd = sd + H(x-1,y-1);
            }
          }
          sd = sd/pow(i-j,gamma)-mus(i-j-1);
          if(sd>0){
            opt_candidate(j-1) = opt(j-1) + sd;
          }else{
            opt_candidate(j-1) = opt(j-1);
          }
          if(opt_candidate(j-1)>=opt(i)){
            opt(i) = opt_candidate(j-1);
            pos(i) = j-1;
          }
        }
      }
      return pos;
    }
  ')
  pos <- DPFindPosition(H,mus,g)
  
  
  # Loops implemented in R is too slow. I strongly recommend you 
  # to use above cpp function to execute dynamic programming.
  # opt <- rep(0,l)
  # pos <- rep(0,l)
  # for(i in 3:l){
  #  opt_candidate <- rep(0,i-2)
  #  for(j in 2:(i-1)){
  #    opt_candidate[j-1] <- opt[j-1] + max(ScaledDensity(H,j,i,g)-mus[i-j],0)
  #    if(opt_candidate[j-1]>=opt[i]){
  #      opt[i] <- opt_candidate[j-1]
  #      pos[i] <- j-1
  #    }
  #  }
  #  print(i)
  # }
  
  
  # Getting TADs boundaries
  c <- 1
  boundary <- l
  thresh <- 6
  while(1){
    if(boundary[c]<=thresh | boundary[c]==pos[boundary[c]]){
      break
    }else{
      c <- c + 1
      boundary <- c(boundary,pos[boundary[c-1]])
    }
  }
  boundary <- rev(boundary)
  return(boundary)
}


MeanDensity <- function(H,g=0.5){
  
  ##################################################################
  # Mean scaled density calculation
  # H: Input HiC contact matrix
  # g: Resolution factor, default 0.5
  ##################################################################
  
  # Faster calculation version
  l <- nrow(H)
  mus <- rep(0,l-1)
  for(i in 1:(l-2)){
    ind <- 1:(i+1)
    subH <- H[ind,ind]
    s <- (sum(subH)-sum(diag(subH)))/2
    s_all <- s
    s_pre <- s
    for(j in 2:(l-i)){
      ind_pre <- (2:(i+1))+j-2
      s <- s_pre - sum(H[j-1,ind_pre]) + sum(H[j+i,ind_pre])
      s_all <- s_all + s
      s_pre <- s
    }
    print(i)
    mus[i] <- s_all/(i^g)/(l-i)
  }
  mus[l-1] <- (sum(H)-sum(diag(H)))/2/((l-1)^g)
  
  # Slow calculation version
  # l <- nrow(H)
  # mus <- rep(0,l-1)
  # for(i in 1:(l-1)){
  #  s_all <- 0
  #  for(j in 1:(l-i)){
  #    # Sub-block index
  #    ind <- (1:(i+1))-1+j
  #    subH <- H[ind,ind]
  #    s <- (sum(subH)-sum(diag(subH)))/2
  #    s_all <- s_all + s
  #  }
  #  mus[i] <- s_all/(i^g)/(l-i)
  # }
  
  return(mus)
}

ScaledDensity <- function(H,k,l,g){
  
  ##################################################################
  # Scaled density calculation
  # H: Input HiC contact matrix
  # k: Starting loci
  # l: Ending loci
  # g: Resolution factor, default 0.5
  ##################################################################
  
  subH <- H[k:l,k:l]
  s <- (sum(subH)-sum(diag(subH)))/2
  s <- s/((l-k)^g)
  return(s)
}










