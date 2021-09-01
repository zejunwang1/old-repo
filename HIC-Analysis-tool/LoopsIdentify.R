
###################################################################################

# HiC loops identification at whole chromosome matrix level / dynamic resolution 
# TADs matrix level
# Step1: use DI+HMM algorithm for TADs identification 
# Step2: dynamic resolution TADs region matrix 
# Step3: loops identification at TADs level 

# Implemented by
# Zejun Wang
# Tsinghua University, Beijing
# mails: xiaolijun_thu@163.com

###################################################################################

TADLoopsIdentify <- function(h,isNorm=0,matrixThresh=0.7,diagMin=3,thresh=2,val=9,pvalueThresh=0.05){
  # identify chromatin loops in a TAD region
  # h: input HiC-TAD raw contact matrix
  # isNorm: 0 - not normalize, 1 - normalize, default 0
  # matrixThresh: peaks find matrix value threshold, default 0.7
  # diagMin: peak minimize distance threshold, default 3
  # thresh: peak start position threshold, default 2
  # val: peak contact value threshold, default 9
  # pvalueThresh: peak pvalue threshold, default 0.05
  # return loop pairs
  
  # save unmapping region
  idx <- which(colSums(h)==0)
  N <- nrow(h)
  
  if(isNorm==0){
    # no normalization
    hn <- h
  }else{
    # do Knight-Ruiz normalization
    htmp1 <- KnightRuizNorm(h)
    if(length(idx)!=0){
      pos <- BoundaryPosition(1:nrow(htmp1),idx)
      htmp2 <- matrix(0,nrow = nrow(htmp1),ncol = N)
      hn <- matrix(0,nrow = N,ncol = N)
      htmp2[,pos] <- htmp1
      hn[pos,] <- htmp2
      rm(htmp1)
      rm(htmp2)
    }else{
      hn <- htmp1
      rm(htmp1)
    }
  }
  # ensure symmetric property
  hn <- (hn + t(hn))/2
  
  # generate domains matrix
  domain1 <- matrix(1,nrow = 5,ncol = 5)
  domain1[3,3] <- 0
  domain2 <- matrix(1,nrow = 7,ncol = 7)
  domain2[4,4] <- 0
  domain3 <- matrix(1,nrow = 9,ncol = 9)
  domain3[5,5] <- 0
  domain4 <- matrix(1,nrow = 11,ncol = 11)
  domain4[6,6] <- 0
  domain5 <- matrix(1,nrow = 13,ncol = 13)
  domain5[7,7] <- 0
  
  # median filtering
  htmp <- medianFilter(hn,0.5,domain1,5)
  htmp <- htmp + medianFilter(hn,0.5,domain2,5)
  htmp <- htmp + medianFilter(hn,0.5,domain3,5)
  htmp <- htmp + medianFilter(hn,0.5,domain4,5)
  hmn <- hn - htmp / 4
  # ensure symmetric property
  hmn <- (hmn + t(hmn))/2
  
  hmn[hmn<0] <- 0
  tmp <- cor(hmn)
  tmp[is.na(tmp)] <- -1
  hmnNew <- hmn*tmp
  
  # fast peak search algorithm
  p <- fastPeakSearch(hmnNew,matrixThresh)
  s <- seq(1,length(p),2)
  x <- p[s]
  s <- seq(2,length(p),2)
  y <- p[s]
  
  flag <- (y-x>diagMin) & (x>thresh) & (y<N-thresh+1) & (h[x+N*(y-1)]>val)
  x <- x[flag]
  y <- y[flag]
  
  # remove peaks in unmapping region
  if(length(x)>0){
    meancutoff <- mean(colSums(h))/5
    unmapping <- which(colSums(h)<meancutoff)
    if(length(unmapping)!=0){
      xdist <- vector()
      ydist <- vector()
      for(i in 1:length(x)){
        xdist <- c(xdist,min(abs(unmapping-x[i])))
        ydist <- c(ydist,min(abs(unmapping-y[i])))
      }
      flag <- (xdist>2) & (ydist>2)
      x <- x[flag]
      y <- y[flag]
    }
  }
  
  # compute p-value
  if(length(x)>0){
    pvalue <- peakPValue(hmnNew,x,y,thresh)
    idx <- (pvalue<pvalueThresh)
    x <- x[idx]
    y <- y[idx]
    pvalue <- pvalue[idx]
    peak <- as.data.frame(cbind(x,y,pvalue))
  }else{
    pvalue <- numeric()
    peak <- as.data.frame(cbind(x,y,pvalue))
  }
  return(peak)
}
  
fastPeakSearch <- function(A,thresh){
  # fast peak search algorithm
  # A: input 2D matrix
  # thresh: a number between 0 and max(A) to remove background
  # return a vector of coordinates of peaks(x1,y1,x2,y2,...)
  
  edg <- 5
  s <- dim(A)
  xy <- which(A[edg:(s[1]-edg),edg:(s[2]-edg)]>thresh,arr.ind = TRUE)
  x <- xy[,1] + edg - 1
  y <- xy[,2] + edg - 1
  
  # initialize outputs
  p <- vector()
  
  # find peaks
  for(j in 1:length(y)){
    xj <- x[j]
    yj <- y[j]
    if(A[xj,yj]>=A[xj-1,yj-1] & A[xj,yj]>=A[xj-1,yj] &
       A[xj,yj]>=A[xj-1,yj+1] & A[xj,yj]>=A[xj,yj-1] &
       A[xj,yj]>=A[xj,yj+1] & A[xj,yj]>=A[xj+1,yj-1] &
       A[xj,yj]>=A[xj+1,yj] & A[xj,yj]>=A[xj+1,yj+1] &
       A[xj,yj]>=A[xj-2,yj-2] & A[xj,yj]>=A[xj-2,yj-1] &
       A[xj,yj]>=A[xj-2,yj] & A[xj,yj]>=A[xj-2,yj+1] &
       A[xj,yj]>=A[xj-2,yj+2] & A[xj,yj]>=A[xj-1,yj-2] &
       A[xj,yj]>=A[xj-1,yj+2] & A[xj,yj]>=A[xj,yj-2] &
       A[xj,yj]>=A[xj,yj+2] & A[xj,yj]>=A[xj+1,yj-2] &
       A[xj,yj]>=A[xj+1,yj+2] & A[xj,yj]>=A[xj+2,yj-2] &
       A[xj,yj]>=A[xj+2,yj-1] & A[xj,yj]>=A[xj+2,yj] &
       A[xj,yj]>=A[xj+2,yj+1] & A[xj,yj]>=A[xj+2,yj+2]){
      p <- c(p,yj,xj)
    }
  }
  return(p)
}

medianFilter <- function(A,order,domain,thresh){
  # find median filtering matrix given specific domain
  # A: input 2D raw matrix
  # domain: template matrix
  # thresh: a number between 0 and max(A) to remove background
  
  # defaults
  B <- A
  domainSize <- dim(domain)
  if(order<1){
    order <- ceiling(domainSize[1]*domainSize[2]*order)
  }
  
  center <- floor((domainSize + 1)/2)
  rc <- which(domain==1,arr.ind = TRUE)
  r <- rc[,1]
  c <- rc[,2]
  r <- r - center[1]
  c <- c - center[2]
  padSize <- max(max(abs(r)),max(abs(c)))
  originalSize <- dim(A)
  
  # extend matrix
  l <- nrow(A)
  tmp <- matrix(0,nrow = l,ncol = padSize)
  A <- cbind(tmp,A)
  A <- cbind(A,tmp)
  l <- ncol(A)
  tmp <- matrix(0,nrow = padSize,ncol = l)
  A <- rbind(tmp,A)
  A <- rbind(A,tmp)
  
  Ma <- dim(A)[1]
  offsets <- c*Ma + r
  
  xy <- which(A>thresh,arr.ind = TRUE)
  x <- xy[,1]
  y <- xy[,2]
  upTri <- (y>x & x>padSize)
  x <- x[upTri]
  y <- y[upTri]
  
  # find window's median element
  for(i in 1:length(y)){
    visit <- A[(x[i]-padSize):(x[i]+padSize),(y[i]-padSize):(y[i]+padSize)]
    vals <- as.vector(visit)
    vals <- sort(vals[-(length(vals)+1)/2])
    B[x[i]-padSize,y[i]-padSize] <- vals[order]
  }
  B[lower.tri(B)] <- 0
  dg <- diag(B)
  B <- B + t(B) - diag(dg)
  
  return(B)
}

peakPValue <- function(h,x,y,thresh){
  # compute peak's pvalue using diag exponential distribution fit
  # h: input 2D matrix
  # x,y: peak location
  # thresh: peak start position threshold
  
  N <- nrow(h)
  hmod <- matrix(0,nrow = N+N-1,ncol = N)
  logVec <- c(rep(0,N-1),rep(1,N),rep(0,N-1))
  indMat <- matrix(rep(1:N,N+N-1),nrow = N+N-1,byrow = TRUE)
  indMat <- indMat + matrix(rep(0:(N+N-2),N),ncol = N)
  logMat <- matrix(logVec[indMat],nrow = N+N-1)
  hmod[which(logMat==1)] <- h
  
  l <- ncol(hmod)
  EXPpars <- rep(0,max(y-x))
  for(i in min(y-x):max(y-x)){
    tmp <- hmod[l-i,(i+1):l]
    q <- quantile(tmp,c(0.05,0.99999))
    idx <- which(tmp>max(q[1],0) & tmp<q[2])
    tmp <- tmp[idx]
    if(length(tmp)>2*thresh){
      # exponential distribution fit
      fp <- fitdistr(tmp,"exponential")
      EXPpars[i] <- as.numeric(fp)[1]
    }
  }
  pvalue <- exp(-EXPpars[y-x]*h[x+N*(y-1)])
  
  # BHFDR calibration
  pvalue <- p.adjust(pvalue,"BH")
  return(pvalue)
}

LoopsIdentifyByTADs <- function(reads,chrName,res=100000,normType=4,window=2000000,initMethod=1,modelMethod=1,
                          multiRes=c(40,20,10,5)*1000,isNorm=0,matrixThresh=0.7,diagMin=3,thresh=2,val=9,
                          pvalueThresh=0.01){
  # chromosome contact loops identification at TAD level
  # reads: input one chromosome HiC bedfile
  # chrName: input chromosome name
  # normType: normalization type, 1-VCNorm, 2-SqrtVCNorm, 3-SinkhornKnoppNorm, 
  #           4-KnightRuizNorm, default 4
  # res: whole matrix resolution, default 100Kb
  # window: direction index computation parameter, default 2Mb
  # initMethod: parameters initialization method, 1 - heuristic initialization
  #             2 - mixture gaussian fit initialization
  #             3 - random initialization, default 1
  # modelMethod: model selection method, 1-AIC, 2-BIC, 3-Loglikelihood, default 1
  # multiRes: dynamic resolution vector, default 40Kb,20Kb,10Kb,5Kb
  # isNorm: whether to do normalization for local TAD matrix, 0-no, 1-yes, default 0
  # matrixThresh: peaks find matrix value threshold, default 0.7
  # diagMin: peak minimize distance threshold, default 3
  # thresh: peak start position threshold, default 2
  # val: peak contact value threshold, default 9
  # pvalueThresh: peak pvalue threshold, default 0.05
  # return peaks list at different resolution
  
  # generate HiC contact matrix
  message("Generate HiC raw contact matrix...\n")
  readsJudge(reads)
  H <- readsToMatrix(reads,res)
  
  # Save unmapping region index
  idx <- which(colSums(H)==0)
  
  # Normalization
  message("Do normalization...\n")
  if(normType==1){
    Hn <- VCNorm(H)
  }else if(normType==2){
    Hn <- SqrtVCNorm(H)
  }else if(normType==3){
    Hn <- SinkhornKnoppNorm(H)
  }else if(normType==4){
    Hn <- KnightRuizNorm(H)
  }else{
    stop("wrong normalization type parameter")
  }
  Hn <- shrinkVariation(Hn)
  
  # DI+HMM model
  message("DI+HMM model to identify TADs...\n")
  TAD_boundary <- HMMTADIdentify(Hn,res,window,initMethod,modelMethod)
  
  # extract TAD region reads 
  Position <- boundaryPosition(TAD_boundary,idx)
  TADReads <- extractTADReads(reads,TAD_boundary,idx,res)
  Position[1] <- 0
  
  # Loops identification at TAD level
  peaks <- list()
  for(i in 1:length(multiRes)){
    peaks <- c(peaks,list(list()))
  }
  
  message("Loops identification at TAD level...\n")
  for(k in 1:length(TADReads)){
    message(paste("Identifying contact loops in TAD ",k,"...",sep = ""))
    s <- Position[k]
    e <- Position[k+1]
    h <- multiResolutionTADMatrix(TADReads[[k]],s,e,res,multiRes)
    temp <- Position[k]
    for(i in 1:length(h)){
      offset <- floor(temp*res/multiRes[i])
      localPeaks <- TADLoopsIdentify(h[[i]],isNorm,matrixThresh,diagMin,thresh,val,pvalueThresh)
      localPeaks$x <- localPeaks$x + offset
      localPeaks$y <- localPeaks$y + offset
      if(nrow(localPeaks)!=0){
        localPeaks <- cbind(chrName,localPeaks)
        colnames(localPeaks) <- c("chr","x","y","pvalue")
      }
      peaks[[i]] <- c(peaks[[i]],list(localPeaks))
    }
  }
  peaks <- loopsProcess(peaks,multiRes)
  return(peaks)
}

LoopsIdentifyByChromosome <- function(reads,chrName,res=10000,isNorm=0,matrixThresh=0.7,diagMin=3,thresh=2,val=9,
                                      pvalueThresh=0.01){
  # chromosome contact loops identification in whole chromosome
  # reads: input one chromosome reads data.frame
  # chrName: input chromosome name
  # res: resolution parameter, default 10Kb
  # isNorm: whether to do normalization for whole chromosome contact matrix, 
  #         0-no, 1-yes, default 0
  # matrixThresh: peaks find matrix value threshold, default 0.7
  # diagMin: peak minimize distance threshold, default 3
  # thresh: peak start position threshold, default 2
  # val: peak contact value threshold, default 9
  # pvalueThresh: peak pvalue threshold, default 0.05
  # return loops data.frame with x,y and pvalue
  
  # generate HiC contact matrix
  message("Generate HiC raw contact matrix...\n")
  readsJudge(reads)
  H <- readsToMatrix(reads,res)
  
  peaks <- TADLoopsIdentify(H,isNorm,matrixThresh,diagMin,thresh,val,pvalueThresh)
  peaks <- cbind(chrName,peaks)
  colnames(peaks) <- c("chr","x2","y2","pvalue")
  peaks$x1 <- peaks$x2 - 1
  peaks$y1 <- peaks$y2 - 1
  z <- peaks$x1
  peaks$x1 <- peaks$x2
  peaks$x2 <- z
  z <- peaks$y2
  peaks$y2 <- peaks$x1
  peaks$x1 <- z
  z <- peaks$pvalue
  peaks$pvalue <- peaks$y1
  peaks$y1 <- z
  colnames(peaks) <- c("chr","x1","x2","y1","y2","pvalue")
  peaks$x1 <- res*peaks$x1
  peaks$x2 <- res*peaks$x2
  peaks$y1 <- res*peaks$y1
  peaks$y2 <- res*peaks$y2
  return(peaks)
}

loopsProcess <- function(loops,multiRes = c(40000,20000,10000,5000)){
  # merge chromatin loops at TAD level
  # multiRes: multi-resolution vector, default 40Kb,20Kb,10Kb,5Kb
  # loops: input chromatin loops
  
  for(i in 1:length(loops)){
    temp <- data.frame()
    for(j in 1:length(loops[[i]])){
      temp <- rbind(temp,loops[[i]][[j]])
    }
    colnames(temp) <- c("chr","x2","y2","pvalue")
    tmp1 <- temp$x2
    tmp2 <- temp$y2
    if(multiRes[i]==40000){
      temp$x1 <- tmp1 - 1
      temp$x2 <- tmp1 + 1
      temp$y1 <- tmp2 - 1
      temp$y2 <- tmp2 + 1
    }else if(multiRes[i]==20000){
      temp$x1 <- tmp1 - 1
      temp$x2 <- tmp1 + 1
      temp$y1 <- tmp2 - 1
      temp$y2 <- tmp2 + 1
    }else if(multiRes[i]==10000){
      temp$x1 <- tmp1 - 2
      temp$x2 <- tmp1 + 2
      temp$y1 <- tmp2 - 2
      temp$y2 <- tmp2 + 2
    }else{
      temp$x1 <- tmp1 - 2
      temp$x2 <- tmp1 + 2
      temp$y1 <- tmp2 - 2
      temp$y2 <- tmp2 + 2
    }
    #temp$x1 <- temp$x2 - 1
    #temp$y1 <- temp$y2 - 1
    z <- temp$x2
    temp$x2 <- temp$x1
    temp$x1 <- z
    z <- temp$y2
    temp$y2 <- temp$x1
    temp$x1 <- z
    z <- temp$pvalue
    temp$pvalue <- temp$y1
    temp$y1 <- z
    colnames(temp) <- c("chr","x1","x2","y1","y2","pvalue")
    temp$x1 <- temp$x1*multiRes[i]
    temp$x2 <- temp$x2*multiRes[i]
    temp$y1 <- temp$y1*multiRes[i]
    temp$y2 <- temp$y2*multiRes[i]
    temp$type <- paste(multiRes[i]/1000,"Kb",sep = "")
    loops[[i]] <- temp
  }
  return(loops)
}

loopsMerge <- function(loops,multiRes = c(40000,20000,10000,5000)){
  # merge loops at different resolution level
  # loops: input chromatin loops
  # multiRes: multi-resolution vector, default 40Kb,20Kb,10Kb,5Kb
  
  if(length(multiRes)>4){
    stop("input invalid multi-resolution parameter")
  }
  
  pos1 <- as.matrix(cbind(loops[[1]]$x1,loops[[1]]$x2,loops[[1]]$y1,loops[[1]]$y2))
  pos2 <- as.matrix(cbind(loops[[2]]$x1,loops[[2]]$x2,loops[[2]]$y1,loops[[2]]$y2))
  
  require(Rcpp)
  cppFunction('
    NumericVector loopsMatch(NumericMatrix pos1,NumericMatrix pos2){
      int n1 = pos1.nrow();
      int n2 = pos2.nrow();
      NumericVector res(n1);
      bool flag;
      for(int i=0;i<n1;i++){
        flag = FALSE;
        for(int j=0;j<n2;j++){
          if((!(pos2(j,0)>pos1(i,1) | pos2(j,1)<pos1(i,0))) & (!(pos2(j,2)>pos1(i,3) | pos2(j,3)<pos1(i,2)))){
            flag = TRUE;
            break;
          }
        }
        if(flag==TRUE){
          res(i) = 0;
        }else{
          res(i) = 1;
        }
      }
      return res;
    }
  ')
  
  # merge loops in 40Kb,20Kb
  flag <- loopsMatch(pos1,pos2)
  idx <- which(flag==1)
  loops1 <- loops[[1]][idx,]
  loops1 <- rbind(loops1,loops[[2]])
  
  # merge loops in 10Kb,5Kb
  l <- length(multiRes)
  pos1 <- as.matrix(cbind(loops[[l-1]]$x1,loops[[l-1]]$x2,loops[[l-1]]$y1,loops[[l-1]]$y2))
  pos2 <- as.matrix(cbind(loops[[l]]$x1,loops[[l]]$x2,loops[[l]]$y1,loops[[l]]$y2))
  flag <- loopsMatch(pos1,pos2)
  idx <- which(flag==1)
  loops2 <- loops[[l-1]][idx,]
  loops2 <- rbind(loops2,loops[[l]])
  
  # get final loops set
  pos1 <- as.matrix(cbind(loops1$x1,loops1$x2,loops1$y1,loops1$y2))
  pos2 <- as.matrix(cbind(loops2$x1,loops2$x2,loops2$y1,loops2$y2))
  flag <- loopsMatch(pos1,pos2)
  idx <- which(flag==1)
  loops1 <- loops1[idx,]
  loops <- rbind(loops1,loops2)
  row.names(loops) <- 1:nrow(loops)
  return(loops)
}
















