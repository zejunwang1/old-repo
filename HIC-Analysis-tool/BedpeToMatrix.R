
###################################################################################
# Bedpe file to HiC contact matrix
# Powerlaw curve plot
# Multi-resolution region matrix generation

# These functions implement following requirements:
# (1) one-chromosome / whole-genome bedpe file converting to 2D contact matrix
#     at fixed resolution
# (2) one-chromosome / whole-genome bedpe file converting to sparse matrix
#     at fixed resolution
# (3) Pearson correlation coefficient of two HiC matrix
# (4) Powerlaw curve plot
# (5) Extract region reads according to TADs boundaries and dynamic-resolution
#     region matrix generation
# (6) Hierarchical TADs identification
# (7) 1M bin whole genome square / sparse matrix generation


# Input bedpe file should be the following format (the last column can be missing):
# [chromosome1] [position1] [strand1] [chromosome2] [position2] [strand2] [type]
#     chr19        99490        +         chr19      2452045        -     Intra
#     chr19       103769        -         chr19     15286108        -     Intra
#     chr19       103803        -         chr19     54758931        -     Intra
#      ...          ...        ...         ...         ...         ...     ...

# Implemented by
# Zejun Wang
# Tsinghua University, Beijing
# mails: xiaolijun_thu@163.com

###################################################################################

readsJudge <- function(reads){
  # judge reads whether belong to one chromosome
  # reads: input reads data.frame
  
  if(class(reads)[2]!="data.frame"){
    stop("wrong input format")
  }
  
  chra <- unique(reads$V1)
  chrb <- unique(reads$V4)
  if(length(chra)!=1 | length(chrb)!=1 | chra[1]!=chrb[1]){
    stop("input reads include multiple chromosomes")
  }
}

readsToMatrix <- function(reads,res=100000){
  # one chromosome reads to hic raw contact matrix
  # reads: input chromosome reads data.frame
  # res: resolution, default 100Kb
  
  readsJudge(reads)

  reads$V2 <- floor(reads$V2/res) + 1
  reads$V5 <- floor(reads$V5/res) + 1
  l <- max(reads$V2,reads$V5)

  flag <- duplicated(reads,by = c("V1","V2","V4","V5"))
  reads[,':=' (freq = .N), by = c("V1","V2","V4","V5")]
  reads <- reads[!flag,]
  temp <- as.matrix(cbind(reads$V2,reads$V5,reads$freq))

  require(Rcpp)
  cppFunction('
    NumericMatrix toMatrixCore(NumericMatrix M,int l){
      NumericMatrix H(l,l);
      for(int i=1;i<=M.nrow();i++){
        H(M(i-1,0)-1,M(i-1,1)-1) = M(i-1,2);
      }
      return H;
    }
  ')
  H <- toMatrixCore(temp,l)
  dg <- diag(H)
  H <- H + t(H) - diag(dg)
  return(H)
}

readsToSparseMatrix <- function(reads,res=100000){
  # convert one chromosome reads to hic sparse matrix
  # reads: input chromosome reads data.frame
  # res: resolution, default 100Kb
  
  readsJudge(reads)
  reads$V2 <- floor(reads$V2/res) + 1
  reads$V5 <- floor(reads$V5/res) + 1
  l <- max(reads$V2,reads$V5)
  
  flag <- duplicated(reads,by = c("V1","V2","V4","V5"))
  reads[,':=' (freq = .N), by = c("V1","V2","V4","V5")]
  reads <- reads[!flag,]
  
  idx <- which(reads$V2>reads$V5)
  temp <- reads[idx,]$V2
  reads[idx,]$V2 <- reads[idx,]$V5
  reads[idx,]$V5 <- temp
  
  sparseH <- sparseMatrix(i = reads$V2,j = reads$V5,x = reads$freq,dims = c(l,l),symmetric = TRUE)
  return(sparseH)
}


bedpeToMatrixOneChrom <- function(bedfile,res=100000){
  # convert one chromosome bedpe file to hic raw contact matrix
  # bedfile: input bedpe file path
  # res: resolution, default 100Kb

  reads <- fread(bedfile)
  H <- readsToMatrix(reads,res)
  return(H)
}

bedpeToSparseMatrixOneChrom <- function(bedfile,res=100000){
  # convert one chromosome bedpe file to hic raw sparse matrix
  # bedfile: input bedpe file path
  # res: resolution, default 100Kb

  reads <- fread(bedfile)
  sparseH <- readsToSparseMatrix(reads,res)
  return(sparseH)
}

bedpeToMatrixWholeChrom <- function(bedfile,res=100000){
  # convert whole genome bedpe file to hic raw contact matrix list
  # bedfile: input bedpe file path
  # res: resolution, default 100Kb

  reads <- fread(bedfile)
  reads <- readsPreprocess(reads)

  H <- list()
  chr <- unique(reads$V1)
  
  # min reads number threshold
  thresh <- 1000
  name <- vector()
  for(i in 1:length(chr)){
    message(paste("Extracting contact matrix for ",chr[i],"...",sep = ""))
    idx <- which(reads$V1==chr[i] & reads$V4==chr[i])
    if(length(idx)>thresh){
      subreads <- reads[idx,]
      intra <- readsToMatrix(subreads,res)
      H <- c(H,list(intra))
      name <- c(name,chr[i])
    }
  }

  names(H) <- name
  return(H)
}

bedpeToSparseMatrixWholeChrom <- function(bedfile,res=100000){
  # convert whole genome bedpe file to hic sparse matrix list
  # bedfile: input bedpe file path
  # res: resolution, default 100Kb

  reads <- fread(bedfile)
  reads <- readsPreprocess(reads)

  sparseH <- list()
  chr <- unique(reads$V1)
  
  # min reads number threshold
  thresh <- 1000
  name <- vector()
  for(i in 1:length(chr)){
    message(paste("Extracting sparse matrix for ",chr[i],"...",sep = ""))
    idx <- which(reads$V1==chr[i] & reads$V4==chr[i])
    if(length(idx)>thresh){
      subreads <- reads[idx,]
      intra <- readsToSparseMatrix(subreads,res)
      name <- c(name,chr[i])
      sparseH <- c(sparseH,list(intra))
    }
  }

  names(sparseH) <- name
  return(sparseH)
}

readsPreprocess <- function(reads){
  # reads: input reads data.frame

  if(class(reads)[2]!="data.frame" | (length(reads)!=6 & length(reads)!=7)){
    stop("wrong input format")
  }

  # select intra reads
  idx <- which(reads$V1==reads$V4)
  reads <- reads[idx,]
  
  # remove invalid reads
  idx <- which(reads$V1=="chrM" | reads$V1=="chrMT" | reads$V4=="chrM" | reads$V4=="chrMT")
  reads <- reads[-idx,]
  return(reads)
}

computeCorrelation <- function(hic1,hic2,method=1){
  # compute correlation coefficient of two hic sparse matrix
  # hic1/hic2: input hic sparse matrix
  # method: 1-pearson, 2-spearman, 3-kendall

  # convert sparse matrix to data.frame
  hic1 <- data.frame(summary(hic1))
  hic2 <- data.frame(summary(hic2))
  colnames(hic1) <- c("i","j","freq")
  colnames(hic2) <- c("i","j","freq")
  df <- merge(hic1,hic2,all = T,by = c("i","j"))
  df[is.na(df)] <- 0
  
  if(method==1){
    type <- "pearson"
  }else if(method==2){
    type <- "spearman"
  }else if(method==3){
    type <- "kendall"
  }else{
    stop("wrong method parameter")
  }
  coeff <- cor(df$freq.x,df$freq.y,method = type)

  return(coeff)
}


extractTADReads <- function(reads,boundary,idx,res){
  # extract TAD reads according to TAD boundaries
  # reads: input one chromosome reads data.frame
  # idx: input unmapping region index
  # boundary: input TAD boundaries
  # res: resolution

  readsJudge(reads)

  pos <- boundaryPosition(boundary,idx)
  pos[1] <- 0
  temp1 <- floor(reads$V2/res) + 1
  temp2 <- floor(reads$V5/res) + 1
  regionReads <- list()
  for(i in 1:(length(pos)-1)){
    regionReads <- c(regionReads,list(data.frame()))
  }

  for(i in 1:(length(pos)-1)){
    ind1 <- (temp1>pos[i] & temp1<=pos[i+1])
    ind2 <- (temp2>pos[i] & temp2<=pos[i+1])
    ind <- which(ind1==TRUE & ind2==TRUE & ind1==ind2)
    regionReads[[i]] <- rbind(regionReads[[i]],reads[ind,])
  }

  return(regionReads)
}

boundaryPosition <- function(boundary,idx){
  # compute true chromosome position for TAD boundaries
  # boundary: input TADs boundaries
  # idx: input unmapping region index
  
  l <- length(boundary)
  for(i in 1:length(idx)){
    ind <- binarySearch(boundary,idx[i])
    if(ind>l){
      ind <- l
    }
    boundary[ind:l] <- boundary[ind:l] + 1
  }
  return(boundary)
}

binarySearch <- function(nums,val){
  # binary search element in a vector
  # nums: input vector
  # val: target value
  # return index of element firstly larger than target
  
  low <- 1
  high <- length(nums)
  while(low<=high){
    mid <- floor((low+high)/2)
    if(nums[mid]==val){
      return(mid)
    }else if(nums[mid]<val){
      low <- mid + 1
    }else{
      high <- mid - 1
    }
  }
  return(low)
}

multiResolutionTADMatrix <- function(reads,s=0,e=0,TADRes=0,res=c(100,40,20,10,5)*1000){
  # generate multi-resolution TAD region matrix
  # reads: input TAD region reads data.frame by extractTADReads function
  # s: TAD start position
  # e: TAD end position
  # TADRes: TAD-level resolution
  # res: multi-resolution vector, default 100Kb,40Kb,20Kb,10Kb,5Kb

  H <- list()
  for(i in 1:length(res)){
    message(paste("Generate contact matrix in resolution ",res[i]/1000,"Kb...",sep = ""))
    temp <- reads
    temp$V2 <- floor(temp$V2/res[i]) + 1
    temp$V5 <- floor(temp$V5/res[i]) + 1
    flag <- duplicated(temp,by = c("V1","V2","V4","V5"))
    temp[,':='(freq = .N),by = c("V1","V2","V4","V5")]
    temp <- temp[!flag,]

    # get local matrix dimension
    if(s==0 & e==0 & TADRes==0){
      offset <- min(temp$V2,temp$V5) - 1
      l <- max(temp$V2,temp$V5) - offset
    }else{
      offset <- floor(s*TADRes/res[i])
      l <- max(temp$V2,temp$V5) - offset
    }
    
    temp$V2 <- temp$V2 - offset
    temp$V5 <- temp$V5 - offset
    temp <- as.matrix(cbind(temp$V2,temp$V5,temp$freq))
    
    loc <- toMatrixCore(temp,l)
    dg <- diag(loc)
    loc <- loc + t(loc) - diag(dg)
    H <- c(H,list(loc))
  }
  rm(temp)
  names(H) <- res

  return(H)
}

resolutionForTAD <- function(H,threshold=1,method=1){
  # fit a reasonable resolution for sub-TADs identification
  # H: input a list of multi-resolution TAD matrix
  # threshold: TAD mean contact density threshold, default 1
  # method: 1-not remove unmapping region, 2-remove unmapping region, default 1
  # return resolution according to mean contact density >= threshold
  
  l <- length(H)
  md <- rep(0,l)
  for(i in 1:l){
    H[[i]] <- H[[i]] - diag(diag(H[[i]]))
    if(method==2){
      # remove unmapping region
      idx <- which(colSums(H[[i]])==0)
      if(length(idx)!=0){
        H[[i]] <- H[[i]][-idx,]
        H[[i]] <- H[[i]][,-idx]
      }
    }
    md[i] <- mean(H[[i]])
  }
  names(md) <- names(H)
  md <- sort(md)
  idx <- binarySearch(md,threshold)
  res <- as.numeric(names(md)[idx])
  return(res)
}

HierarchicalTAD <- function(reads,res=100000,normType=4,window=1000000,initMethod=1,modelMethod=1,
                             multiRes=c(100,40,20,10,5)*1000){
  # hierarchical TAD identification
  # reads: input one chromosome reads data.frame
  # normType: normalization type, 1-VCNorm, 2-SqrtVCNorm, 3-SinkhornKnoppNorm,
  #           4-KnightRuizNorm, default 4
  # res: whole contact matrix resolution, default 100Kb
  # window: direction index computation parameter
  # initMethod: parameters initialization method, 1 - heuristic initialization
  #             2 - mixture gaussian fit initialization
  #             3 - random initialization, default 1
  # modelMethod: model selection method, 1-AIC, 2-BIC, 3-Loglikelihood, default 1
  # multiRes: multi-resolution vector, default 100Kb,40Kb,20Kb,10Kb,5Kb
  # return hierarchical TAD boundary list

  readsJudge(reads)

  # get raw contact matrix in low resolution
  H <- readsToMatrix(reads,res)

  # save unmapping region index
  idx <- which(colSums(H)==0)

  # Normalization
  if(normType==1){
    H <- VCNorm(H)
  }else if(normType==2){
    H <- SqrtVCNorm(H)
  }else if(normType==3){
    H <- SinkhornKnoppNorm(H)
  }else if(normType==4){
    H <- KnightRuizNorm(H)
  }else{
    stop("wrong normalization type parameter")
  }
  H <- shrinkVariation(H)

  # DI+HMM model
  TAD_boundary <- HMMTADIdentify(H,res,window,initMethod,modelMethod)

  # Extract region reads according to TADs boundaries
  Position <- boundaryPosition(TAD_boundary,idx)
  hier_bd <- list()
  hier_bd <- c(hier_bd,list(Position),list(idx),list(res))
  TADReads <- extractTADReads(reads,TAD_boundary,idx,res)
  rm(reads)

  # generate multi-resolution TAD matrix
  subTAD <- list()
  name <- character()
  Position[1] <- 0
  for(k in 1:length(TADReads)){
    message(paste("\nComputing sub-TADs for ",TADReads[[1]]$V1[1]," in region ",k,"...",sep = ""))
    s <- Position[k]
    e <- Position[k+1]
    h <- multiResolutionTADMatrix(TADReads[[k]],s,e,res,multiRes)
    offset <- Position[k]

    # fit a reasonable resolution
    subres <- resolutionForTAD(h)
    ind <- which(as.numeric(names(h))==subres)
    h <- h[[ind]]

    # remove low mapping region
    threshold <- nrow(h)/10
    sub_idx <- which(colSums(h)<=threshold)
    if(length(sub_idx)!=0){
      h <- h[-sub_idx,]
      h <- h[,-sub_idx]
    }

    # Local normalization
    if(normType==1){
      h <- VCNorm(h)
    }else if(normType==2){
      h <- SqrtVCNorm(h)
    }else if(normType==3){
      h <- SinkhornKnoppNorm(h)
    }else if(normType==4){
      h <- KnightRuizNorm(h)
    }else{
      stop("wrong normalization type parameter")
    }
    h <- shrinkVariation(h)

    # subTAD identification using heuristic initialization
    sub_window <- 100000
    sub_boundary <- HMMTADIdentify(h,subres,sub_window,initMethod,modelMethod)
    if(length(sub_idx)!=0){
      subPosition <- boundaryPosition(sub_boundary,sub_idx)
    }else{
      subPosition <- sub_boundary
    }

    # get local subTAD position
    if(k!=1){
      subPosition[1] <- 0
    }
    x <- diff(subPosition)
    if(k!=1){
      subPosition[1] <- offset*res/subres
    }
    for(i in 1:(length(subPosition)-1)){
      subPosition[i+1] <- subPosition[i] + x[i]
    }

    subTAD <- c(subTAD,list(subPosition))
    name <- c(name,paste("region",k,"_boundary",sep = ""))

    # save unmapping region
    if(length(sub_idx)!=0){
      for(i in 1:length(sub_idx)){
        sub_idx[i] <- offset*res/subres + sub_idx[i]
      }
      subTAD <- c(subTAD,list(sub_idx))
      name <- c(name,paste("region",k,"_unmapping",sep = ""))
    }
    subTAD <- c(subTAD,list(subres))
    name <- c(name,paste("region",k,"_resolution",sep = ""))
  }
  names(subTAD) <- name
  hier_bd <- c(hier_bd,list(subTAD))
  names(hier_bd) <- c("wholeChromosome","unmapping","resolution","subTAD")
  return(hier_bd)
}

sampleReads <- function(reads,num){
  # sample reads
  # reads: input reads data.frame
  # num: a positive number, the number of items to choose from
  
  if(nrow(reads)<num){
    stop("reads number not enough")
  }
  idx <- sample(1:nrow(reads),num)
  reads <- reads[idx,]
  return(reads)
}

mergeReads <- function(reads1,reads2){
  # merge reads
  # reads1/reads2: input reads data.frame
  
  if(length(reads1)!=length(reads2)){
    stop("column number not match")
  }
  reads <- rbind(reads1,reads2)
  flag <- duplicated(reads)
  reads <- reads[!flag,]
  return(reads)
}

extractChromReads <- function(reads,chrName){
  # extract intra chromosome reads
  # reads: input whole genome reads data.frame
  # chrName: chromosome name
  
  reads <- readsPreprocess(reads)
  idx <- which(reads$V1==chrName)
  reads <- reads[idx,]
  return(reads)
}

extractInterReads <- function(reads,chra,chrb){
  # extract inter chromosome reads
  # reads: input whole genome reads data.frame
  # chra/chrb: chromosome name
  
  if(class(reads)[2]!="data.frame" | (length(reads)!=6 & length(reads)!=7)){
    stop("wrong input format")
  }
  
  idx <- which((reads$V1==chra & reads$V4==chrb) | (reads$V1==chrb & reads$V4==chra))
  reads <- reads[idx,]
  return(reads)
}

readsToMatrixWholeGenome <- function(reads,type=1){
  # convert whole genome reads to contact matrix
  # reads: input whole genome reads data.frame
  # type: human or mouse, 1-human, 2-mouse, default 1
  
  reads <- readsPreprocess(reads)
  
  if(type!=1 & type!=2){
    stop("not human or mouse genome data")
  }
  
  # get chromosome length vector
  if(type==1){
    require(BSgenome.Hsapiens.UCSC.hg19)
    chrs <- seqlevels(BSgenome.Hsapiens.UCSC.hg19)
    chrs <- chrs[grep("chr\\d+$|chrX$",chrs)]
    chroml <- rep(0,length(chrs))
    for(i in 1:length(chrs)){
      message("Computing length of human chromosome ",i,"...",sep = "")
      chroml[i] <- length(BSgenome.Hsapiens.UCSC.hg19[[chrs[i]]])
    }
  }else{
    require(BSgenome.Mmusculus.UCSC.mm10)
    chrs <- seqlevels(BSgenome.Mmusculus.UCSC.mm10)
    chrs <- chrs[grep("chr\\d+$|chrX$",chrs)]
    chroml <- rep(0,length(chrs))
    for(i in 1:length(chrs)){
      message("Computing length of mouse chromosome ",i,"...",sep = "")
      chroml[i] <- length(BSgenome.Mmusculus.UCSC.mm10[[chrs[i]]])
    }
  }
  
  # delete reads belong to chrY
  idx <- which(reads$V1=="chrY" | reads$V4=="chrY")
  if(length(idx)!=0){
    reads <- reads[-idx,]
  }
  
  # delete invalid reads
  idx <- which(reads$V1=="chrM" | reads$V4=="chrM" | reads$V1=="chrMT" | reads$V4=="chrMT")
  if(length(idx)!=0){
    reads <- reads[-idx,]
  }
  
  # set 1Mb resolution
  res <- 1000000
  chroml <- floor(chroml/res) + 1
  l <- sum(chroml[1:(length(chroml))])
  chroml <- cumsum(chroml)
  chroml <- c(0,chroml)
  require(Matrix)
  reads$V2 <- floor(reads$V2/res) + 1
  reads$V5 <- floor(reads$V5/res) + 1
  flag <- duplicated(reads,by = c("V1","V2","V4","V5"))
  reads[,':='(freq = .N),by = c("V1","V2","V4","V5")]
  reads <- reads[!flag,]
  
  n <- length(unique(reads$V1))-1
  for(i in 2:n){
    chr <- paste("chr",i,sep = "")
    idx <- which(reads$V1==chr)
    reads[idx,]$V2 <- reads[idx,]$V2 + chroml[i]
    idx <- which(reads$V4==chr)
    reads[idx,]$V5 <- reads[idx,]$V5 + chroml[i]
  }
  idx <- which(reads$V1=="chrX")
  reads[idx,]$V2 <- reads[idx,]$V2 + chroml[n+1]
  idx <- which(reads$V4=="chrX")
  reads[idx,]$V5 <- reads[idx,]$V5 + chroml[n+1]
  
  idx <- which(reads$V2>reads$V5)
  if(length(idx)!=0){
    temp <- reads[idx,]$V1
    reads[idx,]$V1 <- reads[idx,]$V4
    reads[idx,]$V4 <- temp
    temp <- reads[idx,]$V2
    reads[idx,]$V2 <- reads[idx,]$V5
    reads[idx,]$V5 <- temp
  }
  
  # get whole genome square matrix at 1Mb resolution
  temp <- as.matrix(cbind(reads$V2,reads$V5,reads$freq))
  h1 <- toMatrixCore(temp,l)
  
  # get whole genome sparse matrix at 1Mb resolution
  h2 <- sparseMatrix(i = reads$V2,j = reads$V5,x = reads$freq,dims = c(l,l),symmetric = TRUE)
  
  h <- list(h1,h2)
  colnames(h) <- c("square_matrix","sparse_matrix")
  return(h)
}

bedpeToMatixWholeGenome <- function(bedfile,type=1){
  # convert whole genome bedpe file to square matrix and sparse matrix at 1Mb resolution
  # bedfile: input hic whole genome bedpe file
  # type: human or mouse, 1-human, 2-mouse, default 1

  reads <- fread(bedfile)
  h <- readsToMatrixWholeGenome(reads,type)
  return(h)
}

extractRegionReads <- function(reads,start_pos,end_pos){
  # extract local region reads
  # reads: input one chromosome reads
  # start_pos: region start position
  # end_pos: region end position
  
  readsJudge(reads)
  if(start_pos>=end_pos){
    stop("invalid position parameter")
  }
  
  idx <- which(reads$V2>=start_pos & reads$V2<end_pos & reads$V5>=start_pos & reads$V5<end_pos)
  reads <- reads[idx,]
  return(reads)
}

fixedResolutionRegion <- function(reads,start_pos,end_pos,res=40000){
  # return fixed resolution local region contact matrix
  # reads: input one chromosome reads
  # start_pos: region start position
  # end_pos: region end position
  # res: resolution, default 40Kb
  
  reads <- extractReads(reads,start_pos,end_pos)
  
  if((end_pos-start_pos)/res<10){
    stop("invalid local region parameter")
  }
  
  offset <- floor(start_pos/res)
  reads$V2 <- floor(reads$V2/res) + 1 - offset
  reads$V5 <- floor(reads$V5/res) + 1 - offset
  l <- max(reads$V2,reads$V5)
  flag <- duplicated(reads,by = c("V1","V2","V4","V5"))
  reads[,':='(freq = .N),by = c("V1","V2","V4","V5")]
  reads <- reads[!flag,]
  
  temp <- as.matrix(cbind(reads$V2,reads$V5,reads$freq))
  h <- toMatrixCore(temp,l)
  dg <- diag(h)
  h <- h + t(h) - diag(dg)
  return(h)
}

multiResolutionRegion <- function(reads,start_pos,end_pos,res = c(100,40,20,10,5)*1000){
  # return multiple resolution local region contact matrix
  # reads: input one chromosome reads
  # start_pos: region start position
  # end_pos: region end position
  # res: resolution vector, default 100Kb,40Kb,20Kb,10Kb,5Kb
  
  h <- list()
  for(i in 1:length(res)){
    temp <- fixedResolutionRegion(reads,start_pos,end_pos,res[i])
    h <- c(h,list(temp))
  }
  return(h)
}


fitPowerLaw <- function(reads,resInterval=c(seq(10,90,10),seq(100,900,100),seq(1000,10000,1000))*1000){
  # least square fit for the relationship between log(density) and log(distance)
  # log(density) = a*log(distance) + b
  # reads: input one chromosome reads, data.frame format
  # resInterval: resolution interval, default 10Kb~10Mb
  # return parameter a,b
  
  # judge reads
  readsJudge(reads)
  
  res <- 5000
  d <- resInterval/res
  reads[,':='(dist = abs(V5-V2))]
  
  # compute density
  N <- nrow(reads)
  reads$dist <- floor(reads$dist/res) + 1
  dens <- rep(0,length(d))
  for(k in 1:length(d)){
    dens[k] <- length(which(reads$dist==d[k])) / N
  }
  
  # least square fit
  df <- data.frame(log10(dens))
  df$dist = log10(resInterval)
  colnames(df) <- c("dens","dist")
  fit <- lsfit(df$dist,df$dens)
  
  return(fit$coefficients)
}





