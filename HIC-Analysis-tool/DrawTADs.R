
###################################################################################
# TADs visualization 

# Ensure package rasterVis be installed, you can install this package by the 
# following command in R console:
# install.packages("rasterVis")

# Implemented by 
# Zejun Wang
# Tsinghua University, Beijing
# mails: xiaolijun_thu@163.com

###################################################################################



DrawTADs <- function(H,TAD_boundaries){
  
  ###############################################################
  # H: Input 2D HiC normalized contact matrix
  # TAD_boundaries: TAD boundary vector
  ###############################################################
  
  l <- nrow(H)
  w <- 3         # line width
  # Color theme
  myTheme <- rasterTheme(9,region = brewer.pal(9,"YlOrRd"))
  
  ind <- 1    # domain start position
  val <- max(H)       # maximum value
  for(i in 2:length(TAD_boundaries)){
    pos1 <- min(ind+w,l)
    pos2 <- max(TAD_boundaries[i]-w,1)
    H[ind:pos1,ind:TAD_boundaries[i]] <- val
    H[ind:TAD_boundaries[i],ind:pos1] <- val
    H[pos2:TAD_boundaries[i],ind:TAD_boundaries[i]] <- val
    H[ind:TAD_boundaries[i],pos2:TAD_boundaries[i]] <- val
    ind <- TAD_boundaries[i]
  }
  
  # Plot heatmap and TADs
  levelplot(H,par.settings = myTheme,
            scales = list(x = list(cex = 1.0),
                          y = list(cex = 1.0)),
            xlab = list(label = ""),
            ylab = list(label = ""),
            colorkey = list(labels = list(cex = 1.0)))
}

DrawHeatmap <- function(H){
  # Plot HiC contact matrix heatmap
  # Color theme
  myTheme <- rasterTheme(9,region = brewer.pal(9,"YlOrRd"))
  
  # Plot heatmap
  levelplot(H,par.settings = myTheme,
            scales = list(x = list(cex = 1.0),
                          y = list(cex = 1.0)),
            xlab = list(label = ""),
            ylab = list(label = ""),
            colorkey = list(labels = list(cex = 1.0)))
}


DrawLocalIntraRegion <- function(reads,start,end,res,normType=0){
  # Draw intra-chromosome local region heatmap
  # start: start position
  # end: end position
  # res: local region resolution
  
  if(class(reads)[2]!="data.frame"){
    stop("input reads format error")
  }
  
  if(length(reads)!=6 & length(reads)!=7){
    stop("input reads format error")
  }
  
  idx <- which(reads$V1!=reads$V4)
  if(length(unique(reads$V1))!=1 | length(idx)!=0){
    stop("input reads format error, including multiple chromosomes")
  }
  
  # Judge lower and upper
  if(start<min(reads$V2,reads$V5)){
    stop("start position beyond lower bound")
  }
  if(end>max(reads$V2,reads$V5)){
    stop("end position beyond upper bound")
  }
  
  thresh <- 10
  if((end-start)/res<10){
    stop("invalid local region parameters")
  }
  
  # Select valid reads
  idx <- which(reads$V2>=start & reads$V2<end & reads$V5>=start & reads$V5<end)
  reads <- reads[idx,]
  offset <- floor(start/res)
  reads$V2 <- floor(reads$V2/res) + 1
  reads$V5 <- floor(reads$V5/res) + 1
  reads$V2 <- reads$V2 - offset
  reads$V5 <- reads$V5 - offset
  flag <- duplicated(reads,by = c("V1","V2","V4","V5"))
  reads[,':='(freq = .N),by = c("V1","V2","V4","V5")]
  reads <- reads[!flag,]
  
  # Generate local matrix
  l <- max(reads$V2,reads$V5)
  mat <- matrix(0,nrow = l,ncol = l)
  for(i in 1:nrow(reads)){
    mat[reads$V2[i],reads$V5[i]] <- reads$freq[i]
  }
  mat <- mat + t(mat) - diag(diag(mat))
  
  # Normalization
  if(normType==0){
    # Decrease the variance
    h <- DecVariation(mat)
  }else if(normType==1){
    # VC Normalization
    h <- VCNorm(mat)
    h <- DecVariation(h)
  }else if(nromType==2){
    # Sqrt VC Normalization
    h <- SqrtVCNorm(mat)
    h <- DecVariation(h)
  }else if(normType==3){
    # Sinkhorn-Knopp Normalization
    h <- SinkhornKnoppNorm(mat)
    h <- DecVariation(h)
  }else{
    # Knight-Ruiz Normalization
    h <- KnightRuizNorm(mat)
    h <- DecVariation(h)
  }
  
  # Draw local heatmap
  require(rasterVis)
  myTheme <- rasterTheme(9,region = brewer.pal(9,"YlOrRd"))
  levelplot(h,par.settings = myTheme,
            scales = list(x = list(cex = 1.0),
                          y = list(cex = 1.0)),
            xlab = list(label = ""),
            ylab = list(label = ""),
            colorkey = list(labels = list(cex = 1.0)))
}





