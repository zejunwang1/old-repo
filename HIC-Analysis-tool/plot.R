
plotHeatmap <- function(H,normType = 0,method = 1){
  # plot HiC contact matrix heatmap
  # H: input raw contact matrix
  # normType: normalization parameter, 0-no normalization, 1-VC normalization, 
  #           2-SqrtVC normalization, 3-SinkhornKnopp normalization, 
  #           4-KnightRuiz normalization, default 0
  # method: 1-use log function, 2-use sqrt function, default 1
  
  # preprocess
  H <- preprocess(H)
  
  # normalization
  if(normType==0){
    H <- H
  }else if(normType==1){
    H <- VCNorm(H)
  }else if(normType==2){
    H <- SqrtVCNorm(H)
  }else if(normType==3){
    H <- SinkhornKnoppNorm(H)
  }else if(normType==4){
    H <- KnightRuizNorm(H)
  }else{
    stop("wrong normalization parameter")
  }
  
  # decrease the variation
  H <- shrinkVariation(H,method)
  # color theme
  myTheme <- rasterTheme(9,region = brewer.pal(9,"YlOrRd"))
  
  # plot heatmap
  levelplot(H,par.settings = myTheme,
            scales = list(x = list(cex = 1.0),
                          y = list(cex = 1.0)),
            xlab = list(label = ""),
            ylab = list(label = ""),
            colorkey = list(labels = list(cex = 1.0)))
}

plotTAD <- function(H,TAD_boundaries,w = 3){
  # H: input 2D HiC normalized contact matrix
  # TAD_boundaries: TAD boundary vector
  # w: line width, default 3
  
  l <- nrow(H)
  # color theme
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
  
  # plot heatmap and TADs
  levelplot(H,par.settings = myTheme,
            scales = list(x = list(cex = 1.0),
                          y = list(cex = 1.0)),
            xlab = list(label = ""),
            ylab = list(label = ""),
            colorkey = list(labels = list(cex = 1.0)))
}

plotNormSum <- function(df){
  # plot row sum values of different normalization results
  # df: input the result of function normSum
  
  if(class(df)!="data.frame"){
    stop("wrong input format")
  }
  
  l <- nrow(df)
  df0 <- data.frame(1:l,df$Original,"Original")
  colnames(df0) <- c("bin","rowSumValue","normType")
  df1 <- data.frame(1:l,df$VC,"VC")
  colnames(df1) <- c("bin","rowSumValue","normType")
  df2 <- data.frame(1:l,df$SqrtVC,"SqrtVC")
  colnames(df2) <- c("bin","rowSumValue","normType")
  df3 <- data.frame(1:l,df$SK,"SinkhornKnopp")
  colnames(df3) <- c("bin","rowSumValue","normType")
  df4 <- data.frame(1:l,df$KR,"KnightRuiz")
  colnames(df4) <- c("bin","rowSumValue","normType")
  
  df <- rbind(df0,df1,df2,df3,df4)
  p <- ggplot(data = df,mapping = aes(x = bin,y = rowSumValue,color = normType,shape = normType))
  p + geom_point() + geom_line() + theme_bw() + labs(x = "bin",y = "log(rowSumValue)")
}

plotPowerLaw <- function(reads,resInterval=c(seq(10,90,10),seq(100,900,100),seq(1000,10000,1000))*1000){
  # plot powerlaw curve according to one chromosome reads
  # reads: input one chromosome reads data.frame
  # resInterval: resolution interval, default 10Kb~10Mb
  
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
  
  # plot powerlaw curve
  df <- data.frame(log10(dens))
  df$chr <- reads$V1[1]
  df$dist <- log10(resInterval)
  colnames(df) <- c("dens","chr","dist")
  
  p <- ggplot(data = df,mapping = aes(x = dist,y = dens,color = chr,shape = chr))
  p + geom_point() + geom_line() + theme_bw() +
    labs(x = "log10 (genomic distance)",y = "log10 (density)") +
    scale_x_continuous(breaks = c(4.0,5.0,6.0,7.0),labels = c("10kb","100kb","1Mb","10Mb")) +
    theme(legend.title = element_text(size = 18),
          legend.text = element_text(size = 14),axis.text.x = element_text(size = 14),
          axis.text.y = element_text(size = 14),axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16))
}

plotCorrelation <- function(hic1,hic2){
  # plot correlation of two hic sparse matrix
  # hic1/hic2: input two sparse matrix
  
  # convert sparse matrix to data.frame
  hic1 <- data.frame(summary(hic1))
  hic2 <- data.frame(summary(hic2))
  colnames(hic1) <- c("i","j","freq")
  colnames(hic2) <- c("i","j","freq")
  df <- merge(hic1,hic2,all = T,by = c("i","j"))
  df[is.na(df)] <- 0
  
  # plot correlation
  p <- ggplot(data = df,mapping = aes(x = freq.x,y = freq.y))
  p + geom_point() + geom_smooth(method=lm) + theme_bw() + labs(x = "HiC1",y = "HiC2") + 
    theme(legend.title = element_text(size = 18),
          legend.text = element_text(size = 14),axis.text.x = element_text(size = 14),
          axis.text.y = element_text(size = 14),axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16))
}

plotLoops <- function(h,loops,r=2,method=1){
  # plot chromatin loops at fixed resolution
  # h: input hic contact matrix
  # loops: input fixed resolution chromatin loops, the format should be data.frame,
  #        which includes three columns: x,y,pvalue
  # r: radius, default 1
  # method: 1-use log function, 2-use sqrt function, default 1
  
  if(length(loops)!=3){
    stop("input wrong loops format")
  }
  
  flag <- (colnames(loops)==c("x","y","pvalue"))
  if(length(which(flag==FALSE))!=0){
    stop("input wrong loops format")
  }
  
  val <- max(h)
  for(k in 1:nrow(loops)){
    i <- loops$x[k] - r
    s <- seq(i,i+2*r,1)
    j <- loops$y[k] - r
    h[s,j] <- val
    j <- loops$y[k] + r
    h[s,j] <- val
    j <- loops$y[k] - r
    s <- seq(j,j+2*r,1)
    i <- loops$x[k] - r
    h[i,s] <- val
    i <- loops$x[k] + r
    h[i,s] <- val
  }
  
  # shrink the variation
  h <- shrinkVariation(h,method)
  # plot heatmap and loops
  levelplot(h,par.settings = myTheme,
            scales = list(x = list(cex = 1.0),
                          y = list(cex = 1.0)),
            xlab = list(label = ""),
            ylab = list(label = ""),
            colorkey = list(labels = list(cex = 1.0)))
}

plotLoessFitCurve <- function(H){
  # plot loess fit curve
  # H: input HiC contact matrix
  
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
  dg$expected <- model$fitted
  p <- ggplot(data = dg,mapping = aes(x = dis,y = avg))
  p + geom_point() + geom_line(aes(x = dis,y = expected,color = "red")) + 
    labs(x = "Distance/resolution",y = "Average") + 
    theme(legend.title = element_text(size = 18),
          legend.text = element_text(size = 14),axis.text.x = element_text(size = 14),
          axis.text.y = element_text(size = 14),axis.title.x = element_text(size = 16),
          axis.title.y = element_text(size = 16))
}


h <- multiResolutionTADMatrix(regionReads[[27]],pos[27],pos[28],100000,c(seq(100,10,-10),5,1)*1000)
interaction_num <- rep(0,length(h))
for(i in 1:length(h)){
  v <- as.vector(h[[i]])
  interaction_num[i] <- length(unique(v))
}




