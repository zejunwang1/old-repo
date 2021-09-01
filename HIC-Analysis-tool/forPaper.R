
reads <- fread("D:\\TAD\\K562_data\\HIC070-chr20.bedpe")
h <- readsToMatrix(reads,100000)
h <- KnightRuizNorm(h)
H <- ToeplitzNorm(h)
P <- cor(H,method = "pearson")
res1 <- eigen(P)
plot(res1$vectors[,1],type = "h",col = "blue")


dgs <- colSums(H)
dgs[dgs==0] <- 1
L <- diag(dgs) - H
Ln <- (diag(dgs^(-1/2)))%*%L%*%(diag(dgs^(-1/2)))
Ln <- (Ln+t(Ln))/2
res2 <- eigen(Ln)
plot(res2$vectors[,length(res2$values)-1],type = "h",col = "blue")

DI <- data.frame(1:length(DI),DI)
colnames(DI) <- c("bin","DI")
DI$chr <- 10
p <- ggplot(data = DI,mapping = aes(x = bin,y = DI))
p + geom_line(color = "red") + labs(x = "bin",y = "DI") + theme_bw() + 
  theme(legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16))


root <- "D:\\TAD\\K562_data\\"
boundary1 <- list()
boundary2 <- list()
boundary3 <- list()
res <- 100000
for(i in 1:22){
  message(paste("processing for chromosome ",i,sep = ""))
  reads <- fread(paste(root,"HIC070-chr",i,".bedpe",sep = ""))
  h <- readsToMatrix(reads,res)
  h <- KnightRuizNorm(h)
  h <- shrinkVariation(h)
  bd1 <- HMMTADIdentify(h,res,1000000,1,1)
  boundary1 <- c(boundary1,list(bd1))
  bd2 <- HMMTADIdentify(h,res,1000000,2,1)
  boundary2 <- c(boundary2,list(bd2))
  bd3 <- HMMTADIdentify(h,res,1000000,3,1)
  boundary3 <- c(boundary3,list(bd3))
}

n1 <- rep(0,23)
n2 <- rep(0,23)
n3 <- rep(0,23)
for(i in 1:23){
  n1[i] <- length(boundary1[[i]]) - 1
  n2[i] <- length(boundary2[[i]]) - 1
  n3[i] <- length(boundary3[[i]]) - 1
}
chr <- "chr1"
for(i in 2:22){
  chr <- c(chr,paste("chr",i,sep = ""))
}
chr <- c(chr,"chrX")
n1 <- data.frame(n1,chr)
n2 <- data.frame(n2,chr)
n3 <- data.frame(n3,chr)
colnames(n1) <- c("num","chr")
colnames(n2) <- c("num","chr")
colnames(n3) <- c("num","chr")
n <- rbind(n1,n2,n3)
p <- ggplot(data = n,mapping = aes(x = idx,y = num,group = chr))
p + geom_bar(stat='identity')

x1 <- n1$num
x2 <- n2$num
x3 <- n3$num
x <- rbind(x1,x2,x3)
row.names(x) <- NULL
colnames(x) <- chr
df <- data.frame(as.matrix(x[,1]))
colnames(df) <- "num"
df$chr <- "chr1"
df$idx <- c(1,2,3)
m <- 3
for(i in 2:ncol(x)){
  m <- m + 2
  temp <- data.frame(as.matrix(x[,i]))
  colnames(temp) <- "num"
  temp$chr <- colnames(x)[i]
  temp$idx <- c(m+1,m+2,m+3)
  df <- rbind(df,temp)
  m <- m + 3
}

p <- ggplot(data = df,mapping = aes(x = idx,y = num,group = chr,color = chr))
p + geom_bar(stat = "identity") + labs(y = "TAD number") + 
  theme(legend.title = element_text(size = 18),
        legend.text = element_text(size = 14),axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),axis.title.x = element_text(size = 16),
        axis.title.y = element_text(size = 16))


# plot the venn diagram
x1 <- x1 + 1
x2 <- x2 + 1
x3 <- x3 + 1
num1 <- sum(x1)
num2 <- sum(x2)
num3 <- sum(x3)
overlap12 <- 0
overlap13 <- 0
overlap23 <- 0
overlap123 <- 0
thresh <- 3
for(i in 1:length(boundary1)){
  message(paste("computing overlap for chromosome ",i,"...",sep = ""))
  set1 <- boundary1[[i]]
  set2 <- boundary2[[i]]
  set3 <- boundary3[[i]]
  for(k1 in 1:length(set1)){
    flag <- 0
    for(k2 in 1:length(set2)){
      if(abs(set1[k1]-set2[k2])<=thresh){
        flag <- 1
        break
      }
    }
    if(flag==1){
      overlap12 <- overlap12 + 1
    }
  }
  
  for(k1 in 1:length(set1)){
    flag <- 0
    for(k3 in 1:length(set3)){
      if(abs(set1[k1]-set3[k3])<=thresh){
        flag <- 1
        break
      }
    }
    if(flag==1){
      overlap13 <- overlap13 + 1
    }
  }
  
  for(k2 in 1:length(set2)){
    flag <- 0
    for(k3 in 1:length(set3)){
      if(abs(set2[k2]-set3[k3])<=thresh){
        flag <- 1
        break
      }
    }
    if(flag==1){
      overlap23 <- overlap23 + 1
    }
  }
  
  for(k1 in 1:length(set1)){
    flag1 <- 0
    for(k2 in 1:length(set2)){
      if(abs(set1[k1]-set2[k2])<=thresh){
        flag1 <- 1
        break
      }
    }
    
    flag2 <- 0
    for(k3 in 1:length(set3)){
      if(abs(set1[k1]-set3[k3])<=thresh){
        flag2 <- 1
        break
      }
    }
    
    if(flag1==1 & flag2==1){
      overlap123 <- overlap123 + 1
    }
  }
}

dev.new()
draw.pairwise.venn(area1 = num1,area2 = num2,cross.area = overlap12,
                   category = c("ARHiC","HiCCups"),cex = rep(2,3),cat.cex = rep(1,2),
                   col = c("red","green"),fill = c("red","green"),
                   cat.col = c("red","green"))

dev.new()
draw.pairwise.venn(area1 = num1,area2 = num3,cross.area = overlap13,
                   category = c("A","C"),
                   col = c("red","blue"),fill = c("red","blue"),
                   cat.col = c("red","blue"))

dev.new()
draw.pairwise.venn(area1 = num2,area2 = num3,cross.area = overlap23,
                   category = c("B","C"),
                   col = c("green","blue"),fill = c("green","blue"),
                   cat.col = c("green","blue"))

dev.new()
draw.triple.venn(area1 = num1,area2 = num2,area3 = num3,n12 = overlap12,
                 n23 = overlap23,n13 = overlap13,n123 = overlap123,
                   category = c("A","B","C"),
                   col = c("red","green","blue"),fill = c("red","green","blue"),
                   cat.col = c("red","green","blue"),reverse = FALSE)

v1 <- as.vector(H[[1]])
v2 <- as.vector(H[[2]])
v3 <- as.vector(H[[3]])
v4 <- as.vector(H[[4]])
v5 <- as.vector(H[[5]])
l1 <- length(unique(v1))
l2 <- length(unique(v2))
l3 <- length(unique(v3))
l4 <- length(unique(v4))
l5 <- length(unique(v5))
idx <- which(v1==0)
if(length(idx)>0){
  v1 <- v1[-idx]
}
idx <- which(v2==0)
if(length(idx)>0){
  v2 <- v2[-idx]
}

h1 <- shrinkVariation(H[[1]])
h2 <- shrinkVariation(H[[2]])
h3 <- shrinkVariation(H[[3]])
h4 <- shrinkVariation(H[[4]])
h5 <- shrinkVariation(H[[5]])
h6 <- shrinkVariation(H[[6]])
v1 <- as.vector(h1)
v2 <- as.vector(h2)
v3 <- as.vector(h3)
v4 <- as.vector(h4)
v5 <- as.vector(h5)
v6 <- as.vector(h6)
idx <- which(v1==0)
if(length(idx)>0){
  v1 <- v1[-idx]
}
idx <- which(v2==0)
if(length(idx)>0){
  v2 <- v2[-idx]
}
idx <- which(v3==0)
if(length(idx)>0){
  v3 <- v3[-idx]
}
idx <- which(v4==0)
if(length(idx)>0){
  v4 <- v4[-idx]
}
idx <- which(v5==0)
if(length(idx)>0){
  v5 <- v5[-idx]
}


bd <- list()
res <- c(100000,40000,20000,10000)
h <- readsToMatrix(reads,10000)
idx <- which(colSums(h)==0)
h <- KnightRuizNorm(h)
h <- shrinkVariation(h)
temp <- HMMTADIdentify(h,100000,1000000,1,1)
bd <- c(bd,list(temp))

hbd <- HierarchicalTAD(reads)







