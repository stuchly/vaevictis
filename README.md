# Vaevictis - this is a playground!!

This is a playground - master could be broken at any moment, use specific commit to be sure you get working code (see Installation).

Experimental combination of various ideas for dimensional reduction nad manifold analysis

Currently (re)implements ideas from following published results:

The ivis algorithm as described in the paper [Structure-preserving visualisation of high dimensional single-cell datasets](https://www.nature.com/articles/s41598-019-45301-0).


The scvis algorithm as described in the paper [Interpretable dimensionality reduction of single cell transcriptome data with deep generative models](https://www.nature.com/articles/s41467-018-04368-5)

## Preprocess
It is recommended to standardize the data prior to analysis (e.g. divide the channels by 0.99 percentil) to avoid numerical problems. This step is not done internally to avoid issues with application already trained model on new data.

## Installation
November 12, 2020 - reticulate version 1.16 or devel version - install_github("rstudio/reticulate") - is needed
Vaevictis runs on top of TensorFlow. 

In R 
```
reticulate::py_install("git+https://github.com/stuchly/vaevictis.git@50152ac30587a3b8f32ad67397526bcddf6a2f8e",pip=TRUE)
```

## Example (R)
```
vae<-reticulate::import("vaevictis")
red=vae$dimred(as.matrix(iris[,1:4]))
plot(red[[1]],col=iris[,5])
red1<-red[[2]](as.matrix(iris[,1:4])) #apply trained function
red[[3]]$save(config_file = "config.json",weights_file = "weights.h5") # save model
loaded<-vae$loadModel(config_file = "config.json",weights_file = "weights.h5") #load model
red2<-loaded[[2]](as.matrix(iris[,1:4])) # apply trained function
plot(red2,col=iris[,5])

library(flowCore)
fcs<-read.FCS("path_to_fcs_file")
efcs<-exprs(fcs)[,c(24,29,30,31,33,37,38,44,46,48,49,50,52,56,57,59)] ## marker selection
efcs<-asinh(efcs/5.0) #cytof transform
vv=reticulate::import("vaevictis")
red<-vv$dimred(efcs,ww=c(0.,1.,0.,0.)) #ivis only - much faster, but population less compact
plot(red[[1]],pch=".") #reduced data in the first slot
```
