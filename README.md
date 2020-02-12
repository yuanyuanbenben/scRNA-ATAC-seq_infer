# scRNA-ATAC-seq_infer
***

>Test data for fitting and inference of a problem about single cell RNA-seq and ATAC-seq data.

***

### cycleGAN
#### paper:

[Unpaired Image-to-Image Translationusing Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

#### Loss

![](http://latex.codecogs.com/gif.latex?\\mathcal{L}{gan}(G,Dy,X,Y)=E^{y\sim p(y)}[\log Dy(y)]+E^{x\sim p(x)}[\log(1-Dy(G(x)))])

![](http://latex.codecogs.com/gif.latex?\\mathcal{L}_{cyc}(G,F)=E_{x\sim p_{data}(x)}[\|F(G(x))-x\|_1]+E_{y\sim p_{data}(y)}[\|F(G(y))-y\|_1])

![](http://latex.codecogs.com/gif.latex?\\Rightarrow\mathcal{L}(G,F,D_X,D_Y)=\mathcal{L}_{GAN}(G,D_Y,X,Y)+\mathcal{L}_{GAN}(F,D_X,Y,X)+\lambda \mathcal{L}_{cyc}(G,F))


where ![](http://latex.codecogs.com/gif.latex?\\lambda) controls the relative important of the two objectives.

#### Details

![](http://latex.codecogs.com/gif.latex?G^*,F^* = \arg \min_{G,F} \max_{D_X,D_Y} \mathcal{L}(G,F,D_X,D_Y))

