# scRNA-ATAC-seq_infer
***

>Test data for fitting and inference of a problem about single cell RNA-seq and ATAC-seq data.

***

### cycleGAN
#### paper:

[Unpaired Image-to-Image Translationusing Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

#### Loss

$$\mathcal{L}_{gan}(G,D_Y,X,Y)=E_{y\sim p_{data}(y)}[\log D_Y(y)]+E_{x\sim p_{data}(x)}[\log(1-D_Y(G(x)))]$$

$$\mathcal{L}_{cyc}(G,F)=E_{x\sim p_{data}(x)}[\|F(G(x))-x\|_1]+E_{y\sim p_{data}(y)}[\|F(G(y))-y\|_1]$$

$$\Rightarrow\mathcal{L}(G,F,D_X,D_Y)=\mathcal{L}_{GAN}(G,D_Y,X,Y)+\mathcal{L}_{GAN}(F,D_X,Y,X)+\lambda \mathcal{L}_{cyc}(G,F)$$


where $\lambda$ controls the relative important of the two objectives.

#### Details

$$G^*,F^* = \arg \min_{G,F} \max_{D_X,D_Y} \mathcal{L}(G,F,D_X,D_Y)$$

using least-squares loss instead of negative log likelihood in practice.

such as train G to minimize $E_{x\sim p_{data}(x)}[(D(G(x))-1)^2]$ and train D to minimize $E_{y\sim p_{data}(y)}[(D(y)-1])^2+E_{x\sim p_{data}(x)}[D(G(x))^2]$

### Hierarchical GNN

paper:

[Hierarchical Graph Representation Learning withDifferentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf)

####