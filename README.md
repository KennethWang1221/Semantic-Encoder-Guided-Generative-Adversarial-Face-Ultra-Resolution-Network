# Semantic Encoder Guided Generative Adversarial Face Ultra-Resolution Network
 **The Repository of SEGA-FURN**



## Overview ##

Face super-resolution is a domain-specific single image super-resolution, which aims to generate High-Resolution (HR) face images from their Low-Resolution (LR) counterparts. In this paper, we propose a novel face super-resolution method, namely Semantic Encoder guided Generative Adversarial Face Ultra-Resolution Network (SEGA-FURN) to ultra-resolve an unaligned tiny LR face image to its HR counterpart with multiple ultra-upscaling factors (e.g., 4× and 8×). In our method, the proposed semantic encoder has the ability to capture the embedded semantics to guide adversarial learning, and the novel architecture named Residual in Internal Dense Block (RIDB) is able to extract hierarchical features for generator. Moreover, we propose a joint discriminator which not only discriminates image data but also discriminates embedded semantics, learning the joint probability distribution of the image space and latent space, and we use a Relativistic average Least Squares loss (RaLS) as the adversarial loss, which can alleviate the gradient vanishing problem and enhance the stability of the training procedure. According to extensive experiments on large face datasets, it is obvious that our proposed method achieves superior super-resolution results and significantly outperforms other state-of-the-art methods in both qualitative and quantitative comparisons. The overview of our method is shown in Fig.1



![CVPR_overview_1](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Face-Ultra-Resolution-Network/blob/main/CVPR_overview_1.png)

Proposed SEGA-FURN and its components: Semantic Encoder $E$, Generator $G$, Joint Discriminator $D$ and Feature Extractor $\phi$. For $D$, ESLDSN represents the Embedded Semantics-Level Discriminative Sub-Net, ILDSN represents the Image-Level Discriminative Sub-Net, and FCM denotes Fully Connected Module. As for the generator $G$, there is three stages: Shallow Feature Module (SFM), Multi-level Residual Dense Module (MRDM), and Upsampling Module (UM). $I^{HR}$ and $I^{LR}$ denote HR face images and LR face images respectively. $I^{SR}$ is SR images from $G$. Furthermore, $E(\cdot)$ denotes the embedded semantics obtained from $E$. $D(\cdot)$ represents the output probability of $D$. $\phi(I^{HR})$ and $\phi(I^{SR})$ describes the features learned by $\phi$.

# contributions

we propose a novel GAN-based SR method, namely Semantic Encoder guided Generative Adversarial Face Ultra-Resolution Network (SEGA-FURN), as shown in Fig. 1. Main contributions of this paper can be summarized as follows:

1. Our proposed method is able to ultra-resolve an unaligned tiny face image to a Super-Resolved (SR) face image with multiple upscaling factors (e.g., 4× and 8×). In addition, our methods does not need any prior information or facial landmark points.

2. We design a semantic encoder to reverse the image information back to the embedded semantics reflecting facial semantic attributes. The embedded semantics combined with image data are fed into a joint discriminator. Such innovation can let semantic encoder guide the discriminative process, which is beneficial to enhance the discriminative ability of the discriminator.

3. We propose a Residual in Internal Dense Block (RIDB) as the basic architecture for the generator. This innovation provides an effective way to take advantage of hierarchical features, resulting in increased feature extraction capability of the SEGA-FURN.

4. We propose a joint discriminator which is capable of learning the joint probability constructed by embedded semantics and visual information (HR and LR images), resulting in a powerful discriminative ability. Furthermore, in order to remedy the problem of vanishing gradient and improve the model stability, we make use of RaLS objective loss to optimize the training process.

   

# Proposed Method

In this section, we present the proposed method SEGA-FURN in detail. First, we describe the novel architecture of SEGA-FURN through four main parts: generator, semantic encoder, joint discriminator and feature extractor. Next, we introduce the objective loss function RaLS for optimizing the generator and discriminator respectively. Finally, we provide the overall perceptual loss used in SEGA-FURN. The overview of our method is shown in Fig. 1. In addition, the architecture of discriminator and generator can be seen in Fig. 2. Moreover, the structures of the proposed DNB and RIDB are presented in Fig. 3.



![CVPR_arch_temp4](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Face-Ultra-Resolution-Network/blob/main/CVPR_arch_temp4.png)

Figure 2. **Red dotted rectangle**: The architecture of Generator. **Blue dotted rectangle**: The architecture of the Joint Discriminator. $F_{SF}$ denotes shallow features, $F_{MDBM}$ denotes the outputs of MDBM, $F_{GF}$ represents global features, and $F_{MHF}$ represents multiple hierarchical features. K, n, and s are the kernel size, number of filters and strides respectively. N is the number of neurons in dense layer.



## Generator

As shown at the top of Fig. 2, the proposed generator mainly consists of three stages: Shallow Feature Module (SFM), Multi-level Dense Block Module (MDBM), and Upsampling Module (UM). The LR face image $I^{LR}$ is fed into the SFM as the initial input. At the end, SR face image $I^{SR}$ is obtained from the UM. As for the SFM, we utilize one convolutional (Conv) layer to extract the shallow feature maps. It can be expressed as follows: 

$F_{SF}=H_{SFM}(I^{LR})$ 

where $H_{SFM}$ represents the Conv operation in the SFM, $F_{SF}$ denotes the shallow (low-level) features, which are used for global residual learning and serve as the input to the MDBM. The following module MDBM is built up by multiple Dense Nested Blocks (DNB) formed by several RIDBs, which will be discussed in the next subsection. We introduce local residual feature extraction (LRFE) and local residual learning (LRL) in the DNB to enhance super-resolution ability and lighten training difficulty. The procedure of LRFE in $i$-th DNB can be formulated as:

$F_{DNB,i\_LF} = H_{DNB,i}(H_{DNB,i-1}(\cdot \cdot \cdot(H_{DNB,1}(F_{SF}))\cdot \cdot \cdot))$

where $H_{DNB,i}$ acts as local residual feature extraction in the $i$-th DNB, which is composed of multiple blocks of RIDBs, $F_{DNB,i\_LF}$ is defined as the Local Features (LF) of the $i$-th DNB. Specifically, as for each DNB, it includes 3 RIDBs cascaded by residual connections and one scale layer, as shown in Fig. 3. It can be formulated as:

$F_{DNB,i\_LF} = \alpha F_{i,j}(F_{i,j-1}(\cdot \cdot \cdot F_{i,1}(F_{DNB,i-1})\cdot \cdot \cdot))$

where $F_{i,j}$ represents the $j$-th RIDB of the $i$-th DNB. We assign $\alpha$ to be 0.2 in the scale layer. In order to take effective use of the local residual features, we perform the local residual learning in $i$-th DNB. The final output deeper features of the $i$-th DNB can be obtained by:

$\ F_{DNB,i} = F_{DNB,i\_LF}+F_{DNB,i-1}$

where $F_{DNB,i}$ denotes the output deeper features of $i$-th DNB, which is obtained by residual connection. With the help of LRFE and LRL, the generator is able to make full use of deeper features and also efficiently propagate these features from lower to higher layers. In our generator, there are four DNBs in MDBM, so in this case the output of 4-th DNB ($i=4$) equals to the output of MDBM. Thus, the $F_{MDBM}$ can be expressed as $F_{DNB,i=4}$. To take advantage of multi-level representations, we apply the obtained $F_{MDBM}$ to Global Feature Fusion (GFF), where GFF is proposed to extract the global features $F_{GF}$ by fusing feature maps produced by $F_{MDBM}$, the formulation is:

$F_{GF}=H_{GFF}(F_{MDBM})$

where $H_{GFF}$ is a composite function of Conv layer followed by the Batch Normalization (BN) layer. It aims to further extract richer features for global residual learning. Next, in order to help the generator fully use hierarchical features and alleviate gradient vanishing problem, we adopt the Global Residual Learning (GRL) to fuse the shallow features and global features.

$F_{MHF} = F_{SFM}+F_{GF}$

where $F_{MHF}$ denotes multiple hierarchical features. Next, the $F_{MHF}$ is passed to the UM followed by one Conv layer. Then, the fused hierarchical feature is transformed from the LR space to the HR space through upsampling layers in the UM. The super-resolution process can be formulated as:

$I^{SR}=f_{UM}(F_{MHF})=H_{SEGA-FURN}(I^{LR})$

where $f_{UM}$ represents the upsampling operation, $H_{SEGA-FURN}$ denotes the function of our method SEGA-FURN. Finally, we obtain the SR image $I^{SR}$.



## Residual in Internal Dense block

As mentioned in Sec.1, the novel architecture RIDB is proposed for the generator, which is used to form the DNB (as shown in Fig. 3). 



![CVPR_RIDB_green_temp2](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Face-Ultra-Resolution-Network/blob/main/CVPR_RIDB_green_temp2.png)

Figure 3. **Top:** Dense Nested Block (DNB) consists of multiple RIDBs. **Bottom:** The proposed Residual in Internal Dense Block (RIDB).



The proposed RIDB is able to extract hierarchical features and address the vanishing-gradient problem, which is the commonly encountered issue in SRGAN, ESRGAN, SRDenseNet, URDGN, RDN. The proposed RIDB is made up of four internal dense blocks and all the internal dense blocks are cascaded through residual connections performing identity mapping. The structure of the RIDB is expressed as:

$F_{RIDB,p} = F_{p,q}(F_{p,q-1}(...F_{p,1}(F_{RIDB,p-1})...))+F_{RIDB,p-1}$

where $F_{RIDB,p-1}$ and $F_{RIDB,p}$ denote the input and output of the $p$-th RIDB respectively, $F_{p,q}$ represents the $q$-th internal dense block of $p$-th RIDB. In addition, an internal dense block is a composition of two groups of the Conv layer followed by the LeakyReLU activation layer. And the two groups are linked by dense skip connections. Each internal dense block can be calculated as follows:

$F_{q,k} =\delta(W_{q,k}[F_{q,k=1},F_{q,k=2}])$

where $F_{q,k}$ represents the output of $k$-th Conv layer of $q$-th internal dense block, $[F_{q,k=1},F_{q,k=2}]$ refers to the concatenation of feature maps in $q$-th internal dense block. $W_{q,k}$ denotes the weights of the $k$-th Conv layer, $\delta$ denotes the LeakyReLU activation. Moreover, the residual learning and more dense connections in the RIDB effectively guarantee the feature maps of each layer are propagated into all succeeding layers, promoting an effective way to extract hierarchical features. Thus, our proposed method is capable of obtaining abundant hierarchical feature information and alleviating the vanishing-gradient problem.

## Semantic Encoder

The proposed semantic encoder is supposed to extract embedded semantics (as shown in Fig. 1), which is used to project visual information (HR, LR) back to the latent space. The motivation is that the GAN-based SR models (SRGAN, ESRGAN, URDGN) only exploit visual information during discriminative procedure, ignoring the semantic information reflected by latent representation. Therefore, the proposed semantic encoder will complement the missing critical property. Previous GAN's work (BiGAN, ALI) has proved that the semantic representation is beneficial to the discriminator.

Based on this observation, the proposed semantic encoder is designed to inversely map the image to the embedded semantics. Significantly, the most important advantage of semantic encoder is that it is able to guide the discriminative process since the embedded semantics obtained from the semantic encoder can reflect semantic attributes, such as the facial features (shape and gender) and the spatial relationship between various components of the face (eyes, mouth). It can be emphasized that the embedded semantics is fed into joint discriminator along with HR and LR images. Thanks to this property, the semantic encoder can guide discriminator to optimize, thereby enhancing its discriminative ability. More details can be found in the supplementary material.

## Joint Discriminator

As shown in Fig 1, the proposed joint discriminator takes the tuple incorporating both visual information and embedded semantics as the input, where Embedded Semantics-Level Discriminative Sub-Net (ESLDSN) receives the input embedded semantics while the image information is sent to Image-Level Discriminative Sub-Net (ILDSN). Next, through the operation of the Fully Connected Module (FCM) on concatenated vector, the final probability is predicted. Thus, the joint discriminator has the ability to learn the joint probability distribution of image data ($I^{HR},I^{SR}$) and embedded semantics ($E(I^{HR}),E(I^{LR})$). There are two sets of paths entering into the joint discriminator. The set of path shown in red indicates real tuple which consists of real sample $I^{HR}$ from dataset and its embedded semantics $E(I^{HR})$.  For the blue path, fake tuple is constructed from SR image $I^{SR}$ generated from generator and $E(I^{LR})$ obtained from LR image through semantic encoder. As a result, different from SRGAN, ESRGAN, URDGN, FSRGAN, our joint discriminator has the ability to evaluate the difference between real tuple $(I^{HR},E(I^{HR}))$ and fake tuple $(I^{SR},E(I^{LR}))$. 

Moreover, in order to alleviate the problem of gradient vanishing and enhance the model stability, we adopt the Relativistic average Least Squares GAN (RaLSGAN) objective loss for the joint discriminator by applying the RaD to the least squares loss function (LSGAN). Let's denote the real tuple by $X_{real}=(I^{HR},E(I^{HR}))$ and denote the fake tuple by $X_{fake} = (I^{SR},E(I^{LR}))$. The process that makes joint discriminator to be relativistic can be expressed as follows:

$\tilde{C}(X_{real}) = (C(X_{real}) - E_{x_{f}}[C(X_{fake})]) \\
\tilde{C}(X_{fake}) = (C(X_{fake}) - E_{x_{r}}[C(X_{real})])$

where $\tilde C(\cdot)$ denotes the probability predicted by joint discriminator, $E_{x_{f}}$ and $E_{x_{r}}$ describe the average of the SR images (fake) and HR images (real) in a training batch. Moreover, the least squares loss is used to measure the distance between HR and SR images. According to Eqn. 15, we optimize the joint discriminator by adversarial loss $L_{D}^{RaLS}$ and the generator is updated by $L_{G}^{RaLS}$, as in Eqn. 16.

$L_{D}^{RaLS}=\mathbb{E}_{I^{HR}\sim p_{(I^{HR})}}[( \tilde{C}( X_{real})-1)^{2}]+\mathbb{E}_{I^{SR}\sim p_{(I^{SR})}}[( \tilde{C}( X_{fake})+1)^{2}]$

$L_{G}^{RaLS}=\mathbb{E}_{I^{SR}\sim p_{(I^{SR})}}[( \tilde{C}( X_{fake})-1)^{2}]+\mathbb{E}_{I^{HR}\sim p_{(I^{HR})}}[( \tilde{C}( X_{real})+1)^{2}]$

where $I_{HR} \sim P_{I^{HR}}$ and $I_{SR} \sim P_{I^{SR}}$ indicate the HR images and SR images distribution respectively. Furthermore, with the help of least squares loss and relativism in RaLS, SEGA-FURN is remarkably more stable and generates authentic and visually pleasant SR images. More details of architecture can be found in the supplementary material.

## Feature Extractor

We further exploit pre-trained VGG19 (VGG19) network as feature extractor $\phi$ in SEGA-FURN to obtain feature representations used to calculate the content loss $L_{content}$, where $L_{content}$ is utilized in SEGA-FURN to eliminate the facial ambiguity and recover missing details of SR images. It is measured as the Euclidean distance between two feature representations of SR images and HR images. Instead of using high-level features as in SRGAN, ESRGAN for content loss, we adopt low-level features before activation layer (i.e., feature representations from `Conv3\_3' layer in the feature extractor), which contains complex edge texture. 

## Loss Function

We involve content loss $L_{content}$ to constrain the intensity and feature similarities between HR and SR images. Furthermore, adversarial loss $L_{G}^{RaLS}$ is adopted to super-resolve SR images containing visually appealing details and faithful to the HR

**Content Loss**

$L_{content}$ is able to reduce the gap between SR image and HR image. It is formulated as:

$L_{content}=\frac{1}{WH}\sum_{q=1}^{W}\sum_{r=1}^{H}( \phi_{i,j}(I^{HR})_{q,r}-\phi_{i,j}( I^{SR})_{q,r})^{2}$

where $W$, $H$ describe the height and width of the feature maps, $\phi(\cdot)$ denotes the output of feature extractor, $\phi_{i,j}$ indicates the feature representations obtained from $j$-th convolution layer before $i$-th maxpooling layer. 

**Perceptual Loss**

The total loss function $L_{perceptual}$ for the generator can be represented as weighted combination of two parts: content loss $L_{content}$ and adversarial loss $L_{G}^{RaLS}$, the formula is described as follows:

$L_{perceptual} = \lambda _{con}L_{content} + \lambda_{adv}L_{G}^{RaLS}$

where $\lambda_{con}$ , $\lambda_{adv}$ are the trade-off weights for the $L_{content}$ and the $L_{G}^{RaLS}$. We set $\lambda_{con}$, $\lambda_{adv}$ empirically to 1 and $10^{-3}$ respectively.

# Experiments

In this section, we first present the details of dataset and training implementation. Then, we demonstrate the experiments and evaluation results. We further compare our method with state-of-the-art methods. Moreover, in order to prove the effectiveness of SEGA-FURN, we conduct ablation experiments to verify the contributions of the components proposed in this work. More experimental details and visual results are presented in the supplementary material.

## Datasets

We conducted experiments on the public large-scale CelebFaces Attributes dataset, CelebA. It consists of 200K celebrity face images of 10,177 celebrities. We used a total of 202,599 images, where we randomly selected 162,048 HR face images as the training set, and all the rest 40,511 images were chosen as the testing set.

## Implementation Details

To verify the effectiveness of our method, we conducted experiments with multiple upscaling factors 4× and 8× respectively. We resized and cropped the images to 256x256 pixels as our HR face images without any alignment operation. In order to obtain two groups of LR downsampled face images, we used bicubic interpolation method with downsampling factor $r$ =4 to produce 64x64 pixels, and factor $r$=8 to produce 32x32 pixels LR images.

We trained our network 30k epochs using the Adam optimizer  by setting $\beta _{1}$ =0.9, $\beta _{2} $=0.999 with the learning rate of $10^{-4}$ and batch size of 8. We alternately updated the generator and discriminator until the model converges. For the quantitative comparison, we adopted Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM) as the evaluation metrics.

## Comparisons with State-of-the-art Methods

We compare the proposed method with the state-of-the-art SR methods.

**Qualitative Comparison**

The 4× and 8× qualitative results are depicted in Fig. 4. The top two rows show the 4× visual results. As for Bicubic interpolation, we observe that its results contain over-smooth visualization effects. SRGAN relatively enhances the SR results compared to Bicubic interpolation, but it still fails to generate fine details especially in facial components, such as eyes, and mouth. It is obvious that ESRGAN produces overly smoothed visual results and misses specific textures. On the contrary, the 4× SR images produced by our method retain facial-specific details and are faithful to HR counterparts.

![CVPR_combine_4x8x](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Face-Ultra-Resolution-Network/blob/main/CVPR_combine_4x8x.png)

Figure 4. Qualitative comparison against state-of-the-art methods. **The top two rows**: the results of 4× upscaling factor from 32x32 pixels to 256x256 pixels. **The bottom two rows**: the results of 8× upsampling factor from 16x16 pixels to 256x256 pixels. From left to right: (a) HR images, (b) LR inputs, (c) Bicubic interpolation, (d) Results of SRGAN, (e) Results of ESRGAN, and (f) Our method.

To reveal the powerful super-resolution ability of our proposed method, we further conduct experiments with 8× ultra upscaling factor. As shown in the bottom two rows in Fig. 4, it apparently presents that the SR visual quality obtained by Bicubic interpolation, SRGAN and ESRGAN is decreased, since the  magnification is increased, resulting in the correspondence between HR and LR images incompatible. The outputs of Bicubic interpolation generate unpleasant noises. SRGAN encounters mode collapse problem during super-resolution process, so that it produces severe distortions in SR images. As for ESRGAN, it produces the SR images which show broken textures, noticeable artifacts around facial components. In contrast, our method is capable of producing photo-realistic SR images which preserve perceptually sharper edges and fine facial textures. 



**Quantitative Comparison**

![Table1](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Face-Ultra-Resolution-Network/blob/main/Table1.png)

Table 1. Quantitative comparison on CelebA dataset for upscaling factor 4x and 8x, in terms of average PSNR(dB) and SSIM. Numbers in bold are the best evaluation results among state-of-the-art methods.

The quantitative results with multiple ultra upscaling factors 4× and 8× are shown in Table 1. It is obvious that our method attains the best in both PSNR and SSIM evaluations, 30.14dB/0.87 for 4× and 25.21dB/0.73 for 8×, among all other methods. FaceAttr is the second best method for 4×, 29.78dB/0.82, however, it degrades dramatically and performs poorly when the upscaling factor increases to 8×, obtaining 21.82dB/0.62. In contrast, our proposed method ranks the first for both upscaling factors 4× and 8×, which reflects the robustness of the proposed SEGA-FURN. Moreover, it is notable that our proposed method not only boosts PSNR/SSIM by a large margin of 1.04dB/0.08 over the classic method URDGN with upscaling factor 4× but also is higher than URDGN which is the second best for the 8× upscaling. This observation shows a stable ability of our method with multiple upscaling factors. In addition, we compare with SRGAN and ESRGAN which also use generative adversarial structure. It is obvious that our method not only improves SR image quality from perceptual aspect but also achieves an impressive numerical results.

# Ablation Study

we further implemented ablation studies to investigate the performance of the proposed method. As shown in Table 2, we list several variants based on different proposed components. First, among them, RIDB-Net can be used as the baseline variant, which only contains single component RIDB. Second, the RIDB-RaLS-Net is constructed by removing the Semantic Encoder (SE) from the SEGA-FURN. Next, RIDB-SE-Net means to remove RaLS loss of SEGA-FURN, and RIDB-SE-RaLS-Net equals to SEGA-FURN including all of the three components. In addition, we provide the visual results of these variants in Fig. 5, and quantitative comparison in Table 3.

![CVPR_ablation_comb4x8x_3](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Face-Ultra-Resolution-Network/blob/main/CVPR_ablation_comb4x8x_3.png)

Figure 5. Qualitative comparison of ablation studies. **The top two rows:** the results of upscaling factor 4×. **The bottom two rows:** the results of upscaling factor 8×. From left to right: (a) HR images, (b) LR inputs, (c) Results of RIDB-Net, (d) Results of RIDB-RaLS-Net (e) Results of RIDB-SE-Net, and (f) Results of RIDB-SE-RaLS-Net (SEGA-FURN).



![Table2](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Face-Ultra-Resolution-Network/blob/main/Table2.png)

​			Table 2. Description of SEGA-FURN variants with different components in experiments.



![Table4](https://github.com/KennethXiang/Semantic-Encoder-Guided-Generative-Adversarial-Face-Ultra-Resolution-Network/blob/main/Table4.png)

​	Table 3. Quantitative comparison of different variants on CelebA dataset for upscaling factor 4x and 8x



**Effect of RIDB**

We compare the proposed RIDB with other feature extraction blocks, such as Residual Block (RB) from SRGAN and Residual in Residual Dense Block (RRDB) of ESRGAN . As shown in Fig. 4 and Table 1, it is noticeable that the SR results generated by our generator employing RIDB outperforms SRGAN utilizing RB and ESRGAN adopting RRDB both in qualitative and quantitative comparisons. The reason is that our RIDB introduces densely connected structure to combine different level features, but there is no dense connections in RB. In addition, different from RRDB, the proposed RIDB designs multi-level residual learning within each basic internal dense blocks, which is able to boost the flow of features through the generator and provide hierarchical features for the super-resolution process. Based on these observations and investigations, it is persuasive to validate the effectiveness of the proposed RIDB. 

**Effect of SE**

The Ablation (A) and (C) performed by RIDB-Net and RIDB-SE-Net aim to illustrate the advantage of SE and also verify the effectiveness of the joint discriminator. The RIDB-SE-Net can obtain embedded semantics extracted by SE and further feed these semantics along with image data to the joint discriminator. In training process, the embedded semantics is capable of providing useful semantic information for the joint discriminator. Such innovation can enhance the discriminative ability of the joint discriminator. Compared with RIDB-Net which does not employ SE and joint discriminator, the RIDB-SE-Net achieves significant improvements in terms of quantitative comparisons. Furthermore, as shown in Fig. 5, there is also a noticeable refinement in detailed texture. The enhanced performance can verify that the extracted embedded semantics has superior impact on SR results and the SE along with the joint discriminator play a critical role of the proposed method.

**Effect of RaLS**

The ablation (A) and (B) are conducted to demonstrate the effect of RaLS loss. We replace the RaLS loss of RIDB-Net with the generic GAN loss, Binary Cross Entropy (BCE) and keep all the other components the same. As shown in Table 3, it is obvious that once we remove the RaLS loss in RIDB-Net, the quantitative results is lower than RIDB-RaLS-Net which has RaLS loss. As expected, BCE used in RIDB-Net shows unrefined textures. In contrast, when RaLS is utilized in variant, the visual results are perceptually pleasing with more natural textures and edges. Thus, it can demonstrate that the RaLS loss is capable of greatly improving the performance of super-resolution.

**Final Effect**

From the comparison between ablation (D) and other studies, it is obvious that the large enhancement is noticeable by integrating all these three components. Finally, we refer the RIDB-SE-RaLS-Net to SEGA-FURN which is the ultimate proposed method.



# Conclusions

In this paper, we proposed a novel Semantic Encoder guided Generative Adversarial Face Ultra-resolution Network (SEGA-FURN) to super-resolve a tiny LR unaligned face image to its HR version with multiple large ultra-upscaling factors (e.g., 4× and 8×). Owing to the proposed Semantic Encoder, Residual in Internal Dense Block and the Joint Discriminator adopting RaLS loss, our method successfully produced photo-realistic SR face images. Extensive experiments and analysis demonstrated that SEGA-FURN is superior to the state-of-the-art methods.



