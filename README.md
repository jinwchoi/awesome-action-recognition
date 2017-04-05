# Computer Vision and Machine Learing papers, codes, and project webpages

Jinwoo's list of Computer Vision and Machine Learning papers, codes, project webs and others.

## Table of Contents

 - [Action Recognition](#ar)

## Action Recognition

### Spatio-Temporal Action Detection
* [Tube Convolutional Neural Network (T-CNN) for Action Detection in Videos](https://128.84.21.199/pdf/1703.10664.pdf) - [R. Hou](http://www.cs.ucf.edu/~rhou/) et al, arXiv2017.
* [Online Real time Multiple Spatiotemporal Action Localisation and Prediction](https://arxiv.org/pdf/1611.08563v3.pdf) - [G. Singh](http://gurkirt.github.io/) et al, arXiv2016.
* [Multi-region two-stream R-CNN for action detection](https://www.robots.ox.ac.uk/~vgg/rg/papers/peng16eccv.pdf) - Xiaojian Peng and Cordelia Schmid. ECCV2016. [[code]](https://github.com/pengxj/action-faster-rcnn)
* [Spot On: Action Localization from Pointly-Supervised Proposals](http://jvgemert.github.io/pub/spotOnECCV16.pdf) - P. Mettes et al, ECCV2016.
* [Deep Learning for Detecting Multiple Space-Time Action Tubes in Videos](https://arxiv.org/abs/1608.01529) - S. Saha et al, BMVC2016. [[code]](https://bitbucket.org/sahasuman/bmvc2016_code) [[project web]](http://sahasuman.bitbucket.org/bmvc2016/)
* [Learning to track for spatio-temporal action localization](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Weinzaepfel_Learning_to_Track_ICCV_2015_paper.pdf) - P. Weinzaepfel et al. ICCV2015.
* [Action detection by implicit intentional motion clustering](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Chen_Action_Detection_by_ICCV_2015_paper.pdf) - W. Chen and J. Corso, ICCV2015.
* [Action Tubes](https://people.eecs.berkeley.edu/~gkioxari/ActionTubes/action_tubes.pdf) - G. Gkioxari and J. Malik CVPR2015. [[code]](https://github.com/gkioxari/ActionTubes) [[project web]](https://people.eecs.berkeley.edu/~gkioxari/ActionTubes/)
* [APT: Action localization proposals from dense trajectories](http://jvgemert.github.io/pub/gemertBMVC15APTactionProposals.pdf) - J. Gemert et al, BMVC2015. [[code]](https://github.com/jvgemert/apt)
* [Spatio-Temporal Object Detection Proposals](https://hal.inria.fr/hal-01021902/PDF/proof.pdf) - D. Oneata et al, ECCV2014. [[code]](https://bitbucket.org/doneata/proposals) [[project web]](http://lear.inrialpes.fr/~oneata/3Dproposals/)
* [Action localization with tubelets from motion](http://isis-data.science.uva.nl/cgmsnoek/pub/jain-tubelets-cvpr2014.pdf) - M. Jain et al, CVPR2014.
* [Spatiotemporal deformable part models for action detection](http://crcv.ucf.edu/papers/cvpr2013/cvpr2013-sdpm.pdf) - Y. Tian eet al, CVPR2013.

### Temporal Action Detection
* [DAPs: Deep Action Proposals for Action Understanding](https://ivul.kaust.edu.sa/Documents/Publications/2016/DAPs%20Deep%20Action%20Proposals%20for%20Action%20Understanding.pdf) - V. Escorcia et al, ECCV2016. [[code]](https://github.com/escorciav/daps) [[raw data]](https://github.com/escorciav/daps)
* [Online Action Detection using Joint Classification-Regression Recurrent Neural Networks](https://arxiv.org/abs/1604.05633) - Y. Li et al, ECCV2016. Noe: RGB-D Action Detection
* [Temporal Action Localization in Untrimmed Videos via Multi-stage CNNs](http://dvmmweb.cs.columbia.edu/files/dvmm_scnn_paper.pdf) - Z. Shou et al, CVPR2016. [[code]](https://github.com/zhengshou/scnn) Note: Aka S-CNN.
* [Fast Temporal Activity Proposals for Efficient Detection of Human Actions in Untrimmed Videos](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Heilbron_Fast_Temporal_Activity_CVPR_2016_paper.pdf) - F. Heilbron et al, CVPR2016. [[code]](https://github.com/cabaf/sparseprop) Note: Depends on [C3D](http://vlg.cs.dartmouth.edu/c3d/), aka SparseProp.
* [Actionness Estimation Using Hybrid Fully Convolutional Networks](https://arxiv.org/abs/1604.07279) - L. Wang et al, CVPR2016. [[code]](https://github.com/wanglimin/actionness-estimation/) Note: The code is not a complete verision. It only contains a demo, not training. [[project web]](http://wanglimin.github.io/actionness_hfcn/index.html)
* [End-to-end Learning of Action Detection from Frame Glimpses in Videos](http://vision.stanford.edu/pdf/yeung2016cvpr.pdf) - S. Yeung et al, CVPR2016. [[code]](https://github.com/syyeung/frameglimpses) [[project web]](http://ai.stanford.edu/~syyeung/frameglimpses.html) Note: This method uses reinforcement learning
* [Fast Action Proposals for Human Action Detection and Search](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yu_Fast_Action_Proposals_2015_CVPR_paper.pdf) - G. Yu and J. Yuan, CVPR2015. Note: code for FAP is NOT available online. Note: Aka FAP.
* [Bag-of-fragments: Selecting and encoding video fragments for event detection and recounting](https://staff.fnwi.uva.nl/t.e.j.mensink/publications/mettes15icmr.pdf) - P. Mettes et al, ICMR2015.
* [Action localization in videos through context walk](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Soomro_Action_Localization_in_ICCV_2015_paper.pdf) - K. Soomro et al, ICCV2015.

### Spatio-Temporal ConvNets
* [Deep Temporal Linear Encoding Networks](https://arxiv.org/abs/1611.06678) - A. Diva et al, arXiv 2016.
* [Temporal Convolutional Networks: A Unified Approach to Action Segmentation and Detection](https://arxiv.org/abs/1608.08242) - C. Lea et al, arXiv 2016. [[code]](https://github.com/colincsl/TemporalConvolutionalNetworks)
* [Long-term Temporal Convolutions](https://arxiv.org/pdf/1604.04494v1.pdf) - G. Varol et al, arXiv 2016. [[project web]](http://www.di.ens.fr/willow/research/ltc/) [[code]](https://github.com/gulvarol/ltc) 
* [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf) - L. Wang et al, arXiv 2016. [[code]](https://github.com/yjxiong/temporal-segment-networks)

### Action Classification
* [Dynamic Image Networks for Action Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16a/bilen16a.pdf) - H. Bilen et al, CVPR2016. [[code]](https://github.com/hbilen/dynamic-image-nets) [[project web]](http://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16a/)
* [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf) - J. Donahue et al, CVPR2015. [[code]](https://github.com/LisaAnne/lisa-caffe-public/tree/lstm_video_deploy) [[project web]](http://jeffdonahue.com/lrcn/)
* [Describing Videos by Exploiting Temporal Structure](http://arxiv.org/pdf/1502.08029v4.pdf) - L. Yao et al, ICCV2015. [[code]](https://github.com/yaoli/arctic-capgen-vid) note: from the same group of RCN paper “Delving Deeper into Convolutional Networks for Learning Video Representations"
* [Two-Stream SR-CNNs for Action Recognition in Videos](http://wanglimin.github.io/papers/ZhangWWQW_CVPR16.pdf) - L. Wang et al, BMVC2016.
* [Real-time Action Recognition with Enhanced Motion Vector CNNs](http://arxiv.org/abs/1604.07669) - B. Zhang et al, CVPR2016. [[code]](https://github.com/zbwglory/MV-release)
* [Action Recognition with Trajectory-Pooled Deep-Convolutional Descriptors](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Action_Recognition_With_2015_CVPR_paper.pdf) - L. Wang et al, CVPR2015. [[code]](https://github.com/wanglimin/TDD)

### Miscellaneous
* [Slicing Convolutional Neural Network for Crowd Video Understanding](http://www.ee.cuhk.edu.hk/~jshao/papers_jshao/jshao_cvpr16_scnn.pdf) - J. Shao et al, CVPR2016. [[code]](https://github.com/amandajshao/Slicing-CNN)
* [Learning Spatiotemporal Features with 3D Convolutional Networks](http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf) - D. Tran et al, ICCV2015. [[code]](https://github.com/facebook/C3D) [[project web]](http://vlg.cs.dartmouth.edu/c3d/) Note: Aka C3D.

### Action Recognition Datasets
* [ActivityNet](http://activity-net.org/) Note: They provide a download script and evaluation code [here](https://github.com/activitynet) .
* [Charades](http://allenai.org/plato/charades/)
* [THUMOS14](http://crcv.ucf.edu/THUMOS14/) Note: It overlaps with [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset. 
* [THUMOS15](http://www.thumos.info/home.html) Note: It overlaps with [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset.
* [HOLLYWOOD2](http://www.di.ens.fr/~laptev/actions/hollywood2/): [Spatio-Temporal annotations](https://staff.fnwi.uva.nl/p.s.m.mettes/index.html#data)
* [UCF-101](http://crcv.ucf.edu/data/UCF101.php) and its variants: UCF-Sports, UCF-50, etc
* [HMDB](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
* [J-HMDB](http://jhmdb.is.tue.mpg.de/)
* [KTH](http://www.nada.kth.se/cvap/actions/)
* [MSR Action](http://research.microsoft.com/en-us/um/people/zliu/ActionRecoRsrc/) Note: It overlaps with [KTH](http://www.nada.kth.se/cvap/actions/) datset.
* [Sports-1M](http://cs.stanford.edu/people/karpathy/deepvideo/classes.html)
* [YouTube-8M](https://research.google.com/youtube8m/)
