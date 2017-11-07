# Awesome Action Recognition: [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of action recognition and related area (e.g. object recognition, pose estimation) resources, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

## Contents
 - [Action Recognition](#action-recognition)
 - [Object Recognition](#object-recognition)
 - [Pose Estimation](#pose-estimation)

## Action Recognition

### Spatio-Temporal Action Detection
* [Action Tubelet Detector for Spatio-Temporal Action Localization](https://arxiv.org/abs/1705.01861) - V. Kalogeiton et al, ICCV2017. [[code]](https://github.com/vkalogeiton/caffe/tree/act-detector) [[project web]](http://thoth.inrialpes.fr/src/ACTdetector/)
* [Tube Convolutional Neural Network (T-CNN) for Action Detection in Videos](https://128.84.21.199/pdf/1703.10664.pdf) - [R. Hou](http://www.cs.ucf.edu/~rhou/) et al, ICCV2017. [[project web]](http://crcv.ucf.edu/projects/TCNN/)
* [Chained Multi-stream Networks Exploiting Pose, Motion, and Appearance for Action Classification and Detection](https://arxiv.org/abs/1704.00616) - M. Zolfaghari et al, ICCV2017. [[project web]](https://lmb.informatik.uni-freiburg.de/projects/action_chain/)
* [Online Real time Multiple Spatiotemporal Action Localisation and Prediction](https://arxiv.org/pdf/1611.08563v3.pdf) - [G. Singh](http://gurkirt.github.io/) et al, ICCV2017.
* [AMTnet: Action-Micro-Tube regression by end-to-end trainable deep architecture](https://arxiv.org/pdf/1704.04952.pdf) - S. Saha et al, ICCV2017.
* [Am I Done? Predicting Action Progress in Videos](https://arxiv.org/abs/1705.01781) - F. Becattini et al, BMVC2017.
* [Generic Tubelet Proposals for Action Localization](https://arxiv.org/abs/1705.10861) - J. He et al, arXiv2017.
* [Incremental Tube Construction for Human Action Detection](https://arxiv.org/pdf/1704.01358.pdf) - H. S. Behl et al, arXiv2017.
* [Multi-region two-stream R-CNN for action detection](https://www.robots.ox.ac.uk/~vgg/rg/papers/peng16eccv.pdf) - [X. Peng](http://xjpeng.weebly.com/) and C. Schmid. ECCV2016. [[code]](https://github.com/pengxj/action-faster-rcnn)
* [Spot On: Action Localization from Pointly-Supervised Proposals](http://jvgemert.github.io/pub/spotOnECCV16.pdf) - P. Mettes et al, ECCV2016.
* [Deep Learning for Detecting Multiple Space-Time Action Tubes in Videos](https://arxiv.org/abs/1608.01529) - S. Saha et al, BMVC2016. [[code]](https://bitbucket.org/sahasuman/bmvc2016_code) [[project web]](http://sahasuman.bitbucket.org/bmvc2016/)
* [Learning to track for spatio-temporal action localization](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Weinzaepfel_Learning_to_Track_ICCV_2015_paper.pdf) - P. Weinzaepfel et al. ICCV2015.
* [Action detection by implicit intentional motion clustering](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Chen_Action_Detection_by_ICCV_2015_paper.pdf) - W. Chen and J. Corso, ICCV2015.
* [Finding Action Tubes](https://people.eecs.berkeley.edu/~gkioxari/ActionTubes/action_tubes.pdf) - G. Gkioxari and J. Malik CVPR2015. [[code]](https://github.com/gkioxari/ActionTubes) [[project web]](https://people.eecs.berkeley.edu/~gkioxari/ActionTubes/)
* [APT: Action localization proposals from dense trajectories](http://jvgemert.github.io/pub/gemertBMVC15APTactionProposals.pdf) - J. Gemert et al, BMVC2015. [[code]](https://github.com/jvgemert/apt)
* [Spatio-Temporal Object Detection Proposals](https://hal.inria.fr/hal-01021902/PDF/proof.pdf) - D. Oneata et al, ECCV2014. [[code]](https://bitbucket.org/doneata/proposals) [[project web]](http://lear.inrialpes.fr/~oneata/3Dproposals/)
* [Action localization with tubelets from motion](http://isis-data.science.uva.nl/cgmsnoek/pub/jain-tubelets-cvpr2014.pdf) - M. Jain et al, CVPR2014.
* [Spatiotemporal deformable part models for action detection](http://crcv.ucf.edu/papers/cvpr2013/cvpr2013-sdpm.pdf) - [Y. Tian](http://www.cs.ucf.edu/~ytian/index.html) et al, CVPR2013. [[code]](http://www.cs.ucf.edu/~ytian/sdpm.html)
* [Action localization in videos through context walk](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Soomro_Action_Localization_in_ICCV_2015_paper.pdf) - K. Soomro et al, ICCV2015.
* [Fast Action Proposals for Human Action Detection and Search](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yu_Fast_Action_Proposals_2015_CVPR_paper.pdf) - G. Yu and J. Yuan, CVPR2015. Note: code for FAP is NOT available online. Note: Aka FAP.

### Temporal Action Detection
* [CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos](https://arxiv.org/abs/1703.01515/) - Z. Shou et al, CVPR2017. [[code]](https://bitbucket.org/columbiadvmm/cdc)
* [SST: Single-Stream Temporal Action Proposals](http://vision.stanford.edu/pdf/buch2017cvpr.pdf) - S. Buch et al, CVPR2017. [[code]](https://github.com/shyamal-b/sst)
* [R-C3D: Region Convolutional 3D Network for Temporal Activity Detection](https://arxiv.org/abs/1703.07814) - H. Xu et al, arXiv2017. [[project web]](http://ai.bu.edu/r-c3d/)
* [DAPs: Deep Action Proposals for Action Understanding](https://ivul.kaust.edu.sa/Documents/Publications/2016/DAPs%20Deep%20Action%20Proposals%20for%20Action%20Understanding.pdf) - V. Escorcia et al, ECCV2016. [[code]](https://github.com/escorciav/daps) [[raw data]](https://github.com/escorciav/daps)
* [Online Action Detection using Joint Classification-Regression Recurrent Neural Networks](https://arxiv.org/abs/1604.05633) - Y. Li et al, ECCV2016. Noe: RGB-D Action Detection
* [Temporal Action Localization in Untrimmed Videos via Multi-stage CNNs](http://dvmmweb.cs.columbia.edu/files/dvmm_scnn_paper.pdf) - Z. Shou et al, CVPR2016. [[code]](https://github.com/zhengshou/scnn) Note: Aka S-CNN.
* [Fast Temporal Activity Proposals for Efficient Detection of Human Actions in Untrimmed Videos](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Heilbron_Fast_Temporal_Activity_CVPR_2016_paper.pdf) - F. Heilbron et al, CVPR2016. [[code]](https://github.com/cabaf/sparseprop) Note: Depends on [C3D](http://vlg.cs.dartmouth.edu/c3d/), aka SparseProp.
* [Actionness Estimation Using Hybrid Fully Convolutional Networks](https://arxiv.org/abs/1604.07279) - L. Wang et al, CVPR2016. [[code]](https://github.com/wanglimin/actionness-estimation/) Note: The code is not a complete verision. It only contains a demo, not training. [[project web]](http://wanglimin.github.io/actionness_hfcn/index.html)
* [Learning Activity Progression in LSTMs for Activity Detection and Early Detection](http://cs.brown.edu/~ls/Publications/cvpr2016ma.pdf) - S. Ma et al, CVPR2016.
* [End-to-end Learning of Action Detection from Frame Glimpses in Videos](http://vision.stanford.edu/pdf/yeung2016cvpr.pdf) - S. Yeung et al, CVPR2016. [[code]](https://github.com/syyeung/frameglimpses) [[project web]](http://ai.stanford.edu/~syyeung/frameglimpses.html) Note: This method uses reinforcement learning
* [Fast Action Proposals for Human Action Detection and Search](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yu_Fast_Action_Proposals_2015_CVPR_paper.pdf) - G. Yu and J. Yuan, CVPR2015. Note: code for FAP is NOT available online. Note: Aka FAP.
* [Bag-of-fragments: Selecting and encoding video fragments for event detection and recounting](https://staff.fnwi.uva.nl/t.e.j.mensink/publications/mettes15icmr.pdf) - P. Mettes et al, ICMR2015.
* [Action localization in videos through context walk](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Soomro_Action_Localization_in_ICCV_2015_paper.pdf) - K. Soomro et al, ICCV2015.

### Spatio-Temporal ConvNets
* [Deep Temporal Linear Encoding Networks](https://arxiv.org/abs/1611.06678) - A. Diba et al, CVPR2017.
* [Temporal Convolutional Networks: A Unified Approach to Action Segmentation and Detection](https://arxiv.org/pdf/1611.05267.pdf) - C. Lea et al, CVPR 2017. [[code]](https://github.com/colincsl/TemporalConvolutionalNetworks)
* [Long-term Temporal Convolutions](https://arxiv.org/pdf/1604.04494v1.pdf) - G. Varol et al, TPAMI2017. [[project web]](http://www.di.ens.fr/willow/research/ltc/) [[code]](https://github.com/gulvarol/ltc) 
* [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://arxiv.org/pdf/1608.00859.pdf) - L. Wang et al, arXiv 2016. [[code]](https://github.com/yjxiong/temporal-segment-networks)

### Action Classification
* [Fully Context-Aware Video Prediction](https://arxiv.org/pdf/1710.08518v1.pdf) - Byeon et al, arXiv2017.
* [Dynamic Image Networks for Action Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16a/bilen16a.pdf) - H. Bilen et al, CVPR2016. [[code]](https://github.com/hbilen/dynamic-image-nets) [[project web]](http://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16a/)
* [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf) - J. Donahue et al, CVPR2015. [[code]](https://github.com/LisaAnne/lisa-caffe-public/tree/lstm_video_deploy) [[project web]](http://jeffdonahue.com/lrcn/)
* [Describing Videos by Exploiting Temporal Structure](http://arxiv.org/pdf/1502.08029v4.pdf) - L. Yao et al, ICCV2015. [[code]](https://github.com/yaoli/arctic-capgen-vid) note: from the same group of RCN paper “Delving Deeper into Convolutional Networks for Learning Video Representations"
* [Two-Stream SR-CNNs for Action Recognition in Videos](http://wanglimin.github.io/papers/ZhangWWQW_CVPR16.pdf) - L. Wang et al, BMVC2016.
* [Real-time Action Recognition with Enhanced Motion Vector CNNs](http://arxiv.org/abs/1604.07669) - B. Zhang et al, CVPR2016. [[code]](https://github.com/zbwglory/MV-release)
* [Action Recognition with Trajectory-Pooled Deep-Convolutional Descriptors](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Action_Recognition_With_2015_CVPR_paper.pdf) - L. Wang et al, CVPR2015. [[code]](https://github.com/wanglimin/TDD)
* [Convolutional Two-Stream Network Fusion for Video Action Recognition](https://arxiv.org/pdf/1604.06573.pdf) - C. Feichtenhofer et al, CVPR2016. [[code]](https://github.com/feichtenhofer/twostreamfusion)

### Video Representation
* [Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.pdf) - Z. Qui et al, ICCV2017. [[code]](https://github.com/ZhaofanQiu/pseudo-3d-residual-networks)
* [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf) - J. Carreira et al, CVPR2017. [[code]](https://github.com/deepmind/kinetics-i3d)
* [Learning Spatiotemporal Features with 3D Convolutional Networks](http://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf) - D. Tran et al, ICCV2015. [[the official Caffe code]](https://github.com/facebook/C3D) [[project web]](http://vlg.cs.dartmouth.edu/c3d/) Note: Aka C3D. [[Python Wrapper]](https://github.com/chuckcho/C3D/tree/python-wrapper) Note that the official caffe does not support python wrapper. [[TensorFlow]](https://github.com/hx173149/C3D-tensorflow), [[TensorFlow + Keras]](https://github.com/axon-research/c3d-keras), [[Another TensorFlow Implemetation]](https://github.com/frankgu/C3D-tensorflow.git), [[Keras C3D Project web]](https://imatge.upc.edu/web/resources/c3d-model-keras-trained-over-sports-1m): [[Keras code]](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2), [[Pretrained weights]](https://www.dropbox.com/s/ypiwalgtlrtnw8b/c3d-sports1M_weights.h5?dl=0).

### Miscellaneous
* [CortexNet: a Generic Network Family for Robust Visual Temporal Representations](https://arxiv.org/pdf/1706.02735.pdf) A. Canziani and E. Culurciello - arXiv2017. [[code]](https://github.com/atcold/pytorch-CortexNet) [[project web]](https://engineering.purdue.edu/elab/CortexNet/)
* [Slicing Convolutional Neural Network for Crowd Video Understanding](http://www.ee.cuhk.edu.hk/~jshao/papers_jshao/jshao_cvpr16_scnn.pdf) - J. Shao et al, CVPR2016. [[code]](https://github.com/amandajshao/Slicing-CNN)
* [Two-Stream (RGB and Flow) pretrained model weights](https://github.com/craftGBD/caffe-GBD/tree/master/models/action_recognition)

### Action Recognition Datasets
* [20BN-JESTER](https://www.twentybn.com/datasets/jester), [20BN-SOMETHING-SOMETHING](https://www.twentybn.com/datasets/something-something)
* [AVA](https://arxiv.org/abs/1705.08421)
* [ActivityNet](http://activity-net.org/) Note: They provide a download script and evaluation code [here](https://github.com/activitynet) .
* [Charades](http://allenai.org/plato/charades/)
* [THUMOS14](http://crcv.ucf.edu/THUMOS14/) Note: It overlaps with [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset. 
* [THUMOS15](http://www.thumos.info/home.html) Note: It overlaps with [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset.
* [HOLLYWOOD2](http://www.di.ens.fr/~laptev/actions/hollywood2/): [Spatio-Temporal annotations](https://staff.fnwi.uva.nl/p.s.m.mettes/index.html#data)
* [UCF-101](http://crcv.ucf.edu/data/UCF101.php), [annotation provided by THUMOS-14](http://crcv.ucf.edu/ICCV13-Action-Workshop/index.files/UCF101_24Action_Detection_Annotations.zip), and [corrupted annotation list](https://github.com/jinwchoi/Jinwoo-Computer-Vision-and-Machine-Learing-papers-to-read/blob/master/UCF101_Spatial_Annotation_Corrupted_file_list),  [UCF-101 corrected annotations](https://github.com/gurkirt/corrected-UCF101-Annots) and [different version annotaions](https://github.com/jvgemert/apt). And there are also some pre-computed spatiotemporal action detection [results](https://drive.google.com/drive/folders/0B-LzM05qEdk0aG5pTE94VFI1SUk)
* [UCF-50](http://crcv.ucf.edu/data/UCF50.php).
* [UCF-Sports](http://crcv.ucf.edu/data/UCF_Sports_Action.php), note: the train/test split link in the official website is broken. Instead, you can download it from [here](http://pascal.inrialpes.fr/data2/oneata/data/ucf_sports/videos.txt).
* [HMDB](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
* [J-HMDB](http://jhmdb.is.tue.mpg.de/)
* [LIRIS-HARL](http://liris.cnrs.fr/voir/activities-dataset/)
* [KTH](http://www.nada.kth.se/cvap/actions/)
* [MSR Action](https://www.microsoft.com/en-us/download/details.aspx?id=52315) Note: It overlaps with [KTH](http://www.nada.kth.se/cvap/actions/) datset.
* [Sports-1M](http://cs.stanford.edu/people/karpathy/deepvideo/classes.html) - Large scale action recognition dataset.
* [YouTube-8M](https://research.google.com/youtube8m/), [technical report](https://arxiv.org/abs/1609.08675)
* [YouTube-BB](https://research.google.com/youtube-bb/), [technical report](https://arxiv.org/pdf/1702.00824.pdf)
* [DALY](http://thoth.inrialpes.fr/daly/) Daily Action Localization in Youtube videos
* [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [paper](https://arxiv.org/pdf/1705.07750.pdf)

### Video Annotation
* [Efficiently scaling up crowdsourced video annotation](http://cvrr.ucsd.edu/ece285/Spring2014/papers/Vondrick_IJCV2013.pdf) - C. Vondrick et. al, IJCV2013. [[code]](https://github.com/cvondrick/vatic)
* [The Design and Implementation of ViPER](https://www.cs.umd.edu/grad/scholarlypapers/papers/davidm-viper.pdf) - D. Mihalcik and D. Doermann, Technical report.


## Object Recognition

### Object Detection
* [Faster R-CNN](https://arxiv.org/abs/1506.01497) - S. Ren et al, NIPS2015. [[official MatCaffe code]](https://github.com/ShaoqingRen/faster_rcnn), [[PyCaffe]](https://github.com/rbgirshick/py-faster-rcnn), [[TensorFlow]](https://github.com/smallcorgi/Faster-RCNN_TF), [[Another TF implementation]](https://github.com/CharlesShang/TFFRCNN) [[Keras]](https://github.com/yhenon/keras-frcnn) - State-of-the-art object detector.
* [YOLO](https://pjreddie.com/media/files/papers/yolo.pdf) - J. Redmon et al, CVPR2016. [[official code]](https://github.com/pjreddie/darknet.git), [[TensorFLow]](https://github.com/gliese581gg/YOLO_tensorflow) - Fast object detector.
* [YOLO9000](https://arxiv.org/abs/1612.08242) - J. Redmon and A. Farhadi, CVPR2017. [[official code]](https://pjreddie.com/darknet/yolo/) - State-of-the-art object detector which can detect 9000 objects in realtime.
* [SSD](https://arxiv.org/abs/1512.02325) - W. Liu et al, ECCV2016. [[official PyCaffe code]](https://github.com/weiliu89/caffe/tree/ssd), [[TensorFlow]](https://github.com/balancap/SSD-Tensorflow), [[Keras]](https://github.com/rykov8/ssd_keras) - State-of-the-art object detector with realtime processing speed.
* [Mask R-CNN](https://arxiv.org/abs/1703.06870) - K. He et al, [[TensorFlow + Keras]](https://github.com/matterport/Mask_RCNN), [[MXNet]](https://github.com/TuSimple/mx-maskrcnn), [[TensorFlow]](https://github.com/CharlesShang/FastMaskRCNN), [[PyTorch]](https://github.com/felixgwu/mask_rcnn_pytorch) - State-of-the-art object detection/instance segmentation algorithm.



## Pose Estimation

### Pose Estimation
* [OpenPose Library](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - Caffe based realtime pose estimation library from CMU.
* [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050) - Z. Cao et al, CVPR2017. [[code]](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) depends on the [[caffe RT pose]](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose.git) - Earlier version of OpenPose from CMU.


## Licenses
License

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Jinwoo Choi](https://sites.google.com/site/jchoivision/) has waived all copyright and related or neighboring rights to this work.


## Contributing
Please read the [contribution guidelines](contributing.md). Then please feel free to send me [pull requests](https://github.com/jinwchoi/Action-Recognition/pulls) or email (jinchoi@vt.edu) to add links. 
