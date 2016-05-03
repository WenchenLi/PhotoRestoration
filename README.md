# deprecated
We have moved to [here](https://github.com/averyallen71/dcgan-denosing-autoencoder). 
# PhotoRestoration
![alt text](http://www.crosswaysimages.ca/wp-content/uploads/2015/04/photo-restoration.jpg?quality=100.3015041915160)

[proposal slides](https://docs.google.com/presentation/d/14VL0wPYdZIuOWYzobaYvf3Y_LnnijEuD2tCy4h8utjw/edit?usp=sharing)
## Progress

our current focus: 

* data: taking existing face dataset like celebA with cropped face as ground truth training data,
and randomly assign a rectangular mask on the face as the training data, so that is a tuple (masked_image, ground_truth_image)
as training data.

* [model](https://github.com/WenchenLi/PhotoRestoration/blob/master/python/DCGAN-tensorflow/dAE_adversarial.py): convolutional autoencoder/decoder followed by a discriminator taking masked image and groundtruth image to discriminate
the restoration.


##current possible solution still requires "photoshop"
1. [facetune](http://www.facetuneapp.com/)
2. [retouch](http://www.colorpilot.com/retouch.html)
3. [retoucher](http://akvis.com/en/retoucher/index.php)

## Ideas
* Inspired by the network in network architecture proprosed by GoogLeNet as well as the R-CNN transition to faster-R-CNN, we want to design an architecture of the network based on the Generative Adversarial Networks (GAN) and convert the pipline of photo restoration within the network. For inpainting, we've found two CNN related papers.  

* For inpainting to work, our algorithm need to apply inpainting from the nearest neighbor of the missing part or creases, inpainting will not work if the area is big, in other words, our algorithm need to shrink the size of user defined problematic area until it's good enough to do inpainting.

* GAN in the sense generator now is the restored instance of image and discriminator classify whether current  restored instance is a good repairing. The problem is this network needs so many data to train the generator at least, but we can have databases of faces and fake the problematic area.

* Train faster RCNN to detect problematic area of the image in real time.(TODO: solve let the user select the area to fix)
##Our Approach
1. Non-face
	* Small tears, folds:Image Inpainting [4]
	* Missing patches:
		* Fragment-based Image Completion[8]
		* Scene completion(background?) 
2. Face
	* Non-fine-grained features(cheeks, forehead):
		* Small (Image Inpainting [4])
		* Patch (Segmented-based image completion[8])
	* Fine-grained features (Missing patches):
		* Graph Laplace for Occluded Face Completion[6][7]
		* Stronger conditional Generative Adversarial Network [5]
		* The condition is no longer semantic description of faces like age or races, but the faces with “problematic” area.


## resources might be useful

#### general ideas
2. [cvpr 2015 Image and Video Processing and Restoration](http://techtalks.tv/events/350/1619/)
3. [Generate image analogies using neural matching and blending](https://github.com/awentzonline/image-analogies)
4. [Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks](http://nucl.ai/semantic-style-transfer.pdf)

#### datasets
1. Aditya Khosla, Wilma A. Bainbridge, Antonio Torralba and Aude Oliva "Modifying the Memorability of Face Photographs."
International Conference on Computer Vision (ICCV), 2013.

#### inpainting related
1. [Shepard Convolutional Neural Networks](http://papers.nips.cc/paper/5774-shepard-convolutional-neural-networks.pdf)
2. [Image Denoising and Inpainting with Deep Neural Networks](http://papers.nips.cc/paper/4686-image-denoising-and-inpainting-with-deep-neural-networks.pdf)

#### GAN github resources
1. [original work](https://github.com/goodfeli/adversarial) 
2. [tensorflow implementation](https://github.com/carpedm20/DCGAN-tensorflow)
3. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/Newmu/dcgan_code)
4. [Conditional generative adversarial networks for convolutional face generation](https://github.com/hans/adversarial)

#### CNN image search
1.[Cross-dimensional Weighting (CroW) aggregation(with repo,search based on CroW extracted by CNN)](https://github.com/yahoo/crow)

##reference
<!--1. [Scene Completion Using Millions of Photographs. James Hays, Alexei A. Efros. ACM Transactions on Graphics (SIGGRAPH 2007). August 2007, vol. 26, No. 3.](http://graphics.cs.cmu.edu/projects/scene-completion/)-->
<!--2. [Sketch2Photo: Internet Image Montage. ACM SIGGRAPH ASIA 2009, ACM Transactions on Graphics. Tao Chen, Ming-Ming Cheng, Ping Tan, Ariel Shamir, Shi-Min Hu.](http://cg.cs.tsinghua.edu.cn/montage/main.htm)-->
<!--3. [Supervised Learning of Semantics-Preserving Hashing via Deep Neural Networks for Large-Scale Image Search Huei-Fang Yang, Kevin Lin, Chu-Song Chen arXiv preprint arXiv:1507.00101](http://arxiv.org/abs/1507.00101)-->
1. ["Generative Adversarial Networks." Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. ArXiv 2014.](http://arxiv.org/abs/1406.2661)
1. Dale, Kevin, et al. "Image restoration using online photo collections."Computer Vision, 2009 IEEE 12th International Conference on. IEEE, 2009.
2. Geman, Stuart, and Donald Geman. "Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images." Pattern Analysis and Machine Intelligence, IEEE Transactions on 6 (1984): 721-741.
3. Dong, Chao, et al. "Learning a deep convolutional network for image super-resolution." Computer Vision–ECCV 2014. Springer International Publishing, 2014. 184-199.
4. Bertalmio, Marcelo, et al. "Image inpainting." Proceedings of the 27th annual conference on Computer graphics and interactive techniques. ACM Press/Addison-Wesley Publishing Co., 2000.
5.  M. Mirza and S. Osindero. Conditional Generative Adversarial Nets. arXiv:1411.1784 [cs, stat], Nov. 2014. arXiv: 1411.1784 
6.  Deng, Yue, Qionghai Dai, and Zengke Zhang. "Graph Laplace for occluded face completion and recognition." Image Processing, IEEE Transactions on 20.8 (2011): 2329-2338.
7.  J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma, “Robust face recognition via sparse representation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 31, no. 2, pp. 210–227, Feb. 2009.  							8.  Drori, Iddo, Daniel Cohen-Or, and Hezy Yeshurun. "Fragment-based image completion." ACM Transactions on Graphics (TOG). Vol. 22. No. 3. ACM, 2003.
				
			
		


