Instuctions for setting up Jupyter Notebook:

	We used Jupyter Notebook as a research and prototyping tool. 

	Part I. Install Anaconda and Python 3.5

	Part II. Create and activate the conda environment by typing the following in the command line:

	1. git clone https://github.com/blueberrynfig/illegal-dumping-object-detection-engine.git
	2. cd illegal-dumping-object-detection-engine
	3. conda env create -f environment.yml
	4. source activate environment
	5. conda install -c conda-forge tensorflow
	6. jupyter notebook

Instructions for training and classification:

	1. build the retrainer:
	   bazel build tensorflow/examples/image_retraining:retrain

	2. download the inception model manually (optinal if automatic download fails): 
	   wget -O /tmp/imagenet/inception-2015-12-05.tgz http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

	3. retrain the last layer of the inception model: 
	   bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir ~/dumping_photos --output_graph ~/output_graph.pb --output_labels ~/output_labels.txt --bottleneck_dir ~/bottleneck

	4. classify an image:
	   python classify.py --image_dir /Users/wendyhu/mattress2.jpg --model_dir /Users/wendyhu/output_graph.pb --label_dir /Users/wendyhu/output_labels.txt 

SJSU CMPE295B 2017 by Wei-Chung Chen, Xiaoming Chuang, Wendy Hu, Luwen Miao.