//Include libraries
#include "masterInclude.h"

//Read image paths and labels from CSV file
void readImagePath(const string &filename, vector<string> &image_paths, 
	vector<float> &image_labels, char separator = ';'){
		cout << "Reading image paths..." << endl;
		std::ifstream file(filename.c_str(), ifstream::in);
		if (!file) {
			string error_message = "No valid input file was given, please check the given filename.";
			CV_Error(CV_StsBadArg, error_message);
		}
		string line, path, classlabel;
		int count = 0;
		while (getline(file, line)) {
			stringstream liness(line);
			getline(liness, path, separator);
			getline(liness, classlabel);
			if(!path.empty() && !classlabel.empty()) {
				image_paths.push_back(path);
				image_labels.push_back(atoi(classlabel.c_str()));
				if(count % 4 == 0){
					cout << ".";
				}
				count++;
			}
		}
		cout << "\nRead " << image_paths.size() << " image data!" << endl;
}
//Get max channel value
int maxchannel(int chan1, int chan2, int chan3){
	return max(max(chan1, chan2), chan3);
}
//Get min channel value
int minchannel(int chan1, int chan2, int chan3){
	return min(min(chan1, chan2), chan3);
}
//Make image mask based on leaf shape
void makeMask(Mat &image, Mat &result, double threshold = 240.0, int threshold_type = THRESH_BINARY_INV){
	//Clone image
	result = image.clone();
	//Reduce image noise
	medianBlur(result, result, 3);
	if(result.type() != 0)
		cvtColor(result, result, CV_BGR2GRAY);
	//Apply image thresholding
	//Max color value, hard coded to 255 as input image was converted to gray scale
	int colour = 255;
	double t = cv::threshold(result, result, threshold, colour, threshold_type);
}
//Pre-process
void preProcess(Mat &image, Mat &out, Mat &mask){
	makeMask(image, mask);
	resize(image, out, Size(640, 480), 0, 0);
	cvtColor(out, out, CV_BGR2GRAY);
}
//Convert image matrices to Cs channel image matrices
void toCsChannelImg(Mat &image, Mat &result){
	//resize image
	if(image.cols != 640 && image.rows != 480){
		cv::resize(image, result, Size(640, 480), 0, 0);
	}
	else
		result = image.clone();

	//Obtain iterator from the image
	Mat_<Vec3b>::iterator it = result.begin<Vec3b>();
	Mat_<Vec3b>::iterator itend = result.end<Vec3b>();

	//declare 3 channel values and a variable to store Cs value
	int R, G, B;
	uchar Cs;
	float S, delta; //Saturation channel
	//Iterate through the image
	while(it != itend){
		B = (*it)[0];
		G = (*it)[1];
		R = (*it)[2];
		float MIN = (float) minchannel(R, G, B);
		float MAX = (float) maxchannel(R, G, B);
		delta = MAX - MIN;
		if(MAX != 0)
			S = delta/MAX;
		else
			S = 0;
		Cs = ((R + G)/2)*sqrt(S);
		(*it)[0] = Cs;
		(*it)[1] = Cs;
		(*it)[2] = Cs;
		++it;
	}
}

//Extract image keypoints
void extractKeypoints(vector<string> &image_paths, 
	vector<vector<KeyPoint>> &keypoint_set, vector<Mat> &descriptor_set,
	Ptr<FeatureDetector> &detector = FeatureDetector::create("SURF"),
	Ptr<DescriptorExtractor> &extractor = DescriptorExtractor::create("SURF"),
	double mask_threshold = 240.0, int mask_threshold_type = THRESH_BINARY_INV){
		cout << "Extract keypoints and descriptors from images" << endl;
		for(int i = 0; i < image_paths.size(); i++){
			Mat tmp_image = imread(image_paths[i], 1);
			Mat tmp_mask, tmp_descriptors;
			vector<KeyPoint> tmp_keypoints;
			makeMask(tmp_image, tmp_mask, mask_threshold, mask_threshold_type);
			toCsChannelImg(tmp_image, tmp_image);
			detector->detect(tmp_image, tmp_keypoints);
			if(!tmp_keypoints.empty()){
				keypoint_set.push_back(tmp_keypoints);
				extractor->compute(tmp_image, tmp_keypoints, tmp_descriptors);
				descriptor_set.push_back(tmp_descriptors);
			}
			else{
				cerr << "No keypoint detected! Terminate the program!!!" << endl;
			}
			if(i % 4 == 0)
				cout << ".";
			else if(i == image_paths.size() - 1);
				cout << endl;
		}
}
//Build Bag-of-word dictionary from descriptor vector
void buildBowDict(vector<Mat> &descriptor_set, BOWKMeansTrainer bowTrainer, Mat &out_dictionary){
	cout << "Building Bag-of-word dictionary..." << endl;
	for(int i = 0; i < descriptor_set.size(); i++){
		bowTrainer.add(descriptor_set[i]);
	}
	out_dictionary = bowTrainer.cluster();
}
//Extract Bag-of-word descriptors
void extractBowDescriptors(vector<string> image_paths, vector<float> image_labels, vector<vector<KeyPoint>> keypoint_set, 
	Mat dictionary, BOWImgDescriptorExtractor bowDE, Mat &out_trainingData, Mat &out_trainingLabels){
		cout << "Extract Bag-of-word descriptors" << endl;
		bowDE.setVocabulary(dictionary);
		for(int i = 0; i < image_paths.size(); i++){
			Mat tmp_image = imread(image_paths[i], 1);
			Mat tmp_descriptors;
			vector<KeyPoint> tmp_keypoints = keypoint_set[i];
			bowDE.compute(tmp_image, tmp_keypoints, tmp_descriptors);
			out_trainingData.push_back(tmp_descriptors);
			out_trainingLabels.push_back(image_labels[i]);
			if(i % 4 == 0)
				cout << ".";
			else if(i == image_paths.size() - 1)
				cout << endl;
		}
}
//Count number of occurrence for a class
int count_predicted(int base_class, vector<float> predictions){
	int count = 0;
	for(int i = 0; i < predictions.size(); i++){
		if(predictions[i] == base_class)
			count++;
	}
	return count;
}
//Make confusion matrix
Mat makeConfusionMatrix(int num_classes, vector<float> predictions, bool histogram_export){
	Mat confusion(num_classes, num_classes, CV_16SC1, cv::Scalar(0));
	Mat confusion_hist(512, 512, CV_32FC1);

	for(int i = 0; i < confusion.rows; i++){
		for(int j = 0; j < confusion.cols; j++){
			confusion.at<short>(i, j) = count_predicted(j+1, predictions);
		}
	}

	return confusion;
}
//Show single image
void showImage(Mat &image, string windowname){
	//Create a new image window for original image with no resize allowed
	namedWindow(windowname, CV_WINDOW_AUTOSIZE);
	imshow(windowname, image);
}
//Sharpen gray scale image
void sharpen(const cv::Mat &image, cv::Mat &result) {
	// allocate if necessary
	result.create(image.size(), image.type()); 
	for (int j= 1; j<image.rows-1; j++) { // for all rows 
		// (except first and last)
		const uchar* previous= 
			image.ptr<const uchar>(j-1); // previous row
		const uchar* current= 
			image.ptr<const uchar>(j); // current row
		const uchar* next= 
			image.ptr<const uchar>(j+1); // next row
		uchar* output= result.ptr<uchar>(j); // output row
		for (int i=1; i<image.cols-1; i++) {
			*output++= cv::saturate_cast<uchar>(
				5*current[i]-current[i-1]
			-current[i+1]-previous[i]-next[i]); 
		}
	}
	// Set the unprocessed pixels to 0
	result.row(0).setTo(cv::Scalar(0));
	result.row(result.rows-1).setTo(cv::Scalar(0));
	result.col(0).setTo(cv::Scalar(0));
	result.col(result.cols-1).setTo(cv::Scalar(0));
}

void salt(cv::Mat &image, int n) {
	for (int k=0; k<n; k++) {
		int i= rand()%image.cols;
		int j= rand()%image.rows;
		if (image.channels() == 1) { // gray-level image
			image.at<uchar>(j,i)= 255; 
		} else if (image.channels() == 3) { // color image
			image.at<cv::Vec3b>(j,i)[0]= 255; 
			image.at<cv::Vec3b>(j,i)[1]= 255; 
			image.at<cv::Vec3b>(j,i)[2]= 255; 
		}
	}
}

void KyneSeg(Mat &image, Mat &result) {
	if(image.channels() != 1)
		cvtColor(image, result, CV_BGR2GRAY, 0);
	else{
		result = image.clone();
		for(int i = 0; i < result.rows; i++){
			for(int j = 0; j < result.cols; j++){

			}
		}
	}
}

void smqt(Mat &image, Mat &result, int level){
	result = image.clone();
	Scalar mean_intensity = mean(result);
	for(int y = 0; y < result.rows - 1; y++){
		for(int x = 0; x < result.cols - 1; x++){
			Scalar intensity = result.at<uchar>(Point(x, y));
			if(intensity.val[0] > mean_intensity.val[0])
				result.at<uchar>(Point(x, y)) = 1;
			else
				result.at<uchar>(Point(x, y)) = 0;
		}
	}
	showImage(result, "test");
}