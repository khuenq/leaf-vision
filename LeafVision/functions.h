#include "masterInclude.h"

#pragma region Functions
//Show single image
void showImage(Mat &image, string windowname){
	//Create a new image window for original image with no resize allowed
	namedWindow(windowname, CV_WINDOW_AUTOSIZE);
	imshow(windowname, image);
}

//Read multiple image paths and their labels
void readCSV(const string &filename, vector<string> &images, vector<float>& labels, char separator = ';'){
	cout << "Reading image path..." << endl;
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
			images.push_back(path);
			labels.push_back(strtol(classlabel.c_str(), NULL, 0));
			if(count % 4 == 0){
				cout << ".";
			}
			count++;
		}
	}
	cout << "\nRead " << images.size() << " image paths!" << endl;

	//std::ofstream report;
	//report.open("data/report.txt");
	//for(int i = 0; i < labels.size(); i++){
	//	report << labels[i] << endl;
	//}
	//report.close();
}

void readPlantData(const string &filename, vector<string> &scientific, vector<string> &common, char separator = '|'){
	cout << "Reading plant name data..." << endl;
	ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, sc_name, co_name;
	int count = 0;
	while(getline(file, line)){
		stringstream liness(line);
		getline(liness, sc_name, separator);
		getline(liness, co_name);
		if(!sc_name.empty() && !co_name.empty()){
			scientific.push_back(sc_name);
			common.push_back(co_name);
			if(count % 4 == 0){
				cout << ".";
			}
			count++;
		}
	}
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
void makeMask(Mat &image, Mat &result){
	//Clone image
	result = image.clone();

	//Reduce image noise
	cv::medianBlur(result, result, 11);

	//Apply image thresholding
	//Max color value, hard coded to 255 as input image was converted to gray scale
	double t = cv::threshold(result, result, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
}

// Perform in-place unsharped masking operation
void unsharpMask(cv::Mat& image) 
{
	cv::Mat normalized, tmp;
	cv::normalize(image, normalized, 0, 255, NORM_MINMAX);
	cv::GaussianBlur(normalized, tmp, cv::Size(5,5), 5);
	cv::addWeighted(normalized, 2, tmp, -1, 0, normalized);
	image = normalized.clone();
	normalized.release();
	tmp.release();
}

//Convert images to Cs channel images and extract masks
void toCsChannelImg(Mat &image, Mat &result, Mat &mask){
	//resize image
	if(image.cols != 640 && image.rows != 480){
		cv::resize(image, result, Size(640, 480), 0, 0);
	}
	else
		result = image.clone();
	//Make image masks
	makeMask(image, mask);

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

void toEdgeImage(Mat &image, Mat &result, 
	double threshold = 240.0, int threshold_type = THRESH_BINARY_INV){
		//resize image
		if(image.cols != 640 && image.rows != 480){
			cv::resize(image, result, Size(640, 480), 0, 0);
		}
		else
			result = image.clone();

		cvtColor(result, result, CV_BGR2GRAY);

		medianBlur(result, result, 3);

		int colour = 255;

		double t = cv::threshold(result, result, threshold, colour, THRESH_BINARY_INV);
		imwrite("images/segmentation/thresholded.jpg", result);

		cv::morphologyEx(result, result, cv::MORPH_GRADIENT, cv::Mat());
}

void readCSV2(const std::string &filename, std::vector<std::string> &images){
	std::ifstream file(filename.c_str(), std::ifstream::in);
	if (!file) {
		std::string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	std::string line, path;
	while (std::getline(file, line)) {
		std::stringstream liness(line);
		std::getline(liness, path);
		if(!path.empty()) {
			images.push_back(path);
		}
	}
}
#pragma endregion Functions