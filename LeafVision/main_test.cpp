//Include libraries
#include "functions.h"

//Main program execution
int main(int argc, const char* argv){
	vector<string> image_paths;
	vector<float> image_labels;
	readCSV("data/CSV/experiment_testing.csv", image_paths, image_labels);
	for(int i = 0; i < image_paths.size(); i++){
		Mat tmp_image = imread(image_paths[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat tmp_image_mask;
		double thresh = makeMask(tmp_image, tmp_image_mask, 240.0, THRESH_BINARY_INV + THRESH_OTSU);
		string filename = "thresholded/" + image_paths[i].substr(image_paths[i].find_last_of("/") + 1);
		imwrite(filename, tmp_image_mask);
		cout << thresh << " ";
	}
	cout << endl;
	//waitKey(0);
	system("PAUSE");
	return 0;
}