//Included libraries
#include "functions.h"

//Main program entry point
int main(int argc, const char* argv[]){

#pragma region Initialization
	//Tick meter for calculate performance
	TickMeter tm;
	tm.start();
	double init_time = 0, trainbow_time = 0, trainsvm_time = 0, test_time = 0, eval_time = 0;
	cout << "Initializing..." << endl;
	//Number of classes
	const int no_of_classes = 32;
	//Vectors for training image paths
	vector<string> image_paths;
	vector<float> image_labels;
	//Vectors for testing image paths
	vector<string> test_image_paths;
	vector<float> test_image_labels;
	//Vectors for plant name data
	vector<string> scientific_names;
	vector<string> common_names;
	//CSV file used for reading multiple images
	string csv_train_path = "data/CSV/database_training.csv";
	string csv_test_path = "data/CSV/database_testing.csv";
	string species_path = "data/CSV/species.csv";
	//Thresholding parameters
	double threshold = 240.0;
	int threshold_type = THRESH_BINARY_INV;
	//BOW dictionary size
	int bow_dictionary_size = 250;
	//Standard image size
	Size image_size(640, 480);
	//Define data storage
	ifstream bow_dictionary("data/dictionary.xml");
	ifstream trainedsvm("data/trained_svm.xml");
	//Define feature detector, extractor and matcher types
	string detector_type = "SURF";
	string extractor_type = "SURF";
	string matcher_type = "FlannBased";
	Ptr<FeatureDetector> detector = new SurfFeatureDetector();
	Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
	Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher();
	//Training data matrices
	Mat trainingData(0, bow_dictionary_size, CV_32FC1);
	Mat labelsData(0, 1, CV_32FC1);
	//Define parameters for Bag-of-word model
	TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	//Define BOW Trainer and Descriptor Extractor
	BOWKMeansTrainer bowTrainer(bow_dictionary_size, tc, retries, flags);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	//Total extracted descriptors
	int total_desc = 0;
	//Create SVM instance
	SVM svm;
	//Create report open file stream
	ofstream report;
	ofstream results;
	tm.stop();
	init_time = tm.getTimeMilli();;
	cout << init_time << " ms!" << endl;
#pragma endregion Initialization

#pragma region Training
	tm.start();
	//Read image paths from CSV file
	if(!bow_dictionary.good() && !trainedsvm.good())
	{
		cout << "Load train data..." << endl;
		readCSV(csv_train_path, image_paths, image_labels);
	}
	if(!bow_dictionary.good()){
		cout << "Detect " << detector_type << " features and extract " << extractor_type << " descriptors..." << endl;
		for(int i = 0; i < image_paths.size(); i++){
			//Preprocessing
			Mat tmp_image, tmp_mask, tmp_descriptors;
			vector<KeyPoint> tmp_keypoints;
			tmp_image = imread(image_paths[i], CV_LOAD_IMAGE_GRAYSCALE);
			resize(tmp_image, tmp_image, image_size, 0, 0);
			makeMask(tmp_image, tmp_mask, threshold, threshold_type);
			//SURF keypoint detection and descriptor extraction
			detector->detect(tmp_image, tmp_keypoints, tmp_mask);
			if(tmp_keypoints.empty()){
				cerr << "Warning: Could not find key points in image " << image_paths[i] << endl;
			}
			else{
				extractor->compute(tmp_image, tmp_keypoints, tmp_descriptors);
				bowTrainer.add(tmp_descriptors);
			}
			if(i % 4 == 0){
				cout << ".";
			}
		}
		cout << endl;
		//Display descriptor informations
		cout << "\nDescriptor size: " << extractor->descriptorSize() << endl;
		cout << "Descriptor type: " << extractor->descriptorType() << endl;
		cout << "Total extracted " << extractor_type << " descriptors: " << bowTrainer.descripotorsCount() << endl;
		total_desc = bowTrainer.descripotorsCount();
		//Cluster the descriptors to make dictionary
		cout << "Clustering the descriptors to make Bag-of-word dictionary..." << endl;
		//Build dictionary
		Mat dictionary = bowTrainer.cluster();
		//Set dictionary
		bowDE.setVocabulary(dictionary);
		cout << "Dictionary size: " << bow_dictionary_size << endl;

		//Save the dictionary to file
		cout << "Saving dictionary to file..." << endl;
		//Define file storage for storing data
		FileStorage fs_bow_dictionary("data/dictionary.xml", FileStorage::WRITE);
		fs_bow_dictionary << "Dictionary" << dictionary;
		fs_bow_dictionary.release();
	}
	else{
		//If the dictionary file is found, open it to read to the dictionary matrix
		cout << "Reading BOW dictionary from file..." << endl;
		FileStorage fs_bow_dictionary("data/dictionary.xml", FileStorage::READ);
		Mat dictionary;
		fs_bow_dictionary["Dictionary"] >> dictionary;
		fs_bow_dictionary.release();
		bowDE.setVocabulary(dictionary);
	}

	//Extract BoW descriptor for each training image
	if(!trainedsvm.good() && !bow_dictionary.good()){
		cout << "Detecting and compute image BOW descriptors for train data..." << endl;
		for(int j = 0; j < image_paths.size(); j++){
			//Pre-processing
			Mat tmp_image, tmp_mask, tmp_descriptors;
			vector<KeyPoint> tmp_keypoints;
			tmp_image = imread(image_paths[j], CV_LOAD_IMAGE_GRAYSCALE);
			resize(tmp_image, tmp_image, image_size, 0, 0);
			makeMask(tmp_image, tmp_mask, threshold, threshold_type);
			//SURF keypoint detection and descriptor extraction
			detector->detect(tmp_image, tmp_keypoints, tmp_mask);
			if(tmp_keypoints.empty()){
				cerr << "Warning: Could not find key points in image " << image_paths[j] << endl;
			}
			else{
				Mat bowDescriptor;
				bowDE.compute(tmp_image, tmp_keypoints, bowDescriptor);
				trainingData.push_back(bowDescriptor);
				labelsData.push_back(image_labels[j]);
			}
			if(j % 4 == 0){
				cout << ".";
			}
		}
	}
	cout << endl;
	tm.stop();
	trainbow_time = tm.getTimeMilli();
	cout << trainbow_time << " ms!" << endl;

	tm.start();
	//Load trained SVM file if exist
	cout << "Loading trained SVM file if exists..." << endl;

	if (trainedsvm.good())
	{
		svm.load("data/trained_svm.xml");
		cout << "Loaded trained SVM model!" << endl;
	}
	else{
		//Train the SVM with defined parameters
		cout << "No trained SVM model file existed!" << endl;
		cout << "Training classifier..." << endl;
		svm.train_auto(trainingData, labelsData, Mat(), Mat(), svm.get_params());
		//Saving trained SVM to file
		cout << "Saving trained SVM to file..." << endl;
		svm.save("data/trained_svm.xml");
		cout << "Training completed!!!" << endl;
	}
	tm.stop();
	trainsvm_time = tm.getTimeMilli();
	cout << trainsvm_time << " ms!" << endl;
#pragma endregion Training

#pragma region Testing
	tm.start();
	//Processing testing data
	cout << "Load test data..." << endl;
	readCSV(csv_test_path, test_image_paths, test_image_labels);
	readPlantData(species_path, scientific_names, common_names);
	//Create SVM output prediction for later evaluation
	vector<float> rating;
	//Confusion matrix
	Mat confusion(no_of_classes, no_of_classes, CV_16SC1, cv::Scalar(0));
	cout << "Identification results" << endl;
	results.open("report/results.txt");
	cout << "Image -> Predicted | Actual | Scientific name | Common name";
	results << "Image -> Predicted | Actual | Scientific name | Common name";
	for(int k = 0; k < test_image_paths.size(); k++){
		//Pre-processing
		Mat tmp_image, tmp_mask, tmp_descriptors;
		vector<KeyPoint> tmp_keypoints;
		tmp_image = imread(test_image_paths[k], CV_LOAD_IMAGE_GRAYSCALE);
		resize(tmp_image, tmp_image, image_size, 0, 0);
		makeMask(tmp_image, tmp_mask, threshold, threshold_type);
		//SURF keypoint detection and descriptor extraction
		detector->detect(tmp_image, tmp_keypoints, tmp_mask);
		//Compute BOW descriptor for test data
		Mat bowDescriptor;
		bowDE.compute(tmp_image, tmp_keypoints, bowDescriptor);
		float predict_label = svm.predict(bowDescriptor);
		rating.push_back(predict_label - test_image_labels[k]);
		confusion.at<short>(test_image_labels[k]-1, predict_label-1) += 1;
		cout << "\n" << test_image_paths[k].substr(test_image_paths[k].find_last_of("/")+1) 
			<< " -> " << predict_label << " | " << test_image_labels[k] << " | " 
			<< scientific_names[predict_label - 1] << " | " << common_names[predict_label - 1];
		results << "\n" << test_image_paths[k].substr(test_image_paths[k].find_last_of("/")+1) 
			<< " -> " << predict_label << " | " << test_image_labels[k] << " | " 
			<< scientific_names[predict_label - 1] << " | " << common_names[predict_label - 1];
	}
	results.close();
	cout << endl;
	tm.stop();
	test_time = tm.getTimeMilli();
	cout << test_time << " ms!" << endl;
#pragma endregion Testing

#pragma region Evaluation
	tm.start();
	//Evaluate the classifier
	cout << endl;
	cout << "Evaluating classifier..." << endl;

	//Calculate the error rate
	double errorRate = countNonZero(rating);
	cout << "Number of prediction errors: " << errorRate << endl;

	//Write report
	report.open("report/report.txt");
	report << "Identification report";
	report << "\n====================================================================";
	if(!bow_dictionary.good() && !trainedsvm.good())
		report << "\nNumber of training images: " << image_paths.size();
	report << "\nNumber of testing images: " << test_image_paths.size();
	report << "\nStandard image size: " << image_size.width << "x" << image_size.height;
	report << "\nDetected feature type: " << detector_type;
	report << "\nExtracted " << extractor_type << " descriptor size: " << extractor->descriptorSize();
	report << "\nExtracted " << extractor_type << " descriptor type: " << extractor->descriptorType();
	if(!bow_dictionary.good() && !trainedsvm.good())
		report << "\nTotal " << extractor_type << " extracted descriptors: " << total_desc;
	report << "\nBag-of-word vocabulary (dictionary) size: " << bow_dictionary_size;
	report << "\nExtracted Bag-of-word descriptor size: " << bowDE.descriptorSize();
	report << "\nExtracted Bag-of-word descriptor type: " << bowDE.descriptorType();
	report << "\nNumber of support vectors: " << svm.get_support_vector_count() << endl;
	report << "\n====================================================================";
	report << "\n Identification results (confusion matrix)\n\n";
	for(int x = 0; x < confusion.rows; x++){
		for(int y = 0; y < confusion.cols; y++){
			report << confusion.at<short>(x, y) << " ";
		}
		report << endl;
	}
	report << "\n====================================================================";
	report << "\nNumber of prediction errors: " << errorRate;
	tm.stop();
	eval_time = tm.getTimeMilli();
	report << "\nTotal time taken: " << (init_time + trainbow_time + trainsvm_time + test_time + eval_time) << " ms!" << endl;
	cout << eval_time << " ms!" << endl;
	cout << "Total time taken: " << (init_time + trainbow_time + trainsvm_time + test_time + eval_time) << " ms!" << endl;
	report.close();
#pragma endregion Evaluation

	system("PAUSE");
}