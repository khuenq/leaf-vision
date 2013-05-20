//Included libraries
#include "functions.h"

//Function prototypes
void train(string csv_train_path = "data/csv/flavia_train.csv", int bow_dictionary_size = 900, 
	int minHessian = 400, int nOctaves = 4, int nOctaveLayers = 3,  
	bool extended = false, bool upright = false);
void test(string csv_test_path = "data/csv/flavia_test.csv", 
	int minHessian = 400, int nOctaves = 4, int nOctaveLayers = 3,  
	bool extended = false, bool upright = false);
void test_special(string csv_test_path = "data/csv/flavia_test.csv", 
	int minHessian = 400, int nOctaves = 4, int nOctaveLayers = 3,  
	bool extended = false, bool upright = false);

//Main program entry point
int main(int argc, const char* argv[]){
	std::ofstream time_report;
	time_report.open("data/results/time_report.txt");
	std::clock_t begin = std::clock();
	train();
	//test();
	test_special();
	std::clock_t end = std::clock();
	double elapsed_hours = double(double(end - begin) / CLOCKS_PER_SEC)/3600;
	cout << "Time taken: " << elapsed_hours << "h" << endl;
	time_report << "Time taken: " << elapsed_hours << "h" << endl;
	time_report.close();
	system("PAUSE");
	return 0;
}

void train(string csv_train_path, int bow_dictionary_size, 
	int minHessian, int nOctaves, int nOctaveLayers,  
	bool extended, bool upright){

	//Define data storage
	std::ifstream bow_dictionary("data/dictionary.xml");
	std::ifstream trainedsvm("data/trained_svm.xml");
	if(!bow_dictionary.good() || !trainedsvm.good()){

#pragma region Initialization
		cout << "Initializing..." << endl;
		//Vectors for training image paths
		std::vector<std::string> image_paths;
		std::vector<float> image_labels;
		//Define data storage
		std::ifstream bow_dictionary("data/dictionary.xml");
		std::ifstream trainedsvm("data/trained_svm.xml");
		//Specify feature detection machine
		cv::Ptr<cv::FeatureDetector> detector = new cv::SurfFeatureDetector(minHessian, nOctaves, nOctaveLayers, extended, upright);
		cv::Ptr<cv::DescriptorExtractor> extractor =  new cv::SurfDescriptorExtractor(minHessian, nOctaves, nOctaveLayers, extended, upright);
		cv::Ptr<cv::DescriptorMatcher> matcher = new cv::FlannBasedMatcher();
		//Training data matrices
		cv::Mat trainingData(0, bow_dictionary_size, CV_32FC1);
		cv::Mat labelsData(0, 1, CV_32FC1);
		//Define parameters for Bag-of-word model
		cv::TermCriteria tc(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10, 0.001);
		int retries = 1;
		int flags = cv::KMEANS_PP_CENTERS;
		//Define BOW Trainer and Descriptor Extractor
		cv::BOWKMeansTrainer bowTrainer(bow_dictionary_size, tc, retries, flags);
		cv::BOWImgDescriptorExtractor bowDE(extractor, matcher);
		
		// Set up SVM's parameters
		CvSVMParams params;
		params.svm_type    = CvSVM::C_SVC;
		params.kernel_type = CvSVM::RBF;
		params.C = 3.1250000000000000e+002;
		params.gamma = 5.0625000000000009e-001;
		params.term_crit   = cvTermCriteria(3, 1000, 1.1920928955078125e-007);

		//Create SVM instance
		cv::SVM svm;
#pragma endregion Initialization

#pragma region Training
		readCSV(csv_train_path, image_paths, image_labels);
		if(!bow_dictionary.good()){
			cout << "Detecting SURF features..." << endl;
			for(int i = 0; i < image_paths.size(); i++){
				//Preprocessing
				cv::Mat tmp_image, tmp_descriptors;
				cv::vector<cv::KeyPoint> tmp_keypoints;
				tmp_image = cv::imread(image_paths[i], CV_LOAD_IMAGE_GRAYSCALE);
				cv::resize(tmp_image, tmp_image, cv::Size(tmp_image.cols/2.5, tmp_image.rows/2.5), 0, 0);
				unsharpMask(tmp_image);

				//SURF keypoint detection and descriptor extraction
				detector->detect(tmp_image, tmp_keypoints);

				if(tmp_keypoints.empty()){
					cout << "Warning: Could not find key points in image " << image_paths[i] << "!" << endl;
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
			cout << "Descriptor size: " << extractor->descriptorSize() << endl;
			cout << "Descriptor type: " << extractor->descriptorType() << endl;
			cout << "Total extracted descriptors: " << bowTrainer.descripotorsCount() << endl;
			cout << "Clustering..." << endl;
			//Build dictionary
			cv::Mat dictionary = bowTrainer.cluster();
			//Set dictionary
			bowDE.setVocabulary(dictionary);

			//Define file storage for storing data
			cout << "Saving dictionary..." << endl;
			cv::FileStorage fs_bow_dictionary("data/dictionary.xml", cv::FileStorage::WRITE);
			fs_bow_dictionary << "Dictionary" << dictionary;
			fs_bow_dictionary.release();
		}
		else{
			cout << "Reading BOW dictionary from file..." << endl;
			cv::FileStorage fs_bow_dictionary("data/dictionary.xml", cv::FileStorage::READ);
			cv::Mat dictionary;
			fs_bow_dictionary["Dictionary"] >> dictionary;
			fs_bow_dictionary.release();
			bowDE.setVocabulary(dictionary);
		}
		//Get BOW descriptors
		cout << "Extracting BOW descriptors..." << endl;
		for(int j = 0; j < image_paths.size(); j++){
			//Pre-processing
			cv::Mat tmp_image;
			cv::vector<cv::KeyPoint> tmp_keypoints;
			tmp_image = cv::imread(image_paths[j], CV_LOAD_IMAGE_GRAYSCALE);
			cv::resize(tmp_image, tmp_image, cv::Size(tmp_image.cols/2.5, tmp_image.rows/2.5), 0, 0);
			unsharpMask(tmp_image);
			
			detector->detect(tmp_image, tmp_keypoints);
			if(tmp_keypoints.empty()){
				cout << "Warning: Could not find key points in image " << image_paths[j] << "!" << endl;
			}
			else{
				cv::Mat bowDescriptor;
				bowDE.compute(tmp_image, tmp_keypoints, bowDescriptor);
				trainingData.push_back(bowDescriptor);
				labelsData.push_back(image_labels[j]);
			}

			if(j % 4 == 0){
				cout << ".";
			}
		}
		cout << endl;
		//Train the SVM with defined parameters
		cout << "Training SVM" << endl;
		svm.train(trainingData, labelsData, cv::Mat(), cv::Mat(), params);
		//Saving trained SVM to file
		cout << "Saving SVM trained file" << endl;
		svm.save("data/trained_svm.xml");
		cout << "Training completed!" << endl;
		trainingData.release();
		labelsData.release();
#pragma endregion Training
	}
	else
	{
		cout << "The system has already been trained!" << endl;
	}
	bow_dictionary.close();
	trainedsvm.close();
}

void test(string csv_test_path, int minHessian, int nOctaves, int nOctaveLayers,  
	bool extended, bool upright){
	std::ifstream bow_dictionary("data/dictionary.xml");
	std::ifstream trainedsvm("data/trained_svm.xml");
	if(!bow_dictionary.good() || !trainedsvm.good()){
		cout << "You must train the system before classifying leaf!" << endl;
	}
	else
	{
		cout << "Classifying images process started!" << endl;
		cout << "Initialization..." << endl;
		//Initialization
		//Specify feature detection machine
		cv::Ptr<cv::FeatureDetector> detector = new cv::SurfFeatureDetector(minHessian, nOctaves, nOctaveLayers, extended, upright);
		cv::Ptr<cv::DescriptorExtractor> extractor =  new cv::SurfDescriptorExtractor(minHessian, nOctaves, nOctaveLayers, extended, upright);
		cv::Ptr<cv::DescriptorMatcher> matcher = new cv::FlannBasedMatcher();
		cv::Ptr<cv::BOWImgDescriptorExtractor> bowDE = new cv::BOWImgDescriptorExtractor(extractor, matcher);
		//Create SVM instance
		cv::SVM svm;
		//Prediction results
		FileStorage fs_prediction_results("data/results/prediction_results.xml", FileStorage::WRITE);
		//Matrix for later evaluations
		cv::Mat predictionResults(0, 1, CV_32FC1);

		cout << "Loading trained data..." << endl;
		//Preparing data
		cv::FileStorage fs_bow_dictionary("data/dictionary.xml", cv::FileStorage::READ);
		cv::Mat dictionary;
		fs_bow_dictionary["Dictionary"] >> dictionary;

		bowDE->setVocabulary(dictionary);
		svm.load("data/trained_svm.xml");	
		
		std::vector<std::string> image_paths;
		cout << "Reading image paths!" << endl;
		readCSV2(csv_test_path, image_paths);

		cout << "Predicting image labels!" << endl;
		fs_prediction_results << "surf_descriptor_size" << bowDE->getVocabulary().cols;
		fs_prediction_results << "bow_dictionary_size" << bowDE->getVocabulary().rows;
		fs_prediction_results << "bow_descriptor_size" << bowDE->descriptorSize();
		fs_prediction_results << "bow_descriptor_type" << bowDE->descriptorType();
		fs_prediction_results << "svm_C" << svm.get_params().C;
		fs_prediction_results << "svm_type" << svm.get_params().svm_type;
		fs_prediction_results << "svm_kernel_type" << svm.get_params().kernel_type;
		fs_prediction_results << "svm_termcrit_type" << svm.get_params().term_crit.type;
		fs_prediction_results << "svm_termcrit_epsilon" << svm.get_params().term_crit.epsilon;
		fs_prediction_results << "svm_termcrit_max_iteration" << svm.get_params().term_crit.max_iter;
		fs_prediction_results << "svm_total_support_vector" << svm.get_support_vector_count();

		fs_prediction_results << "prediction_results" << "[";
		for(int x = 0; x < image_paths.size(); x++){
			//Temp image
			cv::Mat tmp_image;
			std::vector<cv::KeyPoint> tmp_image_keypoints;
			tmp_image = cv::imread(image_paths[x], CV_LOAD_IMAGE_GRAYSCALE);

			//Processing image
			cv::resize(tmp_image, tmp_image, cv::Size(tmp_image.cols/2.5, tmp_image.rows/2.5), 0, 0);
			unsharpMask(tmp_image);
			
			detector->detect(tmp_image, tmp_image_keypoints);

			//Compute BOW descriptor for test data
			cv::Ptr<cv::Mat> bowDescriptor = new cv::Mat();
			bowDE->compute(tmp_image, tmp_image_keypoints, *bowDescriptor);

			//Predict label
			float predict_label = svm.predict(*bowDescriptor);
			std::string filename = image_paths[x].substr(image_paths[x].find_last_of('/', std::string::npos) + 1);
			fs_prediction_results << filename << (int) predict_label;

			tmp_image.release();
			if(x % 4 == 0)
				cout << ".";
		}
		fs_prediction_results << "]";
		fs_prediction_results.release();
		fs_bow_dictionary.release();
		
		cout << endl;
		cout << "Completed!!!" << endl;
	}
	bow_dictionary.close();
	trainedsvm.close();
}

void test_special(string csv_test_path, int minHessian, int nOctaves, int nOctaveLayers,  
	bool extended, bool upright){
	std::ifstream bow_dictionary("data/dictionary.xml");
	std::ifstream trainedsvm("data/trained_svm.xml");
	if(!bow_dictionary.good() || !trainedsvm.good()){
		cout << "You must train the system before classifying leaf!" << endl;
	}
	else
	{
		cout << "Classifying images process started!" << endl;
		cout << "Initialization..." << endl;
		//Initialization
		//Specify feature detection machine
		cv::Ptr<cv::SurfFeatureDetector> detector = new cv::SurfFeatureDetector(minHessian, nOctaves, nOctaveLayers, extended, upright);
		cv::Ptr<cv::SurfDescriptorExtractor> extractor =  new cv::SurfDescriptorExtractor(minHessian, nOctaves, nOctaveLayers, extended, upright);
		cv::Ptr<cv::FlannBasedMatcher> matcher = new cv::FlannBasedMatcher();
		cv::Ptr<cv::BOWImgDescriptorExtractor> bowDE = new cv::BOWImgDescriptorExtractor(extractor, matcher);
		//Create SVM instance
		cv::SVM svm;
		//Prediction results
		cv::FileStorage fs_prediction_results("data/results/prediction_results.xml", FileStorage::WRITE);
		//Ground truth and result vector
		std::vector<float> image_labels;
		std::vector<float> results;

		cout << "Loading trained data..." << endl;
		//Preparing data
		cv::FileStorage fs_bow_dictionary("data/dictionary.xml", cv::FileStorage::READ);
		cv::Mat dictionary;
		fs_bow_dictionary["Dictionary"] >> dictionary;

		bowDE->setVocabulary(dictionary);
		svm.load("data/trained_svm.xml");	

		std::vector<std::string> image_paths;
		cout << "Reading image paths!" << endl;
		readCSV(csv_test_path, image_paths, image_labels);

		cout << "Predicting image labels!" << endl;

		fs_prediction_results << "surf_parameters" << "{";
			fs_prediction_results << "surf_min_hessian" << detector->hessianThreshold;
			fs_prediction_results << "surf_noctaves" << detector->nOctaves;
			fs_prediction_results << "surf_noctave_layers" << detector->nOctaveLayers;
			fs_prediction_results << "surf_extended" << detector->extended;
			fs_prediction_results << "surf_upright" << detector->upright;
		fs_prediction_results << "}";
		
		fs_prediction_results << "bag_of_words_parameters" << "{";
			fs_prediction_results << "bow_dictionary_size" << bowDE->getVocabulary().rows;
			fs_prediction_results << "bow_descriptor_size" << bowDE->descriptorSize();
			fs_prediction_results << "bow_descriptor_type" << bowDE->descriptorType();
		fs_prediction_results << "}";

		fs_prediction_results << "svm_parameters" << "{";
			fs_prediction_results << "svm_C" << svm.get_params().C;
			fs_prediction_results << "svm_type" << svm.get_params().svm_type;
			fs_prediction_results << "svm_kernel_type" << svm.get_params().kernel_type;
			fs_prediction_results << "svm_class_weights" << svm.get_params().class_weights;
			fs_prediction_results << "svm_coef_zero" << svm.get_params().coef0;
			fs_prediction_results << "svm_degree" << svm.get_params().degree;
			fs_prediction_results << "svm_gamma" << svm.get_params().gamma;
			fs_prediction_results << "svm_nu" << svm.get_params().nu;
			fs_prediction_results << "svm_p" << svm.get_params().p;
			fs_prediction_results << "svm_var_count" << svm.get_var_count();
			fs_prediction_results << "svm_termcrit_type" << svm.get_params().term_crit.type;
			fs_prediction_results << "svm_termcrit_epsilon" << svm.get_params().term_crit.epsilon;
			fs_prediction_results << "svm_termcrit_max_iteration" << svm.get_params().term_crit.max_iter;
			fs_prediction_results << "svm_total_support_vector" << svm.get_support_vector_count();
		fs_prediction_results << "}";

		fs_prediction_results << "prediction_results" << "[";
		for(int x = 0; x < image_paths.size(); x++){
			//Temp image
			cv::Mat tmp_image;
			std::vector<cv::KeyPoint> tmp_image_keypoints;
			tmp_image = cv::imread(image_paths[x], CV_LOAD_IMAGE_GRAYSCALE);

			//Processing image
			cv::resize(tmp_image, tmp_image, cv::Size(tmp_image.cols/2.5, tmp_image.rows/2.5), 0, 0);
			unsharpMask(tmp_image);
			
			detector->detect(tmp_image, tmp_image_keypoints);

			//Compute BOW descriptor for test data
			cv::Ptr<cv::Mat> bowDescriptor = new cv::Mat();
			bowDE->compute(tmp_image, tmp_image_keypoints, *bowDescriptor);

			//Predict label
			float predict_label = svm.predict(*bowDescriptor);
			results.push_back(image_labels[x] - predict_label);
			std::string filename = image_paths[x].substr(image_paths[x].find_last_of('/', std::string::npos) + 1);
			fs_prediction_results << filename << (int) predict_label;

			tmp_image.release();
			if(x % 4 == 0)
				cout << ".";
		}
		fs_prediction_results << "]";
		fs_prediction_results << "precision_rate" << ((int) (image_paths.size() - cv::countNonZero(results)));
		fs_prediction_results.release();
		fs_bow_dictionary.release();

		cout << endl;
		cout << "Completed!" << endl;
		cout << "Error rate: " << cv::countNonZero(results) << endl;
	}
	bow_dictionary.close();
	trainedsvm.close();
}