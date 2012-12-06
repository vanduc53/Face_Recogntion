#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;
CvRect *faceDetection(IplImage *pImg);

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void main() {

	CvCapture* capture;
	IplImage* frame;
	capture = cvCreateCameraCapture(0);
	char *filename = "C:/OpenCV2.4.2/data/haarcascades/haarcascade_frontalface_alt.xml";
	cascade = (CvHaarClassifierCascade*)cvLoad(filename, 0, 0, 0);
	storage = cvCreateMemStorage(0);

    // Get the path to your CSV:
    string fn_haar = "C:/OpenCV2.4.3/data/haarcascades/haarcascade_frontalface_default.xml";
    string fn_csv  = "E:/Projects/Face recognition/TUT/FaceReg/FaceReg/train.txt";
    int deviceId = -1;
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
        read_csv(fn_csv, images, labels);
    
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(deviceId);

 

    for(;;) {

        frame = cvQueryFrame(capture);
		
        // Clone the current frame:
		Mat original(frame);//frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
 //       vector< Rect_<int> > faces;
 //       haar_cascade.detectMultiScale(gray, faces, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH, Size(100, 100), Size(480,480));
      
		CvSeq *faces = cvHaarDetectObjects(frame, cascade, storage, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH, cvSize(100, 100));
		if(faces->total == 1) 
		{
			CvRect* rectFace = (CvRect*) cvGetSeqElem(faces, 0);
			cvRectangle( frame, cvPoint( rectFace->x, rectFace->y ), cvPoint( rectFace->x + rectFace->width, rectFace->y + rectFace->height ), CV_RGB( 0, 255, 0 ), 2, 8, 0 );
		
		}

		// At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
/*        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            //
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
            //
            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
            // face you have just found:
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
//            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
           
			// Create the text we will annotate the box with:
            std::stringstream box_text ;
			box_text << "Prediction = ";// << prediction;
           
			
			
			// Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(original, box_text.str().c_str(), Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
       
		
		}
  */      // Show the result:
        cvShowImage("face_recognizer", frame);
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)      break;
    }
   
}

CvRect *faceDetection(IplImage *pImg)
{
	CvRect* r = 0;
	int minSize = pImg->width/5;
	CvSeq *faces = cvHaarDetectObjects(pImg, cascade, storage, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH, cvSize( minSize, minSize)); //one biggest face
	
	// Draw green rectangle on each face
	for( int i = 0 ; i < ( faces ? faces->total : 0 ) ; i++ ) {
		r = ( CvRect*)cvGetSeqElem(faces, i);
		cvRectangle( pImg, cvPoint( r->x, r->y ), cvPoint( r->x + r->width, r->y + r->height ), CV_RGB( 0, 255, 0 ), 2, 8, 0 );
	}
	return r;
}
