#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;

CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;

IplImage* cropImg;
CvRect* RectFace;

bool faceDetection( IplImage *img );

#define FACE_CROP_WIDTH 110
#define FACE_CROP_HEIGHT 110
#define THRES_FACE 1.2

static void readDataFile(const string& filename, vector<Mat>& faces, vector<int>& IDs, char separator = ';') {
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
            faces.push_back(imread(path, 0));
            IDs.push_back(atoi(classlabel.c_str()));
        }
    }
}

void main() {

	CvCapture* capture;
	IplImage* frame;
	cropImg = cvCreateImage(cvSize(FACE_CROP_WIDTH, FACE_CROP_HEIGHT), 8, 3);

	capture = cvCreateCameraCapture(0);
	char *filename = "C:/OpenCV2.4.3/data/haarcascades/haarcascade_frontalface_alt.xml";
	cascade = (CvHaarClassifierCascade*)cvLoad(filename, 0, 0, 0);
	storage = cvCreateMemStorage(0);

	//load training file
	string trainFile  = "E:/Projects/Face recognition/TUT/FaceReg/FaceReg/train.txt";

	// storing faces and names respectively
    vector<Mat> faces;
    vector<int> IDs;
	readDataFile(trainFile, faces, IDs);
	
    //Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->set("threshold", 5000.0);

	model->train(faces, IDs);
//	model->save("facedata.xml");
//	model->load("facedata.xml");

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 10);

	cvNamedWindow("Camera", 1);
	cvNamedWindow("Face", 1);

	while(1)
	{
		//acquire frame
		frame = cvQueryFrame(capture);
		if(faceDetection(frame))
		{
		
		Mat grayFace(cropImg);
		cvtColor(grayFace, grayFace, CV_BGR2GRAY);	

		int predicted_label = -1;
		double predicted_confidence = 0.0;

		model->predict(grayFace, predicted_label, predicted_confidence);  // int prediction = 
//		model->get("threshold");
		
		stringstream name;
	

		stringstream TextShow; 
		TextShow << "Prediction = " << predicted_label << "conf " << predicted_confidence;

		cvPutText(frame, TextShow.str().c_str(), cvPoint(RectFace->x, RectFace->y), &font, cvScalar(0, 0, 255));

		imshow("ssas", grayFace);

		}
		cvShowImage("Camera", frame);
		cvShowImage("Face", cropImg);
		char c = cvWaitKey(3);
		if(c==27) break;
	}

	cvReleaseCapture(&capture);
	cvDestroyWindow("Camera");
	cvDestroyWindow("Face");
	cvReleaseHaarClassifierCascade( &cascade );
	cvReleaseMemStorage( &storage );
}

bool faceDetection(IplImage *pImg)
{
	bool faceDetected = false;
	CvRect* region = 0;
	int minSize = pImg->width/5;
	
	cvClearMemStorage(storage);
	CvSeq *faces = cvHaarDetectObjects(pImg, cascade, storage, 1.1, 3, 
					CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH, 
					cvSize( minSize, minSize)); //one biggest face
	
	//faceDetected flag is on when detected
	if(faces->total == 1) {
		faceDetected = true;
		region = ( CvRect*)cvGetSeqElem(faces, 0);
		RectFace = region;

	//Crop face from photo stream
		cvSetImageROI(pImg, cvRect(region->x, region->y, region->width, region->height));
		IplImage* 	cropImg_ = cvCreateImage(cvSize(region->width, region->height), 8, 3);
		cvCopy(pImg, cropImg_);
		cvResize(cropImg_, cropImg, CV_INTER_CUBIC);
		cvResetImageROI(pImg);

	//Draw rectangle on the face
		cvRectangle( pImg, cvPoint( region->x, region->y ), 
			cvPoint( region->x + region->width, region->y + region->height ), 
			CV_RGB( 255, 0, 0 ), 2, 8, 0 );
		cvReleaseImage(&cropImg_);
	}
	
	return faceDetected;
}
