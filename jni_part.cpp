// Madjid MAIDI
// First Release : 01/01/2014
// Last Update : 01/09/2017


#include <myHeader.h>

extern "C" {
    Mat objectRecognition(const Mat& inputFrame);
    vector<string> collectClassNames(char filename[]);
    JNIEXPORT int JNICALL Java_fr_esme_myapplication_CameraMainActivity_LoadData(
            JNIEnv *env, jobject assetManager, jstring MarkerDirPath, jstring markerfolder )
    {
        const char *jMarkerDirPath = env->GetStringUTFChars( MarkerDirPath, 0 );

        const char *jmarkerfolder = env->GetStringUTFChars( markerfolder, 0 );

        sprintf( marker_fpath_android, "%s", (char*)jMarkerDirPath );// added 23/04/2015

        sprintf( marker_folder, "%s", (char*)jmarkerfolder );// added 03/11/2015

        LOGI("%s is the android filepath", marker_fpath_android);

        // start programming here...
        sprintf(yoloConfigFilename,"%s/yolo.cfg",marker_fpath_android);
        sprintf(yoloWeightsFilename,"%s/yolo.weights",marker_fpath_android);
        sprintf(yoloNamesFilename,"%s/coco.names",marker_fpath_android);
        sprintf(haarcascadeSpeedSignFilename,"%s/haarcascade_traffic_sign.xml",marker_fpath_android);

        trafficSignClassifier = CascadeClassifier(haarcascadeSpeedSignFilename);
        yoloNet = readNetFromDarknet(yoloConfigFilename, yoloWeightsFilename);
        yoloClassNames = collectClassNames(yoloNamesFilename);

        /*LOGI(yoloConfigFilename);
        LOGI(yoloWeightsFilename);
        LOGI(yoloNamesFilename);
        LOGI(haarcascadeSpeedSignFilename);*/

        return 0;
    }

    JNIEXPORT int JNICALL Java_fr_esme_myapplication_CameraMainActivity_imageProcessing(
            JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
    {
        Mat& mGr  = *(Mat*)addrGray;
        Mat& mRgb = *(Mat*)addrRgba;

        // M2: TO DO first
        // workingBGRframe is the working BGR image for usual OCV operations (8UC3)
        Mat workingBGRframe(mRgb.rows, mRgb.cols, CV_8UC3);

        cvtColor(mRgb, workingBGRframe, CV_RGBA2BGR, 3);

        frame = workingBGRframe;

        if ( frame.empty() ) return -1;

        //*************************************************************************//
                         // MANDATORY DO NOT CHANGE LINES ABOVE !!!
        //*************************************************************************//

        // Your image is "frame", process it and add on it all desired graphical results (FPS, texts, ...)

        // start programming here...

        double tm = (double)cvGetTickCount();
        frame = objectRecognition(frame);
        tm = (double)cvGetTickCount() - tm;

        // "frame" contains the input image with overlaid results
        // This is why we get back "frame" here !

        workingBGRframe = frame;

        //*************************************************************************//
                         // MANDATORY DO NOT CHANGE LINES BELOW !!!
        //*************************************************************************//
        // the program ends here.
        // M2: copy the workingBGRframe to android image
        // workingBGRframe MUST BE converted into the same SPACE COLOR as the android frame
        cvtColor(workingBGRframe, mRgb, CV_BGR2RGBA, 4);

        return 0;

    } // end JNI function


        /**
        * @brief Performs Yolo NetWork image preprocessing, feed this image to the network and return the output
        *
        * @param inputFrame is the image on which the preprocessing and yolo recognition is going to be performed
        * @return the output of the Yolo Network containing all the object detected, location and probability as a Mat
        */
        Mat yoloImagePreprocessingAndFeedForward(const Mat& inputFrame){
            Mat preprocessedImage = blobFromImage(inputFrame, 1 / 255.F, Size(608, 608), Scalar(117, 117, 117), false, false);
            yoloNet.setInput(preprocessedImage, "data");
            Mat outputProb = yoloNet.forward("detection_out");
            return outputProb;
        }
        /**
        * @brief Draw all the yolo object (according to a minimum threshold confidence)
        *
        * @param yoloNetOutput is the output of the yolo Network
        * @param outputFrame is the image on which all the object will be drawn
        * @return void
        */
        void drawYoloRecognition(Mat& yoloNetOutput, Mat& outputFrame){
            map<string, int> objectsDetected;
            int probabilityStartAtIndex = 5;
            int probabilitySize = yoloNetOutput.cols - probabilityStartAtIndex;
            for (int i = 0; i < yoloNetOutput.rows; i++) {
                Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
                float *linePointer = (float*)yoloNetOutput.ptr(i);
                float *probabilityArrayPointer = &linePointer[probabilityStartAtIndex];
                size_t indexMaxClass = max_element(probabilityArrayPointer, probabilityArrayPointer + probabilitySize) - probabilityArrayPointer;
                float confidenceMaxClass = linePointer[(int)indexMaxClass + probabilityStartAtIndex];

                if (confidenceMaxClass > 0.40) {
                    float xRectangleCenter = linePointer[0] * outputFrame.cols;
                    float yRectangleCenter = linePointer[1] * outputFrame.rows;
                    float rectangleWidth = linePointer[2] * outputFrame.cols;
                    float rectangleHeight = linePointer[3] * outputFrame.rows;

                    Point topLeftVertex(cvRound(xRectangleCenter - rectangleWidth / 2), cvRound(yRectangleCenter - rectangleHeight / 2));
                    Point bottomRightVertex(cvRound(xRectangleCenter + rectangleWidth / 2), cvRound(yRectangleCenter + rectangleHeight / 2));
                    Rect objectRectangle(topLeftVertex, bottomRightVertex);

                    string yoloClassName = indexMaxClass < yoloClassNames.size() ? yoloClassNames[indexMaxClass] : "";
                    objectsDetected[yoloClassName] += 1;
                    String classNameAndProbabilityString = format("%s: %.2f", yoloClassName.c_str(), confidenceMaxClass);
                    rectangle(outputFrame, objectRectangle, color, 5);
                    int xTopLeft = (int)((xRectangleCenter - (rectangleWidth / 2)));
                    int yTopLeft = (int)((yRectangleCenter - (rectangleHeight / 2)));
                    int baseLine = 0;
                    Size labelSize = getTextSize(classNameAndProbabilityString, cv::FONT_HERSHEY_SIMPLEX, 2.0, 1, &baseLine);
                    rectangle(outputFrame, Rect(Point(xTopLeft, yTopLeft),
                        Size(labelSize.width, labelSize.height + baseLine)),
                        color, CV_FILLED);
                    putText(outputFrame, classNameAndProbabilityString, Point(xTopLeft, yTopLeft + labelSize.height),
                        cv::FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 0, 0), 2);
                }
            }
            /*map<string, int>::iterator it;
            for (it = objectsDetected.begin(); it != objectsDetected.end(); ++it) {
                text.push_back(it->first + " : " + to_string(it->second));
            }*/
        }
        /**
        * @brief Performs a danger signs contours detection
        *
        * @param inputFrame  is the image on which the contours detection will be performed
        * @return an array of all contours as an std::vector<std::vector<cv::Point>>
        */
         vector<vector<Point> > detectDangerSigns(const Mat& inputFrame){
            vector<vector<Point> > contours;
            Mat grayScale;
            Mat cannyOutput;

            vector<Vec4i> hierarchy;

            cvtColor(inputFrame, grayScale, CV_BGR2GRAY);
            GaussianBlur(grayScale, grayScale, Size(3, 3), 1);
            Canny(grayScale, cannyOutput, 100, 200);
            findContours(cannyOutput, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, Point(0, 0));
            return contours;
        }
        /**
        * @brief Used to fill all the danger signs in green(the area must be above a certain threshold)
        *
        * @param inputFrame is the image on which the danger signs will be filled
        * @param contours is the array containing all the contours
        * @return void
        */
        void drawDangerSigns(cv::Mat inputFrame, const std::vector<std::vector<cv::Point> > contours){
            std::vector<cv::Point> approxTriangle;
            int minimumAreaThreshold = 3000;
            for (int i = 0; i < contours.size(); i++) {
                cv::approxPolyDP(contours[i], approxTriangle, cv::arcLength(cv::Mat(contours[i]), true)*0.05, true);
                double area = cv::contourArea(approxTriangle);

                if (approxTriangle.size() == 3){
                    if (area > minimumAreaThreshold){
                        cv::drawContours(inputFrame, contours, i, cv::Scalar(0, 0, 255), CV_FILLED);
                        if (find(text.begin(), text.end(), "Danger") != text.end());
                        else text.push_back("Danger");
                    }
                }
            }
        }
        /**
        * @brief Performs a speed signs haarcascade detection
        *
        * @param inputFrame is the image on which the haarcascade detection will be performed
        * @return an array of rectangle corresponding to the location of all the speed signs detected as an std::vector<cv::Rect>
        */
        vector<Rect> detectSpeedSigns(const Mat& inputFrame){
            vector<Rect> signsDetected;
            trafficSignClassifier.detectMultiScale(inputFrame, signsDetected, 1.1, 5, 0, Size(40, 40));
            return signsDetected;
        }
        /**
        * @brief Used to draw all the rectangles detected by the detecteSpeedSigns method
        *
        * @param inputFrame is the image on which the rectangle will be drawn
        * @param signs is the array containing all the rectangle location
        * @return void
        */
        void drawSpeedSigns(Mat& inputFrame, vector<Rect> signs){
            vector<Rect>::iterator signs_it;
            String signName = "Panneau de vitesse";
            for (signs_it = signs.begin(); signs_it != signs.end(); ++signs_it) {
                int x = signs_it->x;
                int y = signs_it->y;
                int width = signs_it->width;
                int height = signs_it->height;

                Point topLeftVertex(x, y);
                Point bottomRightVertex(x + width, y + height);
                Rect objectRectangle(topLeftVertex, bottomRightVertex);
                rectangle(inputFrame, topLeftVertex, bottomRightVertex, Scalar(255, 0, 0), 3);
                putText(inputFrame, signName, topLeftVertex + cv::Point(0, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));
            }
        }
        /**
        * @brief Write text to an image starting from the bottom left
        *
        * @param inputFrame is the image on which the text is going to be displayed
        * @param textArray is an array of String
        * @param color is the text color
        * @return void
        */
        void writeTextToImage(Mat& inputFrame, vector<String> textArray, Scalar color){
            int baseLine = 0;
            for (int i = 0; i < textArray.size(); i++) {
                Size labelSize = getTextSize(textArray.at(i), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(inputFrame, Rect(Point(20, inputFrame.rows - (i * 20) - 32),
                    Size(labelSize.width, labelSize.height + baseLine)),
                    Scalar(0,0,0), CV_FILLED);
                putText(inputFrame, textArray.at(i), Point(20, inputFrame.rows - (i*20) - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
            }
        }
        /**
        * @brief Collect Yolo Network classnames from a text file
        *
        * @param filename is the path
        * @param eraseCharacters is the number of characters that will be erased from the beginning of each line (it depends on the text file structure)
        * @return an array containing all the class names as a std::vector<std::string>
        */
        vector<string> collectClassNames(char filename[]){
            vector<string> classNames;
            ifstream classNameFileStream;
            classNameFileStream.open(filename);
            //if (classNameFileStream.is_open()){
            while (classNameFileStream.eof() == false){
                string line;
                getline(classNameFileStream, line);
                string className = "";
                className = line;
                classNames.push_back(className);
            }
            //}
            classNameFileStream.close();
            return classNames;
        }
            /**
            * @brief Performs several detection and recognition : Yolo object recognition
            *											          Danger signs detection
            *													  Speed signs detection
            *
            * @param inputFrame is the image on which all the detections and recognition will be performed
            * @return an image containing all the detections and recognition as a Mat
            */
             Mat objectRecognition(const Mat& inputFrame){
                    text.clear();

                    Mat outputFrame;
                    inputFrame.copyTo(outputFrame);

                    Mat yoloNetOutput = yoloImagePreprocessingAndFeedForward(outputFrame);
                    drawYoloRecognition(yoloNetOutput, outputFrame);

                    vector<vector<Point> > contours = detectDangerSigns(inputFrame);
                    drawDangerSigns(outputFrame, contours);

                    vector<Rect> signs = detectSpeedSigns(inputFrame);
                    drawSpeedSigns(outputFrame, signs);

                    writeTextToImage(outputFrame, text, Scalar(255,255,255));
                    return outputFrame;
                }
}
