#  Camera Based 2D Feature Tracking

---


## [Rubric](https://review.udacity.com/#!/rubrics/2549/view) Points


[//]: # (Image References)

[image1]: ./writeup_images/keypoints.png "Keypoints"
[image2]: ./writeup_images/matches.png "Matches"
[image3]: ./writeup_images/times.png "Times"


---

### MP.1 Data Buffer Optimization


Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end. 

The initial implementation already included the functionaliy to add images by pushing them at the end of the vector:


```c++
	// push image into data frame buffer
	DataFrame frame;
	frame.cameraImg = imgGray;
	dataBuffer.push_back(frame);
```

The ring buffer was implemented simply by erasing the first element of the vector once the specified buffer size was reached:

```c++
	if (dataBuffer.size() > dataBufferSize)
	{
	    dataBuffer.erase(dataBuffer.begin());
	}
```


#### MP.2 Keypoint Detection

Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.

The Harris detector code was based on the solution provided in the classroom. The rest of the detectors were implemented as follows:

```c++
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;   // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
```

#### MP.3 Keypoint Removal

Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing. 

The "rectangle" class was used to achieve this purpose, however other methods were also investigated:

```c++
        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);

        if (bFocusOnVehicle)
        {
            vector<cv::KeyPoint> tempKeypoints;    
            cv::rectangle(imgGray,vehicleRect, cv::Scalar(0, 255, 0), 2, 8, 0);

            for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
            {   //alternative for rectangle.contains()
                // if(vehicleRect.x < it->pt.x && it->pt.x < vehicleRect.x + vehicleRect.width && 
                //    vehicleRect.y < it->pt.y && it->pt.y < vehicleRect.y + vehicleRect.height)
                if (vehicleRect.contains(it->pt))
                {
                    tempKeypoints.push_back(*it);
                }
            }
            keypoints = tempKeypoints;
            cout << "Keypoint in preceding vehicle n=" << keypoints.size() << endl;
        }
```

#### MP.4 Keypoint Descriptors

Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.

The following code was added to the "descKeypoints" function:

```c++
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create   (threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }
```

#### MP.5 Descriptor Matching

Implement FLANN matching as well as k-nearest neighbour selection. Both methods must be selectable using the respective strings in the main function. 

The following code presents the implementation of FLANN and K-nearest neighbour. The brute force matches was adapted to work with the SIFT descriptor:

```c++
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        if(descriptorType == "SIFT")
        {
            // Hamming distance cannot be used with SIFT
            // This discussion as reference: https://answers.opencv.org/question/10046/feature-2d-feature-matching-fails-with-assert-statcpp/
            normType = cv::NORM_L2;
        }
        else
        {
            normType = cv::NORM_HAMMING;
        }
        
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // make sure that both descriptors matrices are float, otherwise it throws an error
        if (descSource.type() != CV_32F || descRef.type() != CV_32F )
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);        
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches

        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }
```

#### MP.6 Descriptor Distance Ratio

Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

The following code presents the implementation of k nearest neighbour distance ratio:

```c++

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches

        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }
```

#### MP.7 Performance Evaluation 1

Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented. 

The following image presents the keypoint count per image and the corresponding keypoints, as well as examples of the results.

![alt text][image1]


#### MP.8 Performance Evaluation 2

Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

The following table summarizes the average number of keypoints for the 10 images with the multiple combinations of detectors and descriptors. 

![alt text][image2]

Note: AKAZE descriptors can only be used with AKAZE keypoints. The combination of SIFT detector with ORB descriptors caused an "out of memory" error that needs further investigation.

#### MP.9 Performance Evaluation 3

Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.

The results are summarized in the following table:

![alt text][image3]

Note:The ORB detector took an unusually long time for the first detection, the detection of keypoints in the rest of the images was relatively fast. 

The combination FAST detector with ORB descriptors gave the fastest execution time with a balanced number of matched keypoints. This is the first choice if the execution time is fundamental for the application. 

Similarly, the combinations FAST detector with BRIEF and BRISK descriptors, gave the second and third fastest results, respectively. 

For the sace of discussion, if the execution time requirements are not that strict, the BRISK detector and the BRIEF descriptors combination yielded the highest numbers of matched keypoints (unlike the HARRIS detector in which the overlap % introduced some duplicates). The execution time was 10 times higher than the first choice, however the numbers suggest a higher accuracy (to be proven with appropriate metrics). This option would be the best in applications with higher accuracy requirements.


The data for all these evaluation steps is available in this [spreadsheet] (https://docs.google.com/spreadsheets/d/1fUYwmnbgy_bgM5UjmhFAjLRoE0F7IIE_x5lqQTK5QV0/edit?usp=sharing) 

