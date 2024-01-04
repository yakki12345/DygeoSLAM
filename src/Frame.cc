/**
* This file is a modified version of ORB-SLAM2.<https://github.com/raulmur/ORB_SLAM2>
*
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

// The previous image
cv::Mat imGrayPre; // 上一帧的灰度图
std::vector<cv::Point2f> prepoint, nextpoint; // prepoint:cornerSubPix亚像素计算后得到的角点 nextpoint:对上一帧图像进行光流金字塔得到的本帧图像的角点
std::vector<cv::Point2f> F_prepoint, F_nextpoint;// F_nextpoint:先是在step3经过图像块像素差筛选过后的精度较高的角点,最后是在step5经过图像块像素差筛选以及经过重投影与极线的距离筛选后的精确相对最高的角点
std::vector<cv::Point2f> F2_prepoint, F2_nextpoint;// 经过图像块像素差筛选以及经过重投影与极线的距离筛选后的精确相对最高的角点
using namespace cv;  
using namespace std;
using namespace cv::xfeatures2d;
std::vector<uchar> state;// 记录光流点是否跟踪成功的集合，成功status =1,否则为0
std::vector<float> err;// 输出错误的矢量.只为了用在光流金字塔中.
std::vector<std::vector<cv::KeyPoint>> mvKeysPre;//没有使用过
namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels), mImRGB(frame.mImRGB),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor), mImGray(frame.mImGray),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),mImMask(frame.mImMask),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),mIsKeyFrame(frame.mIsKeyFrame),mImDepth(frame.mImDepth)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(bool idPlus)
{
    if(idPlus)
        mnId=nNextId++;
}
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const cv::Mat &maskLeft, const cv::Mat &maskRight,const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();
    //2023.12.28write
    cv::Mat  imGrayT = imLeft;
    std::chrono::steady_clock::time_point tm1 = std::chrono::steady_clock::now();
    ProcessMovingObject(imLeft);//移除动态外点的函数，极限约束法
    std::chrono::steady_clock::time_point tm2 = std::chrono::steady_clock::now();
    movingDetectTime= std::chrono::duration_cast<std::chrono::duration<double> >(tm2 - tm1).count();
    std::swap(imGrayPre, imGrayT);



    //2023.12.28write

    // Delete those ORB points that fall in Mask borders (Included by Berta)
    cv::Mat MaskLeft_dil = maskLeft.clone();
    cv::Mat MaskRight_dil = maskRight.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(maskLeft, MaskLeft_dil, kernel);
    cv::erode(maskRight, MaskRight_dil, kernel);
//这里把mask又膨胀了一圈
//然后是根据膨胀后的mask_dila重新计算特征点和描述子并赋值
    if(mvKeys.empty())// mvKeys存放左图像中的特征点，如果没有特征点说明这一帧图像不能用
        return;

    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    for (size_t i(0); i < mvKeys.size(); ++i)
    {   //// 遍历之前提取过的特征点，找到特征点坐标在mask_dil上对应的像素值，如果=1就说明不是黑色区域内
        int val = (int)MaskLeft_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
        if (val == 1)
        {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }
//从这里往下不用修改了
    mvKeys = _mvKeys;
    mDescriptors = _mDescriptors;

    std::vector<cv::KeyPoint> _mvKeysRight;
    cv::Mat _mDescriptorsRight;

    for (size_t i(0); i < mvKeysRight.size(); ++i)
    {
        int val = (int)MaskRight_dil.at<uchar>(mvKeysRight[i].pt.y,mvKeysRight[i].pt.x);
        if (val == 1)
        {
            _mvKeysRight.push_back(mvKeysRight[i]);
            _mDescriptorsRight.push_back(mDescriptorsRight.row(i));
        }
    }

    mvKeysRight = _mvKeysRight;
    mDescriptorsRight = _mDescriptorsRight;


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imMask, const double &timeStamp,  ORBextractor* extractor, ORBVocabulary* voc,
             cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mImMask(imMask), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mImGray(imGray),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mIsKeyFrame(false), mImDepth(imDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    // Delete those ORB points that fall in Mask borders (Included by Berta)
    cv::Mat Mask_dil = imMask.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(imMask, Mask_dil, kernel);

    if(mvKeys.empty())
        return;

    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    for (size_t i(0); i < mvKeys.size(); ++i)
    {
        int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
        if (val == 1)
        {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }

    mvKeys = _mvKeys;
    mDescriptors = _mDescriptors;

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imMask, const cv::Mat &imRGB,
             const double &timeStamp,  ORBextractor* extractor, ORBVocabulary* voc,
             cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mImMask(imMask), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mImGray(imGray),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mIsKeyFrame(false), mImDepth(imDepth), mImRGB(imRGB)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);
    
    // cv::Mat  imGrayT = imGray;
    // std::chrono::steady_clock::time_point tm1 = std::chrono::steady_clock::now();
    // ProcessMovingObject(imGray);//移除动态外点的函数，极限约束法
    // std::chrono::steady_clock::time_point tm2 = std::chrono::steady_clock::now();
    // movingDetectTime= std::chrono::duration_cast<std::chrono::duration<double> >(tm2 - tm1).count();
    // std::swap(imGrayPre, imGrayT);

    // Delete those ORB points that fall in Mask borders (Included by Berta)
    cv::Mat Mask_dil = imMask.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(imMask, Mask_dil, kernel);

    if(mvKeys.empty())
        return;

    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    for (size_t i(0); i < mvKeys.size(); ++i)
    {
        int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
        if (val == 1)
        {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }

    mvKeys = _mvKeys;
    mDescriptors = _mDescriptors;

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &mask, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    // Delete those ORB points that fall in mask borders
    cv::Mat Mask_dil = mask.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(mask, Mask_dil, kernel);

    if(mvKeys.empty())
        return;

    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    for (size_t i(0); i < mvKeys.size(); ++i)
    {
        int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
        if (val == 1)
        {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }

    mvKeys = _mvKeys;
    mDescriptors = _mDescriptors;

    if(mvKeys.empty())
        return;

    N = mvKeys.size();

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


void Frame::ProcessMovingObject(const cv::Mat &imgray)//确定需要移除的动态点的函数
{
// step 1 :计算角点(像素级->亚像素级)
// step 2 :计算光流金字塔(确定角点1,2的匹配关系)
// step 3 :对于光流法得到的角点进行筛选(像素块内像素差的和小于阈值)
// step 4 :计算F矩阵(再对点进行了一次筛选)
// step 5 :根据角点到级线的距离小于0.1筛选最匹配的角点
// step 6:找到需要被删去的异常点

    //首先,函数将容器上一次的结果清空
	F_prepoint.clear();
	F_nextpoint.clear();
	F2_prepoint.clear();
	F2_nextpoint.clear();
	T_M.clear();

// 	// 然后使用cv::goodFeaturesToTrack计算Harris角点
	
//     cv::goodFeaturesToTrack(imGrayPre, prepoint, 1000, 0.01, 8, cv::Mat(), 3, true, 0.04);
//     // cv::InputArray image输入图像（CV_8UC1 CV_32FC1）cv::OutputArray corners, 输出角点vector int maxCorners, 最大角点数目 double qualityLevel, 质量水平系数（小于1.0的正数，一般在0.01-0.1之间） 
//     //double minDistance, 最小距离，小于此距离的点忽略
//     //cv::InputArray mask = noArray(),  mask=0的点忽略 nt blockSize = 3, 使用的邻域数  bool useHarrisDetector = false, false =‘Shi Tomasi metric’ double k = 0.04，Harris角点检测时使用
//     cv::cornerSubPix(imGrayPre, prepoint, cv::Size(10, 10), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
// 	//因为cv::goodFeaturesToTrack()提取到的角点只能达到像素级别,所以我们需要使用cv::cornerSubPix()对检测到的角点作进一步的优化计算，可使角点的精度达到亚像素级别。
//     //cv::InputArray image，输入图像 cv::InputOutputArray corners, 角点（既作为输入也作为输出）cv::Size winSize,区域大小为 NXN; N=(winSize*2+1)
//     //cv::Size zeroZone, 类似于winSize，但是总具有较小的范围，Size(-1,-1)表示忽略
//     //cv::TermCriteria criteria 停止优化的标准
//     // 计算光流金字塔，光流金字塔是光流法的一种常见的处理方式，能够避免位移较大时丢失追踪的情况
//     cv::calcOpticalFlowPyrLK(imGrayPre, imgray, prepoint, nextpoint, state, err, cv::Size(22, 22), 5, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));
// //  cv::calcOpticalFlowPyrLK(imGrayPre,  // 输入图像1
// //                           imgray,        // 输入图像2 （t时间之后的）
// //                           prepoint,      // 输入图像1 的角点
// //                           nextpoint,     // 输出图像2 的角点
// //                           state,         // 记录光流点是否跟踪成功，成功status =1,否则为0
// //                           err, cv::Size(22, 22),
// //                           5,             // 5层金字塔
// //                           cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));


// //对于光流法得到的角点进行筛选。DS-SLAM将筛选的结果放入 F_prepoint F_nextpoint 两个数组当中。光流角点是否跟踪成功保存在status数组当中
// 	for (int i = 0; i < state.size(); i++)// state存储追踪到的图像2的角点数目
//     {
//         if(state[i] != 0)// 光流跟踪成功的点
//         {
//             int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
//             int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
//             int x1 = prepoint[i].x, y1 = prepoint[i].y; // 角点1
//             int x2 = nextpoint[i].x, y2 = nextpoint[i].y;   // 角点2
//             // 认为超过规定区域的,太靠近边缘。 跟踪的光流点的status 设置为0 ,一会儿会丢弃这些点
//             if ((x1 < limit_edge_corner || x1 >= imgray.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= imgray.cols - limit_edge_corner
//             || y1 < limit_edge_corner || y1 >= imgray.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= imgray.rows - limit_edge_corner))
//             {
//                 state[i] = 0;
//                 continue;
//             }
//             // 对于光流跟踪的结果进行验证，匹配对中心3*3的图像块的像素差（sum）太大，那么也舍弃这个匹配点
//             // 如果3*3图像块内像素差的和大于2120(limit_of_check,经验值可以调整),就认为匹配不正确
//             double sum_check = 0;
//             for (int j = 0; j < 9; j++)
//                 sum_check += abs(imGrayPre.at<uchar>(y1 + dy[j], x1 + dx[j]) - imgray.at<uchar>(y2 + dy[j], x2 + dx[j]));
//             if (sum_check > limit_of_check) state[i] = 0;
//             // 好的光流点存入 F_prepoint F_nextpoint 两个数组当中
//             if (state[i])
//             {    // 筛选后上下两帧匹配的角点(所以数量是相等的)
//                 F_prepoint.push_back(prepoint[i]);
//                 F_nextpoint.push_back(nextpoint[i]);
//             }
//         }
//     }
    // F-Matrix
    Ptr<Feature2D> f2d = cv::xfeatures2d::SIFT::create();
    	if (!imgray.data || !imGrayPre.data)
	{
		cout << "Reading picture error！" << endl;return;}
    double t0 = getTickCount();//当前滴答数
	vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(imGrayPre, keypoints_1);
	f2d->detect(imgray, keypoints_2);
	cout << "The keypoints number of imGrayPre is:" << keypoints_1.size() << endl;
	cout << "The keypoints number of imgray is:" << keypoints_2.size() << endl;
    //Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute(imgray, keypoints_1, descriptors_1);
	f2d->compute(imGrayPre, keypoints_2, descriptors_2);
    double freq = getTickFrequency();
	double tt = ((double)getTickCount() - t0) / freq;
	cout << "Extract SIFT Time:" <<tt<<"ms"<< endl;
    //drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    //Matching descriptor vector using BFMatcher
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	cout << "The number of match:" << matches.size()<<endl;
    //计算匹配结果中距离最大和距离最小值
	double min_dist = matches[0].distance, max_dist = matches[0].distance;
	for (int m = 0; m < matches.size(); m++)
	{
		if (matches[m].distance<min_dist)
		{
			min_dist = matches[m].distance;
		}
		if (matches[m].distance>max_dist)
		{
			max_dist = matches[m].distance;
		}	
	}
	cout << "min dist=" << min_dist << endl;
	cout << "max dist=" << max_dist << endl;
    vector<DMatch> goodMatches;
	for (int m = 0; m < matches.size(); m++)
	{
		if (matches[m].distance > 1.1*min_dist&&matches[m].distance<0.6*max_dist)
		{
			goodMatches.push_back(matches[m]);
		}
	}
	cout << "The number of good matches:" <<goodMatches.size()<< endl;
    //RANSAC匹配过程
	vector<DMatch> m_Matches;
	m_Matches = goodMatches;
	int ptCount = goodMatches.size();
	if (ptCount < 100)
	{
		cout << "Don't find enough match points" << endl;
		return ;
    }
    vector <KeyPoint> RAN_KP1, RAN_KP2;
    for (size_t i = 0; i < m_Matches.size(); i++)
	{
		RAN_KP1.push_back(keypoints_1[goodMatches[i].queryIdx]);
		RAN_KP2.push_back(keypoints_2[goodMatches[i].trainIdx]);
		//RAN_KP1是要存储img01中能与img02匹配的点
		//goodMatches存储了这些匹配点对的img01和img02的索引值
	}
	//坐标变换
	vector <Point2f> F_prepoint, F_nextpoint;
	for (size_t i = 0; i < m_Matches.size(); i++)
	{
		F_prepoint.push_back(RAN_KP1[i].pt);
		F_nextpoint.push_back(RAN_KP2[i].pt);
	}

    cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
    //筛选之后的光流点计算 F 矩阵(再对点进行了一次筛选)
    cv::Mat F = cv::findFundamentalMat(F_prepoint, F_nextpoint, mask, cv::FM_RANSAC, 0.1, 0.99);
//     CV_EXPORTS Mat findFundamentalMat( InputArray points1, InputArray points2,
// OutputArray mask, int method = FM_RANSAC,mask存储RANSAC后每个点的状态
// double param1 = 3., double param2 = 0.99 );
// @param param1用于RANSAC的参数。它是从一个点到一条外极线的最大距离（以像素为单位），超过该距离的点被视为离群点，不用于计算最终的基本矩阵。
// 它可以设置为1-3，这取决于点定位的精度、图像分辨率和图像噪声。
// @param param2参数仅用于RANSAC或LMedS方法。它规定了估计矩阵正确的理想置信水平（概率）。
// @param mask由N个元素组成的输出数组，其中每个元素的异常值设置为0，其他点设置为1。该数组仅在RANSAC和LMedS方法中计算。
    cout << "the F_pre num is"<< mask.rows<< endl;
    for (int i = 0; i < mask.rows; i++)// mask.rows表示
    {
        if (mask.at<uchar>(i, 0) == 0);
        else
        {  
         // 基线的A,B,C
            // Circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
            double A = F.at<double>(0, 0)*F_prepoint[i].x + F.at<double>(0, 1)*F_prepoint[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*F_prepoint[i].x + F.at<double>(1, 1)*F_prepoint[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*F_prepoint[i].x + F.at<double>(2, 1)*F_prepoint[i].y + F.at<double>(2, 2);
            double dd = fabs(A*F_nextpoint[i].x + B*F_nextpoint[i].y + C) / sqrt(A*A + B*B); //Epipolar constraints(论文公式3,点到直线距离)
            if (dd <= 0.1)//角点2到直线的距离小于0.1(米?),则符合要求
            {
                F2_prepoint.push_back(F_prepoint[i]);// 更加精确的符合要求的角点
                F2_nextpoint.push_back(F_nextpoint[i]);
            }
        }
    }
    // F_prepoint = F2_prepoint;
    // F_nextpoint = F2_nextpoint;
//这两个变量貌似没有再用过了
    for (int i = 0; i < F_prepoint.size(); i++)
    {
        
          // 直线的一般式:Ax+By+C=0
            double A = F.at<double>(0, 0)*F_prepoint[i].x + F.at<double>(0, 1)*F_prepoint[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*F_prepoint[i].x + F.at<double>(1, 1)*F_prepoint[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*F_prepoint[i].x + F.at<double>(2, 1)*F_prepoint[i].y + F.at<double>(2, 2);
            double dd = fabs(A*F_nextpoint[i].x + B*F_nextpoint[i].y + C) / sqrt(A*A + B*B); 

            // Judge outliers 认为大于阈值的点是动态点，存入T_M
            if (dd >= limit_dis_epi) {T_R.push_back(F_nextpoint[i]);}
            else {T_M.push_back(F_nextpoint[i]);}//为了找异常点,所以通过精度不是很高的匹配点来搜索
        
    }
    cv::Mat img_show = imgray.clone();
    cvtColor(img_show,img_show,cv::COLOR_GRAY2BGR);
    cout<<"badpoint  "<<T_M.size()<<endl;
    cout<<"goodpoint  "<<T_R.size()<<endl;
    for( auto kp:T_R ){
        cv::circle(img_show, kp, 5, cv::Scalar(0, 255, 0), 1);//RGB，red
        cv::circle(img_show, kp, 1, cv::Scalar(0, 255, 0), -1);

    }
    for( auto kp:T_M ){
        cv::circle(img_show, kp, 5, cv::Scalar(0, 0, 255), 1);//RGB，red
        cv::circle(img_show, kp, 1, cv::Scalar(0, 0, 255), -1);

    }
    imshow("showGeoImg", img_show);
    F_prepoint = F2_prepoint;
    F_nextpoint = F2_nextpoint;


}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            //cout << "Depth: " << d << " m" << endl;
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
