/**
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/

#include "Geometry.h"
#include <algorithm>
#include "Frame.h"
#include "Tracking.h"

namespace DynaSLAM
{

Geometry::Geometry()
{
    vAllPixels = cv::Mat(640*480,2,CV_32F);
    int m(0);
    for (int i(0); i < 640; i++){
        for (int j(0); j < 480; j++){
            vAllPixels.at<float>(m,0) = i;
            vAllPixels.at<float>(m,1) = j;
            m++;
        }
    }
}

void Geometry::GeometricModelCorrection(const ORB_SLAM2::Frame &currentFrame,
                                        cv::Mat &imDepth, cv::Mat &mask){
    if(currentFrame.mTcw.empty()){
        std::cout << "Geometry not working." << std::endl;
    }
    // 如果当前帧的位姿不为空
    // mDB是关键帧，mNumElem表示存储的关键帧数量，ELEM_INITIAL_MAP=5，论文里用到这个数
    // Geometry.cc是作者在ORB-SLAM2之外额外写的函数
    else if (mDB.mNumElem >= ELEM_INITIAL_MAP){
        // 获取距离差从大到小的5帧关键帧
     // vRefFrames 在提取动态点之前获得
        vector<ORB_SLAM2::Frame> vRefFrames = GetRefFrames(currentFrame);
        vector<DynKeyPoint> vDynPoints = ExtractDynPoints(vRefFrames,currentFrame);
        mask = DepthRegionGrowing(vDynPoints,imDepth);
        CombineMasks(currentFrame,mask);
    }
}

void Geometry::InpaintFrames(const ORB_SLAM2::Frame &currentFrame,
                             cv::Mat &imGray, cv::Mat &imDepth,
                             cv::Mat &imRGB, cv::Mat &mask){
    FillRGBD(currentFrame,mask,imGray,imDepth,imRGB);
}

void Geometry::GeometricModelUpdateDB(const ORB_SLAM2::Frame &currentFrame){
    if (currentFrame.mIsKeyFrame)
    {
        mDB.InsertFrame2DB(currentFrame);
    }
}

vector<ORB_SLAM2::Frame> Geometry::GetRefFrames(const ORB_SLAM2::Frame &currentFrame){
//计算了当前帧与各关键帧之间的距离
// 获取当前帧位姿的旋转
    cv::Mat rot1 = currentFrame.mTcw.rowRange(0,3).colRange(0,3);
    // 计算欧拉角（先不管是怎么算的
    cv::Mat eul1 = rotm2euler(rot1);
    // 获取当前帧位姿的平移
    cv::Mat trans1 = currentFrame.mTcw.rowRange(0,3).col(3);
    cv::Mat vDist;
    cv::Mat vRot;

    // 同时考虑新帧与每个关键帧之间的距离和旋转
    // mDB存储的是关键帧，最多存20帧
    for (int i(0); i < mDB.mNumElem; i++){
        cv::Mat rot2 = mDB.mvDataBase[i].mTcw.rowRange(0,3).colRange(0,3);
        cv::Mat eul2 = rotm2euler(rot2);
        // 计算当前帧与关键帧之间欧拉角距离的二范数
        double distRot = cv::norm(eul2,eul1,cv::NORM_L2);
        // 旋转角之差
        vRot.push_back(distRot);

        cv::Mat trans2 = mDB.mvDataBase[i].mTcw.rowRange(0,3).col(3);
        // 求二范数距离
        double dist = cv::norm(trans2,trans1,cv::NORM_L2);
        // 平移距离之差
        vDist.push_back(dist);
    }
    // 找出平移距离差/欧拉角距离差的最大值
    // 每个距离除以最大值（可能把数据改小方便计算吧

    double minvDist, maxvDist;
    cv::minMaxLoc(vDist, &minvDist, &maxvDist);
    vDist /= maxvDist;

    double minvRot, maxvRot;
    cv::minMaxLoc(vRot, &minvRot, &maxvRot);
    vRot /= maxvRot;
 // 距离指标vDist,论文中有公式 0.7*Precesion+0.3*Recall,但好像不是这个
    vDist = 0.7*vDist + 0.3*vRot;
    // 对距离进行排序，距离描述当前帧与关键帧的相似度。
    // 对矩阵的每一列进行降序排序  eg.vIndex = [0;1;2;3;4],即5*1。vDist的顺序不变，vIndex存储vDist内值从大到小的序号
    cv::Mat vIndex;
    cv::sortIdx(vDist,vIndex,CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

// #define MAX_REF_FRAMES 5 最大参考帧定义为5
    // 对于每个输入帧，我们选择之前具有最大重叠的关键帧。
    // 这个被完成通过同时考虑新帧与每个关键帧之间的距离和旋转，类似Tan等人[9]。重叠关键帧的数量已经在我们的实验中被设置为5
    // vDist.rows就是帧的数量
    mnRefFrames = std::min(MAX_REF_FRAMES,vDist.rows);

    vector<ORB_SLAM2::Frame> vRefFrames;

    for (int i(0); i < mnRefFrames; i++)
    {
        int ind = vIndex.at<int>(0,i);
        // 取出关键帧中距离最
        vRefFrames.push_back(mDB.mvDataBase[ind]);// 距离从大到小
    }

    return(vRefFrames);// 返回距离最大的mnRefFrames个关键帧
}
vector<Geometry::DynKeyPoint> Geometry::ExtractDynPoints(const vector<ORB_SLAM2::Frame> &vRefFrames,
                                                         const ORB_SLAM2::Frame &currentFrame){
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = currentFrame.fx;
    K.at<float>(1,1) = currentFrame.fy;
    K.at<float>(0,2) = currentFrame.cx;
    K.at<float>(1,2) = currentFrame.cy;

    cv::Mat vAllMPw;              //存放所有的世界坐标系下地图点
    cv::Mat vAllMatRefFrame;      //存放参考帧中，关键点的u,v,1
    cv::Mat vAllLabels;
    cv::Mat vAllDepthRefFrame;

    //遍历每一个参考帧
    for (int i(0); i < mnRefFrames; i++)
    {
        ORB_SLAM2::Frame refFrame = vRefFrames[i];

        // Fill matrix with points
        // 这里是用来存储单独一个参考帧的信息 坐标u,v,1
        cv::Mat matRefFrame(refFrame.N,3,CV_32F);
        cv::Mat matDepthRefFrame(refFrame.N,1,CV_32F);
        cv::Mat matInvDepthRefFrame(refFrame.N,1,CV_32F);
        //参考帧中的有效关键点有k个，vLabels.at<float>(k,0)表示第k个有效关键点对应着原来该参考帧中第i个关键点
        cv::Mat vLabels(refFrame.N,1,CV_32F);
        int k(0);
        //遍历这一参考帧的所有关键点
        for(int j(0); j < refFrame.N; j++){
            const cv::KeyPoint &kp = refFrame.mvKeys[j];
            const float &v = kp.pt.y;
            const float &u = kp.pt.x;
            const float d = refFrame.mImDepth.at<float>(v,u);
            //这里的深度单位应该是m
            ///STEP1: 对参考帧的点的深度进行筛选
            if (d > 0 && d < 6){
                matRefFrame.at<float>(k,0) = refFrame.mvKeysUn[j].pt.x;
                matRefFrame.at<float>(k,1) = refFrame.mvKeysUn[j].pt.y;
                matRefFrame.at<float>(k,2) = 1.;
                matInvDepthRefFrame.at<float>(k,0) = 1./d;
                matDepthRefFrame.at<float>(k,0) = d;
                vLabels.at<float>(k,0) = i;
                k++;
            }
        }
        
        if(k==0){
            continue;  //add!
        }

        //matRefFrame是一个k*3维度的矩阵,存的是参考帧关键点的像素坐标 u,v,1
        matRefFrame = matRefFrame.rowRange(0,k);
        matInvDepthRefFrame = matInvDepthRefFrame.rowRange(0,k);
        matDepthRefFrame = matDepthRefFrame.rowRange(0,k);
        vLabels = vLabels.rowRange(0,k);  //一个k*1维的mat
        //参考帧的关键点在相机坐标系的坐标 维度3*k   得到归一化坐标 X/Z, Y/Z, 1
        cv::Mat vMPRefFrame = K.inv()*matRefFrame.t();
        cout <<"vMPRefFrame size " <<vMPRefFrame.size() <<endl;
        //把两个vMPRefFrame和matInvDepthRefFrame拼合在一起 变成 X/Z, Y/Z, 1, 1/Z
        cv::vconcat(vMPRefFrame,matInvDepthRefFrame.t(),vMPRefFrame);  //维度变为 4*k
        cv::Mat vMPw = refFrame.mTcw.inv() * vMPRefFrame;  //关键点在世界坐标系的归一化坐标  4*k
        cv::Mat _vMPw = cv::Mat(4,vMPw.cols,CV_32F);
        cv::Mat _vLabels = cv::Mat(vLabels.rows,1,CV_32F);
        cv::Mat _matRefFrame = cv::Mat(matRefFrame.rows,3,CV_32F);
        cv::Mat _matDepthRefFrame = cv::Mat(matDepthRefFrame.rows,1,CV_32F);
        
        int h(0);
        mParallaxThreshold = 30;   //视差角
        //STEP2： 根据地图点和两帧上的投影点的夹角（视差角）大小进行筛选
        for (int j(0); j < k; j++)
        {
            cv::Mat mp = cv::Mat(3,1,CV_32F);
            //这里又从归一化坐标变为了X，Y，Z
            mp.at<float>(0,0) = vMPw.at<float>(0,j)/matInvDepthRefFrame.at<float>(0,j);
            mp.at<float>(1,0) = vMPw.at<float>(1,j)/matInvDepthRefFrame.at<float>(0,j);
            mp.at<float>(2,0) = vMPw.at<float>(2,j)/matInvDepthRefFrame.at<float>(0,j);
            cv::Mat tRefFrame = refFrame.mTcw.rowRange(0,3).col(3);   //参考帧相机在世界坐标系下的位置
            cv::Mat tCurrentFrame = currentFrame.mTcw.rowRange(0,3).col(3); //当前帧相机在世界坐标系下的位置
            //对应图中的 X-KF
            cv::Mat nMPRefFrame = mp - tRefFrame;
            //对应图中的 X-CF
            cv::Mat nMPCurrentFrame = mp - tCurrentFrame;

            double dotProduct = nMPRefFrame.dot(nMPCurrentFrame);
            double normMPRefFrame = cv::norm(nMPRefFrame,cv::NORM_L2);
            double normMPCurrentFrame = cv::norm(nMPCurrentFrame,cv::NORM_L2);
            //X-KF和X-CF进行点乘然后单位化，求的就是视差角的cos值
            double angle = acos(dotProduct/(normMPRefFrame*normMPCurrentFrame))*180/M_PI;
            //cout << "parallax angle= " << angle <<endl;
            //小于30度才保存地图点，参考帧的点; 大于30度的点在论文中被认为是“有遮挡情况的静态点”，如果不筛除，后续就会被错误归为动态点
            if (angle < mParallaxThreshold)
            {
                //j表示满足前面深度要求的地图点的遍历序号，h表示后续还能满足视差角条件的地图点的遍历序号
                _vMPw.at<float>(0,h) = vMPw.at<float>(0,j);  //X
                _vMPw.at<float>(1,h) = vMPw.at<float>(1,j);  //Y
                _vMPw.at<float>(2,h) = vMPw.at<float>(2,j);  //Z
                _vMPw.at<float>(3,h) = vMPw.at<float>(3,j);  // 1/Z
                _vLabels.at<float>(h,0) = vLabels.at<float>(j,0);
                _matRefFrame.at<float>(h,0) = matRefFrame.at<float>(j,0);  //u
                _matRefFrame.at<float>(h,1) = matRefFrame.at<float>(j,1);  //v
                _matRefFrame.at<float>(h,2) = matRefFrame.at<float>(j,2);  //1
                _matDepthRefFrame.at<float>(h,0) = matDepthRefFrame.at<float>(j,0);
                h++;   //对于当前帧和参考帧，有h个有效的地图点满足视差角
            }
        }

        if(h==0){
            continue;   // add!
        }

        vMPw = _vMPw.colRange(0,h);
        vLabels = _vLabels.rowRange(0,h);
        matRefFrame = _matRefFrame.rowRange(0,h);
        matDepthRefFrame = _matDepthRefFrame.rowRange(0,h);

        //把单帧计算的地图点结果放进All-系列变量中
        if (vAllMPw.empty())
        {
            vAllMPw = vMPw;
            vAllMatRefFrame = matRefFrame;
            vAllLabels = vLabels;
            vAllDepthRefFrame = matDepthRefFrame;
        }
        else
        {
            if (!vMPw.empty())
            {
                hconcat(vAllMPw,vMPw,vAllMPw);
                vconcat(vAllMatRefFrame,matRefFrame,vAllMatRefFrame);
                vconcat(vAllLabels,vLabels,vAllLabels);
                vconcat(vAllDepthRefFrame,matDepthRefFrame,vAllDepthRefFrame);
            }
        }

    }

    cv::Mat vLabels = vAllLabels;

    //STEP3: 将筛选后参考帧的所有地图点投影到当前帧，如果这些地图点的深度不超过7m才保留
    if (!vAllMPw.empty())
    {
        //把筛选后的所有参考帧在世界坐标系下的地图点投影到当前帧相机坐标系下 /有4行
        //世界坐标系下三维点[X/Z，Y/Z，Z/Z，1/Z]-> 得到当前帧坐标系下的归一化坐标
        cv::Mat vMPCurrentFrame = currentFrame.mTcw * vAllMPw;
       

        // Divide by last column
        for (int i(0); i < vMPCurrentFrame.cols; i++)
        {
            //vMPCurrentFrame 存的是 [X, Y, Z, 1]（由参考帧投过来的）
            vMPCurrentFrame.at<float>(0,i) /= vMPCurrentFrame.at<float>(3,i);
            vMPCurrentFrame.at<float>(1,i) /= vMPCurrentFrame.at<float>(3,i);
            vMPCurrentFrame.at<float>(2,i) /= vMPCurrentFrame.at<float>(3,i);
            vMPCurrentFrame.at<float>(3,i) /= vMPCurrentFrame.at<float>(3,i);
        }
        cv::Mat matProjDepth = vMPCurrentFrame.row(2);

        cv::Mat _vMPCurrentFrame = cv::Mat(vMPCurrentFrame.size(),CV_32F);
        cv::Mat _vAllMatRefFrame = cv::Mat(vAllMatRefFrame.size(),CV_32F);
        cv::Mat _vLabels = cv::Mat(vLabels.size(),CV_32F);
        cv::Mat __vAllDepthRefFrame = cv::Mat(vAllDepthRefFrame.size(),CV_32F);
        int h(0);
        cv::Mat __matProjDepth = cv::Mat(matProjDepth.size(),CV_32F);
        
        for (int i(0); i < matProjDepth.cols; i++)
        {
            //只保留计算（投影）得到的当前帧中距离不超过7m的地图点和对应的像素点
            if (matProjDepth.at<float>(0,i) < 7)
            {
                __matProjDepth.at<float>(0,h) = matProjDepth.at<float>(0,i);

                _vMPCurrentFrame.at<float>(0,h) = vMPCurrentFrame.at<float>(0,i);  //X
                _vMPCurrentFrame.at<float>(1,h) = vMPCurrentFrame.at<float>(1,i);  //Y
                _vMPCurrentFrame.at<float>(2,h) = vMPCurrentFrame.at<float>(2,i);  //Z
                _vMPCurrentFrame.at<float>(3,h) = vMPCurrentFrame.at<float>(3,i);  //1

                _vAllMatRefFrame.at<float>(h,0) = vAllMatRefFrame.at<float>(i,0);  //u
                _vAllMatRefFrame.at<float>(h,1) = vAllMatRefFrame.at<float>(i,1);  //v
                _vAllMatRefFrame.at<float>(h,2) = vAllMatRefFrame.at<float>(i,2);  //1

                _vLabels.at<float>(h,0) = vLabels.at<float>(i,0);

                __vAllDepthRefFrame.at<float>(h,0) = vAllDepthRefFrame.at<float>(i,0);

                h++;
            }
        }

        matProjDepth = __matProjDepth.colRange(0,h);
        vMPCurrentFrame = _vMPCurrentFrame.colRange(0,h);  //一共有h个关键点入选，
        vAllMatRefFrame = _vAllMatRefFrame.rowRange(0,h);
        vLabels = _vLabels.rowRange(0,h);
        vAllDepthRefFrame = __vAllDepthRefFrame.rowRange(0,h);

        cv::Mat aux;
        //维度是3*4的矩阵
        cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);
        //vMPCurrentFrame 存的是 [X, Y, Z, 1]
        cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame; //转换到像素坐标  z*(u,v,1)

        cv::Mat mat2CurrentFrame(matCurrentFrame.cols,2,CV_32F);
        cv::Mat v2AllMatRefFrame(matCurrentFrame.cols,3,CV_32F);
        cv::Mat mat2ProjDepth(matCurrentFrame.cols,1,CV_32F);
        cv::Mat v2Labels(matCurrentFrame.cols,1,CV_32F);
        cv::Mat _vAllDepthRefFrame(matCurrentFrame.cols,1,CV_32F);

        //STEP4: 如果把“由多个参考帧的地图点投影到当前帧”的信息换算成一个深度图，投影得到的深度要比测量深度大
        int j = 0;
        for (int i(0); i < matCurrentFrame.cols; i++)
        {
            //这个是由参考帧算出来的 u,v值
            float x = ceil(matCurrentFrame.at<float>(0,i)/matCurrentFrame.at<float>(2,i));
            float y = ceil(matCurrentFrame.at<float>(1,i)/matCurrentFrame.at<float>(2,i));
            //如果这个像素坐标在当前帧深度图的特定范围内（这里把深度图的最外20pixel除去了，和后面使用的滑窗有关）
            if (IsInFrame(x,y,currentFrame))
            {
                //当前帧实际测量出的深度值d
                const float d = currentFrame.mImDepth.at<float>(y,x);
                if (d > 0)
                {
                    //TODO： 如果这里参考帧的投影有重复的，即有相同的u，v坐标，但深度不同。这里没有对重复的筛选？
                    mat2CurrentFrame.at<float>(j,0) = x;
                    mat2CurrentFrame.at<float>(j,1) = y;
                    // 存入所有有效的像素坐标 u,v,1;  depth
                    v2AllMatRefFrame.at<float>(j,0) = vAllMatRefFrame.at<float>(i,0);
                    v2AllMatRefFrame.at<float>(j,1) = vAllMatRefFrame.at<float>(i,1);
                    v2AllMatRefFrame.at<float>(j,2) = vAllMatRefFrame.at<float>(i,2);  // =1
                    //投影得到的当前帧对应点的深度
                    float d1 = matProjDepth.at<float>(0,i);
                    mat2ProjDepth.at<float>(j,0) = d1;  //深度的投影值
                    v2Labels.at<float>(j,0) = vLabels.at<float>(i,0);
                    j++;
                }
            }
        }
        vAllDepthRefFrame = _vAllDepthRefFrame.rowRange(0,j);
        vAllMatRefFrame = v2AllMatRefFrame.rowRange(0,j);
        matProjDepth = mat2ProjDepth.rowRange(0,j);
        matCurrentFrame = mat2CurrentFrame.rowRange(0,j);
        vLabels = v2Labels.rowRange(0,j);

        //在IsInFrame函数中 mDmax初始化为20，所以这里新建了一个维度为[41*41, 2]大小的矩阵
        //这个小矩阵u1,  每一行存放着能遍历一个41*41矩阵的坐标id： i，j
        // [-20,-20]  [-20,-19] .....   [-20,20]
        // [-19,-20]  [-19,-19] .....   [-19,20]
        //  .....
        cv::Mat u1((2*mDmax+1)*(2*mDmax+1),2,CV_32F);
        int m(0);
        for (int i(-mDmax); i <= mDmax; i++){
            for (int j(-mDmax); j <= mDmax; j++){
                u1.at<float>(m,0) = i;
                u1.at<float>(m,1) = j;
                m++;
            }
        }

        cv::Mat matDepthCurrentFrame(matCurrentFrame.rows,1,CV_32F);
        cv::Mat _matProjDepth(matCurrentFrame.rows,1,CV_32F);
        cv::Mat _matCurrentFrame(matCurrentFrame.rows,2,CV_32F);

        int _s(0);
        for (int i(0); i < matCurrentFrame.rows; i++)
        {
            int s(0);
            cv::Mat _matDiffDepth(u1.rows,1,CV_32F);
            cv::Mat _matDepth(u1.rows,1,CV_32F);
            //这里是按照一个patch一个patch来计算的，patch之间会有重叠的地方，有重复的计算
            //这样是为了后面计算一个patch内的深度Diff阈值和标准差
            for (int j(0); j < u1.rows; j++)
            {
                int x = (int)matCurrentFrame.at<float>(i,0) + (int)u1.at<float>(j,0);
                int y = (int)matCurrentFrame.at<float>(i,1) + (int)u1.at<float>(j,1);
                float _d = currentFrame.mImDepth.at<float>(y,x);   //实际测量值
                //如果实际测量值大于0并且小于投影的深度   TODO ///
                if ((_d > 0) && (_d < matProjDepth.at<float>(i,0)))
                {
                    _matDepth.at<float>(s,0) = _d;
                    _matDiffDepth.at<float>(s,0) = matProjDepth.at<float>(i,0) - _d;
                    s++;   //记录计算的DiffDepth的个数，即对多少个像素点，有投影深度>实际深度，这些像素点对应潜在动态点
                }
            }

            //潜在动态点个数大于0时
            if (s > 0)
            {
                _matDepth = _matDepth.rowRange(0,s);
                _matDiffDepth = _matDiffDepth.rowRange(0,s);
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                //存储DiffDepth的最大最小值以及对应的index
                cv::minMaxLoc(_matDiffDepth,&minVal,&maxVal,&minIdx,&maxIdx);
                int xIndex = minIdx.x;
                int yIndex = minIdx.y;
                matDepthCurrentFrame.at<float>(_s,0) = _matDepth.at<float>(yIndex,0);  //实际深度
                _matProjDepth.at<float>(_s,0) = matProjDepth.at<float>(i,0);  //对应的投影深度
                //对应的像素点坐标
                _matCurrentFrame.at<float>(_s,0) = matCurrentFrame.at<float>(i,0);
                _matCurrentFrame.at<float>(_s,1) = matCurrentFrame.at<float>(i,1);
                _s++;
            }
        }

        matDepthCurrentFrame = matDepthCurrentFrame.rowRange(0,_s);
        matProjDepth = _matProjDepth.rowRange(0,_s);
        matCurrentFrame = _matCurrentFrame.rowRange(0,_s);

        mDepthThreshold = 0.6;


        cv::Mat matDepthDifference = matProjDepth - matDepthCurrentFrame;

        mVarThreshold = 0.001; //0.040;
        //STEP5： 根据距离差值的阈值，筛选出最终的动态点
        vector<Geometry::DynKeyPoint> vDynPoints;

        for (int i(0); i < matCurrentFrame.rows; i++)
        {
            //深度的差值要大于阈值
            if (matDepthDifference.at<float>(i,0) > mDepthThreshold)
            {
                int xIni = (int)matCurrentFrame.at<float>(i,0) - mDmax;
                int yIni = (int)matCurrentFrame.at<float>(i,1) - mDmax;
                int xEnd = (int)matCurrentFrame.at<float>(i,0) + mDmax + 1;
                int yEnd = (int)matCurrentFrame.at<float>(i,1) + mDmax + 1;
                cv::Mat patch = currentFrame.mImDepth.rowRange(yIni,yEnd).colRange(xIni,xEnd);
                cv::Mat mean, stddev;
                cv::meanStdDev(patch,mean,stddev);
                double _stddev = stddev.at<double>(0,0);
                double var = _stddev*_stddev;
                //这个patch内当前帧测量深度的方差（深度值的变化幅度小）要小于一个阈值，理解是这一块不能本身就有很复杂的遮挡关系
                if (var < mVarThreshold)
                {
                    DynKeyPoint dynPoint;
                    dynPoint.mPoint.x = matCurrentFrame.at<float>(i,0);
                    dynPoint.mPoint.y = matCurrentFrame.at<float>(i,1);
                    dynPoint.mRefFrameLabel = vLabels.at<float>(i,0);  //对应着原来的某个关键点
                    vDynPoints.push_back(dynPoint);
                }
            }
        }

        return vDynPoints;
    }
    else
    {
        //如果参考帧中没有能满足大于30度视差角的地图点：
        vector<Geometry::DynKeyPoint> vDynPoints;
        return vDynPoints;
    }
}


// vector<Geometry::DynKeyPoint> Geometry::ExtractDynPoints(const vector<ORB_SLAM2::Frame> &vRefFrames,
//                                                          const ORB_SLAM2::Frame &currentFrame){
//                                                          //动态提取点函数
//     //获取与当前帧相比较的参考帧(最大数量为5)，并把参考帧的归一化相机坐标转换回世界坐标（像素坐标转换为世界坐标） 
//     //相机内参矩阵K
//     cv::Mat K = cv::Mat::eye(3,3,CV_32F);
//     K.at<float>(0,0) = currentFrame.fx;
//     K.at<float>(1,1) = currentFrame.fy;
//     K.at<float>(0,2) = currentFrame.cx;
//     K.at<float>(1,2) = currentFrame.cy;

//     cv::Mat vAllMPw;
//     cv::Mat vAllMatRefFrame;
//     cv::Mat vAllLabels;
//     cv::Mat vAllDepthRefFrame;
// //// vRefFrames内是按与当前帧距离从大到小排序的
//     for (int i(0); i < mnRefFrames; i++)
//     {
//         ORB_SLAM2::Frame refFrame = vRefFrames[i];

//         // Fill matrix with points
//         // refFrame.N是参考帧中关键点的数量，CV_32F表示像素可以拥有[0, 1.0]间的任意float值
//         // cv::Size sz(w=cols,h=rows),所以size()=[cols * rows] ,例如某一次运行matRefFrame.size()=[3 x 666]
//         // 但mat的初始化不一样：cv::MAT(h=rows,w= cols)
//         // 存储参考帧的特征点的u,v,1
//         cv::Mat matRefFrame(refFrame.N,3,CV_32F);
//         // 存储特征点的深度 Z
//         cv::Mat matDepthRefFrame(refFrame.N,1,CV_32F);
//         // 逆深度 1/Z
//         cv::Mat matInvDepthRefFrame(refFrame.N,1,CV_32F);
//         // 存储参考帧每个特征点的index
//         cv::Mat vLabels(refFrame.N,1,CV_32F);//n个特征点，n行1列
//         int k(0);
//         for(int j(0); j < refFrame.N; j++){
//             const cv::KeyPoint &kp = refFrame.mvKeys[j];
//             const float &v = kp.pt.y;
//             const float &u = kp.pt.x;
//             // at<float>(y, x)
//             const float d = refFrame.mImDepth.at<float>(v,u);// 特征点深度 
//             if (d > 0 && d < 6){
//                 // 取出当前参考帧每一个特征点的信息，后面计算要用
//                 // at<float>(row, col)/(行，列)/（y, x）
//                 matRefFrame.at<float>(k,0) = refFrame.mvKeysUn[j].pt.x;
//                 matRefFrame.at<float>(k,1) = refFrame.mvKeysUn[j].pt.y;
//                 matRefFrame.at<float>(k,2) = 1.;
//                 matInvDepthRefFrame.at<float>(k,0) = 1./d;
//                 matDepthRefFrame.at<float>(k,0) = d;
//                 vLabels.at<float>(k,0) = i;//vLabels里k行0列的标签是vRefFrames里第i个参考帧，表示这个特征点的标签是/属于第i参考帧
//                 k++;
//                 if(k==0)
//                     continue;
//             }
//         }
//         // 筛选过后的特征点信息(因为原始特征点可能深度不符合要求)  
//         // k=本参考帧内符合距离要求的特征点数量,减少内存占用
//         matRefFrame = matRefFrame.rowRange(0,k);
//         matInvDepthRefFrame = matInvDepthRefFrame.rowRange(0,k);
//         matDepthRefFrame = matDepthRefFrame.rowRange(0,k);
//         vLabels = vLabels.rowRange(0,k);
//         // 获取归一化3D坐标，K-1[u;v;1] = [X/Z, Y/Z ,1]
//         // （3*3） * （3*n）=(rows=3 * cols=n)
//         //  vMPRefFrame = [ X1/Z1   X2/Z2   .... 
//         //                                      Y1/Z1   Y2/Z2   ....
//         //                                          1             1        .... ]
//         cv::Mat vMPRefFrame = K.inv()*matRefFrame.t();// .t()是transpose()的含义
//         // 垂直连接，按行合并
//         // 每一行是[X/Z, Y/Z ,1, 1/Z] 
//         //  vMPRefFrame = [ X1/Z1   X2/Z2   .... 
//         //                  Y1/Z1   Y2/Z2   ....
//         //                   1        1     .... 
//         //                  1/Z1    1/Z2    .... ]
//         cv::vconcat(vMPRefFrame,matInvDepthRefFrame.t(),vMPRefFrame);
//         // 把参考帧的归一化相机坐标转换回世界坐标 
//         //  vMPw.size()=[n x 4]，4行n列
//         cv::Mat vMPw = refFrame.mTcw.inv()*vMPRefFrame;
//         cv::Mat _vMPw = cv::Mat(4,vMPw.cols,CV_32F);//cv::MAT(rows,cols)；
//         cv::Mat _vLabels = cv::Mat(vLabels.rows,1,CV_32F);
//         cv::Mat _matRefFrame = cv::Mat(matRefFrame.rows,3,CV_32F);
//         cv::Mat _matDepthRefFrame = cv::Mat(matDepthRefFrame.rows,1,CV_32F);
// //初筛选：计算视差角 ，大于30度的去掉
//         int h(0);
//         // 角度阈值，超过30度认为是动态点
//         mParallaxThreshold = 30;
//         for (int j(0); j < k; j++)//当前帧符合条件的特征点
//         {
//             cv::Mat mp = cv::Mat(3,1,CV_32F);
//             //  matInvDepthRefFrame.size()=[cols=1 x rows=n] 变换回X，Y，Z
//             // vMPw.size()=[n*4],  matInvDepthRefFrame.size()=[1*n]
//             // mp:参考帧点[j]的世界坐标X，Y，Z
//             // mp 是 3D坐标 [X, Y, Z]T
//             mp.at<float>(0,0) = vMPw.at<float>(0,j)/matInvDepthRefFrame.at<float>(0,j);
//             mp.at<float>(1,0) = vMPw.at<float>(1,j)/matInvDepthRefFrame.at<float>(0,j);
//             mp.at<float>(2,0) = vMPw.at<float>(2,j)/matInvDepthRefFrame.at<float>(0,j);
//             //取出平移量（0到2行，第四列，就是参考帧的平移向量），同样下面也是取的平移向量，（3*1）
//             cv::Mat tRefFrame = refFrame.mTcw.rowRange(0,3).col(3);//参考帧
//             // 计算两个点之间的夹角 https://www.cnblogs.com/xuyi911204/p/13154612.html
//             cv::Mat tCurrentFrame = currentFrame.mTcw.rowRange(0,3).col(3);//当前帧
//             cv::Mat nMPRefFrame = mp - tRefFrame;
//             cv::Mat nMPCurrentFrame = mp - tCurrentFrame;
//             //更改部分开始
//             //已知mp是当前参考帧这个点的世界坐标了
//             //需要得到当前这个点在当前帧的像素坐标
//             cv::Mat mp2 = cv::Mat(4,1,CV_32F);
//             mp2.at<float>(0,0)=mp.at<float>(0,0);
//             mp2.at<float>(1,0)=mp.at<float>(1,0);
//             mp2.at<float>(2,0)=mp.at<float>(2,0);
//             mp2.at<float>(3,0)=1.0;//换成齐次坐标
//             cv::Mat nmpc1=currentFrame.mTcw*mp2;
//             cv::Mat nmpc2=K*nmpc1;
//             //已经是当前帧的（uv1）了
//             // x1*x2+y1*y2
//             double dotProduct = nMPRefFrame.dot(nMPCurrentFrame);// opencv mat做点乘
//             // 返回数组中所有元素的二范数,sqrt(x*x+y*y )    
//             double normMPRefFrame = cv::norm(nMPRefFrame,cv::NORM_L2);
//             double normMPCurrentFrame = cv::norm(nMPCurrentFrame,cv::NORM_L2);
//             double angle = acos(dotProduct/(normMPRefFrame*normMPCurrentFrame))*180/M_PI;
//             // RGBD 计算角度差值 是否小于阈值,小于的是静态点，保留
//             if (angle < mParallaxThreshold)
//             {// size()=[cols * rows]，at<float>(row, col)
//                 _vMPw.at<float>(0,h) = vMPw.at<float>(0,j);//MPw是世界坐标下参考帧中的静态特征点
//                 _vMPw.at<float>(1,h) = vMPw.at<float>(1,j);
//                 _vMPw.at<float>(2,h) = vMPw.at<float>(2,j);
//                 _vMPw.at<float>(3,h) = vMPw.at<float>(3,j);
//                 // _vMPw存储的是静态点（）
//                 _vLabels.at<float>(h,0) = vLabels.at<float>(j,0);
//                 _matRefFrame.at<float>(h,0) = matRefFrame.at<float>(j,0);
//                 _matRefFrame.at<float>(h,1) = matRefFrame.at<float>(j,1);
//                 _matRefFrame.at<float>(h,2) = matRefFrame.at<float>(j,2);//_matRefFrame是像素坐标下参考帧中的静态特征点
//                 _matDepthRefFrame.at<float>(h,0) = matDepthRefFrame.at<float>(j,0);
//                 h++;
//                 if(h==0)
//                     continue;// h：在本参考帧内符合距离要求d>0&&d<6的点k个里面继续筛选，符合角度差的为h个
//             }
//         }

//         vMPw = _vMPw.colRange(0,h);//包括序号0不包括序号h列，共h列
//         //cout << "_vMPw.size()="<<_vMPw.size()<<",h="<<h<<endl;
//         //cout << "_vMPw.rows="<<_vMPw.rows<<endl;
//         // 取出小于阈值的静态点
//         //cout <<"rowRange end"<<endl;
//         vLabels = _vLabels.rowRange(0,h);
//         matRefFrame = _matRefFrame.rowRange(0,h);
//         matDepthRefFrame = _matDepthRefFrame.rowRange(0,h);

//         if (vAllMPw.empty())
//         {   //  第一次进来
//             vAllMPw = vMPw;// 存的静态点。point_world，点的世界坐标X，Y，Z
//             vAllMatRefFrame = matRefFrame;
//             vAllLabels = vLabels;
//             vAllDepthRefFrame = matDepthRefFrame;
//         }
//         else
//         {   // 不是第一次进来
//             if (!vMPw.empty())
//             {   // hconcat水平拼接,vconcat垂直拼接 
//                 //vMPw=[4行n列]，matRefFrame=[n行1列] 
//                 hconcat(vAllMPw,vMPw,vAllMPw);// 继续在之前的静态点基础上叠加
//                 //  vAllMPw(world)  = [ X1/Z1   X2/Z2   .... 
//                 //                      Y1/Z1   Y2/Z2   ....
//                 //                      1         1     .... 
//                 //                      1/Z1    1/Z2     .... ] 世界坐标系下的坐标
//                 vconcat(vAllMatRefFrame,matRefFrame,vAllMatRefFrame);
//                 vconcat(vAllLabels,vLabels,vAllLabels);
//                 vconcat(vAllDepthRefFrame,matDepthRefFrame,vAllDepthRefFrame);
//             }
//         }
//     }
//     //Part3：在世界坐标系、当前坐标系之间进行3D坐标或者像素坐标的转换，获取从KF投影到CF的深度z_proj和CF的直接深度z'
//     // 遍历完了所有参考帧  for (int i(0); i < mnRefFrames; i++)
//     cv::Mat vLabels = vAllLabels;

//     if (!vAllMPw.empty())
//     {
//         // 世界坐标系下三维点[X/Z，Y/Z，Z/Z，1/Z]转换到当前帧坐标系
//         // [X/Z, Y/Z ,1, 1/Z] ,还是归一化的坐标，没乘以深度
//         // cout << vAllMPw.rowRange(0,4).col(0);
//         // getchar();
//         cv::Mat vMPCurrentFrame = currentFrame.mTcw * vAllMPw;

//         // Divide by last column
//         // vMPCurrentFrame：rows=4,cols=h,就是符合要求的点的数量
//         for (int i(0); i < vMPCurrentFrame.cols; i++)
//         {// 除以逆深度=乘以深度，变成当前帧下【X，Y，Z，1】坐标 【rows=4,cols=h]
//             vMPCurrentFrame.at<float>(0,i) /= vMPCurrentFrame.at<float>(3,i);
//             vMPCurrentFrame.at<float>(1,i) /= vMPCurrentFrame.at<float>(3,i);
//             vMPCurrentFrame.at<float>(2,i) /= vMPCurrentFrame.at<float>(3,i);
//             vMPCurrentFrame.at<float>(3,i) /= vMPCurrentFrame.at<float>(3,i);
//         }
//         // 投影深度就是取出了所有点投影过来的深度z_proj [rows=1,cols=n]
//         cv::Mat matProjDepth = vMPCurrentFrame.row(2);

//         cv::Mat _vMPCurrentFrame = cv::Mat(vMPCurrentFrame.size(),CV_32F);
//         cv::Mat _vAllMatRefFrame = cv::Mat(vAllMatRefFrame.size(),CV_32F);
//         cv::Mat _vLabels = cv::Mat(vLabels.size(),CV_32F);
//         cv::Mat __vAllDepthRefFrame = cv::Mat(vAllDepthRefFrame.size(),CV_32F);
//         int h(0);
//         cv::Mat __matProjDepth = cv::Mat(matProjDepth.size(),CV_32F);
//         for (int i(0); i < matProjDepth.cols; i++)// cols,符合条件的特征点的数量
//         {
//             if (matProjDepth.at<float>(0,i) < 7)// 如果这个点的深度<7
//             {
//                 __matProjDepth.at<float>(0,h) = matProjDepth.at<float>(0,i);

//                 _vMPCurrentFrame.at<float>(0,h) = vMPCurrentFrame.at<float>(0,i);
//                 _vMPCurrentFrame.at<float>(1,h) = vMPCurrentFrame.at<float>(1,i);
//                 _vMPCurrentFrame.at<float>(2,h) = vMPCurrentFrame.at<float>(2,i);
//                 _vMPCurrentFrame.at<float>(3,h) = vMPCurrentFrame.at<float>(3,i);

//                 _vAllMatRefFrame.at<float>(h,0) = vAllMatRefFrame.at<float>(i,0);
//                 _vAllMatRefFrame.at<float>(h,1) = vAllMatRefFrame.at<float>(i,1);
//                 _vAllMatRefFrame.at<float>(h,2) = vAllMatRefFrame.at<float>(i,2);

//                 _vLabels.at<float>(h,0) = vLabels.at<float>(i,0);

//                 __vAllDepthRefFrame.at<float>(h,0) = vAllDepthRefFrame.at<float>(i,0);

//                 h++;
//                 if(h==0)
//                     continue;
//             }
//         }

//         matProjDepth = __matProjDepth.colRange(0,h);
//         vMPCurrentFrame = _vMPCurrentFrame.colRange(0,h);
//         vAllMatRefFrame = _vAllMatRefFrame.rowRange(0,h);
//         vLabels = _vLabels.rowRange(0,h);
//         vAllDepthRefFrame = __vAllDepthRefFrame.rowRange(0,h);

//         cv::Mat aux;
//         cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);
//         // 当前帧下的3D归一化坐标变换回当前帧相机的 ( 像素坐标*Z)【rows=3,cols=h]
//         // vMPCurrentFrame：当前帧坐标系下的坐标[X, Y, Z, 1]
//         // matCurrentFrame：[u*Z, v*Z, 1*Z]
//         cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame;
//         // 下面这些是临时变量
//         cv::Mat mat2CurrentFrame(matCurrentFrame.cols,2,CV_32F);
//         cv::Mat v2AllMatRefFrame(matCurrentFrame.cols,3,CV_32F);
//         cv::Mat mat2ProjDepth(matCurrentFrame.cols,1,CV_32F);
//         cv::Mat v2Labels(matCurrentFrame.cols,1,CV_32F);
//         cv::Mat _vAllDepthRefFrame(matCurrentFrame.cols,1,CV_32F);

//         int j = 0;
//         for (int i(0); i < matCurrentFrame.cols; i++)
//         {// x,y是像素坐标
//             float x = ceil(matCurrentFrame.at<float>(0,i)/matCurrentFrame.at<float>(2,i));
//             float y = ceil(matCurrentFrame.at<float>(1,i)/matCurrentFrame.at<float>(2,i));
//             /*IsInFrame 判断是否在当前帧一个范围之内*/
//             /*图像边缘部分的点不纳入考虑*/
//             /*mDmax = 20; return (x > (mDmax + 1) && x < (Frame.mImDepth.cols - mDmax - 1) && y > (mDmax + 1) && y < (Frame.mImDepth.rows - mDmax - 1));*/
//             if (IsInFrame(x,y,currentFrame))
//             {
//                 const float d = currentFrame.mImDepth.at<float>(y,x);
//                 if (d > 0)// 如果有深度（有些地方测不到深度
//                 {   // 把信息存一存 
//                     mat2CurrentFrame.at<float>(j,0) = x;
//                     mat2CurrentFrame.at<float>(j,1) = y;
//                     v2AllMatRefFrame.at<float>(j,0) = vAllMatRefFrame.at<float>(i,0);
//                     v2AllMatRefFrame.at<float>(j,1) = vAllMatRefFrame.at<float>(i,1);
//                     v2AllMatRefFrame.at<float>(j,2) = vAllMatRefFrame.at<float>(i,2);
//                     _vAllDepthRefFrame.at<float>(j,0) = vAllDepthRefFrame.at<float>(i,0);
//                     // 点在当前坐标系下的深度
//                     float d1 = matProjDepth.at<float>(0,i);
//                     mat2ProjDepth.at<float>(j,0) = d1;
//                     v2Labels.at<float>(j,0) = vLabels.at<float>(i,0);
//                     j++;
//                 }
//             }
//         }
//         // 筛选符合条件d<7的特征点完毕，赋值
//         // 在这里matCurrentFrame变成了当前帧坐标系下的像素坐标[u,v,1]
//         vAllDepthRefFrame = _vAllDepthRefFrame.rowRange(0,j);
//         vAllMatRefFrame = v2AllMatRefFrame.rowRange(0,j);
//         matProjDepth = mat2ProjDepth.rowRange(0,j);
//         matCurrentFrame = mat2CurrentFrame.rowRange(0,j);
//         vLabels = v2Labels.rowRange(0,j);

//         cv::Mat u1((2*mDmax+1)*(2*mDmax+1),2,CV_32F);
//         int m(0);
//         for (int i(-mDmax); i <= mDmax; i++){
//             for (int j(-mDmax); j <= mDmax; j++){
//                 u1.at<float>(m,0) = i;
//                 u1.at<float>(m,1) = j;
//                 m++;// 存储一个patch的x,y坐标
//             }
//         }

//         cv::Mat matDepthCurrentFrame(matCurrentFrame.rows,1,CV_32F);
//         cv::Mat _matProjDepth(matCurrentFrame.rows,1,CV_32F);
//         cv::Mat _matCurrentFrame(matCurrentFrame.rows,2,CV_32F);

//         int _s(0);
//         for (int i(0); i < matCurrentFrame.rows; i++)// 对每一个点作操作
//         {
//             int s(0);
//             // z_projet ，深度偏差
//             cv::Mat _matDiffDepth(u1.rows,1,CV_32F);
//             cv::Mat _matDepth(u1.rows,1,CV_32F);

//             for (int j(0); j < u1.rows; j++)
//             {// matCurrentFrame：当前帧坐标系下的像素坐标[u,v,1] ，rows=3,cols=m
//                 int x = (int)matCurrentFrame.at<float>(i,0) + (int)u1.at<float>(j,0);
//                 int y = (int)matCurrentFrame.at<float>(i,1) + (int)u1.at<float>(j,1);
//                 float _d = currentFrame.mImDepth.at<float>(y,x);
//                 // 看点周围一个patch的深度是否>0并且比中心点的深度小
//                 if ((_d > 0) && (_d < matProjDepth.at<float>(i,0)))
//                 {
//                     // 符合条件就存储这个深度并且计算深度差
//                     _matDepth.at<float>(s,0) = _d;
//                     _matDiffDepth.at<float>(s,0) = matProjDepth.at<float>(i,0) - _d;
//                     s++;
//                 }
//             }
//             if (s > 0)// 如果patch内存在有深度并且深度值小于当前点深度的像素
//             {
//                 _matDepth = _matDepth.rowRange(0,s);
//                 _matDiffDepth = _matDiffDepth.rowRange(0,s);
//                 double minVal, maxVal;
//                 cv::Point minIdx, maxIdx;
//                 cv::minMaxLoc(_matDiffDepth,&minVal,&maxVal,&minIdx,&maxIdx);
//                 // 找到这个特征点周围深度差最小的点的x,y坐标
//                 int xIndex = minIdx.x;
//                 int yIndex = minIdx.y;
//                 // 把深度差最小的点的深度存起来(因为深度差大的认为是动态点)
//                 // matDepthCurrentFrame存储的是用T投影过来的深度
//                 // 这时候把当前帧投影深度设置为更小的一个深度，但差别不大
//                 matDepthCurrentFrame.at<float>(_s,0) = _matDepth.at<float>(yIndex,0);
//                 _matProjDepth.at<float>(_s,0) = matProjDepth.at<float>(i,0);
//                 _matCurrentFrame.at<float>(_s,0) = matCurrentFrame.at<float>(i,0);
//                 _matCurrentFrame.at<float>(_s,1) = matCurrentFrame.at<float>(i,1);
//                 _s++;
//             }
//         }

//     // matProjDepth：投影过来的深度（已作值修改）z_proj
//         // matDepthCurrentFrame ：当前帧下直接取的深度z'
//         //  matCurrentFrame：当前帧坐标系下的像素坐标[u,v,1] ，rows=3,cols=_s
//         matDepthCurrentFrame = matDepthCurrentFrame.rowRange(0,_s);
//         matProjDepth = _matProjDepth.rowRange(0,_s);
//         matCurrentFrame = _matCurrentFrame.rowRange(0,_s);
// //Part4：计算dz = z_proj-z'，以及周围一个patch的深度方差，由此最终筛选动态点
//         //RGBD 深度阈值
//         mDepthThreshold = 0.6;
//         // 对应论文里的dz = z_proj-z'
//         cv::Mat matDepthDifference = matProjDepth - matDepthCurrentFrame;
//     // 方差的阈值
//         mVarThreshold = 0.001; //0.040;

//         vector<Geometry::DynKeyPoint> vDynPoints;

//         for (int i(0); i < matCurrentFrame.rows; i++)
//         {
//             if (matDepthDifference.at<float>(i,0) > mDepthThreshold)// 大于深度阈值
//             {   // 计算该点周围一个Patch内的方差，mDmax=20
//                 int xIni = (int)matCurrentFrame.at<float>(i,0) - mDmax;
//                 int yIni = (int)matCurrentFrame.at<float>(i,1) - mDmax;
//                 int xEnd = (int)matCurrentFrame.at<float>(i,0) + mDmax + 1;
//                 int yEnd = (int)matCurrentFrame.at<float>(i,1) + mDmax + 1;
//                 cv::Mat patch = currentFrame.mImDepth.rowRange(yIni,yEnd).colRange(xIni,xEnd);
//                 // 计算patch内深度的均值和方差
//                 // 对应论文中【如果关键点被设置为动态，但是它周围深度地图具有较大方差，
//                 // 我们改变编号为静态。】的部分
//                 cv::Mat mean, stddev;
//                 cv::meanStdDev(patch,mean,stddev);
//                 double _stddev = stddev.at<double>(0,0);
//                 double var = _stddev*_stddev;
//                 // 如果patch内的方差不大，认为是动态点
//                 if (var < mVarThreshold)
//                 {
//                     DynKeyPoint dynPoint;
//                     dynPoint.mPoint.x = matCurrentFrame.at<float>(i,0);
//                     dynPoint.mPoint.y = matCurrentFrame.at<float>(i,1);
//                     dynPoint.mRefFrameLabel = vLabels.at<float>(i,0);
//                     vDynPoints.push_back(dynPoint);
//                 }
//             }
//         }

//         return vDynPoints;
//     }
//     else
//     {
//         vector<Geometry::DynKeyPoint> vDynPoints;
//         return vDynPoints;
//     }
// }


cv::Mat Geometry::DepthRegionGrowing(const vector<DynKeyPoint> &vDynPoints,const cv::Mat &imDepth){

    cv::Mat maskG = cv::Mat::zeros(480,640,CV_32F);

    if (!vDynPoints.empty())
    {
        mSegThreshold = 0.20;

        for (size_t i(0); i < vDynPoints.size(); i++){
            int xSeed = vDynPoints[i].mPoint.x;
            int ySeed = vDynPoints[i].mPoint.y;
            const float d = imDepth.at<float>(ySeed,xSeed);
            if (maskG.at<float>(ySeed,xSeed)!=1. && d > 0)
            {
                cv::Mat J = RegionGrowing(imDepth,xSeed,ySeed,mSegThreshold);
                maskG = maskG | J;
            }
        }

        int dilation_size = 15;
        cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                               cv::Point( dilation_size, dilation_size ) );
        maskG.cv::Mat::convertTo(maskG,CV_8U);
        cv::dilate(maskG, maskG, kernel);
    }
    else
    {
        maskG.cv::Mat::convertTo(maskG,CV_8U);
    }

    cv::Mat _maskG = cv::Mat::ones(480,640,CV_8U);
    maskG = _maskG - maskG;

    return maskG;
}



void Geometry::CombineMasks(const ORB_SLAM2::Frame &currentFrame, cv::Mat &mask)
{
    cv::Mat _maskL = cv::Mat::ones(currentFrame.mImMask.size(),currentFrame.mImMask.type());
    _maskL = _maskL - currentFrame.mImMask;

    cv::Mat _maskG = cv::Mat::ones(mask.size(),mask.type());
    _maskG = _maskG - mask;

    cv::Mat _mask = _maskL | _maskG;

    cv::Mat __mask = cv::Mat::ones(_mask.size(),_mask.type());
    __mask = __mask - _mask;
    mask = __mask;

}

float Area(float x1, float x2, float y1, float y2){
    float xc1 = max(x1-0.5,x2-0.5);
    float xc2 = min(x1+0.5,x2+0.5);
    float yc1 = max(y1-0.5,y2-0.5);
    float yc2 = min(y1+0.5,y2+0.5);
    return (xc2-xc1)*(yc2-yc1);
}

void Geometry::FillRGBD(const ORB_SLAM2::Frame &currentFrame,cv::Mat &mask,cv::Mat &imGray,cv::Mat &imDepth){

    cv::Mat imGrayAccumulator = imGray.mul(mask);
    imGrayAccumulator.convertTo(imGrayAccumulator,CV_32F);
    cv::Mat imCounter;
    mask.convertTo(imCounter,CV_32F);
    cv::Mat imDepthAccumulator = imDepth.mul(imCounter);
    imDepthAccumulator.convertTo(imDepthAccumulator,CV_32F);
    cv::Mat imMinDepth = cv::Mat::zeros(imDepth.size(),CV_32F)+100.0;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = currentFrame.fx;
    K.at<float>(1,1) = currentFrame.fy;
    K.at<float>(0,2) = currentFrame.cx;
    K.at<float>(1,2) = currentFrame.cy;

    for (int i(0); i < mDB.mNumElem; i++){

        ORB_SLAM2::Frame refFrame = mDB.mvDataBase[i];

        cv::Mat vPixels(640*480,2,CV_32F);
        cv::Mat mDepth(640*480,1,CV_32F);

        int n(0);
        for (int j(0); j < 640*480; j++){
            int x = (int)vAllPixels.at<float>(j,0);
            int y = (int)vAllPixels.at<float>(j,1);
            if ((int)refFrame.mImMask.at<uchar>(y,x) == 1){
                const float d = refFrame.mImDepth.at<float>(y,x);
                if (d > 0 && d < 7){
                    vPixels.at<float>(n,0) = vAllPixels.at<float>(j,0);
                    vPixels.at<float>(n,1) = vAllPixels.at<float>(j,1);
                    mDepth.at<float>(n,0) = 1./d;
                    n++;
                }
            }
        }

        vPixels = vPixels.rowRange(0,n);
        mDepth = mDepth.rowRange(0,n);
        hconcat(vPixels,cv::Mat::ones(n,1,CV_32F),vPixels);
        cv::Mat vMPRefFrame = K.inv() * vPixels.t();
        vconcat(vMPRefFrame,mDepth.t(),vMPRefFrame);

        cv::Mat vMPw = refFrame.mTcw.inv() * vMPRefFrame;
        cv::Mat vMPCurrentFrame = currentFrame.mTcw * vMPw;

        // Divide by last column
        for (int j(0); j < vMPCurrentFrame.cols; j++)
        {
            vMPCurrentFrame.at<float>(0,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(1,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(2,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(3,j) /= vMPCurrentFrame.at<float>(3,j);
        }

        cv::Mat matProjDepth = vMPCurrentFrame.row(2);
        cv::Mat aux;
        cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);
        cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame;

        cv::Mat vProjPixels(matCurrentFrame.cols,2,CV_32F);
        cv::Mat _matProjDepth(matCurrentFrame.cols,1,CV_32F);
        cv::Mat _vPixels(matCurrentFrame.cols,2,CV_32F);

        int p(0);
        for (int j(0); j < matCurrentFrame.cols; j++)
        {
            float x = matCurrentFrame.at<float>(0,j)/matCurrentFrame.at<float>(2,j);
            float y = matCurrentFrame.at<float>(1,j)/matCurrentFrame.at<float>(2,j);
            bool inFrame = (x > 1 && x < (currentFrame.mImDepth.cols - 1) && y > 1 && y < (currentFrame.mImDepth.rows - 1));
            if (inFrame && (mask.at<uchar>(y,x) == 0)){
                vProjPixels.at<float>(p,0) = x;
                vProjPixels.at<float>(p,1) = y;
                _matProjDepth.at<float>(p,0) = matProjDepth.at<float>(0,j);
                _vPixels.at<float>(p,0) = vPixels.at<float>(j,0);
                _vPixels.at<float>(p,1) = vPixels.at<float>(j,1);
                p++;
            }
        }
        vProjPixels = vProjPixels.rowRange(0,p);
        matProjDepth = _matProjDepth.rowRange(0,p);
        vPixels = _vPixels.rowRange(0,p);

        for (int j(0); j< p; j++)
        {


            int _x = (int)vPixels.at<float>(j,0);
            int _y = (int)vPixels.at<float>(j,1);
            float x = vProjPixels.at<float>(j,0);//x of *
            float y = vProjPixels.at<float>(j,1);//y of *
            /*
                -----------
                | A  | B  |
                ----*------ y
                | C  | D  |
                -----------
                     x
            */
            float x_a = floor(x);
            float y_a = floor(y);
            float x_b = ceil(x);
            float y_b = floor(y);
            float x_c = floor(x);
            float y_c = ceil(y);
            float x_d = ceil(x);
            float y_d = ceil(y);

            float weight = 0;

            if( IsInImage(x_a,y_a,imGrayAccumulator)){
                if(abs(imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a))+=weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a))+=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a))+=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                else if ((imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a))=weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a))=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a))=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                imMinDepth.at<float>(y_a,x_a) = min(imMinDepth.at<float>(y_a,x_a),matProjDepth.at<float>(j,0));
            }
            if( IsInImage(x_b,y_b,imGrayAccumulator) && (x_a != x_b))
            {
                if(abs(imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b))+=weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b))+=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b))+=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                else if ((imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b)) = weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                imMinDepth.at<float>(y_b,x_b) = min(imMinDepth.at<float>(y_b,x_b),matProjDepth.at<float>(j,0));
            }
            if( IsInImage(x_c,y_c,imGrayAccumulator) && (y_a != y_c) && (x_b != x_c && y_b != y_c))
            {
                if(abs(imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c))+=weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c))+=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c))+=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                else if ((imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c)) = weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                imMinDepth.at<float>(y_c,x_c) = min(imMinDepth.at<float>(y_c,x_c),matProjDepth.at<float>(j,0));

            }
            if( IsInImage(x_d,y_d,imGrayAccumulator) && (x_a != x_d && y_a != y_d) && (y_b != y_d) && (x_d != x_c))
            {
                if (abs(imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d))+=weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d))+=weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d))+=weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                else if ((imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d)) = weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                imMinDepth.at<float>(y_d,x_d) = min(imMinDepth.at<float>(y_d,x_d),matProjDepth.at<float>(j,0));
            }
        }
    }

    imGrayAccumulator = imGrayAccumulator.mul(1/imCounter);
    imDepthAccumulator = imDepthAccumulator.mul(1/imCounter);

    imGrayAccumulator.convertTo(imGrayAccumulator,CV_8U);
    imGray = imGray*0;
    imGrayAccumulator.copyTo(imGray,mask);
    imDepth = imDepth*0;
    imDepthAccumulator.copyTo(imDepth,mask);

}

void Geometry::FillRGBD(const ORB_SLAM2::Frame &currentFrame,cv::Mat &mask,cv::Mat &imGray,cv::Mat &imDepth,cv::Mat &imRGB){

    cv::Mat imGrayAccumulator = imGray.mul(mask);
    imGrayAccumulator.convertTo(imGrayAccumulator,CV_32F);
    cv::Mat bgr[3];
    cv::split(imRGB,bgr);
    cv::Mat imRAccumulator = bgr[2].mul(mask);
    imRAccumulator.convertTo(imRAccumulator,CV_32F);
    cv::Mat imGAccumulator = bgr[1].mul(mask);
    imGAccumulator.convertTo(imGAccumulator,CV_32F);
    cv::Mat imBAccumulator = bgr[0].mul(mask);
    imBAccumulator.convertTo(imBAccumulator,CV_32F);
    cv::Mat imCounter;
    mask.convertTo(imCounter,CV_32F);
    cv::Mat imDepthAccumulator = imDepth.mul(imCounter);
    imDepthAccumulator.convertTo(imDepthAccumulator,CV_32F);
    cv::Mat imMinDepth = cv::Mat::zeros(imDepth.size(),CV_32F)+100.0;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = currentFrame.fx;
    K.at<float>(1,1) = currentFrame.fy;
    K.at<float>(0,2) = currentFrame.cx;
    K.at<float>(1,2) = currentFrame.cy;

    for (int i(0); i < mDB.mNumElem; i++){

        ORB_SLAM2::Frame refFrame = mDB.mvDataBase[i];
        cv::Mat bgr[3];
        cv::split(refFrame.mImRGB,bgr);
        cv::Mat imR = bgr[2];
        cv::Mat imG = bgr[1];
        cv::Mat imB = bgr[0];

        cv::Mat vPixels(640*480,2,CV_32F);
        cv::Mat mDepth(640*480,1,CV_32F);

        int n(0);
        for (int j(0); j < 640*480; j++){
            int x = (int)vAllPixels.at<float>(j,0);
            int y = (int)vAllPixels.at<float>(j,1);
            if ((int)refFrame.mImMask.at<uchar>(y,x) == 1){
                const float d = refFrame.mImDepth.at<float>(y,x);
                if (d > 0){
                    vPixels.at<float>(n,0) = vAllPixels.at<float>(j,0);
                    vPixels.at<float>(n,1) = vAllPixels.at<float>(j,1);
                    mDepth.at<float>(n,0) = 1./d;
                    n++;
                }
            }
        }

        vPixels = vPixels.rowRange(0,n);
        mDepth = mDepth.rowRange(0,n);
        hconcat(vPixels,cv::Mat::ones(n,1,CV_32F),vPixels);
        cv::Mat vMPRefFrame = K.inv() * vPixels.t();
        vconcat(vMPRefFrame,mDepth.t(),vMPRefFrame);

        cv::Mat vMPw = refFrame.mTcw.inv() * vMPRefFrame;
        cv::Mat vMPCurrentFrame = currentFrame.mTcw * vMPw;

        // Divide by last column
        for (int j(0); j < vMPCurrentFrame.cols; j++)
        {
            vMPCurrentFrame.at<float>(0,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(1,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(2,j) /= vMPCurrentFrame.at<float>(3,j);
            vMPCurrentFrame.at<float>(3,j) /= vMPCurrentFrame.at<float>(3,j);
        }

        cv::Mat matProjDepth = vMPCurrentFrame.row(2);
        cv::Mat aux;
        cv::hconcat(cv::Mat::eye(3,3,CV_32F),cv::Mat::zeros(3,1,CV_32F),aux);
        cv::Mat matCurrentFrame = K*aux*vMPCurrentFrame;

        cv::Mat vProjPixels(matCurrentFrame.cols,2,CV_32F);
        cv::Mat _matProjDepth(matCurrentFrame.cols,1,CV_32F);
        cv::Mat _vPixels(matCurrentFrame.cols,2,CV_32F);

        int p(0);
        for (int j(0); j < matCurrentFrame.cols; j++)
        {
            float x = matCurrentFrame.at<float>(0,j)/matCurrentFrame.at<float>(2,j);
            float y = matCurrentFrame.at<float>(1,j)/matCurrentFrame.at<float>(2,j);
            bool inFrame = (x > 1 && x < (currentFrame.mImDepth.cols - 1) && y > 1 && y < (currentFrame.mImDepth.rows - 1));
            if (inFrame && (mask.at<uchar>(y,x) == 0))
            {
                vProjPixels.at<float>(p,0) = x;
                vProjPixels.at<float>(p,1) = y;
                _matProjDepth.at<float>(p,0) = matProjDepth.at<float>(0,j);
                _vPixels.at<float>(p,0) = vPixels.at<float>(j,0);
                _vPixels.at<float>(p,1) = vPixels.at<float>(j,1);
                p++;
            }
        }
        vProjPixels = vProjPixels.rowRange(0,p);
        matProjDepth = _matProjDepth.rowRange(0,p);
        vPixels = _vPixels.rowRange(0,p);

        for (int j(0); j< p; j++)
        {


            int _x = (int)vPixels.at<float>(j,0);
            int _y = (int)vPixels.at<float>(j,1);
            float x = vProjPixels.at<float>(j,0);//x of *
            float y = vProjPixels.at<float>(j,1);//y of *
            /*
                -----------
                | A  | B  |
                ----*------ y
                | C  | D  |
                -----------
                     x
            */
            float x_a = floor(x);
            float y_a = floor(y);
            float x_b = ceil(x);
            float y_b = floor(y);
            float x_c = floor(x);
            float y_c = ceil(y);
            float x_d = ceil(x);
            float y_d = ceil(y);

            float weight = 0;

            if( IsInImage(x_a,y_a,imGrayAccumulator)){
                if(abs(imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a)) += weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_a),int(x_a)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                else if ((imMinDepth.at<float>(y_a,x_a)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_a,y,y_a);
                    imCounter.at<float>(int(y_a),int(x_a)) = weight;
                    imGrayAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_a),int(x_a)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_a),int(x_a)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_a),int(x_a)) = 1;
                }
                imMinDepth.at<float>(y_a,x_a) = min(imMinDepth.at<float>(y_a,x_a),matProjDepth.at<float>(j,0));
            }

            if( IsInImage(x_b,y_b,imGrayAccumulator) && (x_a != x_b))
            {
                if(abs(imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b)) += weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_b),int(x_b)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                else if ((imMinDepth.at<float>(y_b,x_b)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_b,y,y_b);
                    imCounter.at<float>(int(y_b),int(x_b)) = weight;
                    imGrayAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_b),int(x_b)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_b),int(x_b)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_b),int(x_b)) = 1;
                }
                imMinDepth.at<float>(y_b,x_b) = min(imMinDepth.at<float>(y_b,x_b),matProjDepth.at<float>(j,0));
            }
            if( IsInImage(x_c,y_c,imGrayAccumulator) && (y_a != y_c) && (x_b != x_c && y_b != y_c))
            {
                if(abs(imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c)) += weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_c),int(x_c)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                else if ((imMinDepth.at<float>(y_c,x_c)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_c,y,y_c);
                    imCounter.at<float>(int(y_c),int(x_c)) = weight;
                    imGrayAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_c),int(x_c)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_c),int(x_c)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_c),int(x_c)) = 1;
                }
                imMinDepth.at<float>(y_c,x_c) = min(imMinDepth.at<float>(y_c,x_c),matProjDepth.at<float>(j,0));

            }
            if( IsInImage(x_d,y_d,imGrayAccumulator) && (x_a != x_d && y_a != y_d) && (y_b != y_d) && (x_d != x_c))
            {
                if (abs(imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) < MIN_DEPTH_THRESHOLD )
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d)) += weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_d),int(x_d)) += weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d)) += weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                else if ((imMinDepth.at<float>(y_d,x_d)-matProjDepth.at<float>(j,0)) > 0)
                {
                    weight = Area(x,x_d,y,y_d);
                    imCounter.at<float>(int(y_d),int(x_d)) = weight;
                    imGrayAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)refFrame.mImGray.at<uchar>(_y,_x);
                    imRAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imR.at<uchar>(_y,_x);
                    imGAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imG.at<uchar>(_y,_x);
                    imBAccumulator.at<float>(int(y_d),int(x_d)) = weight*(float)imB.at<uchar>(_y,_x);
                    imDepthAccumulator.at<float>(int(y_d),int(x_d)) = weight*matProjDepth.at<float>(j,0);
                    mask.at<uchar>(int(y_d),int(x_d)) = 1;
                }
                imMinDepth.at<float>(y_d,x_d) = min(imMinDepth.at<float>(y_d,x_d),matProjDepth.at<float>(j,0));
            }
        }
    }

    imGrayAccumulator = imGrayAccumulator.mul(1/imCounter);
    imRAccumulator = imRAccumulator.mul(1/imCounter);
    imRAccumulator.convertTo(imRAccumulator,CV_8U);
    cv::Mat imR = cv::Mat::zeros(imRAccumulator.size(),imRAccumulator.type());
    imRAccumulator.copyTo(imR,mask);
    imGAccumulator = imGAccumulator.mul(1/imCounter);
    imGAccumulator.convertTo(imGAccumulator,CV_8U);
    cv::Mat imG = cv::Mat::zeros(imGAccumulator.size(),imGAccumulator.type());
    imGAccumulator.copyTo(imG,mask);
    imBAccumulator = imBAccumulator.mul(1/imCounter);
    imBAccumulator.convertTo(imBAccumulator,CV_8U);
    cv::Mat imB = cv::Mat::zeros(imBAccumulator.size(),imBAccumulator.type());
    imBAccumulator.copyTo(imB,mask);
    imDepthAccumulator = imDepthAccumulator.mul(1/imCounter);

    std::vector<cv::Mat> arrayToMerge;
    arrayToMerge.push_back(imB);
    arrayToMerge.push_back(imG);
    arrayToMerge.push_back(imR);
    cv::merge(arrayToMerge, imRGB);

    imGrayAccumulator.convertTo(imGrayAccumulator,CV_8U);
    imGray = imGray*0;
    imGrayAccumulator.copyTo(imGray,mask);
    imDepth = imDepth*0;
    imDepthAccumulator.copyTo(imDepth,mask);

}

void Geometry::GetClosestNonEmptyCoordinates(const cv::Mat &mask, const int &x, const int &y, int &_x, int &_y)
{
    cv::Mat neigbIni(4,2,CV_32F);
    neigbIni.at<float>(0,0) = -1;
    neigbIni.at<float>(0,1) = 0;
    neigbIni.at<float>(1,0) = 1;
    neigbIni.at<float>(1,1) = 0;
    neigbIni.at<float>(2,0) = 0;
    neigbIni.at<float>(2,1) = -1;
    neigbIni.at<float>(3,0) = 0;
    neigbIni.at<float>(3,1) = 1;

    cv::Mat neigb = neigbIni;

    bool found = false;
    int f(2);

    while (!found)
    {
        for (int j(0); j< 4; j++)
        {
            int xn = x + neigb.at<float>(j,0);
            int yn = y + neigb.at<float>(j,1);
            bool ins = ((xn >= 0) && (yn >= 0) && (xn <= mask.cols) && (yn <= mask.rows));
            if (ins && ((int)mask.at<uchar>(yn,xn) == 1))
            {
                found = true;
                _x = xn;
                _y = yn;
            }
        }
        neigb = f*neigbIni;
        f++;
    }

}


void Geometry::DataBase::InsertFrame2DB(const ORB_SLAM2::Frame &currentFrame){

    if (!IsFull()){
        mvDataBase[mFin] = currentFrame;
        mFin = (mFin + 1) % MAX_DB_SIZE;
        mNumElem += 1;
    }
    else {
        mvDataBase[mIni] = currentFrame;
        mFin = mIni;
        mIni = (mIni + 1) % MAX_DB_SIZE;
    }
}

bool Geometry::DataBase::IsFull(){
    return (mIni == (mFin+1) % MAX_DB_SIZE);
}

cv::Mat Geometry::rotm2euler(const cv::Mat &R){
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6;
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    cv::Mat res = (cv::Mat_<double>(1,3) << x, y, z);
    return res;
}


bool Geometry::isRotationMatrix(const cv::Mat &R){
    cv::Mat Rt;
    transpose(R,Rt);
    cv::Mat shouldBeIdentity = Rt*R;
    cv::Mat I = cv::Mat::eye(3,3,shouldBeIdentity.type());
    return norm(I,shouldBeIdentity) < 1e-6;
}


bool Geometry::IsInFrame(const float &x, const float &y, const ORB_SLAM2::Frame &Frame)
{
    mDmax = 20;
    return (x > (mDmax + 1) && x < (Frame.mImDepth.cols - mDmax - 1) && y > (mDmax + 1) && y < (Frame.mImDepth.rows - mDmax - 1));
}
bool Geometry::IsInImage(const float &x, const float &y, const cv::Mat image)
{
    return (x >= 0 && x < (image.cols) && y >= 0 && y < image.rows);
}

cv::Mat Geometry::RegionGrowing(const cv::Mat &im,int &x,int &y,const float &reg_maxdist){

    cv::Mat J = cv::Mat::zeros(im.size(),CV_32F);

    float reg_mean = im.at<float>(y,x);
    int reg_size = 1;

    int _neg_free = 10000;
    int neg_free = 10000;
    int neg_pos = -1;
    cv::Mat neg_list = cv::Mat::zeros(neg_free,3,CV_32F);

    double pixdist=0;

    //Neighbor locations (footprint)
    cv::Mat neigb(4,2,CV_32F);
    neigb.at<float>(0,0) = -1;
    neigb.at<float>(0,1) = 0;
    neigb.at<float>(1,0) = 1;
    neigb.at<float>(1,1) = 0;
    neigb.at<float>(2,0) = 0;
    neigb.at<float>(2,1) = -1;
    neigb.at<float>(3,0) = 0;
    neigb.at<float>(3,1) = 1;

    while(pixdist < reg_maxdist && reg_size < im.total())
    {
        for (int j(0); j< 4; j++)
        {
            //Calculate the neighbour coordinate
            int xn = x + neigb.at<float>(j,0);
            int yn = y + neigb.at<float>(j,1);

            bool ins = ((xn >= 0) && (yn >= 0) && (xn < im.cols) && (yn < im.rows));
            if (ins && (J.at<float>(yn,xn) == 0.))
            {
                neg_pos ++;
                neg_list.at<float>(neg_pos,0) = xn;
                neg_list.at<float>(neg_pos,1) = yn;
                neg_list.at<float>(neg_pos,2) = im.at<float>(yn,xn);
                J.at<float>(yn,xn) = 1.;
            }
        }

        // Add a new block of free memory
        if((neg_pos + 10) > neg_free){
            cv::Mat _neg_list = cv::Mat::zeros(_neg_free,3,CV_32F);
            neg_free += 10000;
            vconcat(neg_list,_neg_list,neg_list);
        }

        // Add pixel with intensity nearest to the mean of the region, to the region
        cv::Mat dist;
        for (int i(0); i < neg_pos; i++){
            double d = abs(neg_list.at<float>(i,2) - reg_mean);
            dist.push_back(d);
        }
        double max;
        cv::Point ind, maxpos;
        cv::minMaxLoc(dist, &pixdist, &max, &ind, &maxpos);
        int index = ind.y;

        if (index != -1)
        {
            J.at<float>(y,x) = -1.;
            reg_size += 1;

            // Calculate the new mean of the region
            reg_mean = (reg_mean*reg_size + neg_list.at<float>(index,2))/(reg_size+1);

            // Save the x and y coordinates of the pixel (for the neighbour add proccess)
            x = neg_list.at<float>(index,0);
            y = neg_list.at<float>(index,1);

            // Remove the pixel from the neighbour (check) list
            neg_list.at<float>(index,0) = neg_list.at<float>(neg_pos,0);
            neg_list.at<float>(index,1) = neg_list.at<float>(neg_pos,1);
            neg_list.at<float>(index,2) = neg_list.at<float>(neg_pos,2);
            neg_pos -= 1;
        }
        else
        {
            pixdist = reg_maxdist;
        }

    }

    J = cv::abs(J);
    return(J);
}

}
