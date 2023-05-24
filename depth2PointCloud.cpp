#include <unistd.h>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/common/time.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include<opencv2/opencv.hpp>
#include<vector>
#include<fstream>

//#define REALSENSE_ON
//#define SPATIAL_FILTER
#define CALIB_PATH "../wall/calib.yaml"
#define IMAGE_PATH "../wall/"
#define WIDTH  (320)
#define HEIGHT (240)
#define Rectify

using namespace cv;
using namespace std;

cv::Mat left_matrix;
cv::Mat left_coeff;
cv::Mat right_matrix;
cv::Mat right_coeff;
cv::Mat R;
cv::Mat T;
cv::Mat Rl;
cv::Mat Rr;
cv::Mat Pl;
cv::Mat Pr;
cv::Mat Q;
double BF;

//#define CLOUD_FIlTER
//#define CLOUD_PLANE

bool getStereoCameraParams()
{
	FileStorage fs(CALIB_PATH, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["left_matrix"] >> left_matrix;
		fs["left_coeff"] >> left_coeff;

		fs["right_matrix"] >> right_matrix;
		fs["right_coeff"] >> right_coeff;

		fs["R"] >> R;
		fs["T"] >> T;
		fs["Rl"] >> Rl;
		fs["Rr"] >> Rr;
		fs["Pl"] >> Pl;
		fs["Pr"] >> Pr;
		fs["Q"] >> Q;

		std::cout << Q << endl;

		fs.release();
	}
	else
	{
		return 0;
	}

	return 1;
}
ushort calDepth(int x, int y, double D, double c0, double c1, double c2)
{
	double depth = (double)(D *(-1) / (c0 * x + c1 * y + c2)) * 1000;
	//cout << depth << endl;

	ushort d = static_cast<ushort>(depth);
	return d;
}


void distPoint2Plane(float A, float B, float C, float D, int rs, int re, int cs, int ce, cv::Mat depthImage)
{
	ushort* ptr;
	float dd = sqrtf(A*A + B*B + C*C);
	float sum_n = 0;//
	float sum_p = 0;//
	int cnt_n = 0;
	int cnt_p = 0;
	for (int i = rs; i < re; i++)
	{
		ptr = depthImage.ptr<ushort>(i);
		for (int j = cs; j < ce; j++)
		{
			float depth = (float)ptr[j] * 0.001f;
			pcl::PointXYZ currPoint;
			//if (depth >= 0.5 && depth <= 1)
			if (ptr[j] !=0 && ptr[j] != 0xffff)
			{
				float imgTo3d_ratio = depth / Q.at<double>(2, 3);
				float X = (j + Q.at<double>(0, 3)) * imgTo3d_ratio;
				float Y = (i + Q.at<double>(1, 3)) * imgTo3d_ratio;

				float dist = (A * X + B * Y + C * depth + D) / dd;
				if (dist > 0)
				{
					sum_p += dist;
					cnt_p++;
				}
				else
				{
					sum_n += dist;
					cnt_n++;
				}

				//std::cout << dist << endl;
			}
		}
	}

	sum_p = sum_p / cnt_p;
	sum_n = sum_n / cnt_n;
	std::cout << "mean_dist = " << sum_p << "\t" << sum_n << endl;

	return;
}

int access_folder(string folder)
{
    int ret = 0;
    if(access(folder.c_str(),0) == -1)
    {
		int ret = mkdir(folder.c_str(),0777);
		if(ret == -1)
		{
            cout << "create folder error" << endl;
            return ret;
        }
    }
    return 0;
}

void depth2cloud(int rs, int re, int cs, int ce, cv::Mat depthImage, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::string path)
{
	ofstream OutFile(path);
	ushort* ptr;
	for (int i = rs; i < re; i++)
	{
		ptr = depthImage.ptr<ushort>(i);
		for (int j = cs; j < ce; j++)
		{
			float depth = (float)ptr[j] * 0.001f;//转化为单位m
			pcl::PointXYZ currPoint;
			//if (depth >= 0.5 && depth <= 1.5)
			if (ptr[j] !=0 && ptr[j] != 0xffff)
			{
				//cout << depth << endl;
				float imgTo3d_ratio = depth / Q.at<double>(2,3);
				float X = (j + Q.at<double>(0, 3)) * imgTo3d_ratio;
				float Y = (i + Q.at<double>(1, 3)) * imgTo3d_ratio;
				
				currPoint.x = X;
				currPoint.y = Y;
				currPoint.z = depth;
				OutFile << X << ";" << Y << ";" << depth << "\n";
				cloud->points.push_back(currPoint);
			}
			
		}
	}
	OutFile.close();
}

void depth2Disparity(Mat inputImage, double bf)
{
	uint16_t *image = (uint16_t*)inputImage.data;
	int count = 0;
	for(int r = 0; r < HEIGHT; r++)
	{
		for(int c = 0; c < WIDTH; c++)
		{
			uint16_t depth = *image;
			if(depth > 0 && depth <0xffff)
				*image = (uint16_t)(bf/(depth*1.0) + 0.5);
			else
			{
				*image = 0;
				count++;
			}
			if(r == 120) cout << *image << endl;
			image++;
		}
		
	}
	cout << "count "<< count << endl;
}

void disparity2Depth(Mat inputImage, double bf)
{
	uint16_t *image = (uint16_t*)inputImage.data;
	int count = 0;
	for(int r = 0; r < HEIGHT; r++)
	{
		for(int c = 0; c < WIDTH; c++)
		{
			uint16_t disp = *image;
			if(disp > 0 && disp < 0xffff)
				*image = (uint16_t)(bf/(disp*1.0) + 0.5);
			else
			{
				*image = 0;
				count++;
			}
			image++;
		}
	}
	cout<< "count1 " << count << endl;	
}


void recursive_filter_horizontal(uint16_t *frame_data, float alpha, int delta, int holes_filling_radius)
{
	//递归滑动滤波
	int curr_fill = 0;
	uint16_t *image = frame_data;
	int count = 0;
	for(int r = 0; r < HEIGHT; r++)
	{
		//左到右，滤波+填充空洞
		uint16_t *im = image + r * WIDTH;
		uint16_t val0 = im[0];
		curr_fill = 0;
		for(int c = 1; c < WIDTH-1; c++)
		{
			uint16_t val1 = im[1];
			if(val0 > 0 && val0 < 0xffff)
			{
				if(val1 > 0 && val1 < 0xffff)
				{
					curr_fill = 0;
					int diff = abs(val1 - val0);
					if(diff <= delta)
					{
						count++;
						val1 = (uint16_t)(alpha * val1 + (1.0-alpha) * val0 + 0.5);
					}
					
				}
				else
				{
					if(holes_filling_radius)
					{
						if(curr_fill < holes_filling_radius)
						{
							val1 = val0;
							curr_fill += 1;
						}
					}

				}
				im[1] = val1;
			}
			val0 = val1;
			im++;
		}

		//右到左，填充空洞
		im = image + (r+1) * WIDTH - 1;
		val0 = im[0];
		curr_fill = 0;
		for(int c = 1; c < WIDTH-1; c++)
		{
			uint16_t val1 = im[-1];
			if(val0 > 0 && val0 < 0xffff)
			{
				if(val1 > 0 && val1 < 0xffff)
				{
					curr_fill = 0;
					int diff = abs(val1 - val0);
					if(diff <= delta)
					{	count++;
						val1 = (uint16_t)(alpha * val1 + (1.0-alpha) * val0 + 0.5);
					}
				}
				else
				{
					if(holes_filling_radius)
					{
						if(curr_fill < holes_filling_radius)
						{
							val1 = val0;
							curr_fill += 1;
						}
					}
				}
				im[-1] = val1;
			}
			val0 = val1;
			im--;
		}		
	}
	cout << "count2 " << count << endl;
}

void recursive_filter_vertical(uint16_t *frame_data, float alpha, int delta)
{
	uint16_t *image = frame_data;
	uint16_t val0, val1;

	//从上到下
	uint16_t *im = image;
	for(int r = 1; r < HEIGHT; r++)
	{
		for(int c = 0; c < WIDTH; c++)
		{
			val0 = im[0];
			val1 = im[WIDTH];
			if((val0 > 0 && val0 < 0xffff) && (val1 > 0 && val1 < 0xffff))
			{
				int diff = abs(val1 - val0);
				if(diff <= delta)
				{
					im[WIDTH] = (uint16_t)(alpha * val1 + (1.0-alpha) * val0 + 0.5);
				}
			}
			im++;
		}
	}
	//从下到上
	im = image + WIDTH*HEIGHT-1;
	for(int r = 1; r < HEIGHT; r++)
	{
		for(int c = 0; c < WIDTH; c++)
		{
			val0 = im[0];
			val1 = im[-WIDTH];
			if((val0 > 0 && val0 < 0xffff) && (val1 > 0 && val1 < 0xffff))
			{
				int diff = abs(val1 - val0);
				if(diff <= delta)
				{
					im[-WIDTH] = (uint16_t)(alpha * val1 + (1.0 - alpha) * val0 + 0.5);
				}
			}
			im--;
		}
	}

}

//Edge-preserving spatial filter
void EdgePresevingFilter(Mat inputImage, float alpha, int delta, int iteration, int holes_filling_radius)
{
	uint16_t *frame_data = (uint16_t*)inputImage.data;
	if(alpha < 0.0 || alpha > 1.0 || delta == 0) 
	{
		cout << "filter paramter error !" <<endl;
		return;
	} 
	for(int i = 0; i < iteration; i++)
	{
		recursive_filter_horizontal(frame_data, alpha, delta, holes_filling_radius);
		recursive_filter_vertical(frame_data, alpha, delta);
	}
}


void depth2cloudWithSpatialFilter(int rs, int re, int cs, int ce, cv::Mat depthImage, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::string path)
{
	float alpha = 0.3;
	int delta = 32; 
	int holes_fill = 1;
	int iteration = 3;

	ofstream OutFile(path);

	Mat &inputImage = depthImage;
	depth2Disparity(inputImage, BF);
	EdgePresevingFilter(inputImage, alpha, delta, iteration, holes_fill);
	disparity2Depth(inputImage, BF);
	//转化为点云
	ushort* ptr;
	for (int i = rs; i < re; i++)
	{
		ptr = inputImage.ptr<ushort>(i);
		for (int j = cs; j < ce; j++)
		{
			float depth = (float)ptr[j] * 0.001f;//转化为单位m
			pcl::PointXYZ currPoint;
			//if (depth >= 0.5 && depth <= 1.5)
			if (ptr[j] !=0 && ptr[j] != 0xffff)
			{
				//cout << depth << endl;
				float imgTo3d_ratio = depth / Q.at<double>(2,3);
				float X = (j + Q.at<double>(0, 3)) * imgTo3d_ratio;
				float Y = (i + Q.at<double>(1, 3)) * imgTo3d_ratio;
				
				currPoint.x = X;
				currPoint.y = Y;
				currPoint.z = depth;
				OutFile << X << ";" << Y << ";" << depth << "\n";
				cloud->points.push_back(currPoint);
			}
			
		}
	}
	OutFile.close();
}


void depth2cloudWithFilter(int rs, int re, int cs, int ce, cv::Mat depthImage, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::string path)
{
	ofstream OutFile(path);
	//转化为CV_32F数据类型
	Mat fimage = Mat(depthImage.size(), CV_32FC1);
	depthImage.convertTo(fimage, CV_32FC1);
	//双边滤波
	Mat rimage;
	bilateralFilter(fimage, rimage, 21, 150, 150);
	//Mat mimage;
	//medianBlur(fimage, mimage, 7);
	//edgePreservingFilter
	//转化为点云
	float* ptr;
	for (int i = rs; i < re; i++)
	{
		ptr = rimage.ptr<float>(i);
		for (int j = cs; j < ce; j++)
		{
			float depth = ptr[j] * 0.001f;//转化为单位m
			pcl::PointXYZ currPoint;
			//if (depth >= 0.5 && depth <= 1.5)
			if (ptr[j] > 0 && ptr[j] < 0xffff)
			{
				//cout << depth << endl;
				float imgTo3d_ratio = depth / Q.at<double>(2,3);
				float X = (j + Q.at<double>(0, 3)) * imgTo3d_ratio;
				float Y = (i + Q.at<double>(1, 3)) * imgTo3d_ratio;
				
				currPoint.x = X;
				currPoint.y = Y;
				currPoint.z = depth;
				OutFile << X << ";" << Y << ";" << depth << "\n";
				cloud->points.push_back(currPoint);
			}
			
		}
	}
	OutFile.close();
}


int main()
{
#ifndef REALSENSE_ON
	bool ret = getStereoCameraParams();
	bool _verbose;
	if (!ret)
	{
		std::cout << "getStereoCameraParams failed!" << std::endl;
	}
	BF = 64 * Q.at<double>(2,3) / Q.at<double>(3,2);
	printf("%lf\n", BF);

	auto imageSize = Size(WIDTH, HEIGHT);
	Mat mapLx, mapLy, mapRx, mapRy;
	initUndistortRectifyMap(left_matrix, left_coeff, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(right_matrix, right_coeff, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

	std::string src_path = IMAGE_PATH;
	std::vector<cv::String> file_vec;
	cv::glob(src_path + "depth/*.png", file_vec, false);
    char input;
	access_folder(src_path + "left_right_rectify");

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	
	for (int m = 0; m < file_vec.size(); m++)
	{

		std::string file_name = file_vec[m];
		std::cout << file_name << std::endl;

		cv::Mat depthImage = imread(file_name, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if(depthImage.empty())
        {
            std::cout << "depthImage.empty" << endl;
            return -1;
        }
#ifdef Rectify
		std::string leftimagePath = file_name;
		leftimagePath.replace(leftimagePath.find("depth"), 5, "left_right");
		leftimagePath.replace(leftimagePath.find("Depth"), 5, "Resize_Pic_L");
		std::cout << leftimagePath << std::endl;
		cv::Mat left = imread(leftimagePath, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if(left.empty())
        {
            std::cout << "left.empty" << endl;
            return -1;
        }
		cv::Mat left_rectify;
		remap(left, left_rectify, mapLx, mapLy, INTER_LINEAR);
        cv::namedWindow("left_rectify");
        cv::imshow("left_rectify", left_rectify);

		std::string leftimageRectPath  = leftimagePath;
		leftimageRectPath.replace(leftimageRectPath.find("left_right"), 10, "left_right_rectify");
		leftimageRectPath.replace(leftimageRectPath.find("Resize_Pic_L"), 12, "Resize_Pic_L_Rectify");
		cv::imwrite(leftimageRectPath, left_rectify);

		std::string rightimagePath = file_name;
		rightimagePath.replace(rightimagePath.find("depth"), 5, "left_right");
		rightimagePath.replace(rightimagePath.find("Depth"), 5, "Resize_Pic_R");
        std::cout << rightimagePath << std::endl;
		cv::Mat right = imread(rightimagePath, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if(right.empty())
        {
            std::cout << "right.empty" << endl;
            return -1;
        }
		cv::Mat right_rectify;
		remap(right, right_rectify, mapRx, mapRy, INTER_LINEAR);
        cv::namedWindow("right_rectify");
        cv::imshow("right_rectify", right_rectify);

		std::string rightimageRectPath  = rightimagePath;
		rightimageRectPath.replace(rightimageRectPath.find("left_right"), 10, "left_right_rectify");
		rightimageRectPath.replace(rightimageRectPath.find("Resize_Pic_R"), 12, "Resize_Pic_R_Rectify");
		cv::imwrite(rightimageRectPath, right_rectify);
#endif

        std::string cloudPath = file_name;
        cloudPath.replace(cloudPath.find(".png"), 5, ".txt");

		int width = depthImage.cols;
		int height = depthImage.rows;

		cloud->points.clear();
#ifndef SPATIAL_FILTER
		depth2cloud(0, height, 0, width, depthImage, cloud, cloudPath);
#else
		//depth2cloudWithFilter(0, height, 0, width, depthImage, cloud, cloudPath);
		depth2cloudWithSpatialFilter(0, height, 0, width, depthImage, cloud, cloudPath);
#endif
		std::cout << "cloud->points.size() ratio: " << (float)cloud->points.size()/(width*height) << std::endl;

		// 设置并保存点云
		cloud->height = 1;
		cloud->width = cloud->points.size();
		//cout<<"point cloud size = "<<cloud->points.size()<<endl;
		cloud->is_dense = false;

		std::string plyPath = file_name;
        plyPath.replace(plyPath.find(".png"), 5, ".ply");
		pcl::io::savePLYFile(plyPath, *cloud); 
		std::string pcdPath = file_name;
        pcdPath.replace(pcdPath.find(".png"), 5, ".pcd");
		pcl::io::savePCDFile( pcdPath, *cloud );
		cout<<"Point cloud saved."<<endl;

		pcl::visualization::CloudViewer viewer ("cloud");
		viewer.showCloud(cloud);
        //waitKey(10000);
		while (!viewer.wasStopped())
		{
			boost::this_thread::sleep(boost::posix_time::microseconds(100));
		}
		// while (!viewer.wasStopped()){ };   

		waitKey(1);
        
#ifdef CLOUD_FIlTER
        /*StatisticalOutlierRemoval滤波原理：
        *对每个点的邻域进行一个统计分析，并修剪掉那些不符合一定标准的点。稀疏离群点移除方法基于在输入数据中对点到邻近点的距离分布的计算。
        *对每个点，我们计算其到所有邻近点的平均距离。假设得到的结果是一个高斯分布，其形状由均值和标准差决定，平均距离在标准范围（由全局
        *距离平均值和方差定义）之外的点，可被定义为离群点并从数据集中去除掉。
        */
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
		sor.setInputCloud(cloud);
		sor.setMeanK(50);//对每个点分析的邻近点个数设为50
		sor.setStddevMulThresh(1.0);//标准差倍数设为1
		sor.filter(*cloud_filtered);//注意前面加“*”
	
		// pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
		// viewer.showCloud(cloud_filtered);
		// while (!viewer.wasStopped())
		// {

		// }
#endif

#ifdef CLOUD_PLANE
        /*采样一致性分割算法的目的主要是从原点云中提取目标模型，比如说面，球体，圆柱等等，从而为后续的目标识别或者点云匹配等等做准备。
         *使用此算法之前应该先熟悉PCL中的采样一致性（sample consensus）模块，里边包含了模型（平面、直线等）和采样方法（RANSAC、LMedS等）
         *的一些枚举变量，一些核心的算法也包含其中，我们可以使用不同的方法提取不同的模型。 
         */
		// Create the segmentation object
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		// Optional
		seg.setOptimizeCoefficients(true);
		// Mandatory
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setMaxIterations(1000);
		seg.setDistanceThreshold(0.02);

		pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
			return 0;
		}
		cout << coefficients->values[0] << "\t" << coefficients->values[1] << "\t" << coefficients->values[2] << "\t" << coefficients->values[3];
		cout << endl;

		distPoint2Plane(coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3], 0, height, 0, width, depthImage);

		double A = coefficients->values[0] * 1000;
		double B = coefficients->values[1] * 1000;
		double C = coefficients->values[2] * 1000;
		double D = coefficients->values[3] * 1000;
		double c0 = A / Q.at<double>(2,3);
		double c1 = B / Q.at<double>(2,3);
		double c2 = c0 * Q.at<double>(0,3) + c1 * Q.at<double>(1,3) + C;

		cout << D << "\t" << c0 << "\t" << c1 << "\t" << c2 << endl;

		cv::Mat groundDepth = cv::Mat::zeros(height, width, CV_16UC1);

		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				groundDepth.at<ushort>(row, col) = calDepth(col, row, D, c0, c1, c2);
			}
		}
#endif
        // waitKey(0);
	}

	return 0;
#else //REALSENSE
#if 0
	Q = Mat::zeros(4,4,CV_64FC1);
	Q.at<double>(0,3) = -323.136;
	Q.at<double>(1,3) = -239.487;
	Q.at<double>(2,3) = 381.084;
	Q.at<double>(3,2) = 1/50.0485;
	BF = 64 * Q.at<double>(2,3) / Q.at<double>(3,2);
	printf("%lf\n", BF);
	std::string src_path = IMAGE_PATH;
	std::vector<cv::String> file_vec;
	cv::glob(src_path + "depth/*.png", file_vec, false);
    char input;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	
	for (int m = 0; m < file_vec.size(); m++)
	{

		std::string file_name = file_vec[m];
		std::cout << file_name << std::endl;
/*		std::ifstream fin;
		// 注意，这里要指定binary读取模式
		fin.open(file_name, std::ios::binary);
		if (!fin) {
			std::cerr << "open failed: " << file_name << std::endl;
		}
		// seek函数会把标记移动到输入流的结尾
		fin.seekg(0, fin.end);
		// tell会告知整个输入流（从开头到标记）的字节数量
		int length = fin.tellg();
		// 再把标记移动到流的开始位置
		fin.seekg(0, fin.beg);
		std::cout << "file length: " << length << std::endl;

		// load buffer
		char* buffer = new char[length];
		// read函数读取（拷贝）流中的length各字节到buffer
		fin.read(buffer, length);

		// construct opencv mat and show image
		cv::Mat depthImage(cv::Size(640, 480), CV_16UC1, buffer);
*/
		cv::Mat depthImage = imread(file_name, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if(depthImage.empty())
        {
            std::cout << "depthImage.empty" << endl;
            return -1;
        }

       // file_name.replace(file_name.find(".raw"), 5, ".png");
		//cv::imwrite(file_name, depthImage);

		int type = depthImage.type();
		if( CV_16UC1 == type )
			printf("type = %d, %d, %d\n", type, depthImage.cols, depthImage.rows);
        // cv::namedWindow("depthImage");
        // cv::imshow("depthImage", depthImage);
		//waitKey(0);
        std::string cloudPath = file_name;
        cloudPath.replace(cloudPath.find(".png"), 5, ".txt");

		int width = depthImage.cols;
		int height = depthImage.rows;

		cloud->points.clear();
		depth2cloud(0, height, 0, width, depthImage, cloud, cloudPath);
		//depth2cloudWithSpatialFilter(0, height, 0, width, depthImage, cloud, cloudPath);
		std::cout << "cloud->points.size() ratio: " << (float)cloud->points.size()/(width*height) << std::endl;

		// 设置并保存点云
		cloud->height = 1;
		cloud->width = cloud->points.size();
		//cout<<"point cloud size = "<<cloud->points.size()<<endl;
		cloud->is_dense = false;

		std::string plyPath = file_name;
        plyPath.replace(plyPath.find(".png"), 5, ".ply");
		pcl::io::savePLYFile(plyPath, *cloud); 
		std::string pcdPath = file_name;
        pcdPath.replace(pcdPath.find(".png"), 5, ".pcd");
		pcl::io::savePCDFile( pcdPath, *cloud );
		cout<<"Point cloud saved."<<endl;

		pcl::visualization::CloudViewer viewer ("cloud");
		viewer.showCloud(cloud);
        //waitKey(10000);
		while (!viewer.wasStopped())
		{
			boost::this_thread::sleep(boost::posix_time::microseconds(100));
		}

	}

#endif

	Q = Mat::zeros(4,4,CV_64FC1);
	Q.at<double>(0,3) = -3.3191730117797852e+02;
	Q.at<double>(1,3) = -2.0922435569763184e+02;
	Q.at<double>(2,3) = 5.2385772999630058e+02;
	Q.at<double>(3,2) = 3.1190643796076584e-02;
	BF = 16 * Q.at<double>(2,3) / Q.at<double>(3,2);
	printf("%lf\n", BF);
	std::string src_path = IMAGE_PATH;
	std::vector<cv::String> file_vec;
	cv::glob(src_path + "depth/*.png", file_vec, false);
    char input;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	
	for (int m = 0; m < file_vec.size(); m++)
	{

		std::string file_name = file_vec[m];
		std::cout << file_name << std::endl;
/*		std::ifstream fin;
		// 注意，这里要指定binary读取模式
		fin.open(file_name, std::ios::binary);
		if (!fin) {
			std::cerr << "open failed: " << file_name << std::endl;
		}
		// seek函数会把标记移动到输入流的结尾
		fin.seekg(0, fin.end);
		// tell会告知整个输入流（从开头到标记）的字节数量
		int length = fin.tellg();
		// 再把标记移动到流的开始位置
		fin.seekg(0, fin.beg);
		std::cout << "file length: " << length << std::endl;

		// load buffer
		char* buffer = new char[length];
		// read函数读取（拷贝）流中的length各字节到buffer
		fin.read(buffer, length);

		// construct opencv mat and show image
		cv::Mat depthImage(cv::Size(640, 480), CV_16UC1, buffer);
*/
		cv::Mat depthImage = imread(file_name, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if(depthImage.empty())
        {
            std::cout << "depthImage.empty" << endl;
            return -1;
        }

       // file_name.replace(file_name.find(".raw"), 5, ".png");
		//cv::imwrite(file_name, depthImage);

		int type = depthImage.type();
		if( CV_16UC1 == type )
			printf("type = %d, %d, %d\n", type, depthImage.cols, depthImage.rows);
        // cv::namedWindow("depthImage");
        // cv::imshow("depthImage", depthImage);
		//waitKey(0);
        std::string cloudPath = file_name;
        cloudPath.replace(cloudPath.find(".png"), 5, ".txt");

		int width = depthImage.cols;
		int height = depthImage.rows;

		cloud->points.clear();
		depth2cloud(0, height, 0, width, depthImage, cloud, cloudPath);
		//depth2cloudWithSpatialFilter(0, height, 0, width, depthImage, cloud, cloudPath);
		std::cout << "cloud->points.size() ratio: " << (float)cloud->points.size()/(width*height) << std::endl;

		// 设置并保存点云
		cloud->height = 1;
		cloud->width = cloud->points.size();
		//cout<<"point cloud size = "<<cloud->points.size()<<endl;
		cloud->is_dense = false;

		std::string plyPath = file_name;
        plyPath.replace(plyPath.find(".png"), 5, ".ply");
		pcl::io::savePLYFile(plyPath, *cloud); 
		std::string pcdPath = file_name;
        pcdPath.replace(pcdPath.find(".png"), 5, ".pcd");
		pcl::io::savePCDFile( pcdPath, *cloud );
		cout<<"Point cloud saved."<<endl;

		pcl::visualization::CloudViewer viewer ("cloud");
		viewer.showCloud(cloud);
        //waitKey(10000);
		while (!viewer.wasStopped())
		{
			boost::this_thread::sleep(boost::posix_time::microseconds(100));
		}

	}


	return 0;
#endif
}
