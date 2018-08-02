#include "stdio.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <opencv2/opencv.hpp>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <list>

#define cvQueryHistValue_2D( hist, idx0, idx1 )   cvGetReal2D( (hist)->bins, (idx0), (idx1) )


using namespace std;
using namespace cv;

string intToString(int v);
int changeSize(int oldSize);
int changeSizeForAndroid(int oldSize);
CvScalar getMainColor(IplImage *src);
bool Screenshot(IplImage* src, IplImage* dst, CvRect rect);
string RGBToHexadecimal(int R, int G, int B);
string DecimalToHexadecimal(int dec);
CvScalar DCD(IplImage* src);
bool hasCross(Rect rectOne, Rect rectTwo);
bool hasCrossOnXAndY(Rect rectOne, Rect rectTwo);


int main(int argc, char** argv)
{
	 //重新调整图像大小
	//int resize_height = 1334;
	//int resize_width = 750;
	//Mat src = imread("C:/AllFiles/VisualStudioWorkspace/MarkOpenCV/MarkOpenCV/x64/Debug/image/image1.jpg", 1);
	//Mat dst;
	//resize(src, dst, Size(resize_width, resize_height), (0, 0), (0, 0), INTER_NEAREST);
	//imwrite("C:/AllFiles/VisualStudioWorkspace/MarkOpenCV/MarkOpenCV/x64/Debug/image/image1_resize.jpg", dst);
	
	
	string path = "C:/AllFiles/VisualStudioWorkspace/MarkOpenCV/MarkOpenCV/x64/Debug/image/test2.jpg";
	Mat image = imread(path, 1);
	IplImage imageForWidth = IplImage(image);
	int imageWidth = imageForWidth.width;

	// 灰度图
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);
	imwrite("grayImage.jpg", gray_image);

	// 二值化：局部自适应二值化函数
	Mat binary_image;
	adaptiveThreshold(gray_image, binary_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 15);
	imwrite("binaryImage.jpg", binary_image);

	// 横向膨胀
	Mat element = getStructuringElement(MORPH_RECT, Size(binary_image.cols / 50, 1));
	Mat dilateImage;
	dilate(binary_image, dilateImage, element);
	imwrite("dilateImage.jpg", dilateImage);
	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dilateImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNAL只检测外部轮廓，可根据自身需求进行调整

	Mat contoursImage(dilateImage.rows, dilateImage.cols, CV_8U, Scalar(255));
	int index = 0;
	vector<Rect> listBigRects;
	for (; index >= 0; index = hierarchy[index][0]) {
		cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
		// 检测外轮廓
		Rect rect = boundingRect(contours[index]);

		// 添加到列表中，用于做横向标注
		listBigRects.push_back(rect);

		//对外轮廓加矩形框
		//rectangle(contoursImage, rect, Scalar(0, 0, 255), 3);

		int height = rect.height;
		// 假如宽超过高的1.5倍，就当做是文本，需要作转换
		if (height * 1.5f < rect.width)
		{
			if (imageWidth == 750)
			{
				height = changeSize(height);
			}
			else if (imageWidth == 1080)
			{
				height = changeSizeForAndroid(height);
			}
		}
		string pxText = intToString(height) + "px";
		cv::Point point;//特征点，用以画在图像中  
		point.x = rect.x + rect.width - 20;//特征点在图像中横坐标  
		point.y = rect.y - 10;//特征点在图像中纵坐标  
		putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

		// 画线
		cv::Point pointOne;//特征点，用以画在图像中  
		pointOne.x = rect.x + rect.width;//特征点在图像中横坐标  
		pointOne.y = rect.y;//特征点在图像中纵坐标  
		cv::Point pointTwo;//特征点，用以画在图像中  
		pointTwo.x = rect.x + rect.width;//特征点在图像中横坐标  
		pointTwo.y = rect.y + rect.height;//特征点在图像中纵坐标  
		line(image, pointOne, pointTwo, Scalar(0, 255, 0), 1);
	}
	
	imwrite("contours_image.jpg", contoursImage);
	imwrite("mark_image.jpg", image);



	/*
		==================================================================================
		标注边距
	*/
	Mat imageTwo = imread(path, 1);
	

	// 灰度图
	Mat gray_imageTwo;
	cvtColor(imageTwo, gray_imageTwo, CV_BGR2GRAY);
	imwrite("grayImage.jpg", gray_imageTwo);

	// //模糊
	//Mat blur_imageTwo;
	//blur(gray_imageTwo, blur_imageTwo, Size(3, 3));
	//imwrite("2.jpg", blur_imageTwo);

	// 二值化：局部自适应二值化函数
	Mat binary_imageTwo;
	adaptiveThreshold(gray_imageTwo, binary_imageTwo, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 15);
	imwrite("binaryImageTwo.jpg", binary_imageTwo);

	// 纵向膨胀
	Mat elementTwo = getStructuringElement(MORPH_RECT, Size(1, binary_imageTwo.rows /  70));
	Mat dilateImageTwo;
	dilate(binary_imageTwo, dilateImageTwo, elementTwo);
	imwrite("dilateImageTwo.jpg", dilateImageTwo);

	//// 去噪：中值滤波
	//Mat de_noiseTwo = binary_imageTwo.clone();
	//medianBlur(binary_imageTwo, de_noiseTwo, 5);
	//imwrite("4.jpg", de_noiseTwo);

	vector<vector<Point>> contoursTwo;
	vector<Vec4i> hierarchyTwo;
	findContours(dilateImageTwo, contoursTwo, hierarchyTwo, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNAL只检测外部轮廓，可根据自身需求进行调整

	Mat contoursImageTwo(dilateImageTwo.rows, dilateImageTwo.cols, CV_8U, Scalar(255));
	int indexTwo = 0;
	vector<Rect> listRects;
	for (; indexTwo >= 0; indexTwo = hierarchyTwo[indexTwo][0]) {
		cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
		// 检测外轮廓
		Rect rect = boundingRect(contoursTwo[indexTwo]);
		listRects.push_back(rect);
		//对外轮廓加矩形框
		rectangle(contoursImageTwo, rect, Scalar(0, 0, 255), 3);
	}

	/*
		=====================================================================================
		绘制边界
	*/
	
	// 查找
	vector<Rect> elementOnEdge;
	for (int i = 0; i < listBigRects.size(); i++)
	{
		Rect rect;
		for (int j = 0; j < listRects.size(); j++)
		{
			if (hasCrossOnXAndY(listBigRects[i], listRects[j]))
			{
				// 空状态，则直接保存
				if (rect.x == 0 && rect.y == 0 && rect.width == 0 && rect.height == 0)
				{
					rect = listRects[j];
				}
				else
				{
					if (listBigRects[i].x <= imageWidth / 2)
					{
						if (rect.x > listRects[j].x)
						{
							rect = listRects[j];
						}
					}
					else
					{
						if ((rect.x + rect.width) < (listRects[j].x + listRects[j].width))
						{
							rect = listRects[j];
						}
					}
				}
			}
		}
		elementOnEdge.push_back(rect);
	}
	

	/*
		===================================================================================
		绘制最边的元素边距
	*/
	vector<Rect> leftEdgeRects; // 保存所有最左边的元素
	vector<Rect> rightEdgeRects; // 保存所有最右边的元素
	for (int i = 0; i < elementOnEdge.size(); i++)
	{
		bool hasLeftElement = false;
		bool hasRightElement = false;

		for (int j = 0; j < elementOnEdge.size(); j++)
		{
			if (hasCross(elementOnEdge[i], elementOnEdge[j]))
			{
				if (elementOnEdge[i].x <= imageWidth / 2)
				{
					if (elementOnEdge[j].x < elementOnEdge[i].x)
					{
						hasLeftElement = true;
					}
				}
				else
				{
					if ((elementOnEdge[j].x + elementOnEdge[j].width) >(elementOnEdge[i].x + elementOnEdge[i].width))
					{
						hasRightElement = true;
					}
				}
			}
		}

		if (!hasLeftElement && elementOnEdge[i].x <= imageWidth / 2)
		{
			leftEdgeRects.push_back(elementOnEdge[i]);
			// 画线
			cv::Point pointOne;//特征点，用以画在图像中  
			pointOne.x = 0;//特征点在图像中横坐标  
			pointOne.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//特征点在图像中纵坐标  
			cv::Point pointTwo;//特征点，用以画在图像中  
			pointTwo.x = elementOnEdge[i].x;//特征点在图像中横坐标  
			pointTwo.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//特征点在图像中纵坐标  
			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);

			string pxText = intToString(elementOnEdge[i].x) + "px";
			cv::Point point;//特征点，用以画在图像中  
			point.x = pointOne.x;//特征点在图像中横坐标  
			point.y = elementOnEdge[i].y + elementOnEdge[i].height / 2 - 20;//特征点在图像中纵坐标  
			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}

		if (!hasRightElement && elementOnEdge[i].x > imageWidth / 2)
		{
			rightEdgeRects.push_back(elementOnEdge[i]);
			// 画线
			cv::Point pointOne;//特征点，用以画在图像中  
			pointOne.x = elementOnEdge[i].x + elementOnEdge[i].width;//特征点在图像中横坐标  
			pointOne.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//特征点在图像中纵坐标  
			cv::Point pointTwo;//特征点，用以画在图像中  
			pointTwo.x = imageWidth;//特征点在图像中横坐标  
			pointTwo.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//特征点在图像中纵坐标  
			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);

			int width = imageWidth - elementOnEdge[i].x - elementOnEdge[i].width;
			string pxText = intToString(width) + "px";
			cv::Point point;//特征点，用以画在图像中  
			point.x = pointOne.x;//特征点在图像中横坐标  
			point.y = elementOnEdge[i].y + elementOnEdge[i].height / 2 - 20;//特征点在图像中纵坐标  
			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
	}

	/*
		==================================================================================
		绘制中间的元素边距
	*/
	for (int i = 0; i < elementOnEdge.size(); i++)
	{
		// 剔除最边元素
		bool isLeftElement = false;
		for (int left = 0; left < leftEdgeRects.size(); left++)
		{
			if (elementOnEdge[i].x == leftEdgeRects[left].x && elementOnEdge[i].y == leftEdgeRects[left].y &&elementOnEdge[i].width == leftEdgeRects[left].width &&elementOnEdge[i].height == leftEdgeRects[left].height)
			{
				isLeftElement = true;
				break;
			}
		}
		if (isLeftElement)
		{
			continue;
		}
		bool isRightElement = false;
		for (int right = 0; right < rightEdgeRects.size(); right++)
		{
			if (elementOnEdge[i].x == rightEdgeRects[right].x && elementOnEdge[i].y == rightEdgeRects[right].y &&elementOnEdge[i].width == rightEdgeRects[right].width &&elementOnEdge[i].height == rightEdgeRects[right].height)
			{
				isRightElement = true;
				break;
			}
		}
		if (isRightElement)
		{
			continue;
		}

		// 判断旁边是否有小于5像素的元素
		int x = 0;
		for (int j = 0; j < elementOnEdge.size(); j++)
		{
			if (i == j) 
			{
				continue;
			}
			if (hasCross(elementOnEdge[i], elementOnEdge[j]))
			{
				if (elementOnEdge[i].x <= imageWidth / 2)
				{
					// 查找左边是否有元素：1、x轴坐标需要大于左边元素的right   2、y轴需要有交叉   3、x轴的长度要大于5		
					if (elementOnEdge[j].x + elementOnEdge[j].width < elementOnEdge[i].x && (elementOnEdge[i].x - elementOnEdge[j].x - elementOnEdge[j].width) > 10)
					{
						x = elementOnEdge[j].x + elementOnEdge[j].width;
					}
				}
				else
				{
					// 查找右边是否有元素：1、元素的right需要小于其他元素的x轴   2、y轴需要有交叉   3、x轴的长度要大于5		
					if (elementOnEdge[j].x >(elementOnEdge[i].x + elementOnEdge[i].width) && (elementOnEdge[j].x - elementOnEdge[i].x - elementOnEdge[i].width) > 10)
					{
						x = elementOnEdge[j].x;
					}
				}
			}
		}

		if (elementOnEdge[i].x <= imageWidth / 2)
		{
			cv::Point pointOne;//特征点，用以画在图像中  
			pointOne.x = x;//特征点在图像中横坐标  
			pointOne.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//特征点在图像中纵坐标  
			cv::Point pointTwo;//特征点，用以画在图像中  
			pointTwo.x = elementOnEdge[i].x;//特征点在图像中横坐标  
			pointTwo.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//特征点在图像中纵坐标  
			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);

			int width = elementOnEdge[i].x - x;
			string pxText = intToString(width) + "px";
			cv::Point point;//特征点，用以画在图像中  
			point.x = pointOne.x;//特征点在图像中横坐标  
			point.y = elementOnEdge[i].y + elementOnEdge[i].height / 2 - 20;//特征点在图像中纵坐标  
			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
		else
		{
			cv::Point pointOne;//特征点，用以画在图像中  
			pointOne.x = elementOnEdge[i].x + elementOnEdge[i].width;//特征点在图像中横坐标  
			pointOne.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//特征点在图像中纵坐标  
			cv::Point pointTwo;//特征点，用以画在图像中  
			pointTwo.x = x;//特征点在图像中横坐标  
			pointTwo.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//特征点在图像中纵坐标  
			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);

			int width = x - elementOnEdge[i].x - elementOnEdge[i].width;
			string pxText = intToString(width) + "px";
			cv::Point point;//特征点，用以画在图像中  
			point.x = pointOne.x;//特征点在图像中横坐标  
			point.y = elementOnEdge[i].y + elementOnEdge[i].height / 2 - 20;//特征点在图像中纵坐标  
			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
	}

	imwrite("contours_image_two.jpg", contoursImageTwo);
	imwrite("biaozhu_finish.jpg", image);
	imshow("img", image);

	cout << "完成检测";

	/// 等待用户按键
	waitKey();
	return 0;
}

/*
	判断是否在同一水平区域内
*/
bool hasCross(Rect rectOne, Rect rectTwo)
{
	// A位于B上方
	if (rectOne.y < rectTwo.y && (rectOne.y + rectOne.height) < rectTwo.y)
	{
		return false;
	}
	// A位于B下方
	else if (rectOne.y > (rectTwo.y + rectTwo.height))
	{
		return false;
	}
	else
	{
		return true;
	}
}

/*
	判断元素是否横向和纵向都有交集
*/
bool hasCrossOnXAndY(Rect rectOne, Rect rectTwo)
{
	// A位于B上方
	if (rectOne.y < rectTwo.y && (rectOne.y + rectOne.height) < rectTwo.y)
	{
		return false;
	}
	// A位于B下方
	else if (rectOne.y > (rectTwo.y + rectTwo.height))
	{
		return false;
	}
	// A位于B左方
	else if (rectOne.x < rectTwo.x && (rectOne.x + rectOne.width) < rectTwo.x)
	{
		return false;
	}
	// A位于B右方
	else if (rectOne.x > (rectTwo.x + rectTwo.width))
	{
		return false;
	}
	else
	{
		return true;
	}
}

int changeSize(int oldSize)
{
	if (oldSize <= 22)
	{
		return 22;
	}
	else if (oldSize > 22 && oldSize <= 24)
	{
		return 24;
	}
	else if (oldSize > 24 && oldSize <= 26)
	{
		return 26;
	}
	else if (oldSize > 26 && oldSize <= 28)
	{
		return 28;
	}
	else if (oldSize > 28 && oldSize <= 32)
	{
		return 32;
	}
	else
	{
		return 36;
	}
}

int changeSizeForAndroid(int oldSize)
{
	if (oldSize <= 30)
	{
		return 30;
	}
	else if (oldSize > 30 && oldSize <= 36)
	{
		return 36;
	}
	else if (oldSize > 36 && oldSize <= 42)
	{
		return 42;
	}
	else if (oldSize > 42 && oldSize <= 48)
	{
		return 48;
	}
	else if (oldSize > 48 && oldSize <= 54)
	{
		return 54;
	}
	else
	{
		return 60;
	}
}

string intToString(int v)
{
	char buf[32] = { 0 };
	snprintf(buf, sizeof(buf), "%u", v);

	string str = buf;
	return str;
}

CvScalar getMainColor(IplImage *src)
{
	IplImage* hsv = cvCreateImage(cvGetSize(src), 8, 3);
	IplImage* h_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* s_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* v_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* planes[] = { h_plane, s_plane };

	/** H 分量划分为16个等级，S分量划分为8个等级 */
	int h_bins = 16, s_bins = 8;
	int hist_size[] = { h_bins, s_bins };

	/** H 分量的变化范围 */
	float h_ranges[] = { 0, 180 };

	/** S 分量的变化范围*/
	float s_ranges[] = { 0, 255 };
	float* ranges[] = { h_ranges, s_ranges };

	/** 输入图像转换到HSV颜色空间 */
	cvCvtColor(src, hsv, CV_BGR2HSV);
	cvSplit(hsv, h_plane, s_plane, v_plane, 0);

	/** 创建直方图，二维, 每个维度上均分 */
	CvHistogram * hist = cvCreateHist(2, hist_size, CV_HIST_ARRAY, ranges, 1);
	/** 根据H,S两个平面数据统计直方图 */
	cvCalcHist(planes, hist, 0, 0);

	/** 获取直方图统计的最大值，用于动态显示直方图 */
	float max_value;
	cvGetMinMaxHistValue(hist, 0, &max_value, 0, 0);


	/** 设置直方图显示图像 */
	int height = 240;
	int width = (h_bins*s_bins * 6);
	IplImage* hist_img = cvCreateImage(cvSize(width, height), 8, 3);
	cvZero(hist_img);

	/** 用来进行HSV到RGB颜色转换的临时单位图像 */
	IplImage * hsv_color = cvCreateImage(cvSize(1, 1), 8, 3);
	IplImage * rgb_color = cvCreateImage(cvSize(1, 1), 8, 3);
	int bin_w = width / (h_bins * s_bins);

	CvScalar color;
	int currentMaxHeight = 0;
	for (int h = 0; h < h_bins; h++)
	{
		for (int s = 0; s < s_bins; s++)
		{
			int i = h*s_bins + s;
			/** 获得直方图中的统计次数，计算显示在图像中的高度 */
			float bin_val = cvQueryHistValue_2D(hist, h, s);
			int intensity = cvRound(bin_val*height / max_value);

			/** 获得当前直方图代表的颜色，转换成RGB用于绘制 */
			if (currentMaxHeight < intensity)
			{
				cvSet2D(hsv_color, 0, 0, cvScalar(h*180.f / h_bins, s*255.f / s_bins, 255, 0));
				cvCvtColor(hsv_color, rgb_color, CV_HSV2BGR);
				CvScalar col = cvGet2D(rgb_color, 0, 0);
				/*if (col.val[0] == col.val[1] == col.val[2] == 0)
				{
					continue;
				}*/
				color = col;
				currentMaxHeight = intensity;
			}
		}
	}

	return color;
}

bool Screenshot(IplImage* src, IplImage* dst, CvRect rect)
{
	try {
		cvSetImageROI(src, rect);
		cvCopy(src, dst, 0);
		cvResetImageROI(src);
		return true;
	}
	catch (cv::Exception e)
	{
	}
}

string DecimalToHexadecimal(int dec) {
	if (dec < 1) return "00";

	int hex = dec;
	string hexStr = "";

	while (dec > 0)
	{
		hex = dec % 16;

		if (hex < 10)
			hexStr = hexStr.insert(0, string(1, (hex + 48)));
		else
			hexStr = hexStr.insert(0, string(1, (hex + 55)));

		dec /= 16;
	}

	return hexStr;
}

string RGBToHexadecimal(int R, int G, int B)
{
	string rs = DecimalToHexadecimal(R);
	string gs = DecimalToHexadecimal(G);
	string bs = DecimalToHexadecimal(B);

	return '#' + rs + gs + bs;
}