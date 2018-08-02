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
	 //���µ���ͼ���С
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

	// �Ҷ�ͼ
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);
	imwrite("grayImage.jpg", gray_image);

	// ��ֵ�����ֲ�����Ӧ��ֵ������
	Mat binary_image;
	adaptiveThreshold(gray_image, binary_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 15);
	imwrite("binaryImage.jpg", binary_image);

	// ��������
	Mat element = getStructuringElement(MORPH_RECT, Size(binary_image.cols / 50, 1));
	Mat dilateImage;
	dilate(binary_image, dilateImage, element);
	imwrite("dilateImage.jpg", dilateImage);
	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dilateImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNALֻ����ⲿ�������ɸ�������������е���

	Mat contoursImage(dilateImage.rows, dilateImage.cols, CV_8U, Scalar(255));
	int index = 0;
	vector<Rect> listBigRects;
	for (; index >= 0; index = hierarchy[index][0]) {
		cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
		// ���������
		Rect rect = boundingRect(contours[index]);

		// ��ӵ��б��У������������ע
		listBigRects.push_back(rect);

		//���������Ӿ��ο�
		//rectangle(contoursImage, rect, Scalar(0, 0, 255), 3);

		int height = rect.height;
		// ��������ߵ�1.5�����͵������ı�����Ҫ��ת��
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
		cv::Point point;//�����㣬���Ի���ͼ����  
		point.x = rect.x + rect.width - 20;//��������ͼ���к�����  
		point.y = rect.y - 10;//��������ͼ����������  
		putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

		// ����
		cv::Point pointOne;//�����㣬���Ի���ͼ����  
		pointOne.x = rect.x + rect.width;//��������ͼ���к�����  
		pointOne.y = rect.y;//��������ͼ����������  
		cv::Point pointTwo;//�����㣬���Ի���ͼ����  
		pointTwo.x = rect.x + rect.width;//��������ͼ���к�����  
		pointTwo.y = rect.y + rect.height;//��������ͼ����������  
		line(image, pointOne, pointTwo, Scalar(0, 255, 0), 1);
	}
	
	imwrite("contours_image.jpg", contoursImage);
	imwrite("mark_image.jpg", image);



	/*
		==================================================================================
		��ע�߾�
	*/
	Mat imageTwo = imread(path, 1);
	

	// �Ҷ�ͼ
	Mat gray_imageTwo;
	cvtColor(imageTwo, gray_imageTwo, CV_BGR2GRAY);
	imwrite("grayImage.jpg", gray_imageTwo);

	// //ģ��
	//Mat blur_imageTwo;
	//blur(gray_imageTwo, blur_imageTwo, Size(3, 3));
	//imwrite("2.jpg", blur_imageTwo);

	// ��ֵ�����ֲ�����Ӧ��ֵ������
	Mat binary_imageTwo;
	adaptiveThreshold(gray_imageTwo, binary_imageTwo, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 15);
	imwrite("binaryImageTwo.jpg", binary_imageTwo);

	// ��������
	Mat elementTwo = getStructuringElement(MORPH_RECT, Size(1, binary_imageTwo.rows /  70));
	Mat dilateImageTwo;
	dilate(binary_imageTwo, dilateImageTwo, elementTwo);
	imwrite("dilateImageTwo.jpg", dilateImageTwo);

	//// ȥ�룺��ֵ�˲�
	//Mat de_noiseTwo = binary_imageTwo.clone();
	//medianBlur(binary_imageTwo, de_noiseTwo, 5);
	//imwrite("4.jpg", de_noiseTwo);

	vector<vector<Point>> contoursTwo;
	vector<Vec4i> hierarchyTwo;
	findContours(dilateImageTwo, contoursTwo, hierarchyTwo, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNALֻ����ⲿ�������ɸ�������������е���

	Mat contoursImageTwo(dilateImageTwo.rows, dilateImageTwo.cols, CV_8U, Scalar(255));
	int indexTwo = 0;
	vector<Rect> listRects;
	for (; indexTwo >= 0; indexTwo = hierarchyTwo[indexTwo][0]) {
		cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
		// ���������
		Rect rect = boundingRect(contoursTwo[indexTwo]);
		listRects.push_back(rect);
		//���������Ӿ��ο�
		rectangle(contoursImageTwo, rect, Scalar(0, 0, 255), 3);
	}

	/*
		=====================================================================================
		���Ʊ߽�
	*/
	
	// ����
	vector<Rect> elementOnEdge;
	for (int i = 0; i < listBigRects.size(); i++)
	{
		Rect rect;
		for (int j = 0; j < listRects.size(); j++)
		{
			if (hasCrossOnXAndY(listBigRects[i], listRects[j]))
			{
				// ��״̬����ֱ�ӱ���
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
		������ߵ�Ԫ�ر߾�
	*/
	vector<Rect> leftEdgeRects; // ������������ߵ�Ԫ��
	vector<Rect> rightEdgeRects; // �����������ұߵ�Ԫ��
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
			// ����
			cv::Point pointOne;//�����㣬���Ի���ͼ����  
			pointOne.x = 0;//��������ͼ���к�����  
			pointOne.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//��������ͼ����������  
			cv::Point pointTwo;//�����㣬���Ի���ͼ����  
			pointTwo.x = elementOnEdge[i].x;//��������ͼ���к�����  
			pointTwo.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//��������ͼ����������  
			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);

			string pxText = intToString(elementOnEdge[i].x) + "px";
			cv::Point point;//�����㣬���Ի���ͼ����  
			point.x = pointOne.x;//��������ͼ���к�����  
			point.y = elementOnEdge[i].y + elementOnEdge[i].height / 2 - 20;//��������ͼ����������  
			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}

		if (!hasRightElement && elementOnEdge[i].x > imageWidth / 2)
		{
			rightEdgeRects.push_back(elementOnEdge[i]);
			// ����
			cv::Point pointOne;//�����㣬���Ի���ͼ����  
			pointOne.x = elementOnEdge[i].x + elementOnEdge[i].width;//��������ͼ���к�����  
			pointOne.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//��������ͼ����������  
			cv::Point pointTwo;//�����㣬���Ի���ͼ����  
			pointTwo.x = imageWidth;//��������ͼ���к�����  
			pointTwo.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//��������ͼ����������  
			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);

			int width = imageWidth - elementOnEdge[i].x - elementOnEdge[i].width;
			string pxText = intToString(width) + "px";
			cv::Point point;//�����㣬���Ի���ͼ����  
			point.x = pointOne.x;//��������ͼ���к�����  
			point.y = elementOnEdge[i].y + elementOnEdge[i].height / 2 - 20;//��������ͼ����������  
			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
	}

	/*
		==================================================================================
		�����м��Ԫ�ر߾�
	*/
	for (int i = 0; i < elementOnEdge.size(); i++)
	{
		// �޳����Ԫ��
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

		// �ж��Ա��Ƿ���С��5���ص�Ԫ��
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
					// ��������Ƿ���Ԫ�أ�1��x��������Ҫ�������Ԫ�ص�right   2��y����Ҫ�н���   3��x��ĳ���Ҫ����5		
					if (elementOnEdge[j].x + elementOnEdge[j].width < elementOnEdge[i].x && (elementOnEdge[i].x - elementOnEdge[j].x - elementOnEdge[j].width) > 10)
					{
						x = elementOnEdge[j].x + elementOnEdge[j].width;
					}
				}
				else
				{
					// �����ұ��Ƿ���Ԫ�أ�1��Ԫ�ص�right��ҪС������Ԫ�ص�x��   2��y����Ҫ�н���   3��x��ĳ���Ҫ����5		
					if (elementOnEdge[j].x >(elementOnEdge[i].x + elementOnEdge[i].width) && (elementOnEdge[j].x - elementOnEdge[i].x - elementOnEdge[i].width) > 10)
					{
						x = elementOnEdge[j].x;
					}
				}
			}
		}

		if (elementOnEdge[i].x <= imageWidth / 2)
		{
			cv::Point pointOne;//�����㣬���Ի���ͼ����  
			pointOne.x = x;//��������ͼ���к�����  
			pointOne.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//��������ͼ����������  
			cv::Point pointTwo;//�����㣬���Ի���ͼ����  
			pointTwo.x = elementOnEdge[i].x;//��������ͼ���к�����  
			pointTwo.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//��������ͼ����������  
			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);

			int width = elementOnEdge[i].x - x;
			string pxText = intToString(width) + "px";
			cv::Point point;//�����㣬���Ի���ͼ����  
			point.x = pointOne.x;//��������ͼ���к�����  
			point.y = elementOnEdge[i].y + elementOnEdge[i].height / 2 - 20;//��������ͼ����������  
			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
		else
		{
			cv::Point pointOne;//�����㣬���Ի���ͼ����  
			pointOne.x = elementOnEdge[i].x + elementOnEdge[i].width;//��������ͼ���к�����  
			pointOne.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//��������ͼ����������  
			cv::Point pointTwo;//�����㣬���Ի���ͼ����  
			pointTwo.x = x;//��������ͼ���к�����  
			pointTwo.y = elementOnEdge[i].y + elementOnEdge[i].height / 2;//��������ͼ����������  
			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);

			int width = x - elementOnEdge[i].x - elementOnEdge[i].width;
			string pxText = intToString(width) + "px";
			cv::Point point;//�����㣬���Ի���ͼ����  
			point.x = pointOne.x;//��������ͼ���к�����  
			point.y = elementOnEdge[i].y + elementOnEdge[i].height / 2 - 20;//��������ͼ����������  
			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
	}

	imwrite("contours_image_two.jpg", contoursImageTwo);
	imwrite("biaozhu_finish.jpg", image);
	imshow("img", image);

	cout << "��ɼ��";

	/// �ȴ��û�����
	waitKey();
	return 0;
}

/*
	�ж��Ƿ���ͬһˮƽ������
*/
bool hasCross(Rect rectOne, Rect rectTwo)
{
	// Aλ��B�Ϸ�
	if (rectOne.y < rectTwo.y && (rectOne.y + rectOne.height) < rectTwo.y)
	{
		return false;
	}
	// Aλ��B�·�
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
	�ж�Ԫ���Ƿ����������н���
*/
bool hasCrossOnXAndY(Rect rectOne, Rect rectTwo)
{
	// Aλ��B�Ϸ�
	if (rectOne.y < rectTwo.y && (rectOne.y + rectOne.height) < rectTwo.y)
	{
		return false;
	}
	// Aλ��B�·�
	else if (rectOne.y > (rectTwo.y + rectTwo.height))
	{
		return false;
	}
	// Aλ��B��
	else if (rectOne.x < rectTwo.x && (rectOne.x + rectOne.width) < rectTwo.x)
	{
		return false;
	}
	// Aλ��B�ҷ�
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

	/** H ��������Ϊ16���ȼ���S��������Ϊ8���ȼ� */
	int h_bins = 16, s_bins = 8;
	int hist_size[] = { h_bins, s_bins };

	/** H �����ı仯��Χ */
	float h_ranges[] = { 0, 180 };

	/** S �����ı仯��Χ*/
	float s_ranges[] = { 0, 255 };
	float* ranges[] = { h_ranges, s_ranges };

	/** ����ͼ��ת����HSV��ɫ�ռ� */
	cvCvtColor(src, hsv, CV_BGR2HSV);
	cvSplit(hsv, h_plane, s_plane, v_plane, 0);

	/** ����ֱ��ͼ����ά, ÿ��ά���Ͼ��� */
	CvHistogram * hist = cvCreateHist(2, hist_size, CV_HIST_ARRAY, ranges, 1);
	/** ����H,S����ƽ������ͳ��ֱ��ͼ */
	cvCalcHist(planes, hist, 0, 0);

	/** ��ȡֱ��ͼͳ�Ƶ����ֵ�����ڶ�̬��ʾֱ��ͼ */
	float max_value;
	cvGetMinMaxHistValue(hist, 0, &max_value, 0, 0);


	/** ����ֱ��ͼ��ʾͼ�� */
	int height = 240;
	int width = (h_bins*s_bins * 6);
	IplImage* hist_img = cvCreateImage(cvSize(width, height), 8, 3);
	cvZero(hist_img);

	/** ��������HSV��RGB��ɫת������ʱ��λͼ�� */
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
			/** ���ֱ��ͼ�е�ͳ�ƴ�����������ʾ��ͼ���еĸ߶� */
			float bin_val = cvQueryHistValue_2D(hist, h, s);
			int intensity = cvRound(bin_val*height / max_value);

			/** ��õ�ǰֱ��ͼ�������ɫ��ת����RGB���ڻ��� */
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