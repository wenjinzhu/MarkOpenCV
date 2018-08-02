//#include "stdio.h"
//#include "opencv/cv.h"
//#include "opencv/highgui.h"
//#include <opencv2/opencv.hpp>
//#include <tchar.h>
//#include <iostream>
//#include <fstream>
//#include <string.h>
//#include <math.h>
//#include <list>
//
//#define cvQueryHistValue_2D( hist, idx0, idx1 )   cvGetReal2D( (hist)->bins, (idx0), (idx1) )
//
//
//using namespace std;
//using namespace cv;
//
//string intToString(int v);
//int changeSize(int oldSize);
//CvScalar getMainColor(IplImage *src);
//bool Screenshot(IplImage* src, IplImage* dst, CvRect rect);
//string RGBToHexadecimal(int R, int G, int B);
//string DecimalToHexadecimal(int dec);
//CvScalar DCD(IplImage* src);
//bool hasCross(Rect rectOne, Rect rectTwo);
//
//
//int main(int argc, char** argv)
//{
//
//	Mat image = imread("C:/AllFiles/VisualStudioWorkspace/MarkOpenCV/MarkOpenCV/x64/Debug/image/test1.jpg", 1);
//
//	// �Ҷ�ͼ
//	Mat gray_image;
//	cvtColor(image, gray_image, CV_BGR2GRAY);
//	imwrite("grayImage.jpg", gray_image);
//
//	// ��ֵ�����ֲ�����Ӧ��ֵ������
//	Mat binary_image;
//	adaptiveThreshold(gray_image, binary_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 15);
//	imwrite("binaryImage.jpg", binary_image);
//
//	// ��������
//	Mat element = getStructuringElement(MORPH_RECT, Size(binary_image.cols / 35, 1));
//	Mat dilateImage;
//	dilate(binary_image, dilateImage, element);
//	imwrite("dilateImage.jpg", dilateImage);
//
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierarchy;
//	findContours(dilateImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNALֻ����ⲿ�������ɸ�������������е���
//
//	Mat contoursImage(dilateImage.rows, dilateImage.cols, CV_8U, Scalar(255));
//	int index = 0;
//	vector<Rect> listBigRects;
//	for (; index >= 0; index = hierarchy[index][0]) {
//		cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
//		// ���������
//		Rect rect = boundingRect(contours[index]);
//
//		// ��ӵ��б��У������������ע
//
//		//���������Ӿ��ο�
//		rectangle(contoursImage, rect, Scalar(0, 0, 255), 3);
//
//		int height = rect.height;
//		// ��������ߵ�1.5�����͵������ı�����Ҫ��ת��
//		if (height * 1.5f < rect.width)
//		{
//			height = changeSize(height);
//		}
//		string pxText = intToString(height) + "px";
//		cv::Point point;//�����㣬���Ի���ͼ����  
//		point.x = rect.x + rect.width / 2;//��������ͼ���к�����  
//		point.y = rect.y - 10;//��������ͼ����������  
//		putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//
//		//cv::Point pointOne;//�����㣬���Ի���ͼ����  
//		//pointOne.x = rect.x;//��������ͼ���к�����  
//		//pointOne.y = rect.y ;//��������ͼ����������  
//		//cv::Point pointTwo;//�����㣬���Ի���ͼ����  
//		//pointTwo.x = rect.x + rect.width;//��������ͼ���к�����  
//		//pointTwo.y = rect.y ;//��������ͼ����������  
//		//cv::Point pointThree;//�����㣬���Ի���ͼ����  
//		//pointThree.x = rect.x ;//��������ͼ���к�����  
//		//pointThree.y = rect.y + rect.height;//��������ͼ����������  
//		//cv::Point pointFour;//�����㣬���Ի���ͼ����  
//		//pointFour.x = rect.x + rect.width;//��������ͼ���к�����  
//		//pointFour.y = rect.y + rect.height;//��������ͼ����������  
//
//		//line(image, pointOne, pointTwo, Scalar(0, 255, 0),  1);
//		//line(image, pointTwo, pointFour, Scalar(0, 255, 0), 1);
//		//line(image, pointOne, pointThree, Scalar(0, 255, 0), 1);
//		//line(image, pointThree, pointFour, Scalar(0, 255, 0), 1);
//	}
//
//	imwrite("contours_image.jpg", contoursImage);
//	imwrite("mark_image.jpg", image);
//
//
//
//	/*
//	==================================================================================
//	��ע�߾�
//	*/
//	Mat imageTwo = imread("C:/AllFiles/VisualStudioWorkspace/MarkOpenCV/MarkOpenCV/x64/Debug/image/test1.jpg", 1);
//	IplImage imageForWidth = IplImage(imageTwo);
//	int imageWidth = imageForWidth.width;
//
//	// �Ҷ�ͼ
//	Mat gray_imageTwo;
//	cvtColor(imageTwo, gray_imageTwo, CV_BGR2GRAY);
//	imwrite("grayImage.jpg", gray_imageTwo);
//
//	//ģ��
//	Mat blur_imageTwo;
//	blur(gray_imageTwo, blur_imageTwo, Size(3, 3));
//	imwrite("2.jpg", blur_imageTwo);
//
//	// ��ֵ�����ֲ�����Ӧ��ֵ������
//	Mat binary_imageTwo;
//	adaptiveThreshold(blur_imageTwo, binary_imageTwo, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 17, 15);
//	imwrite("binaryImageTwo.jpg", binary_imageTwo);
//
//	// ��������
//	Mat elementTwo = getStructuringElement(MORPH_RECT, Size(1, binary_imageTwo.rows / 70));
//	Mat dilateImageTwo;
//	dilate(binary_imageTwo, dilateImageTwo, elementTwo);
//	imwrite("dilateImageTwo.jpg", dilateImageTwo);
//
//	//// ȥ�룺��ֵ�˲�
//	//Mat de_noiseTwo = binary_imageTwo.clone();
//	//medianBlur(binary_imageTwo, de_noiseTwo, 5);
//	//imwrite("4.jpg", de_noiseTwo);
//
//	vector<vector<Point>> contoursTwo;
//	vector<Vec4i> hierarchyTwo;
//	findContours(dilateImageTwo, contoursTwo, hierarchyTwo, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNALֻ����ⲿ�������ɸ�������������е���
//
//	Mat contoursImageTwo(dilateImageTwo.rows, dilateImageTwo.cols, CV_8U, Scalar(255));
//	int indexTwo = 0;
//	vector<Rect> listRects;
//	for (; indexTwo >= 0; indexTwo = hierarchyTwo[indexTwo][0]) {
//		cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
//		// ���������
//		Rect rect = boundingRect(contoursTwo[indexTwo]);
//		listRects.push_back(rect);
//		//���������Ӿ��ο�
//		rectangle(contoursImageTwo, rect, Scalar(0, 0, 255), 3);
//	}
//
//
//	/*
//	===================================================================================
//	������ߵ�Ԫ�ر߾�
//	*/
//	vector<Rect> leftEdgeRects; // ������������ߵ�Ԫ��
//	vector<Rect> rightEdgeRects; // �����������ұߵ�Ԫ��
//	for (int i = 0; i < listRects.size() - 1; i++)
//	{
//		bool hasLeftElement = false;
//		bool hasRightElement = false;
//
//		for (int j = 1; j < listRects.size(); j++)
//		{
//			if (hasCross(listRects[i], listRects[j]))
//			{
//				if (listRects[i].x <= imageWidth / 2)
//				{
//					if (listRects[j].x < listRects[i].x)
//					{
//						hasLeftElement = true;
//					}
//				}
//				else
//				{
//					if ((listRects[j].x + listRects[j].width) >(listRects[i].x + listRects[i].width))
//					{
//						hasRightElement = true;
//					}
//				}
//			}
//		}
//
//		if (!hasLeftElement && listRects[i].x <= imageWidth / 2)
//		{
//			leftEdgeRects.push_back(listRects[i]);
//			// ����
//			cv::Point pointOne;//�����㣬���Ի���ͼ����  
//			pointOne.x = 0;//��������ͼ���к�����  
//			pointOne.y = listRects[i].y + listRects[i].height / 2;//��������ͼ����������  
//			cv::Point pointTwo;//�����㣬���Ի���ͼ����  
//			pointTwo.x = listRects[i].x;//��������ͼ���к�����  
//			pointTwo.y = listRects[i].y + listRects[i].height / 2;//��������ͼ����������  
//			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);
//
//			string pxText = intToString(listRects[i].x) + "px";
//			cv::Point point;//�����㣬���Ի���ͼ����  
//			point.x = pointOne.x;//��������ͼ���к�����  
//			point.y = listRects[i].y + listRects[i].height / 2 - 20;//��������ͼ����������  
//			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
//		}
//
//		if (!hasRightElement && listRects[i].x > imageWidth / 2)
//		{
//			rightEdgeRects.push_back(listRects[i]);
//			// ����
//			cv::Point pointOne;//�����㣬���Ի���ͼ����  
//			pointOne.x = listRects[i].x + listRects[i].width;//��������ͼ���к�����  
//			pointOne.y = listRects[i].y + listRects[i].height / 2;//��������ͼ����������  
//			cv::Point pointTwo;//�����㣬���Ի���ͼ����  
//			pointTwo.x = imageWidth;//��������ͼ���к�����  
//			pointTwo.y = listRects[i].y + listRects[i].height / 2;//��������ͼ����������  
//			line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);
//
//			int width = imageWidth - listRects[i].x - listRects[i].width;
//			string pxText = intToString(width) + "px";
//			cv::Point point;//�����㣬���Ի���ͼ����  
//			point.x = pointOne.x;//��������ͼ���к�����  
//			point.y = listRects[i].y + listRects[i].height / 2 - 20;//��������ͼ����������  
//			putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
//		}
//	}
//
//	/*
//	==================================================================================
//	�����м��Ԫ�ر߾�
//	*/
//	for (int i = 0; i < listRects.size(); i++)
//	{
//		// �޳����Ԫ��
//		bool isLeftElement = false;
//		for (int left = 0; left < leftEdgeRects.size(); left++)
//		{
//			if (listRects[i].x == leftEdgeRects[left].x && listRects[i].y == leftEdgeRects[left].y &&listRects[i].width == leftEdgeRects[left].width &&listRects[i].height == leftEdgeRects[left].height)
//			{
//				isLeftElement = true;
//				break;
//			}
//		}
//		if (isLeftElement)
//		{
//			continue;
//		}
//		bool isRightElement = false;
//		for (int right = 0; right < rightEdgeRects.size(); right++)
//		{
//			if (listRects[i].x == rightEdgeRects[right].x && listRects[i].y == rightEdgeRects[right].y &&listRects[i].width == rightEdgeRects[right].width &&listRects[i].height == rightEdgeRects[right].height)
//			{
//				isRightElement = true;
//				break;
//			}
//		}
//		if (isRightElement)
//		{
//			continue;
//		}
//
//		// �ж��Ա��Ƿ���С��5���ص�Ԫ��
//		int x = 0;
//		bool hasLeftCloseElement = false;
//		bool hasRightCloseElement = false;
//		for (int j = 0; j < listRects.size(); j++)
//		{
//			if (i == j)
//			{
//				continue;
//			}
//			if (hasCross(listRects[i], listRects[j]))
//			{
//				// У���Ƿ����������Ԫ��
//				if (listRects[j].x + listRects[j].width < listRects[i].x && (listRects[i].x - listRects[j].x - listRects[j].width) < 10)
//				{
//					hasLeftCloseElement = true;
//				}
//				// У���Ƿ����ұ�����Ԫ��
//				if ((listRects[j].x > listRects[i].x + listRects[i].width) && (listRects[j].x - listRects[i].x - listRects[i].width) < 10)
//				{
//					hasRightCloseElement = true;
//				}
//				if (listRects[i].x <= imageWidth / 2)
//				{
//					// ��������Ƿ���Ԫ�أ�1��x��������Ҫ�������Ԫ�ص�right   2��y����Ҫ�н���   3��x��ĳ���Ҫ����5		
//					if (listRects[j].x + listRects[j].width < listRects[i].x && (listRects[i].x - listRects[j].x - listRects[j].width) > 10)
//					{
//						x = listRects[j].x + listRects[j].width;
//					}
//				}
//				else
//				{
//					// �����ұ��Ƿ���Ԫ�أ�1��Ԫ�ص�right��ҪС������Ԫ�ص�x��   2��y����Ҫ�н���   3��x��ĳ���Ҫ����5		
//					if (listRects[j].x >(listRects[i].x + listRects[i].width) && (listRects[j].x - listRects[i].x - listRects[i].width) > 10)
//					{
//						x = listRects[j].x;
//					}
//				}
//				//if (listRects[j].x + listRects[j].width < listRects[i].x && (listRects[i].x - listRects[j].x - listRects[j].width) < 5)
//				//{
//				//	// ֻҪ��������Ԫ�أ�������Ҫ���ǻ��Ʊ߾�
//				//	hasCloseElement = true;
//				//}
//				//else if ((listRects[j].x > listRects[i].x + listRects[i].width) && (listRects[j].x - listRects[i].x - listRects[i].width) < 5)
//				//{
//				//	// ֻҪ��������Ԫ�أ�������Ҫ���ǻ��Ʊ߾�
//				//	hasCloseElement = true;
//				//}
//			}
//		}
//
//		if (!hasLeftCloseElement || !hasRightCloseElement)
//		{
//			if (listRects[i].x <= imageWidth / 2 && !hasLeftCloseElement)
//			{
//				cv::Point pointOne;//�����㣬���Ի���ͼ����  
//				pointOne.x = x;//��������ͼ���к�����  
//				pointOne.y = listRects[i].y + listRects[i].height / 2;//��������ͼ����������  
//				cv::Point pointTwo;//�����㣬���Ի���ͼ����  
//				pointTwo.x = listRects[i].x;//��������ͼ���к�����  
//				pointTwo.y = listRects[i].y + listRects[i].height / 2;//��������ͼ����������  
//				line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);
//
//				int width = listRects[i].x - x;
//				string pxText = intToString(width) + "px";
//				cv::Point point;//�����㣬���Ի���ͼ����  
//				point.x = pointOne.x;//��������ͼ���к�����  
//				point.y = listRects[i].y + listRects[i].height / 2 - 20;//��������ͼ����������  
//				putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
//			}
//			if (listRects[i].x > imageWidth / 2 && !hasRightCloseElement)
//			{
//				cv::Point pointOne;//�����㣬���Ի���ͼ����  
//				pointOne.x = listRects[i].x + listRects[i].width;//��������ͼ���к�����  
//				pointOne.y = listRects[i].y + listRects[i].height / 2;//��������ͼ����������  
//				cv::Point pointTwo;//�����㣬���Ի���ͼ����  
//				pointTwo.x = x;//��������ͼ���к�����  
//				pointTwo.y = listRects[i].y + listRects[i].height / 2;//��������ͼ����������  
//				line(image, pointOne, pointTwo, Scalar(0, 0, 255), 1);
//
//				int width = x - listRects[i].x - listRects[i].width;
//				string pxText = intToString(width) + "px";
//				cv::Point point;//�����㣬���Ի���ͼ����  
//				point.x = pointOne.x;//��������ͼ���к�����  
//				point.y = listRects[i].y + listRects[i].height / 2 - 20;//��������ͼ����������  
//				putText(image, pxText, point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
//			}
//		}
//	}
//
//	imwrite("contours_image_two.jpg", contoursImageTwo);
//	imwrite("mark_image.jpg", imageTwo);
//	imwrite("biaozhu_finish.jpg", image);
//	imshow("img", image);
//
//	cout << "��ɼ��";
//
//	/// �ȴ��û�����
//	waitKey();
//	return 0;
//}
//
//bool hasCross(Rect rectOne, Rect rectTwo)
//{
//	// Aλ��B�Ϸ�
//	if (rectOne.y < rectTwo.y && (rectOne.y + rectOne.height) < rectTwo.y)
//	{
//		return false;
//	}
//	// Aλ��B�·�
//	else if (rectOne.y > (rectTwo.y + rectTwo.height))
//	{
//		return false;
//	}
//	else
//	{
//		return true;
//	}
//}
//
//int changeSize(int oldSize)
//{
//	if (oldSize <= 22)
//	{
//		return 22;
//	}
//	else if (oldSize > 22 && oldSize <= 24)
//	{
//		return 24;
//	}
//	else if (oldSize > 24 && oldSize <= 28)
//	{
//		return 28;
//	}
//	else if (oldSize > 28 && oldSize <= 32)
//	{
//		return 32;
//	}
//	else
//	{
//		return 36;
//	}
//}
//
//string intToString(int v)
//{
//	char buf[32] = { 0 };
//	snprintf(buf, sizeof(buf), "%u", v);
//
//	string str = buf;
//	return str;
//}
//
//CvScalar getMainColor(IplImage *src)
//{
//	IplImage* hsv = cvCreateImage(cvGetSize(src), 8, 3);
//	IplImage* h_plane = cvCreateImage(cvGetSize(src), 8, 1);
//	IplImage* s_plane = cvCreateImage(cvGetSize(src), 8, 1);
//	IplImage* v_plane = cvCreateImage(cvGetSize(src), 8, 1);
//	IplImage* planes[] = { h_plane, s_plane };
//
//	/** H ��������Ϊ16���ȼ���S��������Ϊ8���ȼ� */
//	int h_bins = 16, s_bins = 8;
//	int hist_size[] = { h_bins, s_bins };
//
//	/** H �����ı仯��Χ */
//	float h_ranges[] = { 0, 180 };
//
//	/** S �����ı仯��Χ*/
//	float s_ranges[] = { 0, 255 };
//	float* ranges[] = { h_ranges, s_ranges };
//
//	/** ����ͼ��ת����HSV��ɫ�ռ� */
//	cvCvtColor(src, hsv, CV_BGR2HSV);
//	cvSplit(hsv, h_plane, s_plane, v_plane, 0);
//
//	/** ����ֱ��ͼ����ά, ÿ��ά���Ͼ��� */
//	CvHistogram * hist = cvCreateHist(2, hist_size, CV_HIST_ARRAY, ranges, 1);
//	/** ����H,S����ƽ������ͳ��ֱ��ͼ */
//	cvCalcHist(planes, hist, 0, 0);
//
//	/** ��ȡֱ��ͼͳ�Ƶ����ֵ�����ڶ�̬��ʾֱ��ͼ */
//	float max_value;
//	cvGetMinMaxHistValue(hist, 0, &max_value, 0, 0);
//
//
//	/** ����ֱ��ͼ��ʾͼ�� */
//	int height = 240;
//	int width = (h_bins*s_bins * 6);
//	IplImage* hist_img = cvCreateImage(cvSize(width, height), 8, 3);
//	cvZero(hist_img);
//
//	/** ��������HSV��RGB��ɫת������ʱ��λͼ�� */
//	IplImage * hsv_color = cvCreateImage(cvSize(1, 1), 8, 3);
//	IplImage * rgb_color = cvCreateImage(cvSize(1, 1), 8, 3);
//	int bin_w = width / (h_bins * s_bins);
//
//	CvScalar color;
//	int currentMaxHeight = 0;
//	for (int h = 0; h < h_bins; h++)
//	{
//		for (int s = 0; s < s_bins; s++)
//		{
//			int i = h*s_bins + s;
//			/** ���ֱ��ͼ�е�ͳ�ƴ�����������ʾ��ͼ���еĸ߶� */
//			float bin_val = cvQueryHistValue_2D(hist, h, s);
//			int intensity = cvRound(bin_val*height / max_value);
//
//			/** ��õ�ǰֱ��ͼ�������ɫ��ת����RGB���ڻ��� */
//			if (currentMaxHeight < intensity)
//			{
//				cvSet2D(hsv_color, 0, 0, cvScalar(h*180.f / h_bins, s*255.f / s_bins, 255, 0));
//				cvCvtColor(hsv_color, rgb_color, CV_HSV2BGR);
//				CvScalar col = cvGet2D(rgb_color, 0, 0);
//				/*if (col.val[0] == col.val[1] == col.val[2] == 0)
//				{
//				continue;
//				}*/
//				color = col;
//				currentMaxHeight = intensity;
//			}
//		}
//	}
//
//	return color;
//}
//
//bool Screenshot(IplImage* src, IplImage* dst, CvRect rect)
//{
//	try {
//		cvSetImageROI(src, rect);
//		cvCopy(src, dst, 0);
//		cvResetImageROI(src);
//		return true;
//	}
//	catch (cv::Exception e)
//	{
//	}
//}
//
//string DecimalToHexadecimal(int dec) {
//	if (dec < 1) return "00";
//
//	int hex = dec;
//	string hexStr = "";
//
//	while (dec > 0)
//	{
//		hex = dec % 16;
//
//		if (hex < 10)
//			hexStr = hexStr.insert(0, string(1, (hex + 48)));
//		else
//			hexStr = hexStr.insert(0, string(1, (hex + 55)));
//
//		dec /= 16;
//	}
//
//	return hexStr;
//}
//
//string RGBToHexadecimal(int R, int G, int B)
//{
//	string rs = DecimalToHexadecimal(R);
//	string gs = DecimalToHexadecimal(G);
//	string bs = DecimalToHexadecimal(B);
//
//	return '#' + rs + gs + bs;
//}
//
//#define CLUST_NUM 5
//
//CvScalar DCD(IplImage* src)
//{
//	struct Node
//	{
//		CvPoint point;
//		CvScalar color;
//	};
//
//	struct Clust
//	{
//		Node center;
//		vector<Node> buff;
//	};
//
//	//IplImage* src = NULL;
//	//src = cvLoadImage("test.jpg");
//
//	if (!src)
//	{
//		return NULL;
//	}
//
//	int width = src->width;
//	int height = src->height;
//
//	// �����������
//	CvPoint point[CLUST_NUM];
//	CvRNG rng(cvGetTickCount());
//	for (int i = 0; i< CLUST_NUM; i++)
//	{
//		point[i].x = cvRandInt(&rng) % width;
//		point[i].y = cvRandInt(&rng) % height;
//	}
//
//	//�����
//	Clust v[CLUST_NUM];
//
//	//��ʼ��
//	for (int i = 0; i < CLUST_NUM; i++)
//	{
//		v[i].center.point = point[i];
//		v[i].center.color = cvGet2D(src, v[i].center.point.y, v[i].center.point.x);
//
//		cout << point[i].x << ":" << point[i].y << endl;
//
//		Node node;
//		node.point = v[i].center.point;
//		node.color = v[i].center.color;
//		v[i].buff.push_back(node);
//	}
//
//	do
//	{
//		for (int rows = 0; rows < height; rows++)
//		{
//			uchar* ptr = (uchar*)src->imageData + src->widthStep*rows;
//			for (int cols = 0; cols < width; cols++)
//			{
//				int b, g, r;
//				b = ptr[cols*src->nChannels + 0];
//				g = ptr[cols*src->nChannels + 1];
//				r = ptr[cols*src->nChannels + 2];
//
//				//����ÿ�����ص���ÿ���ص����ľ���
//				double dis[CLUST_NUM] = { 0.0 };
//				for (int i = 0; i < CLUST_NUM; i++)
//				{
//					dis[i] = (v[i].center.color.val[0] - b)*(v[i].center.color.val[0] - b) +
//						(v[i].center.color.val[1] - g)*(v[i].center.color.val[1] - g) +
//						(v[i].center.color.val[2] - r)*(v[i].center.color.val[2] - r);
//					dis[i] = sqrt(dis[i]);
//				}
//
//				//��ȡ�����ص��������Ĵصı��
//				int minDisClustId = 0;
//				int i = 0;
//				for (i = 1; i < CLUST_NUM; i++)
//				{
//					if (dis[i] < dis[minDisClustId])
//					{
//						minDisClustId = i;
//					}
//
//				}
//
//				Node node;
//				node.color.val[0] = b;
//				node.color.val[1] = g;
//				node.color.val[2] = r;
//
//				node.point.x = cols;
//				node.point.y = rows;
//
//				v[minDisClustId].buff.push_back(node);
//			}
//		}
//
//		for (int i = 0; i < CLUST_NUM; i++)
//		{
//			double avgR = 0.0;
//			double avgG = 0.0;
//			double avgB = 0.0;
//
//			//����ÿһ�ص�ƽ��RGB
//			vector<Node>::iterator it = v[i].buff.begin();
//			while (it != v[i].buff.end())
//			{
//				avgB += it->color.val[0];
//				avgG += it->color.val[1];
//				avgR += it->color.val[2];
//				it++;
//			}
//
//			avgB /= v[i].buff.size();
//			avgG /= v[i].buff.size();
//			avgR /= v[i].buff.size();
//
//			//��þ���ƽ��RGB����ĵ�
//			it = v[i].buff.begin();
//
//			double curDis = 0.0;
//			double minDis = 0.0;
//			int curId = 0;
//			int minId = 0;
//
//			while (it != v[i].buff.end())
//			{
//				if (it == v[i].buff.begin())
//				{
//					minDis = (it->color.val[0] - avgB)*(it->color.val[0] - avgB) +
//						(it->color.val[1] - avgG)*(it->color.val[1] - avgG) +
//						(it->color.val[2] - avgR)*(it->color.val[2] - avgR);
//
//					minDis = sqrt(minDis);
//					minId = 0;
//					it++;
//				}
//				else
//				{
//					curDis = (it->color.val[0] - avgB)*(it->color.val[0] - avgB) +
//						(it->color.val[1] - avgG)*(it->color.val[1] - avgG) +
//						(it->color.val[2] - avgR)*(it->color.val[2] - avgR);
//					curDis = sqrt(curDis);
//
//					if (curDis < minDis)
//					{
//						minDis = curDis;
//						minId = it - v[i].buff.begin();
//					}
//					it++;
//				}
//			}
//			//����µ�ÿһ�ص����ĵ�
//			v[i].center.point = v[i].buff.at(minId).point;
//			v[i].center.color = v[i].buff.at(minId).color;
//		}
//
//		//�����ֹ���� �Ƿ����е����ĵ㶼û�иı�
//		int flg = 0;
//		for (int i = 0; i < CLUST_NUM; i++)
//		{
//			for (int j = 0; j < CLUST_NUM; j++)
//			{
//				if (point[i].x == v[j].center.point.x &&
//					point[i].y == v[j].center.point.y)
//				{
//					flg++;
//					break;
//				}
//			}
//		}
//
//		if (flg == CLUST_NUM)
//		{
//			break;
//		}
//
//		//���ÿһ�صĵ� �������ĵ�
//		for (int i = 0; i < CLUST_NUM; i++)
//		{
//			//cout << v[i].center.point.x << ":" << v[i].center.point.y << "size:" << v[i].buff.size() << endl;
//			v[i].buff.clear();
//
//			//���ĵ�����Ӧ�Ĵ�
//			Node node;
//			node.point = v[i].center.point;
//			node.color = v[i].center.color;
//			v[i].buff.push_back(node);
//
//			point[i] = v[i].center.point;
//		}
//		cout << "*" << endl;
//	} while (1);
//
//	IplImage* dst = NULL;
//	IplImage* lab = NULL;
//
//	dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
//	lab = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
//
//	if (!dst || !lab)
//	{
//		return NULL;
//	}
//
//	CvScalar color = v[0].center.color;
//	return color;
//
//	//for (int i = 0; i < CLUST_NUM; i++)
//	//{
//	//	CvScalar color;
//	//	color = v[i].center.color;
//	//	cvSetImageROI(dst, cvRect(i*src->width / CLUST_NUM, 0, src->width / CLUST_NUM, src->height));
//	//	cvSet(dst, color);
//	//	cvResetImageROI(dst);
//
//	//	vector<Node>::iterator it = v[i].buff.begin();
//	//	while (it != v[i].buff.end())
//	//	{
//	//		cvSet2D(lab, it->point.y, it->point.x, v[i].center.color);
//	//		it++;
//	//	}
//	//}
//}