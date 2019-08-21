#include "OpenGLDemo.h"
#include "SOIL.h"
// GLUT header
#include <stdlib.h>
#include <GL\glut.h>    // OpenGL GLUT Library Header
#include <opencv2\core\core.hpp>
// Open file dialog
#include "LoadFileDlg.h"

// The GLM code for loading and displying OBJ mesh file
#include "glm.h"

#include "trackball.h"

#include "FeatureExtractionVid.h"
#include "LandmarkDetectorFunc.h"
#include <boost\tokenizer.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <thread>
#include <time.h>
#include <opencv2\core\eigen.hpp>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace Eigen;

// The size of the GLUT window
int window_width = 640;
int window_height = 480;

// The OBJ model
GLMmodel* pModel = NULL;
GLMmodel* pModelCopy = NULL;

// The current modelview matrix
double pModelViewMatrix[16];

// If mouse left button is pressed
bool bLeftBntDown = false;

// Old position of the mouse
int OldX = 0;
int OldY = 0;
double WEIGHT = 5;
double PI = 3.14159;
typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
std::vector<double> allPos;
std::vector<double> xPos;
std::vector<double> yPos;
std::vector<double> zPos;
std::vector<int> mapIndex;

std::vector<double> transPos;
std::vector<double> rotPos;
std::vector<std::vector<int>> allVertexneighbors;
std::vector<int> anchorsIdx;
cv::Mat_<double> laplacianMatrix;
MatrixXd laplacianMatrixEigen;

char* objFilePath = "./Model/faceOnly.obj";
char* objFileFacePath = "./Model/faceOnly.obj";
char* mapFilePath = "./Map/lapMap2.txt";
char* output = "./output_features_vid/001.txt";
int num = 0; 
float modelScale;//human:0.858875

#define ModelVerticesNum 1948

// Initialize the OpenGL
#define GLM_OPTION GLM_SMOOTH \
        | GLM_TEXTURE

int openCVAnalysis();

void init()
{
	//glClearColor(0.0, 0.0, 0.0, 0.0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, (float)window_width / (float)window_height, 0.01f, 200.0f);

	glClearColor(1,1, 1, 1);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);

	glEnable(GL_CULL_FACE);

	// Setup other misc features.
	glEnable(GL_LIGHTING);
	glEnable(GL_NORMALIZE);
	glEnable(GL_TEXTURE_2D);
	glShadeModel(GL_SMOOTH);

	//LoadGLTextures();

	// Setup lighting model.
	GLfloat light_model_ambient[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat light0_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat light0_direction[] = { 0.0f, 0.0f, 10.0f, 0.0f };
	GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

	glLightfv(GL_LIGHT0, GL_POSITION, light0_direction);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_model_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glEnable(GL_LIGHT0);

	// Init the dlg of the open file
	PopFileInitialize(NULL);
}

double transX = 0;
double transY = 0;
double transZ = -1.5;

// Display the Object
void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if (transPos.size() > 0)
	{

		glTranslated(transPos[0] * modelScale / 40, -transPos[1] * modelScale / 40, -transPos[2] * modelScale / 40);
		transX = transPos[0] * modelScale / 40;
		transY = -transPos[1] * modelScale / 40;
		transZ = -transPos[2] * modelScale / 40;
		//std::cout << "x:" << transX << "\ty:" << transY << "\tz:" << transZ << std::endl;
	}
	else
	{
		glTranslated(transX, transY, transZ);
	}
	transPos.clear();

	glMultMatrixd(pModelViewMatrix);

	if (pModel)
	{
		glmDraw(pModel, GLM_OPTION);

	}

	//获得xPos,yPos,zPos,rotPos,transPos
	openCVAnalysis();

	//头部旋转
	addRotation2Model(rotPos);

	
	//laplacian deformation
	if (xPos.size() > 0)
	{
		updateModelVertices(xPos, yPos, zPos);
	}
	
	//vertices visulation
	verticesVisiable();

	xPos.clear();
	yPos.clear();
	zPos.clear();
	allPos.clear();
	rotPos.clear();
	num++;
	//glmDraw( pModel, GLM_FLAT );
	glutSwapBuffers();
}

GLuint  texture[1];
int LoadGLTextures()
{
	glEnable(GL_TEXTURE_2D);
	/* load an image file directly as a new OpenGL texture */
	texture[0] = SOIL_load_OGL_texture
	(
		"zyuv.png",
		SOIL_LOAD_AUTO,
		SOIL_CREATE_NEW_ID,
		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT
	);
	assert(texture[0] == 0);
	if (texture[0] == 0) {
		cout << "load texture failded!";
		return false;
	}


	// Typical Texture Generation Using Data From The Bitmap  
	glBindTexture(GL_TEXTURE_2D, texture[0]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	return true;	// Return Success  
}

//vertices visulization
void verticesVisiable()
{
	if (xPos.size() > 0)
	{
		for (int j = 0; j < 68; j++)
		{
			//drawSphere(xPos[j], -yPos[j], -zPos[j], 0.01, 20, 20);
			drawSphere(pModel->vertices[(mapIndex[j]+1) * 3], pModel->vertices[(mapIndex[j]+1)* 3 + 1], pModel->vertices[(mapIndex[j]+1) * 3 + 2], 0.005, 10, 10);
		}
	}
}

//draw the sphere at the position of the vertices
void drawSphere(GLfloat xx, GLfloat yy, GLfloat zz, GLfloat radius, GLfloat M, GLfloat N)
{
	float step_z = PI / M;
	float step_xy = 2 * PI / N;
	float x[4], y[4], z[4];

	float angle_z = 0.0;
	float angle_xy = 0.0;
	int i = 0, j = 0;
	glBegin(GL_QUADS);
	for (i = 0; i < M; i++)
	{
		angle_z = i * step_z;

		for (j = 0; j < N; j++)
		{
			angle_xy = j * step_xy;

			x[0] = radius * sin(angle_z) * cos(angle_xy);
			y[0] = radius * sin(angle_z) * sin(angle_xy);
			z[0] = radius * cos(angle_z);

			x[1] = radius * sin(angle_z + step_z) * cos(angle_xy);
			y[1] = radius * sin(angle_z + step_z) * sin(angle_xy);
			z[1] = radius * cos(angle_z + step_z);

			x[2] = radius*sin(angle_z + step_z)*cos(angle_xy + step_xy);
			y[2] = radius*sin(angle_z + step_z)*sin(angle_xy + step_xy);
			z[2] = radius*cos(angle_z + step_z);

			x[3] = radius * sin(angle_z) * cos(angle_xy + step_xy);
			y[3] = radius * sin(angle_z) * sin(angle_xy + step_xy);
			z[3] = radius * cos(angle_z);

			for (int k = 0; k < 4; k++)
			{
				glVertex3f(xx + x[k], yy + y[k], zz + z[k]);
			}
		}
	}
	glEnd();
}

// Reshape the Window
void reshape(int w, int h)
{
	// Update the window's width and height
	window_width = w;
	window_height = h;

	// Reset the viewport
	glViewport(0, 0, window_width, window_height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, (float)window_width / (float)window_height, 0.01f, 200.0f);

	glutPostRedisplay();
}

//solve the laplacian mesh
void solveLaplacianMesh(GLMmodel* pModel, std::vector<std::vector<double>> anchors, std::vector<int> anchorsIdx)
{
	int anchorNum = anchorsIdx.size();
	MatrixXd deltaEigen = MatrixXd::Zero(pModel->numvertices + anchorsIdx.size(), 3);

	//initial original matrix
	MatrixXd vertexMatrixEigen = MatrixXd::Zero(pModel->numvertices, 3);

	for (int row = 0; row < pModel->numvertices; row++)
	{
		for (int col = 0; col < 3; col++)
		{
			vertexMatrixEigen(row, col) = pModel->vertices[3 * (row + 1) + col];
			//std::cout << "v:" << pModel->vertices[3 * (row + 1) + col] << '\t';
		}
		//std::cout << std::endl;
	}

	clock_t b = clock();
	//计算delta=L*V  1s
	deltaEigen = laplacianMatrixEigen * vertexMatrixEigen;
	clock_t e = clock();
	//std::cout << "delta: " << e - b << std::endl;	

	//第i行之后赋值anchor坐标值
	for (int i = 0; i < anchorNum; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			deltaEigen(pModel->numvertices + i, j) = WEIGHT*anchors[i][j];
		}
	}

	deltaEigen.transposeInPlace();

	//delta = delta.t();
	cv::Mat_<double> resultVertexX(pModel->numvertices, 1);
	cv::Mat_<double> resultVertexY(pModel->numvertices, 1);
	cv::Mat_<double> resultVertexZ(pModel->numvertices, 1);

	leastSquare(laplacianMatrixEigen, deltaEigen.row(0), OUT resultVertexX);
	leastSquare(laplacianMatrixEigen, deltaEigen.row(1), OUT resultVertexY);
	leastSquare(laplacianMatrixEigen, deltaEigen.row(2), OUT resultVertexZ);


	for (int j = 0; j < resultVertexX.rows; j++)
	{
		vertexMatrixEigen(j, 0) = resultVertexX(j, 0);
		vertexMatrixEigen(j, 1) = resultVertexY(j, 0);
		vertexMatrixEigen(j, 2) = resultVertexZ(j, 0);
	}

	//update
	for (int k = 0; k < vertexMatrixEigen.rows(); k++)
	{
		pModel->vertices[3 * (k + 1)] = vertexMatrixEigen(k, 0);
		//std::cout << vertexMatrixEigen(k, 0) << '\t';

		pModel->vertices[3 * (k + 1) + 1] = vertexMatrixEigen(k, 1);
		//std::cout << vertexMatrixEigen(k, 1) << '\t';

		pModel->vertices[3 * (k + 1) + 2] = vertexMatrixEigen(k, 2);
		//std::cout << vertexMatrixEigen(k, 2) << std::endl;
	}
}

LDLT<MatrixXd> A;
//calculate LLLD
void calculateLTL(MatrixXd lapMat)
{
	clock_t bb = clock();
	A = (lapMat.transpose() * lapMat).ldlt();
	clock_t ee = clock();
	//std::cout << "A: " << ee - bb << std::endl;
}

void leastSquare(MatrixXd lapMat, MatrixXd delta, OUT cv::Mat_<double> resultVertex)
{
	//[8,1] temp delta
	VectorXd tempDelta(delta.cols());

	for (int delCol = 0; delCol < delta.cols(); delCol++)
	{
		tempDelta(delCol) = delta(0, delCol);
	}

	MatrixXd tempVertexMat;
	clock_t bb = clock();
	//least squares
	tempVertexMat = A.solve(lapMat.transpose() * tempDelta);
	clock_t ee = clock();
	//std::cout << "least: " << ee - bb << std::endl;
	eigen2cv(tempVertexMat, resultVertex);
}

//obtain Laplacian matrix
MatrixXd getLaplacianMatrixUmbrella(GLMmodel* pModel, std::vector<int> anchorsIdx)
{
	//initial laplacian matrix
	MatrixXd laplacianMatrixUniform = MatrixXd::Zero(ModelVerticesNum + anchorsIdx.size(), ModelVerticesNum);

	//Diagonal is the degree of each Vertex, adjacent Vertex: -1, the rest is not adjacent: 0 
	for (int row = 0; row < ModelVerticesNum; row++)
	{
		laplacianMatrixUniform(row, row) = allVertexneighbors[row].size();
		for (int i = 0; i < allVertexneighbors[row].size(); i++)
		{
			laplacianMatrixUniform(row, allVertexneighbors[row][i] - 1) = -1;
		}
	}
	for (int i = 0; i < anchorsIdx.size(); i++)
	{
		laplacianMatrixUniform(ModelVerticesNum + i, anchorsIdx[i]) = WEIGHT * 1;
	}
	return laplacianMatrixUniform;
}

//obtain all the neighbors of each vertex(Duplicated)
std::vector<std::vector<int>> getAllVertexNeighbors()
{
	std::vector<std::vector<int>> neighbors;
	std::vector<std::vector<int>> neighborsDuplicate;

	neighbors = getVertexNeighbors();

	return neighborsDuplicate = removeDuplicates(neighbors);
}

//obtain all the neighbors of each vertex
std::vector<std::vector<int>> getVertexNeighbors()
{
	FILE* file = fopen(objFileFacePath, "r");
	const int n = ModelVerticesNum;
	int array[n] = { 0 };
	std::vector<std::vector<int>> neighbors(array, array + n);
	char buf[128];
	while (fscanf(file, "%s", buf) != EOF)
	{
		switch (buf[0])
		{
		case '#':        /* comment */
						 /* eat up rest of line */
			fgets(buf, sizeof(buf), file); //get a line
			break;
		case 'f':
			fgets(buf, sizeof(buf), file);
			std::vector<int> tmp;
			char *p = strtok(buf, " ");
			while (p)
			{
				tmp.push_back(atoi(std::string(p).c_str()));
				p = strtok(NULL, " ");
			}
			for (int i = 0; i < 3; i++)
			{
				if (i == 0)
				{
					neighbors[tmp[i] - 1].push_back(tmp[i + 1]);
					neighbors[tmp[i] - 1].push_back(tmp[i + 2]);
				}
				else if (i == 1)
				{
					neighbors[tmp[i] - 1].push_back(tmp[i - 1]);
					neighbors[tmp[i] - 1].push_back(tmp[i + 1]);
				}
				else
				{
					neighbors[tmp[i] - 1].push_back(tmp[i - 2]);
					neighbors[tmp[i] - 1].push_back(tmp[i - 1]);
				}
			}
			tmp.clear();
			break;
		}
	}
	fclose(file);
	return neighbors;
}

//Duplicate
std::vector<std::vector<int>> removeDuplicates(std::vector<std::vector<int>> vec)
{
	int i = 0;
	//排序
	for (i; i < vec.size(); i++)
	{
		for (int j = 0; j < vec[i].size(); j++)
		{
			for (int k = 0; k < vec[i].size() - j - 1; k++)
			{
				if (vec[i][k] > vec[i][k + 1])
				{
					int t = vec[i][k];
					vec[i][k] = vec[i][k + 1];
					vec[i][k + 1] = t;
				}
			}
		}
		//去重
		if (vec[i].size() != 0)
		{
			int index = 0;
			std::vector<int>::iterator it = vec[i].begin();
			for (int j = 1; j < vec[i].size(); j++)
			{
				if (vec[i][index] != vec[i][j])
				{
					vec[i][++index] = vec[i][j];
				}
			}
			for (it = vec[i].begin() + index + 1; it < vec[i].end();)
			{
				vec[i].erase(it);
			}
			//std::cout << index << std::endl;
		}
	}
	return vec;
}


//read the mapping relationship between Model vertices and feature points 
void readMap()
{
	std::fstream fileIn;
	fileIn.open(mapFilePath);
	if (!fileIn.is_open())
	{
		std::cout << "File : map.txt Not found Exiting " << std::endl;
		//exit(EXIT_FAILURE);
	}
	std::string lineBuffer;
	while (!fileIn.eof())
	{
		getline(fileIn, lineBuffer, '\n');
		mapIndex.push_back(atoi(lineBuffer.c_str()));
		anchorsIdx.push_back(atoi(lineBuffer.c_str()));
	}
}

//calculate the rotation matrix
cv::Matx33d calculateRotationMatrix(std::vector<double> rotPos)
{
	cv::Vec3d euler;
	euler[0] = rotPos[0];
	euler[1] = rotPos[1];
	euler[2] = rotPos[2];
	cv::Matx33d Euler2RotationMatrix = LandmarkDetector::Euler2RotationMatrix(euler);
	return Euler2RotationMatrix;
}

//add rotation to model
void addRotation2Model(std::vector<double> rotPos)
{
	if (rotPos.size() != 0)
	{
		cv::Matx33d rotationMatrix = calculateRotationMatrix(rotPos);

		for (int i = 3; i < pModel->numvertices + 3; i++)
		{
			pModel->vertices[3 * (i - 2)] = pModelCopy->vertices[3 * (i - 2)];
			pModel->vertices[3 * (i - 2) + 1] = pModelCopy->vertices[3 * (i - 2) + 1];
			pModel->vertices[3 * (i - 2) + 2] = pModelCopy->vertices[3 * (i - 2) + 2];
			cv::Matx31d verticesMatrix(pModel->vertices[3 * (i - 2)], pModel->vertices[3 * (i - 2) + 1], pModel->vertices[3 * (i - 2) + 2]);

			//cv::Mat(verticesMatrix).t() = cv::Mat(verticesMatrix).t()* cv::Mat(rotationMatrix).t();

			verticesMatrix = rotationMatrix * verticesMatrix;
			//cout <<"matrix:"<< verticesMatrix(0, 0) << ","<< verticesMatrix(1, 0) <<","<< verticesMatrix(2, 0) << endl;

			pModel->vertices[3 * (i - 2)] = verticesMatrix(0, 0);
			pModel->vertices[3 * (i - 2) + 1] = verticesMatrix(1, 0);
			pModel->vertices[3 * (i - 2) + 2] = verticesMatrix(2, 0);
		}
	}
	else
	{
		std::cout << "No rot found!" << std::endl;
	}
}

//laplacian anchors for deformation
std::vector<std::vector<double>> anchorsForLaplacian(std::vector<double> posX, std::vector<double> poxY, std::vector<double>posZ)
{
	std::vector<std::vector<double>> anchors;
	std::vector<double> position;
	if (posX.size() > 0)
	{
		for (int j = 0; j < 68; j++)
		{
			position.push_back(posX[j]);
			position.push_back(-poxY[j]);//Output point coordinates y, z needs to reverse!!
			position.push_back(-posZ[j]);
			anchors.push_back(position);
			position.clear();
		}
	}
	return anchors;
}

//update vertices position of model
void updateModelVertices(std::vector<double> xPos, std::vector<double> yPos, std::vector<double> zPos)
{
	clock_t laplacian_start_time = clock();
	std::vector<std::vector<double>> anchors;
	anchors = anchorsForLaplacian(xPos, yPos, zPos);
	solveLaplacianMesh(pModel, anchors, anchorsIdx);
	clock_t laplacian_end_time = clock();
	//std::cout << "Laplacian running time is: " << laplacian_end_time - laplacian_start_time << std::endl;
	//if (xPos.size() > 0)
	//{
	//	for (int j = 0; j < 68; j++)
	//	{
	//		pModel->vertices[(atoi(mapIndex[j].c_str()) - 2) * 3] = xPos[j];
	//		pModel->vertices[(atoi(mapIndex[j].c_str()) - 2) * 3 + 1] = -yPos[j];
	//		pModel->vertices[(atoi(mapIndex[j].c_str()) - 2) * 3 + 2] = -zPos[j];
	//	}
	//}
	//else
	//{
	//	std::cout << "Pose Not found!" << std::endl;
	//}
}

//change the origin coordinate of model
void changeOriginPoint()
{
	int nose = 53;//female model center vertex(on the nose)

	float verticesX = pModel->vertices[3 * (nose - 2)];
	float verticesY = pModel->vertices[3 * (nose - 2) + 1];
	float verticesZ = pModel->vertices[3 * (nose - 2) + 2];
	//move the original coordinate to the nose
	for (int i = 3; i < pModel->numvertices + 3; i++)
	{
		pModel->vertices[3 * (i - 2)] -= verticesX;
		pModel->vertices[3 * (i - 2) + 1] -= verticesY;
		pModel->vertices[3 * (i - 2) + 2] -= verticesZ;
		pModelCopy->vertices[3 * (i - 2)] -= verticesX;
		pModelCopy->vertices[3 * (i - 2) + 1] -= verticesY;
		pModelCopy->vertices[3 * (i - 2) + 2] -= verticesZ;
		//printf("%d:%f %f %f\t",i-3,pModel->vertices[3*(i-2)],pModel->vertices[3*(i-2)+1],pModel->vertices[3*(i-2)+2]);
		//printf("%d:%f %f %f\n", i - 3, pModelCopy->vertices[3 * (i - 2)], pModelCopy->vertices[3 * (i - 2) + 1], pModelCopy->vertices[3 * (i - 2) + 2]);
	}

	int p1 = 4;
	int p2 = 53;
	//calculate distance between two points on the model
	double x0 = pModel->vertices[3 * (p1 - 2)];
	double y0 = pModel->vertices[3 * (p1 - 2) + 1];
	double z0 = pModel->vertices[3 * (p1 - 2) + 2];

	double x1 = pModel->vertices[3 * (p2 - 2)];
	double y1 = pModel->vertices[3 * (p2 - 2) + 1];
	double z1 = pModel->vertices[3 * (p2 - 2) + 2];

	modelScale = sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0) + (z1 - z0)*(z1 - z0));
	printf("distance:%f", modelScale);
}

//convert ruler to rotation matrix
cv::Mat_<double> Euler2RotationMatrix(const cv::Vec3d& eulerAngles)
{
	cv::Mat_<double> rotation_matrix(3, 3, 0.0);

	double s1 = sin(eulerAngles[0]);
	double s2 = sin(eulerAngles[1]);
	double s3 = sin(eulerAngles[2]);

	double c1 = cos(eulerAngles[0]);
	double c2 = cos(eulerAngles[1]);
	double c3 = cos(eulerAngles[2]);

	rotation_matrix.at<double>(0, 0) = c2 * c3;
	rotation_matrix.at<double>(0, 1) = -c2 *s3;
	rotation_matrix.at<double>(0, 2) = s2;
	rotation_matrix.at<double>(1, 0) = c1 * s3 + c3 * s1 * s2;
	rotation_matrix.at<double>(1, 1) = c1 * c3 - s1 * s2 * s3;
	rotation_matrix.at<double>(1, 2) = -c2 * s1;
	rotation_matrix.at<double>(2, 0) = s1 * s3 - c1 * c3 * s2;
	rotation_matrix.at<double>(2, 1) = c3 * s1 + c1 * s2 * s3;
	rotation_matrix.at<double>(2, 2) = c1 * c2;

	return rotation_matrix;
}

//open an obj model
void openObjFile(char* filename)
{
	// Center of the model
	float modelCenter[] = { 0.0f, 0.0f, 0.0f };

	// If there is a obj model has been loaded, destroy it
	if (pModel)
	{
		glmDelete(pModel);
		pModel = NULL;
	}

	// Load the new obj model
	pModel = glmReadOBJ(filename);
	pModelCopy = glmReadOBJ(filename);
	//std::cout << "pmodel:" << pModel << ",copy:" << pModelCopy << std::endl;

	// Generate normal for the model
	glmFacetNormals(pModel);

	// Scale the model to fit the screen
	//glmUnitize( pModel, modelCenter );

	//Init the modelview matrix as an identity matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glGetDoublev(GL_MODELVIEW_MATRIX, pModelViewMatrix);

	//标记当前窗口需要重新绘制
	glutPostRedisplay();
}

// Keyboard Messenge
void keyboard(unsigned char key, int x, int y)
{
	// The obj file will be loaded
	char FileName[128] = "";
	char TitleName[128] = "";

	// Center of the model
	float modelCenter[] = { 0.0f, 0.0f, 0.0f };

	switch (key)
	{
	case 'o':
	case 'O':
		//PopFileOpenDlg( NULL, FileName, TitleName );

		// If there is a obj model has been loaded, destroy it
		if (pModel)
		{
			glmDelete(pModel);
			pModel = NULL;
		}

		// Load the new obj model
		pModel = glmReadOBJ(FileName);

		// Generate normal for the model
		glmFacetNormals(pModel);

		// Scale the model to fit the screen
		glmUnitize(pModel, modelCenter);

		// Init the modelview matrix as an identity matrix
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glGetDoublev(GL_MODELVIEW_MATRIX, pModelViewMatrix);

		break;

	case '+':
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glLoadMatrixd(pModelViewMatrix);
		glScaled(1.05, 1.05, 1.05);
		glGetDoublev(GL_MODELVIEW_MATRIX, pModelViewMatrix);
		break;

	case '-':
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glLoadMatrixd(pModelViewMatrix);
		glScaled(0.95, 0.95, 0.95);
		glGetDoublev(GL_MODELVIEW_MATRIX, pModelViewMatrix);
		break;
	case 'w':
	case 'W':
		glmWriteOBJ(pModel, "./out/out.obj",GLM_SMOOTH|GLM_TEXTURE);
		cout << "out.obj saved" << endl;
	default:
		break;
	}

	glutPostRedisplay();
}

//Mouse Messenge
void mouse(int button, int state, int x, int y)
{
	if (pModel)
	{
		if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON)
		{
			OldX = x;
			OldY = y;
			bLeftBntDown = true;
		}
		else if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON)
		{
			bLeftBntDown = false;
		}
	}
}

// Motion Function
void motion(int x, int y)
{
	if (bLeftBntDown && pModel)
	{
		float fOldX = 2.0f*OldX / (float)window_width - 1.0f;
		float fOldY = -2.0f*OldY / (float)window_height + 1.0f;
		float fNewX = 2.0f*x / (float)window_width - 1.0f;
		float fNewY = -2.0f*y / (float)window_height + 1.0f;

		double pMatrix[16];
		trackball_opengl_matrix(pMatrix, fOldX, fOldY, fNewX, fNewY);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glLoadMatrixd(pMatrix);
		glMultMatrixd(pModelViewMatrix);
		glGetDoublev(GL_MODELVIEW_MATRIX, pModelViewMatrix);

		OldX = x;
		OldY = y;
		glutPostRedisplay();
	}
}

// Idle function
void idle(void)
{
	glutPostRedisplay();
}
