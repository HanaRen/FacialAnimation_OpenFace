#pragma once
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "glm.h"

using namespace Eigen;

extern GLMmodel* pModel;

extern char* objFilePath;
extern char* objFileFacePath;
extern char* output;

extern std::vector<double> allPos;
extern std::vector<double> xPos;
extern std::vector<double> yPos;
extern std::vector<double> zPos;

extern float modelScale;

extern std::vector<double> rotPos;
extern std::vector<double> transPos;

extern std::vector<std::vector<int>> allVertexneighbors;
extern std::vector<int> anchorsIdx;
extern cv::Mat_<double> laplacianMatrix;
extern MatrixXd laplacianMatrixEigen;

// Init the OpenGL
void init(void);

// Display the Object
void display(void);

// Reshape the Window
void reshape(int w, int h);

// Mouse Messenge
void mouse(int button, int state, int x, int y);

// Motion Function
void motion(int x, int y);

// Keyboard Messenge
void keyboard(unsigned char key, int x, int y);

// Idle Function
void idle(void);

//Read the human.obj file
void openObjFile(char* filename);

//Read mapping file
void readMap();

cv::Mat_<double> Euler2RotationMatrix(const cv::Vec3d& eulerAngles);

cv::Matx33d calculateRotationMatrix(std::vector<double> rotPos);

void addRotation2Model(std::vector<double> rotPos);

void updateModelVertices(std::vector<double> xPos, std::vector<double> yPos, std::vector<double> zPos);

void drawSphere(GLfloat xx, GLfloat yy, GLfloat zz, GLfloat radius, GLfloat M, GLfloat N);

void changeOriginPoint();

void verticesVisiable();

std::vector<std::vector<int>> getVertexNeighbors();

std::vector<std::vector<int>> removeDuplicates(std::vector<std::vector<int>> vec);

std::vector<std::vector<int>> getAllVertexNeighbors();

void solveLaplacianMesh(GLMmodel* pModel, std::vector<std::vector<double>> anchors, std::vector<int> anchorsIdx);

MatrixXd getLaplacianMatrixUmbrella(GLMmodel* pModel, std::vector<int> anchorsIdx);

void leastSquare(MatrixXd lapMat, MatrixXd delta, OUT cv::Mat_<double> resultVertex);

std::vector<std::vector<double>> anchorsForLaplacian(std::vector<double> posX, std::vector<double> poxY, std::vector<double>posZ);

void calculateLTL(MatrixXd lapMat);

int LoadGLTextures();
