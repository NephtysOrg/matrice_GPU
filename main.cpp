#include "qclcontext.h"
#include "qclprogram.h"
#include "qclkernel.h"
#include "qclcommandqueue.h"
#include "qcldevice.h"
#include "iostream"
#include <math.h>
using namespace std;
#define GPU 0
#define CPU 1

const int leafsize = 1024;
QCLVector<int> inbuffer_A;
QCLVector<int> inbuffer_B;
QCLVector<int> outbuffer;

QCLKernel kernel;
QCLContext context;
QCLProgram program;

int mode = CPU;

void multiply(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > &C, int n);
vector<int> matrixPlane(vector< vector<int> > A,bool col);
void CPUMult(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > &C, int n);
void GPUMult(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > &C, int n);
void strassen(vector< vector<int> > &A, vector< vector<int> > &B, vector< vector<int> > &C, unsigned int tam);
unsigned int nextPowerOfTwo(int n);
void strassenCPU(vector< vector<int> > &A, vector< vector<int> > &B, vector< vector<int> > &C, int tam);
void strassenGPU(vector< vector<int> > &A, vector< vector<int> > &B, vector< vector<int> > &C, int tam);
void sum(vector< vector<int> > &A, vector< vector<int> > &B, vector< vector<int> > &C, int tam);
void subtract(vector< vector<int> > &A, vector< vector<int> > &B, vector< vector<int> > &C, int tam);
void printMatrix(vector< vector<int> > matrix);




vector<int> matrixPlane(vector< vector<int> > A,bool col){
    vector<int> result;
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[i].size(); ++j) {
            if(col){
                result.push_back(A[j][i]);
            }else{
                result.push_back(A[i][j]);
            }

        }
    }
    return result;
}




void CPUMult(vector< vector<int> > A,
             vector< vector<int> > B,
             vector< vector<int> > &C, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void GPUMult(vector< vector<int> > A,
             vector< vector<int> > B,
             vector< vector<int> > &C, int n) {
    vector<int> A_plane = matrixPlane(A,false);
    vector<int> B_plane = matrixPlane(B,true);
    vector<int> outdata(A_plane.size());


    inbuffer_A=context.createVector<int>(n*n,QCLMemoryObject::ReadOnly);
    inbuffer_B=context.createVector<int>(n*n,QCLMemoryObject::ReadOnly);
    outbuffer=context.createVector<int>(n*n,QCLMemoryObject::WriteOnly);

    program=context.buildProgramFromSourceFile("multiplication.cl");
    kernel=program.createKernel("multiplication");
    kernel.setGlobalWorkSize(n,n);
    kernel.setArg(0,outbuffer);
    kernel.setArg(1,inbuffer_A);
    kernel.setArg(2,inbuffer_B);
    kernel.setArg(3,n);

    inbuffer_A.write(&A_plane[0],A_plane.size());
    inbuffer_B.write(&B_plane[0],B_plane.size());

    kernel.run();

    outbuffer.read(&outdata[0],A_plane.size());


    for (int i = 0,pas = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j,pas++) {
            C[i][j] += outdata[pas];
        }
    }

    inbuffer_A.release();
    inbuffer_B.release();
    outbuffer.release();
}


void strassenGPU(vector< vector<int> > &A,
                 vector< vector<int> > &B,
                 vector< vector<int> > &C, int tam) {
    if (tam <= leafsize) {
        GPUMult(A, B, C, tam);
        return;
    }

    // other cases are treated here:
    else {
        int newTam = tam/2;
        vector<int> inner (newTam,0);
        vector< vector<int> >
                a11(newTam,inner), a12(newTam,inner), a21(newTam,inner), a22(newTam,inner),
                b11(newTam,inner), b12(newTam,inner), b21(newTam,inner), b22(newTam,inner),
                c11(newTam,inner), c12(newTam,inner), c21(newTam,inner), c22(newTam,inner),
                p1(newTam,inner), p2(newTam,inner), p3(newTam,inner), p4(newTam,inner),
                p5(newTam,inner), p6(newTam,inner), p7(newTam,inner),
                aResult(newTam,inner), bResult(newTam,inner);

        int i, j;

        //dividing the matrices in 4 sub-matrices:
        for (i = 0; i < newTam; i++) {
            for (j = 0; j < newTam; j++) {
                a11[i][j] = A[i][j];
                a12[i][j] = A[i][j + newTam];
                a21[i][j] = A[i + newTam][j];
                a22[i][j] = A[i + newTam][j + newTam];

                b11[i][j] = B[i][j];
                b12[i][j] = B[i][j + newTam];
                b21[i][j] = B[i + newTam][j];
                b22[i][j] = B[i + newTam][j + newTam];
            }
        }


        // Calculating p1 to p7:

        sum(a11, a22, aResult, newTam); // a11 + a22
        sum(b11, b22, bResult, newTam); // b11 + b22
        strassenGPU(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)

        sum(a21, a22, aResult, newTam); // a21 + a22
        strassenGPU(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)

        subtract(b12, b22, bResult, newTam); // b12 - b22
        strassenGPU(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)

        subtract(b21, b11, bResult, newTam); // b21 - b11
        strassenGPU(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)

        sum(a11, a12, aResult, newTam); // a11 + a12
        strassenGPU(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)

        subtract(a21, a11, aResult, newTam); // a21 - a11
        sum(b11, b12, bResult, newTam); // b11 + b12
        strassenGPU(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)

        subtract(a12, a22, aResult, newTam); // a12 - a22
        sum(b21, b22, bResult, newTam); // b21 + b22
        strassenGPU(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)

        // calculating c21, c21, c11 e c22:

        sum(p3, p5, c12, newTam); // c12 = p3 + p5
        sum(p2, p4, c21, newTam); // c21 = p2 + p4

        sum(p1, p4, aResult, newTam); // p1 + p4
        sum(aResult, p7, bResult, newTam); // p1 + p4 + p7
        subtract(bResult, p5, c11, newTam); // c11 = p1 + p4 - p5 + p7

        sum(p1, p3, aResult, newTam); // p1 + p3
        sum(aResult, p6, bResult, newTam); // p1 + p3 + p6
        subtract(bResult, p2, c22, newTam); // c22 = p1 + p3 - p2 + p6

        // Grouping the results obtained in a single matrix:
        for (i = 0; i < newTam ; i++) {
            for (j = 0 ; j < newTam ; j++) {
                C[i][j] = c11[i][j];
                C[i][j + newTam] = c12[i][j];
                C[i + newTam][j] = c21[i][j];
                C[i + newTam][j + newTam] = c22[i][j];
            }
        }
    }
}


void strassenCPU(vector< vector<int> > &A,
                 vector< vector<int> > &B,
                 vector< vector<int> > &C, int tam) {
    if (tam <= leafsize) {
        CPUMult(A, B, C, tam);
        return;
    }

    // other cases are treated here:
    else {
        int newTam = tam/2;
        vector<int> inner (newTam,0);
        vector< vector<int> >
                a11(newTam,inner), a12(newTam,inner), a21(newTam,inner), a22(newTam,inner),
                b11(newTam,inner), b12(newTam,inner), b21(newTam,inner), b22(newTam,inner),
                c11(newTam,inner), c12(newTam,inner), c21(newTam,inner), c22(newTam,inner),
                p1(newTam,inner), p2(newTam,inner), p3(newTam,inner), p4(newTam,inner),
                p5(newTam,inner), p6(newTam,inner), p7(newTam,inner),
                aResult(newTam,inner), bResult(newTam,inner);

        int i, j;

        //dividing the matrices in 4 sub-matrices:
        for (i = 0; i < newTam; i++) {
            for (j = 0; j < newTam; j++) {
                a11[i][j] = A[i][j];
                a12[i][j] = A[i][j + newTam];
                a21[i][j] = A[i + newTam][j];
                a22[i][j] = A[i + newTam][j + newTam];

                b11[i][j] = B[i][j];
                b12[i][j] = B[i][j + newTam];
                b21[i][j] = B[i + newTam][j];
                b22[i][j] = B[i + newTam][j + newTam];
            }
        }


        // Calculating p1 to p7:

        sum(a11, a22, aResult, newTam); // a11 + a22
        sum(b11, b22, bResult, newTam); // b11 + b22
        strassenCPU(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)

        sum(a21, a22, aResult, newTam); // a21 + a22
        strassenCPU(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)

        subtract(b12, b22, bResult, newTam); // b12 - b22
        strassenCPU(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)

        subtract(b21, b11, bResult, newTam); // b21 - b11
        strassenCPU(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)

        sum(a11, a12, aResult, newTam); // a11 + a12
        strassenCPU(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)

        subtract(a21, a11, aResult, newTam); // a21 - a11
        sum(b11, b12, bResult, newTam); // b11 + b12
        strassenCPU(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)

        subtract(a12, a22, aResult, newTam); // a12 - a22
        sum(b21, b22, bResult, newTam); // b21 + b22
        strassenCPU(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)

        // calculating c21, c21, c11 e c22:

        sum(p3, p5, c12, newTam); // c12 = p3 + p5
        sum(p2, p4, c21, newTam); // c21 = p2 + p4

        sum(p1, p4, aResult, newTam); // p1 + p4
        sum(aResult, p7, bResult, newTam); // p1 + p4 + p7
        subtract(bResult, p5, c11, newTam); // c11 = p1 + p4 - p5 + p7

        sum(p1, p3, aResult, newTam); // p1 + p3
        sum(aResult, p6, bResult, newTam); // p1 + p3 + p6
        subtract(bResult, p2, c22, newTam); // c22 = p1 + p3 - p2 + p6

        // Grouping the results obtained in a single matrix:
        for (i = 0; i < newTam ; i++) {
            for (j = 0 ; j < newTam ; j++) {
                C[i][j] = c11[i][j];
                C[i][j + newTam] = c12[i][j];
                C[i + newTam][j] = c21[i][j];
                C[i + newTam][j + newTam] = c22[i][j];
            }
        }
    }
}

unsigned int nextPowerOfTwo(int n) {
    return pow(2, int(ceil(log2(n))));
}

void strassen(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, unsigned int n) {
    unsigned int m = nextPowerOfTwo(n);
    vector<int> inner(m,0);
    vector< vector<int> > APrep(m, inner), BPrep(m, inner), CPrep(m, inner);

    for(unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            APrep[i][j] = A[i][j];
            BPrep[i][j] = B[i][j];
        }
    }
    if(mode == GPU){
        strassenGPU(APrep, BPrep, CPrep, m);
    }else{
        strassenCPU(APrep, BPrep, CPrep, m);
    }

    for(unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            C[i][j] = CPrep[i][j];
        }
    }
}

void sum(vector< vector<int> > &A,
         vector< vector<int> > &B,
         vector< vector<int> > &C, int tam) {
    int i, j;

    for (i = 0; i < tam; i++) {
        for (j = 0; j < tam; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void subtract(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, int tam) {
    int i, j;

    for (i = 0; i < tam; i++) {
        for (j = 0; j < tam; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}



void printMatrix(vector< vector<int> > matrix) {
    for (int i=0; i < matrix.size(); i++) {
        for (int j=0; j < matrix[i].size(); j++) {
            if (j != 0) {
                cout << " ";
            }
            cout << matrix[i][j];
        }
        cout << endl;
    }
}

using namespace std;

int main(int argc, char *argv[])
{
    int TAILLE=10;
    if(argv[2] != NULL)
        TAILLE=(atoi(argv[2])>0)?atoi(argv[2]):10;
    if(argv[1] != NULL){
        mode = (strcmp(argv[1],"-cpu") == 0)?CPU:mode;
        mode = (strcmp(argv[1],"-gpu") == 0)?GPU:mode;
    }else{
        cout<<"Usage : ./multiplication [-gpu|-cpu] n"<<endl;
        return 0;
    }

    vector<int> inner (TAILLE,0);
    vector< vector<int> > A(TAILLE, inner), B(TAILLE, inner), C(TAILLE, inner);
    for (int i = 0; i < TAILLE; ++i) {
        for (int j = 0; j < TAILLE; ++j) {
            A[i][j] = B[i][j] = 1;
        }
    }
    if(!context.create()){
        qFatal("Could not create OpenCL context for the GPU\n");
        exit(0);
    }

    if(mode == CPU){
        cout<<"Mode : CPU";
    }else{
        cout<<"Mode : GPU";
    }
    cout<<endl;
    cout<<"Taille : "<<TAILLE<<endl;
    clock_t tStart = clock();
    strassen(A, B, C, TAILLE);
    cout<<"time :"<<(clock() - tStart)/(double)(CLOCKS_PER_SEC)<<" ms"<<endl;

    //if(TAILLE <= 5){
        cout<<"A"<<endl;
        printMatrix(A);
        cout<<"B"<<endl;
        printMatrix(B);
        cout<<"C"<<endl;
        printMatrix(C);
    //}
}



