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


void strassen(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, unsigned int tam);
int nextPowerOfTwo(int n);
void strassenR(vector< vector<int> > &A,
               vector< vector<int> > &B,
               vector< vector<int> > &C,
               int tam);
void sum(vector< vector<int> > &A,
         vector< vector<int> > &B,
         vector< vector<int> > &C, int tam);
void subtract(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, int tam);

void printMatrix(vector< vector<int> > matrix);
void printArray(vector<int> matrix);

//void ikjalgorithm(vector< vector<int> > A,
//                  vector< vector<int> > B,
//                  vector< vector<int> > &C, int n) {

//    for (int i = 0; i < n; i++) {
//        for (int k = 0; k < n; k++) {
//            for (int j = 0; j < n; j++) {
//                C[i][j] += A[i][k] * B[k][j];
//            }
//        }
//    }
//}


vector<int> matrixPlane(vector< vector<int> > A){
    cout<<"->matrixPlane()"<<endl;
    vector<int> result;
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[i].size(); ++j) {
            result.push_back(A[i][j]);
        }
    }
    cout<<"<-matrixPlane()"<<endl;
    return result;
}

void gpuMultiplication(vector< vector<int> > A,
                  vector< vector<int> > B,
                  vector< vector<int> > &C, int n) {
    cout<<"->ikjalgorithm()"<<endl;
    //mise à plat

    vector<int> A_plane = matrixPlane(A);
    vector<int> B_plane = matrixPlane(B);

    cout<<"Taille a plat de A : "<<A_plane.size()<<endl;
    cout<<"Taille a plat de B : "<<B_plane.size()<<endl;

    int* indata_A = new int[A_plane.size()];
    int* indata_B = new int[B_plane.size()];

    for (int i = 0; i < A_plane.size(); ++i) {
            cout<<"i="<<i<<endl;
            indata_A[i] = A_plane[i];
            indata_B[i] = B_plane[i];
    }

    int* outdata = new int[A_plane.size()];

    inbuffer_A.write(indata_A,A_plane.size());
    inbuffer_B.write(indata_B,B_plane.size());

    kernel.run();
    context.sync();

    outbuffer.read(outdata,A_plane.size());

    for (int i = 0,pas = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j,pas++) {
            C[i][j] = outdata[pas];
        }
    }
    delete[] indata_A;
    delete[] indata_B;
    delete[] outdata;
    cout<<"->ikjalgorithm()"<<endl;
}

void strassenR(vector< vector<int> > &A,
               vector< vector<int> > &B,
               vector< vector<int> > &C, int tam) {
    cout<<"->strassenR()"<<endl;
    if (tam <= leafsize) {
        gpuMultiplication(A, B, C, tam);
        cout<<"<-strassenR()"<<endl;
        return;
    }

    // other cases are treated here:
    else {
        int newTam = tam/2;
        vector<int> inner (newTam);
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
        strassenR(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)

        sum(a21, a22, aResult, newTam); // a21 + a22
        strassenR(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)

        subtract(b12, b22, bResult, newTam); // b12 - b22
        strassenR(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)

        subtract(b21, b11, bResult, newTam); // b21 - b11
        strassenR(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)

        sum(a11, a12, aResult, newTam); // a11 + a12
        strassenR(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)

        subtract(a21, a11, aResult, newTam); // a21 - a11
        sum(b11, b12, bResult, newTam); // b11 + b12
        strassenR(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)

        subtract(a12, a22, aResult, newTam); // a12 - a22
        sum(b21, b22, bResult, newTam); // b21 + b22
        strassenR(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)

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
    cout<<"<-strassenR()"<<endl;
}

int nextPowerOfTwo(int n) {
    return pow(2, int(ceil(log2(n))));
}

void strassen(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, unsigned int n) {
    cout<<"->strassen()"<<endl;
    unsigned int m = nextPowerOfTwo(n);
    vector<int> inner(m,0);
    vector< vector<int> > APrep(m, inner), BPrep(m, inner), CPrep(m, inner);

    for(unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            APrep[i][j] = A[i][j];
            BPrep[i][j] = B[i][j];
        }
    }

    strassenR(APrep, BPrep, CPrep, m);
    cout<<"Sous strassR"<<endl;
    for(unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            C[i][j] = CPrep[i][j];
        }
    }
    cout<<"<-strassen()"<<endl;
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
                cout << "\t";
            }
            cout << matrix[i][j];
        }
        cout << endl;
    }
}

void printArray(vector<int> matrix) {
    for (int i=0; i < matrix.size(); i++) {
        cout << matrix[i]<< " ,";

    }
    cout << endl;
}




using namespace std;
int main(int argc, char *argv[])
{
    // Declarations
    int work_size = 1024;
    cout <<"Worksize : "<< work_size<<endl;

    int TAILLE=10;
    int mode = CPU;
    if(argv[2] != NULL)
        TAILLE=(atoi(argv[2])>0)?atoi(argv[2]):10;
    if(argv[1] != NULL){
        mode = (strcmp(argv[1],"-cpu") == 0)?CPU:mode;
        mode = (strcmp(argv[1],"-gpu") == 0)?GPU:mode;
    }

    int **A = new int*[TAILLE];
    int **B = new int*[TAILLE];
    int **C = new int*[TAILLE];
    for (int i = 0; i < TAILLE; ++i) {
        A[i] = new int[TAILLE];
        B[i] = new int[TAILLE];
        C[i] = new int[TAILLE];
        for (int j = 0; j < TAILLE; ++j) {
            A[i][j] = B[i][j]= 1 ;
        }
    }
    if(mode == CPU){
        cout<<"CPU mode"<<endl;
        clock_t tStart = clock();
        for (int i = 0; i < TAILLE; i++){
            for (int j=0; j < TAILLE; j++){
                C[i][j]=0;
                for (int k = 0; k < TAILLE; k++){
                    C[i][j] += A[i][k]*B[k][j];
                }
            }
        }
        cout<<"CPU time :"<<(clock() - tStart)/(double)(CLOCKS_PER_SEC/1000)<<" ms"<<endl;
    }else{
        if(!context.create()){
            qFatal("Could not create OpenCL context for the GPU\n");
            exit(0);
        }
        int n = TAILLE;
        vector<int> inner (n);
        vector< vector<int> > A(n, inner), B(n, inner), C(n, inner);
        for (int i = 0; i < TAILLE; ++i) {
            for (int j = 0; j < TAILLE; ++j) {
                A[i][j] = B[i][j] = 1;
            }
        }
        if(TAILLE != nextPowerOfTwo(TAILLE)){
            TAILLE = nextPowerOfTwo(TAILLE);
        }
        inbuffer_A=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::ReadOnly);
        inbuffer_B=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::ReadOnly);
        outbuffer=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::WriteOnly);
        program=context.buildProgramFromSourceFile("multiplication.cl");
        kernel=program.createKernel("multiplication");
        kernel.setGlobalWorkSize(TAILLE,TAILLE);
        kernel.setArg(0,outbuffer);
        kernel.setArg(1,inbuffer_A);
        kernel.setArg(2,inbuffer_B);
        kernel.setArg(3,TAILLE);
        cout<<"GPU mode"<<endl;


        strassen(A, B, C, n);
        printMatrix(C);

    }
    if(TAILLE <= 5){

    }

}



