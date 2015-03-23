#include "qclcontext.h"
#include "qclprogram.h"
#include "qclkernel.h"
#include "qclcommandqueue.h"
#include "qcldevice.h"
#include "iostream"
#define GPU 0
#define CPU 1
using namespace std;
int main(int argc, char *argv[])
{
    // Declarations
    int work_size = CL_DEVICE_MAX_PARAMETER_SIZE/2;
    cout <<"Worksize : "<< work_size<<endl;
    QCLContext context;
    QCLProgram program;
    QCLKernel kernel;
    int TAILLE=10;
    int mode = CPU;
    if(argv[2] != NULL)
        TAILLE=(atoi(argv[2])>0)?atoi(argv[2]):10;
    if(argv[1] != NULL){
        mode = (strcmp(argv[1],"-cpu") == 0)?CPU:mode;
        mode = (strcmp(argv[1],"-gpu") == 0)?GPU:mode;
    }
    srand(time(NULL));
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
        QCLVector<int> inbuffer_A=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::ReadOnly);
        QCLVector<int> inbuffer_B=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::ReadOnly);
        QCLVector<int> outbuffer=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::WriteOnly);
        program=context.buildProgramFromSourceFile("multiplication.cl");
        kernel=program.createKernel("multiplication");
        kernel.setGlobalWorkSize(TAILLE,TAILLE);
        kernel.setArg(0,outbuffer);
        kernel.setArg(1,inbuffer_A);
        kernel.setArg(2,inbuffer_B);
        kernel.setArg(3,TAILLE);
        int* indata_A = new int[TAILLE*TAILLE];
        int* indata_B = new int[TAILLE*TAILLE];
        int* outdata = new int[TAILLE*TAILLE];
        // Mise à plat
        int pas= 0;
        for (int i = 0; i < TAILLE; ++i) {
            for (int j = 0; j < TAILLE; ++j) {
                indata_A[pas] = A[i][j];
                indata_B[pas] = B[j][i];
                pas ++;
            }
        }
        cout<<"GPU mode"<<endl;
        inbuffer_A.write(indata_A,TAILLE*TAILLE);
        inbuffer_B.write(indata_B,TAILLE*TAILLE);
        clock_t tStart = clock();
        kernel.run();
        outbuffer.read(outdata,TAILLE*TAILLE);
        cout<<"GPU time :"<<(clock() - tStart)/(double)(CLOCKS_PER_SEC/1000)<<" ms"<<endl;
        for (int i = 0,pas = 0; i < TAILLE; ++i) {
            for (int j = 0; j < TAILLE; ++j,pas++) {
                C[i][j] = outdata[pas];
            }
        }
        delete[] indata_A;
        delete[] indata_B;
        delete[] outdata;
    }
    if(TAILLE <= 5){
        cout<<"Matrice A"<<endl;
        for (int i=0; i<TAILLE; i++){
            for (int j = 0; j < TAILLE; j++){
                cout<<A[i][j]<<" ";
            }
            cout<<endl;
        }
        cout<<"Matrice B"<<endl;
        for (int i=0; i<TAILLE; i++){
            for (int j = 0; j < TAILLE; j++){
                cout<<B[i][j]<<" ";
            }
            cout<<endl;
        }
        cout<<"Matrice C"<<endl;
        for (int i=0; i<TAILLE; i++){
            for (int j = 0; j < TAILLE; j++){
                cout<<C[i][j]<<" ";
            }
            cout<<endl;
        }
    }
    for (int i = 0; i < TAILLE; ++i) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;
}
