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
    QCLContext context;
    QCLProgram program;
    QCLKernel kernel;


    int TAILLE=10;
    int mode =  CPU;
    if(argv[2] != NULL)
        TAILLE=(atoi(argv[2])>0)?atoi(argv[2]):10;

    if(argv[1] != NULL){
        mode = (strcmp(argv[1],"-cpu") == 0)?CPU:mode;
        mode = (strcmp(argv[1],"-gpu") == 0)?GPU:mode;
    }
    srand(time(NULL));

    int A[TAILLE][TAILLE];
    int B[TAILLE][TAILLE];
    int C[TAILLE][TAILLE];
    for (int i = 0; i < TAILLE; ++i) {
        for (int j = 0; j < TAILLE; ++j) {
            A[i][j] = B[i][j]= 1 ;
        }
    }
    if(mode == CPU){
        cout<<"CPU mode"<<endl;
        for (int i = 0; i < TAILLE; i++){
            for (int j=0; j < TAILLE; j++){
                C[i][j]=0;
                for (int k = 0; k < TAILLE; k++){
                    C[i][j] += A[i][k]*B[k][j];
                }
            }
        }
    }else{
        if(!context.create()){
            qFatal("Could not create OpenCL context for the GPU\n");
            exit(0);
        }

        QCLVector<int>  inbuffer_A=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::ReadOnly);
        QCLVector<int>  inbuffer_B=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::ReadOnly);
        QCLVector<int>  outbuffer=context.createVector<int>(TAILLE*TAILLE,QCLMemoryObject::WriteOnly);

        program=context.buildProgramFromSourceFile("multiplication.cl");
        kernel=program.createKernel("multiplication");
        kernel.setGlobalWorkSize(TAILLE,TAILLE);
        kernel.setArg(0,outbuffer);
        kernel.setArg(1,inbuffer_A);
        kernel.setArg(2,inbuffer_B);
        kernel.setArg(3,TAILLE);

        int indata_A[TAILLE*TAILLE];
        int indata_B[TAILLE*TAILLE];
        int outdata[TAILLE*TAILLE];

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
        kernel.run();
        outbuffer.read(outdata,TAILLE*TAILLE);
        for (int i = 0; i < TAILLE*TAILLE; ++i) {
            cout<<outdata[i]<<" ";
        }
    }


//    cout<<"Matrice A"<<endl;
//    for (int i=0; i<TAILLE; i++){
//        for (int j = 0; j < TAILLE; j++){
//            cout<<A[i][j]<<" ";
//        }
//        cout<<endl;
//    }

//    cout<<"Matrice B"<<endl;
//    for (int i=0; i<TAILLE; i++){
//        for (int j = 0; j < TAILLE; j++){
//            cout<<B[i][j]<<" ";
//        }
//        cout<<endl;
//    }

//    cout<<"Matrice C"<<endl;
//    for (int i=0; i<TAILLE; i++){
//        for (int j = 0; j < TAILLE; j++){
//            cout<<C[i][j]<<" ";
//        }
//        cout<<endl;
//    }
}

