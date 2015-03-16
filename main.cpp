#include "qclcontext.h"
#include "qclprogram.h"
#include "qclkernel.h"

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
    int nbKernels = 10;
    int mode =  CPU;
    if(argv[2] != NULL)
        TAILLE=(atoi(argv[2])>0)?atoi(argv[2]):10;

    if(argv[1] != NULL)
        mode = (strcmp(argv[1],"-cpu") == 0)?CPU:GPU;

    if(!context.create()){
        qFatal("Could not create OpenCL context for the GPU\n");
        exit(0);
    }

    QCLVector<int>  inbuffer=context.createVector<int>(nbKernels,QCLMemoryObject::ReadOnly);
    QCLVector<int>  outbuffer=context.createVector<int>(nbKernels,QCLMemoryObject::WriteOnly);
    program=context.buildProgramFromSourceFile("multiplication.cl");
    kernel=program.createKernel("multiplication");
    kernel.setGlobalWorkSize(nbKernels);
    kernel.setArg(0,outbuffer);
    kernel.setArg(1,inbuffer);
    kernel.setArg(2,nbKernels);

    srand(time(NULL));

    int A[TAILLE][TAILLE];
    int B[TAILLE][TAILLE];
    int C[TAILLE][TAILLE];
    for (int i = 0; i < TAILLE; ++i) {
        for (int j = 0; j < TAILLE; ++j) {
            A[i][j] = B[i][j]= 1 ;
           //(rand() % TAILLE);
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


        cout<<"GPU mode"<<endl;
//        inbuffer.write(indata,nbKernels);
//        kernel.run();
//        outbuffer.read(outdata,nbKernels);

    }


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

