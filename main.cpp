#include "qclcontext.h"
#include "qclprogram.h"
#include "qclkernel.h"
#include "iostream"
using namespace std;
int main(int argc, char *argv[])
{    
    // Declarations
    QCLContext context;
    QCLProgram program;
    QCLKernel kernel;

    int nbKernels=10;

    if(argc==2){
        if(atoi(argv[1])>0){
            nbKernels=atoi(argv[1]);
        }
    }

    if(!context.create()){
        qFatal("Could not create OpenCL context for the GPU\n");
        exit(0);
    }

    QCLVector<int>  inbuffer=context.createVector<int>(nbKernels,QCLMemoryObject::ReadOnly);
    QCLVector<int>  outbuffer=context.createVector<int>(nbKernels,QCLMemoryObject::WriteOnly);
    program=context.buildProgramFromSourceFile("tribulle.cl");
    kernel=program.createKernel("tribulle");
    kernel.setGlobalWorkSize(nbKernels);
    kernel.setArg(0,outbuffer);
    kernel.setArg(1,inbuffer);
    kernel.setArg(2,nbKernels);

    int max=10*nbKernels;
    int indata[nbKernels];
    srand(time(NULL));

    for(int i=0; i<nbKernels; i++){
            indata[i]=(rand() % max);
    }
    int outdata[nbKernels];

    inbuffer.write(indata,nbKernels);
    kernel.run();
    outbuffer.read(outdata,nbKernels);

    for(int i=0; i<nbKernels; i++){
        cout<<"Resultat = "<<outdata[i]<<endl;
    }


}

