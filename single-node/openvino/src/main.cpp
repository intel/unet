#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <iomanip>

#include <inference_engine.hpp>
#include "../include/brainunetopenvino.h"


using namespace InferenceEngine;
using namespace std;
int IMAGE_FILE_INDEX = 60;
InferenceEngine::TargetDevice PLUGIN_NAME = TargetDevice::eCPU;
int image_file_index;
InferenceEngine::TargetDevice plugin_name;


void printUsage(const string &app_name = string())
{
    cout << endl << "To run application in default configuration use:" << endl;
    if (!app_name.empty())
        cout << app_name << " ";
    cout << "-p" << endl << endl;
    cout << "Usage:" << endl;
    if (!app_name.empty())
        cout << app_name << endl;
    cout << "     [-h] [--h] [-help] [--help] : display this usage page. No other commands accepted." << endl;
    cout << "     [-p MYRIAD | myriad | CPU | cpu] : select plugin (defaults is CPU)" << endl;
    cout << "     [-f filename] : specify the image number to be detected and recognized\n" << endl;
}



// Parses the command line to find the plugin to use.
void parseArgs(int argc, char *argv[])

{

    try
    {
        /*if (argc < 2)
        {
            printUsage(argv[0]);
            throw "Insufficient number of command line arguments.";
        } // end if
         */
        // Specify the default file name in case users don't pass it
        image_file_index = IMAGE_FILE_INDEX;
        plugin_name = PLUGIN_NAME;

        // Start parsing out parameters
        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if (arg == "-h")	{ // users want to see help
                printUsage(argv[0]);
            }

            if (arg == "-f")
                        {
                            if (i + 1 < argc)
                            {
                                i++;
                                image_file_index = std::stoi(argv[i]);
                            }
                            else
                                throw i; // invalid parameter
                        } // end if (-f)

            if (arg == "-p")	{ // if users pass this option, mean they want test this particular interface

                if (i + 1 < argc)
                {
                    i++;
                    arg = argv[i];
                    if (arg == "myriad" || arg == "MYRIAD")
                       plugin_name = TargetDevice::eMYRIAD;
                    if (arg == "cpu" || arg == "CPU")
                       plugin_name = TargetDevice::eCPU;
                }
            }
        }
    }
    catch (...)
        {
            cout << "Unexpected ERROR parsing command line." << endl;
            cout << "Use parameter -help for a list of valid parameters." << endl;

            throw "ERROR parsing command line.";
        }
}



int main(int argc, char *argv[])
{
    std::cout<<"Starting program"<<std::endl;
    BrainUnetOpenVino brainunetobj;
    cnpy::NpyArray arr;
    cnpy::NpyArray arr_msks;

    try
       {
           parseArgs(argc, argv);  // request users for default
           brainunetobj.loadNumpyData(arr,arr_msks);
           brainunetobj.makeInference(image_file_index,plugin_name,arr,arr_msks);

      }
    catch (...)
               {
                  cout << "Unspecified ERROR occurred during execution." << endl;
               }

    return 0;

 }
