#include "config_handler.h"
#include <fstream>
#include <iostream>

namespace po = boost::program_options;
using namespace std;

config_handler::config_handler(int ac,char* av[]):
    config("Configuration")

{
    po::options_description config("Configuration");
    string config_filename="./config";

    //Add program option here.
    config.add_options()
        ("-c", po::value<string>(&config_filename)->default_value("./config"),
                  "Where is config file")
        ("rgb-data-root", po::value<string>()->default_value("./dataset"),
                  "Root of rgb data folder")
        ("first-depth-map", po::value<string>()->default_value("./dataset/first_depth.png"),
                  "First Depth Map for initialisation")
        ;
    po::store(po::parse_command_line(ac, av, config), vm);
    po::notify(vm);
    ifstream f(config_filename.c_str());
    if(!f.fail()){
        po::store(po::parse_config_file<char>(config_filename.c_str(),config),vm);
        po::notify(vm);
    }
    else {
        std::cerr<<"File \""<<config_filename<<"\" not found. Default config used."<<endl;
    }
    po::store(po::parse_command_line(ac, av, config), vm);
    po::notify(vm);
}
