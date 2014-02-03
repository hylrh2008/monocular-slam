#ifndef CONFIG_HANDLER_H
#define CONFIG_HANDLER_H
#include <boost/program_options.hpp>
#include <string>
#include <vector>
namespace po = boost::program_options;

class config_handler
{
public:
    config_handler(int argc, char *argv[]);
    po::options_description config;
    po::variables_map vm;
};

#endif // CONFIG_HANDLER_H
