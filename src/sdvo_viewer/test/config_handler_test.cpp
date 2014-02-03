#include <gtest/gtest.h>

#include "config_handler.h"


int g_argc;
char ** g_argv;
TEST(config_handler, basic_config)
{
  config_handler(g_argc,g_argv);
  ASSERT_TRUE(true);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
